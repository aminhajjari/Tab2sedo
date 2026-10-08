import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset, random_split
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, transforms
import itertools
import argparse
import os
import json
from datetime import datetime
from kan_hybrid import KAN, HybridKAN
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import scipy.io.arff as arff
from tqdm import tqdm
#from adopt import ADOPT 

# ========== ARGUMENT PARSER ==========
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--data', type=str, required=True, 
                   help='Path to the dataset (csv/arff/data)')
parser.add_argument('--save_dir', type=str, required=False, default=None,
                   help='Directory to save results (optional, for compatibility)')
parser.add_argument('--num_images', type=int, default=20,
                   help='Number of sample images to save (default: 20)')

args = parser.parse_args()

# ========== PARAMETERS ==========
EPOCH = 50
BATCH_SIZE = 64
NUM_IMAGES_TO_SAVE = min(args.num_images, 20)  # Cap at 20

data_path = args.data
file_name = os.path.basename(os.path.dirname(data_path))


# ===== MANY-CLASS (>20) VERSION: MedMNIST images only, own output folder =====
# (separate from main.py: no FashionMNIST / MNIST, no old Image folder, no old output folder)
MEDMNIST_ROOT = os.environ.get("MEDMNIST_ROOT", "/home/gkianfar/scratch/Amin/Sedo/MedMNIST")
OUTPUT_ROOT   = os.environ.get("OUTPUT_ROOT", "/home/gkianfar/scratch/Amin/Sedo/output_manyclass/medmnist")
POOL_LIMIT    = int(os.environ.get("POOL_LIMIT", 0))         # 0 = all MedMNIST classes as distinct images; 20 = ablation with 20 distinct images
MAX_FEATURES  = int(os.environ.get("MAX_FEATURES", 1100))    # keep top-variance columns above this (amazon has ~10k)
POOL_NAME     = "medmnist" if POOL_LIMIT == 0 else f"medmnist_limit{POOL_LIMIT}"
# per-dataset switches (keyed by the dataset FOLDER name)
CAP_PER_CLASS = {"dionis": 100, "aloi": 54}                  # stratified row cap per class; NO_CAP=1 disables
DROP_COLS_BY_DATASET = {"Bach_Choral_Harmony": [0, 1]}       # id / event-number columns (only if the file has no header)
# MedMNIST sets are appended in this order (classes numbered from 0); loaded only as far as needed
MEDMNIST_ORDER = ["pathmnist", "bloodmnist", "dermamnist", "retinamnist",
                  "octmnist", "tissuemnist", "breastmnist", "pneumoniamnist"]
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"[CONFIG] pool={POOL_NAME}  MEDMNIST_ROOT={MEDMNIST_ROOT}  OUTPUT_ROOT={OUTPUT_ROOT}  MAX_FEATURES={MAX_FEATURES}")
#CLIP_MODEL_PATH = "/home/gkianfar/scratch/Amin/ICC/models/ViT-B-32.pt"


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"\n{'='*70}")
print(f"TABLE2IMAGE - Starting Experiment")
print(f"{'='*70}")
print(f"Dataset: {file_name}")
print(f"Device: {DEVICE}")
print(f"Images to save: {NUM_IMAGES_TO_SAVE}")
print(f"{'='*70}\n")

# ========== DATA LOADING FUNCTION ==========
def load_dataset(file_path):
    """Auto-detect file format and load dataset"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        print(f"[INFO] Loading CSV file: {file_path}")
        return pd.read_csv(file_path)
    
    elif file_ext == '.arff':
        print(f"[INFO] Loading ARFF file: {file_path}")
        try:
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            print(f"[INFO] ARFF attributes: {list(meta.names())[:10]}...")
            return df, meta  # Return metadata too
        except Exception as e:
            print(f"[WARNING] scipy.io.arff failed: {e}")
            try:
                import arff as arff_lib
                with open(file_path, 'r') as f:
                    dataset = arff_lib.load(f)
                df = pd.DataFrame(dataset['data'], 
                                columns=[attr[0] for attr in dataset['attributes']])
                return df, None  # No metadata from backup parser
            except Exception as e2:
                raise Exception(f"All ARFF parsers failed. Errors: (1) {e}, (2) {e2}")
    
    elif file_ext == '.data':
        print(f"[INFO] Loading .data file: {file_path}")
        for sep in [',', ' ', '\t', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None)
                if df.shape[1] > 1:
                    print(f"[INFO] Detected delimiter: '{sep}'")
                    return df, None
            except:
                continue
        raise Exception("Could not determine delimiter for .data file")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


print(f"[INFO] Loading dataset: {data_path}")

# Load dataset (with metadata if ARFF)
file_ext = os.path.splitext(data_path)[1].lower()
if file_ext == '.arff':
    df, arff_meta = load_dataset(data_path)
else:
    df = load_dataset(data_path)
    arff_meta = None

if df.empty:
    raise ValueError("Dataset is empty after loading")
if df.shape[1] < 2:
    raise ValueError(f"Dataset has only {df.shape[1]} column(s), need at least 2")

print(f"[INFO] Initial dataset shape: {df.shape}")
print(f"[INFO] Columns: {df.columns.tolist()[:10]}...")

# Handle missing values
missing_markers = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']
df = df.replace(missing_markers, np.nan)
initial_missing = df.isnull().sum().sum()
print(f"[INFO] Initial missing values: {initial_missing}")

_drop = [c for c in DROP_COLS_BY_DATASET.get(file_name, []) if c in df.columns]
if _drop:
    print(f"[INFO] {file_name}: dropping identifier columns {_drop}")
    df = df.drop(columns=_drop)
elif file_name in DROP_COLS_BY_DATASET:
    print(f"[WARNING] {file_name}: expected headerless columns {DROP_COLS_BY_DATASET[file_name]} not found, nothing dropped")

# ========== IMPROVED TARGET COLUMN DETECTION ==========
target_col = None

# Strategy 1: For ARFF files, use metadata to identify target
if arff_meta is not None:
    print("[INFO] Detecting target column from ARFF metadata...")
    try:
        attr_names = list(arff_meta.names())
        # ARFF convention: last attribute is typically the class/target
        target_col = attr_names[-1]
        print(f"[INFO] ARFF metadata indicates target: '{target_col}'")
        
        # Verify this column exists in dataframe
        if target_col not in df.columns:
            print(f"[WARNING] Metadata target '{target_col}' not found in dataframe. Falling back...")
            target_col = None
    except Exception as e:
        print(f"[WARNING] Could not read ARFF metadata: {e}")
        target_col = None

# Strategy 2: Search for known target column names
if target_col is None:
    target_col_candidates = [
        'target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target',
        'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology',
        'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival',
        'diagnosis', 'y', 'Author', 'Utility'
    ]
    target_col = next((col for col in df.columns if col in target_col_candidates), None)
    if target_col:
        print(f"[INFO] Found target column by name: '{target_col}'")

# Strategy 3: Use last column as fallback
if target_col is None:
    target_col = df.columns[-1]
    if all(isinstance(col, int) for col in df.columns):
        print(f"[INFO] Using last column (index {target_col}) as target.")
    else:
        print(f"[INFO] Using last column '{target_col}' as target.")

print(f"[INFO] Target column: {target_col}")

# ========== EARLY CLASS DISTRIBUTION CHECK ==========
print(f"\n[INFO] Checking class distribution before preprocessing...")
if target_col in df.columns:
    # Show raw distribution
    target_value_counts = df[target_col].value_counts()
    print(f"[INFO] Raw class distribution:")
    for val, count in target_value_counts.items():
        print(f"  Class '{val}': {count} samples")
    
    # Check for classes with too few samples
    min_samples_per_class = 10
    rare_classes = target_value_counts[target_value_counts < min_samples_per_class]
    
    if len(rare_classes) > 0:
        print(f"\n[WARNING] Found {len(rare_classes)} class(es) with <{min_samples_per_class} samples:")
        for cls, count in rare_classes.items():
            print(f"  Class '{cls}': {count} samples")
        
        # Filter out rare classes
        valid_classes = target_value_counts[target_value_counts >= min_samples_per_class].index.tolist()
        
        if len(valid_classes) < 2:
            print(f"[ERROR] Only {len(valid_classes)} valid class(es) remain after filtering. Need at least 2.")
            print(f"[ERROR] Skipping dataset: insufficient samples per class.")
            exit(0)
        
        print(f"[INFO] Filtering dataset to keep only classes with >={min_samples_per_class} samples...")
        original_size = len(df)
        df = df[df[target_col].isin(valid_classes)]
        filtered_size = len(df)
        pct_removed = ((original_size - filtered_size) / original_size) * 100
        print(f"   ⚠️  WARNING: Removed {original_size - filtered_size} samples ({pct_removed:.1f}% of original data)")
        print(f"[INFO] New dataset shape: {df.shape}")
        
        # Show new distribution
        new_distribution = df[target_col].value_counts()
        print(f"[INFO] Filtered class distribution:")
        for val, count in new_distribution.items():
            print(f"  Class '{val}': {count} samples")
else:
    print(f"[ERROR] Target column '{target_col}' not found in dataframe!")
    exit(1)

_cap = CAP_PER_CLASS.get(file_name, 0)
if _cap and not os.environ.get("NO_CAP"):
    _before = len(df)
    df = df.sample(frac=1.0, random_state=42).groupby(target_col, sort=False).head(_cap).reset_index(drop=True)
    print(f"[INFO] {file_name}: stratified cap of {_cap} rows/class -> {_before} -> {len(df)} rows (set NO_CAP=1 to disable)")

missing_threshold = 0.5
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
if target_col in cols_to_drop:
    cols_to_drop.remove(target_col)
if cols_to_drop:
    print(f"[INFO] Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing data")
    df = df.drop(columns=cols_to_drop)
    print(f"[INFO] Shape after dropping: {df.shape}")

if df.shape[1] <= 1:
    raise ValueError("All feature columns were dropped. Dataset unusable.")

if df[target_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[target_col]):
    print(f"[INFO] Converting labels to integers...")
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))
    unique_values = le_target.classes_.tolist()
else:
    y = df[target_col].astype(int).values
    unique_values = sorted(set(y))

num_classes = len(unique_values)
print(f"[INFO] Detected {num_classes} unique classes: {unique_values}")

if num_classes < 2:
    raise ValueError(f"Dataset has only {num_classes} class. Need at least 2.")

X_df = df.drop(columns=[target_col])
print(f"[INFO] Encoding categorical features...")
for col in X_df.columns:
    if not pd.api.types.is_numeric_dtype(X_df[col]):
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
    else:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

if X_df.shape[1] == 0:
    raise ValueError(f"All features dropped for {file_name} — check dtype handling.")

print(f"[INFO] Imputing missing values with median...")
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X_df)
imputed_count = X_df.isnull().sum().sum()
if imputed_count > 0:
    print(f"[INFO] Imputed {imputed_count} missing values")

if X.shape[1] > MAX_FEATURES:
    _keep = np.sort(np.argsort(X.var(axis=0))[-MAX_FEATURES:])
    print(f"[INFO] {X.shape[1]} features > MAX_FEATURES={MAX_FEATURES}: keeping the {MAX_FEATURES} highest-variance columns (label-free)")
    X = X[:, _keep]

unique_values = sorted(set(y))
num_classes = len(unique_values)
value_map = {unique_values[i]: i for i in range(num_classes)}
y = np.array([value_map[val] for val in y])

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4

print(f"\n{'='*70}")
print(f"[SUMMARY] Preprocessed Data:")
print(f"  - Samples: {X.shape[0]}")
print(f"  - Features: {X.shape[1]}")
print(f"  - Classes: {num_classes}")
print(f"  - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  - Tab latent size: {tab_latent_size}")
print(f"{'='*70}\n")

print(f"[INFO] Image pool: MedMNIST 28x28 from {MEDMNIST_ROOT}")

print("[INFO] Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("[INFO] Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

train_tabular_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.long)
)
test_tabular_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.long)
)

print("[INFO] Calculating VIF values...")
def calculate_vif_safe(X_data):
    df_vif = pd.DataFrame(X_data)
    n_features = df_vif.shape[1]
    vif_values = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for i in range(n_features):
            try:
                vif = variance_inflation_factor(df_vif.values, i)
                if np.isnan(vif) or np.isinf(vif):
                    vif = 1.0
            except:
                vif = 1.0
            vif_values.append(vif)
    vif_values = np.array(vif_values)
    vif_values = np.clip(vif_values, 1.0, 100.0)
    return vif_values

X_sample = X_train[:min(1000, len(X_train))]
vif_values = calculate_vif_safe(X_sample)
print(f"[INFO] VIF calculated. Mean: {vif_values.mean():.2f}, Max: {vif_values.max():.2f}")

print("[INFO] Preparing synchronized image-tabular datasets...")
train_tabular_label_counts = torch.bincount(train_tabular_dataset.tensors[1], minlength=num_classes)
test_tabular_label_counts = torch.bincount(test_tabular_dataset.tensors[1], minlength=num_classes)
num_samples_needed = train_tabular_label_counts.tolist()
num_samples_needed_test = test_tabular_label_counts.tolist()
class ArrayImageDataset(Dataset):
    """28x28 uint8 images held in memory -> float tensor [1,28,28] in [0,1] (same as transforms.ToTensor())."""
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        img = torch.from_numpy(np.ascontiguousarray(self.images[i])).float().div_(255.0).unsqueeze(0)
        return img, int(self.labels[i])


def find_medmnist_npz(root, name):
    """Locate the plain 28x28 <name>.npz inside MEDMNIST_ROOT (folder layout is searched, not assumed)."""
    import glob
    for c in (os.path.join(root, name, name + ".npz"), os.path.join(root, name + ".npz")):
        if os.path.isfile(c):
            return c
    folder = os.path.join(root, name)
    found = sorted(glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True), key=os.path.getsize)
    exact = [f for f in found if os.path.basename(f) == name + ".npz"]
    if exact:
        return exact[0]
    plain = [f for f in found if not os.path.basename(f)[:-4].endswith(("_64", "_128", "_224"))]
    if plain:
        return plain[0]
    listing = os.listdir(folder) if os.path.isdir(folder) else "folder not found"
    raise FileNotFoundError(f"No {name}.npz under {folder}. Contents: {listing}")


def load_medmnist_train(path):
    d = np.load(path)
    imgs, labs = d["train_images"], d["train_labels"].reshape(-1).astype(np.int64)
    if imgs.ndim == 4:                                   # RGB (N,28,28,3) -> grayscale, in chunks to save RAM
        gray = np.empty(imgs.shape[:3], dtype=np.uint8)
        for s in range(0, len(imgs), 8192):
            c = imgs[s:s + 8192].astype(np.uint32)
            gray[s:s + 8192] = ((c[..., 0] * 299 + c[..., 1] * 587 + c[..., 2] * 114) // 1000).astype(np.uint8)
        imgs = gray
    assert imgs.shape[1:] == (28, 28), f"{path}: expected 28x28 images, got {imgs.shape[1:]} (wrong size file?)"
    return imgs, labs


def build_image_pool(n_classes):
    """MedMNIST image pool. Returns (combined_dataset, indices_by_label, distinct_image_classes, wrapped).
    Pool class ids are numbered from 0 in MEDMNIST_ORDER. Dataset class c uses pool class c; when the dataset has more
    classes than the pool offers, classes wrap (c % distinct) and share images (the log prints a warning).
    POOL_LIMIT > 0 caps the number of distinct image classes (ablation)."""
    need = min(n_classes, POOL_LIMIT) if POOL_LIMIT > 0 else n_classes
    parts, tgts, offset = [], [], 0
    for name in MEDMNIST_ORDER:
        if offset >= need:
            break
        path = find_medmnist_npz(MEDMNIST_ROOT, name)
        imgs, labs = load_medmnist_train(path)
        n_cls = int(labs.max()) + 1
        assert len(np.unique(labs)) == n_cls, f"{name}: labels are not 0..{n_cls - 1}"
        print(f"[POOL] classes {offset:>3}-{offset + n_cls - 1:<3} <- {name}  ({len(imgs)} images)  {path}")
        parts.append(ArrayImageDataset(imgs, labs)); tgts.append(labs + offset); offset += n_cls
    pool_targets = np.concatenate(tgts)
    distinct = min(offset, POOL_LIMIT) if POOL_LIMIT > 0 else offset
    by_src = {s: np.where(pool_targets == s)[0].tolist() for s in range(offset)}
    wrapped = n_classes > distinct
    if wrapped:
        print(f"[WARNING] {n_classes} classes > {distinct} distinct image classes: classes c and c+{distinct} share images")
    print(f"[INFO] Image pool={POOL_NAME}: {distinct} distinct image classes, {n_classes} dataset classes mapped")
    return ConcatDataset(parts), {c: by_src[c % distinct] for c in range(n_classes)}, distinct, wrapped


combined_dataset, indices_by_label, pool_size, pool_wrapped = build_image_pool(num_classes)
valid_labels = sorted(range(num_classes))

repeated_indices = {
    label: list(itertools.islice(
        itertools.cycle(indices_by_label[label]),
        num_samples_needed[label] + num_samples_needed_test[label]
    ))
    for label in indices_by_label
}

aligned_train_indices = []
aligned_test_indices = []
for label in valid_labels:
    train_tab_indices = np.where(y_train == label)[0].tolist()
    test_tab_indices = np.where(y_test == label)[0].tolist()
    train_img_indices = repeated_indices[label][:num_samples_needed[label]]
    test_img_indices = repeated_indices[label][
        num_samples_needed[label]:num_samples_needed[label] + num_samples_needed_test[label]
    ]
    if len(train_tab_indices) == len(train_img_indices) and \
       len(test_tab_indices) == len(test_img_indices):
        aligned_train_indices.extend(list(zip(train_tab_indices, train_img_indices)))
        aligned_test_indices.extend(list(zip(test_tab_indices, test_img_indices)))
    else:
        raise ValueError(f"Mismatch for label {label}")

train_filtered_tab_set = Subset(train_tabular_dataset, [idx[0] for idx in aligned_train_indices])
train_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_train_indices])
test_filtered_tab_set = Subset(test_tabular_dataset, [idx[0] for idx in aligned_test_indices])
test_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_test_indices])

class SynchronizedDataset(Dataset):
    def __init__(self, tabular_dataset, image_dataset):
        self.tabular_dataset = tabular_dataset
        self.image_dataset = image_dataset
        assert len(self.tabular_dataset) == len(self.image_dataset)
    def __len__(self):
        return len(self.tabular_dataset)
    def __getitem__(self, index):
        tab_data, tab_label = self.tabular_dataset[index]
        img_data, _ = self.image_dataset[index]   # class pairing is guaranteed by build_image_pool
        return tab_data, tab_label, img_data, tab_label

train_synchronized_dataset = SynchronizedDataset(train_filtered_tab_set, train_filtered_img_set)
test_synchronized_dataset = SynchronizedDataset(test_filtered_tab_set, test_filtered_img_set)
train_synchronized_loader = DataLoader(train_synchronized_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_synchronized_loader = DataLoader(test_synchronized_dataset, batch_size=BATCH_SIZE)
print(f"[INFO] Synchronized datasets created. Train batches: {len(train_synchronized_loader)}")

# ========== MODEL DEFINITIONS ==========


class ImageFeatureEncoder(nn.Module):
    """Conv trunk only — feeds HybridKAN's fusion, not a classifier itself."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
    def forward(self, x):
        return self.net(x) 


def supcon_loss(z, labels, temperature=0.1):
    """Supervised contrastive loss on the CVAE latent z."""
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=z.device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()


class KANTabularBranch(nn.Module):
    """
    Drop-in replacement for SimpleMLP: the tabular path is a KAN instead of
    Linear+ReLU. Same interface -> returns (tab_latent, logits).

    grid_range must cover your *scaled* feature range. You use StandardScaler,
    so features are z-scores; (-5, 5) covers essentially all of them. Beyond the
    grid only the spline part goes inactive (the SiLU base still fires), so
    nothing breaks, but keep the range wide enough that real values sit inside it.
    """
    def __init__(self, input_dim, latent_dim, num_classes,
                 grid_size=5, spline_order=3, grid_range=(-5.0, 5.0)):
        super().__init__()
        self.encoder_kan = KAN([input_dim, latent_dim], grid_size, spline_order, grid_range)
        self.head_kan    = KAN([latent_dim, num_classes], grid_size, spline_order, grid_range)

    def forward(self, x):
        tab_latent = self.encoder_kan(x)
        logits = self.head_kan(tab_latent)
        return tab_latent, logits

    @torch.no_grad()
    def feature_score(self, x):
        return self.encoder_kan.feature_score(x)   # native KAN interpretability
    
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, x

class VIFInitialization(nn.Module):
    def __init__(self, input_dim, vif_values):
        super(VIFInitialization, self).__init__()
        self.input_dim = input_dim
        self.vif_values = vif_values
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)
        vif_tensor = torch.tensor(vif_values, dtype=torch.float32)
        vif_tensor = vif_tensor / (vif_tensor.mean() + 1e-6)
        inv_vif = 1.0 / torch.clamp(vif_tensor, min=1.0)
        with torch.no_grad():
            for i in range(self.fc1.weight.data.shape[0]):
                self.fc1.weight.data[i, :] = inv_vif[i % len(inv_vif)] / (self.input_dim + 4)
        print("[INFO] VIF-based weight initialization complete.")
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class CAEWithTabEmbedding(nn.Module):
    def __init__(self, input_dim, tab_latent_size, num_classes, latent_size=8, vif_values=None):
        super(CAEWithTabEmbedding, self).__init__()
        self.mlp = KANTabularBranch(input_dim, tab_latent_size, num_classes)
        if vif_values is not None:
            self.vif_model = VIFInitialization(input_dim, vif_values)
        else:
            self.vif_model = None
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        
        self.hybrid = HybridKAN(
            image_encoder=ImageFeatureEncoder(),
            n_features=input_dim,
            n_outputs=num_classes,
            img_feat_dim=64 * 7 * 7,
            kan_neurons=tab_latent_size,
            fusion="bottleneck",
            bottleneck_dim=16,
            grid_range=(-5.0, 5.0),   # matches your StandardScaler'd feature range
        )
        # self.gate removed — HybridKAN's Final KAN replaces the sigmoid blend
    def encode(self, x, tab_embedding, vif_embedding):
        return self.encoder(torch.cat([x, tab_embedding, vif_embedding], dim=1))
    def decode(self, z, tab_embedding, vif_embedding):
        return self.decoder(torch.cat([z, tab_embedding, vif_embedding], dim=1))
    def forward(self, x, tab_data):
        if self.vif_model is not None:
            vif_embedding = self.vif_model(tab_data)
        else:
            vif_embedding = tab_data
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        recon_img = recon_x.view(-1, 1, 28, 28)
        fused_pred = self.hybrid(tab_data, recon_img)
        return recon_x, tab_pred, fused_pred, z

print("[INFO] Creating model...")
cae = CAEWithTabEmbedding(
    input_dim=n_cont_features,
    tab_latent_size=tab_latent_size,
    num_classes=num_classes,
    latent_size=8,
    vif_values=vif_values
).to(DEVICE)
#optimizer = optim.AdamW(cae.parameters(), lr=0.001, weight_decay=1e-4)
#optimizer = ADOPT(cae.parameters(), lr=0.001, decouple=True, weight_decay=1e-4)
#optimizer = ADOPT(cae.parameters(), lr=0.001, decouple=True)
optimizer = optim.AdamW(cae.parameters(), lr=0.001, weight_decay=1e-4)

print(f"[INFO] Model created with {sum(p.numel() for p in cae.parameters())} parameters")
# ============================================================
# TABLE 2 – TRAINABLE PARAMETER COUNT (C = 2, N = 78)
# TRAINABLE PARAMETER COUNT
# ============================================================
num_params = sum(p.numel() for p in cae.parameters() if p.requires_grad)

print(f"[INFO] Model created with {num_params:,} trainable parameters")
print(f"       Configuration: C={num_classes} classes, N={n_cont_features} features")

# Note if this matches Table 2's reference configuration
if num_classes == 2 and n_cont_features == 78:
    print(f"       ✓ Matches Table 2 specs (Expected: ~627.6K)")
#############################################
def loss_function(recon_x, x, tab_pred, tab_labels, fused_pred, z, con_weight=0.5):
    BCE = F.mse_loss(recon_x, x)
    tab_loss = F.cross_entropy(tab_pred, tab_labels)
    fused_loss = F.cross_entropy(fused_pred, tab_labels)
    con_loss = supcon_loss(z, tab_labels)
    return BCE + tab_loss + fused_loss + con_weight * con_loss

def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        tab_label = tab_label.to(DEVICE).long()
        optimizer.zero_grad()
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        recon_x, tab_pred, fused_pred, z = model(x_rand, tab_data)
        loss = loss_function(recon_x, img_data, tab_pred, tab_label, fused_pred, z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_data_loader)

def test(model, test_data_loader, epoch, best_accuracy, best_auc, best_epoch):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_fused_total = 0
    total = 0
    all_tab_labels, all_tab_preds = [], []
    all_fused_preds = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            tab_label = tab_label.to(DEVICE).long()
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            recon_x, tab_pred, fused_pred, z = model(x_rand, tab_data)
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, fused_pred, z).item()
            tab_probs = F.softmax(tab_pred, dim=1)
            fused_probs = F.softmax(fused_pred, dim=1)
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_probs.cpu().numpy())
            all_fused_preds.extend(fused_probs.cpu().numpy())
            tab_predicted = torch.argmax(tab_pred, dim=1)
            fused_predicted = torch.argmax(fused_pred, dim=1)
            correct_tab_total += (tab_predicted == tab_label).sum().item()
            correct_fused_total += (fused_predicted == tab_label).sum().item()
            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total
    fused_accuracy_total = 100 * correct_fused_total / total
    
    all_tab_preds_arr = np.array(all_tab_preds)
    all_fused_preds_arr = np.array(all_fused_preds)
    all_tab_labels_arr = np.array(all_tab_labels)

    tab_auc, fused_auc = 0.0, 0.0
    if not (np.isnan(all_tab_preds_arr).any() or np.isinf(all_tab_preds_arr).any()):
        try:
            if num_classes == 2:
                tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr[:, 1])
            else:
                tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[WARNING] Tab AUC calculation failed: {e}")

    if not (np.isnan(all_fused_preds_arr).any() or np.isinf(all_fused_preds_arr).any()):
        try:
            if num_classes == 2:
                fused_auc = roc_auc_score(all_tab_labels_arr, all_fused_preds_arr[:, 1])
            else:
                fused_auc = roc_auc_score(all_tab_labels_arr, all_fused_preds_arr, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[WARNING] Fused AUC calculation failed: {e}")

    if fused_accuracy_total > best_accuracy:
        best_accuracy = fused_accuracy_total
        best_auc = fused_auc
        best_epoch = epoch
        print(f"[INFO] New best accuracy: {best_accuracy:.2f}% (AUC: {fused_auc:.4f}) at epoch {epoch}")

    return best_accuracy, best_auc, best_epoch, test_loss, tab_accuracy_total, fused_accuracy_total

# ========== IMAGE SAVING FUNCTION ==========

def save_sample_images(model, test_data_loader, dataset_name, num_classes, num_images=20):
    """
    Save reconstructed images with DIVERSE labels
    Ensures all classes are represented in saved samples
    """
    model.eval()
    num_classes = min(num_classes, 20)   # the grid has 2*num_classes rows, so only draw the first 20 classes
    images_base_dir = os.path.join(OUTPUT_ROOT, "imageout")
    images_dir = os.path.join(images_base_dir, dataset_name)
    os.makedirs(images_dir, exist_ok=True)
    
    # Calculate samples per class (ensure diversity)
    samples_per_class = max(1, num_images // num_classes)
    total_to_save = samples_per_class * num_classes
    
    print(f"\n[INFO] Generating {total_to_save} sample images...")
    print(f"[INFO] Strategy: {samples_per_class} samples × {num_classes} classes")
    
    # Storage for images by class
    class_samples = {label: [] for label in range(num_classes)}
    
    # Collect samples for each class
    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            # Check if we have enough samples for all classes
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
                
            img_data_flat = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            
            # Generate reconstructed images
            random_array = np.random.rand(img_data_flat.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).to(DEVICE)
            recon_x, _, _, _ = model(x_rand, tab_data)
            
            # Store samples by class
            for i in range(len(tab_label)):
                label = tab_label[i].item()
                
                # Only collect if we need more samples for this class
                if label in class_samples and len(class_samples[label]) < samples_per_class:
                    class_samples[label].append({
                        'original': img_data[i].cpu().numpy(),
                        'reconstructed': recon_x[i].cpu().numpy().reshape(28, 28),
                        'label': label
                    })
    
    # Flatten samples for saving
    all_samples = []
    for label in sorted(class_samples.keys()):
        all_samples.extend(class_samples[label])
    
    num_saved = len(all_samples)
    print(f"[INFO] Collected {num_saved} samples across {num_classes} classes")
    
    # Print distribution
    print(f"[INFO] Samples per class:")
    for label in range(num_classes):
        count = len(class_samples[label])
        print(f"  Class {label}: {count} samples")
    
    # ============ CREATE GRID VISUALIZATION ============
    num_cols = min(5, num_classes)  # Show up to 5 classes per row
    num_rows = 2 * num_classes  # 2 rows per class (original + reconstructed)
    
    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(3*num_cols, 2*num_rows))
    
    # Handle edge cases for axes dimensions
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot images organized by class
    for class_idx in range(num_classes):
        samples = class_samples[class_idx][:num_cols]  # Take up to num_cols samples
        
        for sample_idx, sample in enumerate(samples):
            orig_row = class_idx * 2
            recon_row = class_idx * 2 + 1
            
            # Original image
            axes[orig_row, sample_idx].imshow(sample['original'].squeeze(), cmap='gray')
            axes[orig_row, sample_idx].set_title(
                f'Original\nClass {sample["label"]}', 
                fontsize=8, fontweight='bold'
            )
            axes[orig_row, sample_idx].axis('off')
            
            # Reconstructed image
            axes[recon_row, sample_idx].imshow(sample['reconstructed'], cmap='gray')
            axes[recon_row, sample_idx].set_title(
                f'Generated\nClass {sample["label"]}', 
                fontsize=8
            )
            axes[recon_row, sample_idx].axis('off')
        
        # Hide unused subplots in this class row
        for empty_col in range(len(samples), num_cols):
            axes[orig_row, empty_col].axis('off')
            axes[recon_row, empty_col].axis('off')
    
    plt.suptitle(f'{dataset_name} - Image Generation by Class', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    grid_path = os.path.join(images_dir, 'comparison_grid_by_class.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved class-organized grid to: {grid_path}")
    
    # ============ ALSO CREATE RANDOM MIXED GRID ============
    # Show diversity in a single view
    random_samples = np.random.choice(len(all_samples), 
                                     size=min(20, len(all_samples)), 
                                     replace=False)
    
    num_random = len(random_samples)
    num_cols_random = min(5, num_random)
    num_rows_random = 2 * ((num_random + num_cols_random - 1) // num_cols_random)
    
    fig2, axes2 = plt.subplots(num_rows_random, num_cols_random, 
                               figsize=(3*num_cols_random, 3*num_rows_random))
    
    if num_rows_random == 1:
        axes2 = axes2.reshape(1, -1)
    elif num_cols_random == 1:
        axes2 = axes2.reshape(-1, 1)
    
    axes2_flat = axes2.flatten()
    
    for idx, sample_idx in enumerate(random_samples):
        sample = all_samples[sample_idx]
        orig_idx = idx * 2
        recon_idx = idx * 2 + 1
        
        # Original
        if orig_idx < len(axes2_flat):
            axes2_flat[orig_idx].imshow(sample['original'].squeeze(), cmap='gray')
            axes2_flat[orig_idx].set_title(
                f'Original (Class {sample["label"]})', 
                fontsize=8
            )
            axes2_flat[orig_idx].axis('off')
        
        # Reconstructed
        if recon_idx < len(axes2_flat):
            axes2_flat[recon_idx].imshow(sample['reconstructed'], cmap='gray')
            axes2_flat[recon_idx].set_title(
                f'Generated (Class {sample["label"]})', 
                fontsize=8
            )
            axes2_flat[recon_idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(random_samples) * 2, len(axes2_flat)):
        axes2_flat[idx].axis('off')
    
    plt.tight_layout()
    mixed_grid_path = os.path.join(images_dir, 'comparison_grid_mixed.png')
    plt.savefig(mixed_grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved mixed grid to: {mixed_grid_path}")
    
    # ============ SAVE INDIVIDUAL IMAGES ============
    for idx, sample in enumerate(all_samples):
        label = sample['label']
        
        # Original
        orig_path = os.path.join(images_dir, 
                                f'sample_{idx:02d}_class{label}_original.png')
        plt.imsave(orig_path, sample['original'].squeeze(), cmap='gray')
        
        # Reconstructed
        recon_path = os.path.join(images_dir, 
                                 f'sample_{idx:02d}_class{label}_generated.png')
        plt.imsave(recon_path, sample['reconstructed'], cmap='gray')
    
    print(f"[INFO] Saved {num_saved} individual image pairs")
    
    # ============ CREATE CLASS DISTRIBUTION REPORT ============
    report_path = os.path.join(images_dir, 'sample_distribution.txt')
    with open(report_path, 'w') as f:
        f.write(f"Image Sample Distribution Report\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total samples saved: {num_saved}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Target samples per class: {samples_per_class}\n\n")
        f.write(f"Actual distribution:\n")
        f.write(f"-"*50 + "\n")
        for label in range(num_classes):
            count = len(class_samples[label])
            percentage = (count / num_saved * 100) if num_saved > 0 else 0
            f.write(f"  Class {label:2d}: {count:3d} samples ({percentage:5.1f}%)\n")
        f.write(f"\nGenerated files:\n")
        f.write(f"-"*50 + "\n")
        f.write(f"  1. comparison_grid_by_class.png - Organized by class\n")
        f.write(f"  2. comparison_grid_mixed.png    - Random mixed view\n")
        f.write(f"  3. sample_*.png                 - Individual images\n")
    
    print(f"[INFO] Saved distribution report to: {report_path}")
    print(f"[INFO] All images saved to: {images_dir}")
    
    return num_saved, images_dir

# ========== TRAINING LOOP (NO MODEL SAVING) ==========
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

best_accuracy = 0
best_auc = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train_loss = train(cae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_auc, best_epoch, test_loss, tab_acc, fused_acc  = test(
        cae, test_synchronized_loader, epoch, best_accuracy, best_auc, best_epoch
    )
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Tab Acc: {tab_acc:.2f}% | Fused Acc: {fused_acc:.2f}%")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print(f"Best Accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
print(f"Best AUC: {best_auc:.4f}")
print("="*70 + "\n")
cae.eval()
with torch.no_grad():
    xb = torch.tensor(X_train[:512], dtype=torch.float32, device=DEVICE)
    scores = cae.mlp.feature_score(xb).cpu().numpy()
importances = sorted(zip(X_df.columns.tolist(), scores), key=lambda t: t[1], reverse=True)
print("Top KAN feature importances:")
for name, s in importances[:15]:
    print(f"  {name:30s} {s:.4f}")

################################################################
# AUTOMATIC # OF WINS TRACKER - ADD AFTER TRAINING
print("\n" + "="*60)
print("YOUR MODEL BENCHMARK RESULTS")
print("="*60)

# Load/save your results history
RESULTS_FILE = os.path.join(OUTPUT_ROOT, "my_model_wins.json")
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        history = json.load(f)
else:
    history = []

history.append({
    'dataset': file_name,
    'accuracy': round(best_accuracy, 4),
    'auc': round(best_auc, 4),
    'features': n_cont_features,
    'classes': num_classes,
    'image_pool': POOL_NAME,
    'pool_classes': pool_size,
    'shared_images': pool_wrapped,
    'date': datetime.now().strftime("%Y-%m-%d")
})


# Save updated history
with open(RESULTS_FILE, 'w') as f:
    json.dump(history, f, indent=2)

# Calculate your # of wins (datasets where you got top score)
total_datasets = len(history)
your_acc_wins = len([r for r in history if r['accuracy'] >= 0.85])  # Your threshold
your_auc_wins = len([r for r in history if r['auc'] >= 0.92])      # Your threshold

avg_acc = np.mean([r['accuracy'] for r in history])
avg_auc = np.mean([r['auc'] for r in history])

print(f"📊 RESULTS ACROSS {total_datasets} DATASETS:")
print(f"   Avg ACC: {avg_acc:.4f}  |  Avg AUC: {avg_auc:.4f}")
print(f"   # Wins ACC: {your_acc_wins}/{total_datasets}  |  # Wins AUC: {your_auc_wins}/{total_datasets}")

print("\n📋 TABLE 1 STYLE SUMMARY:")
print("| Metric          | YourModel |")
print("|-----------------|-----------|")
print(f"| OpenML ACC Wins | **{your_acc_wins}** |")
print(f"| OpenML AUC Wins | **{your_auc_wins}** |")
print(f"| Avg ACC         | {avg_acc:.4f} |")
print(f"| Avg AUC         | {avg_auc:.4f} |")
print("\n💾 Saved to:", RESULTS_FILE)
print("="*60 + "\n")
##################################################################################


# Save sample images
num_saved, save_dir = save_sample_images(
    cae, test_synchronized_loader, file_name, num_classes, NUM_IMAGES_TO_SAVE
)




# Output results as JSON to stdout (for batch script to capture)
# Output results as JSON to stdout (for batch script to capture)
results = {
    'dataset': file_name,
    'num_samples': len(X),
    'num_features': n_cont_features,
    'num_classes': num_classes,
    'image_pool': POOL_NAME,
    'pool_classes': pool_size,
    'shared_images': pool_wrapped,
    'best_accuracy': best_accuracy,
    'best_auc': best_auc,
    'best_epoch': best_epoch,
    'images_saved': num_saved,
    'images_dir': save_dir,
    'trainable_params': num_params,  
    'matches_table2': (num_classes == 2 and n_cont_features == 78),  
    'timestamp': datetime.now().isoformat()
}
# Print JSON result (batch script will capture this)
print("\n" + "="*70)
print("RESULTS_JSON_START")
print(json.dumps(results))
print("RESULTS_JSON_END")
print("="*70 + "\n")

print(f"✅ Experiment completed successfully!")
print(f"   Dataset: {file_name}")
print(f"   Accuracy: {best_accuracy:.2f}%")
print(f"   AUC: {best_auc:.4f}")
print(f"   Images: {save_dir}")
