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
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import json
from datetime import datetime

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

DATASET_ROOT = "/home/gkianfar/scratch/Amin/ICC/Unzippeddata/Image"
 




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

if num_classes > 20:
    print(f"[ERROR] Dataset has {num_classes} classes (>20). Skipping...")
    exit(1)
if num_classes < 2:
    raise ValueError(f"Dataset has only {num_classes} class. Need at least 2.")

# ============================================================
# PREPROCESSING
# ============================================================

X_df = df.drop(columns=[target_col])

# ------------------------------------------------------------
# DIAGNOSTIC: catch all-NaN or degenerate columns early
# ------------------------------------------------------------
print(f"[DEBUG] X_df dtypes:\n{X_df.dtypes.value_counts()}")
nan_pct = X_df.isnull().mean()
fully_nan_cols = nan_pct[nan_pct == 1.0].index.tolist()
if fully_nan_cols:
    print(f"[WARNING] {len(fully_nan_cols)} column(s) are 100% NaN after loading: {fully_nan_cols}")

# ------------------------------------------------------------
# Convert numeric columns
# ------------------------------------------------------------
print("[INFO] Preparing feature columns...")

for col in X_df.columns:
    if X_df[col].dtype != 'object':
        X_df[col] = pd.to_numeric(
            X_df[col],
            errors='coerce'
        )

# ------------------------------------------------------------
# Convert labels to consecutive integers
# ------------------------------------------------------------
unique_values = sorted(set(y))
num_classes = len(unique_values)

value_map = {
    unique_values[i]: i
    for i in range(num_classes)
}

y = np.array([
    value_map[val]
    for val in y
])

# ------------------------------------------------------------
# IMPORTANT:
# Split BEFORE fitting preprocessing components
# ------------------------------------------------------------
print("[INFO] Splitting into train/test (80/20)...")

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    f"[INFO] Train samples: {len(X_train_df)}, "
    f"Test samples: {len(X_test_df)}"
)

# ============================================================
# CATEGORICAL FEATURE ENCODING
# ============================================================

print("[INFO] Encoding categorical features...")

categorical_columns = X_train_df.select_dtypes(
    include=['object']
).columns.tolist()

for col in categorical_columns:

    # --------------------------------------------------------
    # Learn mapping ONLY from training data
    # --------------------------------------------------------
    train_categories = sorted(
        X_train_df[col].astype(str).unique()
    )

    category_to_int = {
        category: idx
        for idx, category in enumerate(train_categories)
    }

    # --------------------------------------------------------
    # Transform training data
    # --------------------------------------------------------
    X_train_df[col] = (
        X_train_df[col]
        .astype(str)
        .map(category_to_int)
        .astype(float)
    )

    # --------------------------------------------------------
    # Transform test data
    #
    # Unknown categories are mapped to -1
    # instead of learning them from test data.
    # --------------------------------------------------------
    X_test_df[col] = (
        X_test_df[col]
        .astype(str)
        .map(category_to_int)
        .fillna(-1)
        .astype(float)
    )

# ============================================================
# MISSING-VALUE IMPUTATION
# ============================================================

print("[INFO] Imputing missing values with median...")

imputer = SimpleImputer(
    strategy='median',
    keep_empty_features=True
)

# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------
X_train = imputer.fit_transform(
    X_train_df
)

# ------------------------------------------------------------
# TEST DATA IS ONLY TRANSFORMED
# ------------------------------------------------------------
X_test = imputer.transform(
    X_test_df
)

print(
    "[INFO] Missing-value imputation fitted "
    "on training data only."
)

# ============================================================
# STANDARDIZATION
# ============================================================

print("[INFO] Standardizing features...")

if X_train.shape[1] == 0:
    raise ValueError(
        f"[FATAL] X_train has 0 features going into StandardScaler for dataset "
        f"'{file_name}'. Check the [WARNING] all-NaN column log above — "
        f"the categorical encoding step likely produced NaN-only columns."
    )

scaler = StandardScaler()

scaler = StandardScaler()

# FIT ONLY ON TRAINING DATA
X_train = scaler.fit_transform(
    X_train
)

# TRANSFORM TEST DATA
X_test = scaler.transform(
    X_test
)

# ============================================================
# FINAL DATA INFORMATION
# ============================================================

n_cont_features = X_train.shape[1]
tab_latent_size = n_cont_features + 4

print(f"\n{'='*70}")
print(f"[SUMMARY] Preprocessed Data:")
print(f"  - Train samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")
print(f"  - Features: {n_cont_features}")
print(f"  - Classes: {num_classes}")
print(
    f"  - Train class distribution: "
    f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
)
print(
    f"  - Test class distribution: "
    f"{dict(zip(*np.unique(y_test, return_counts=True)))}"
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

print("[INFO] Preparing tabular datasets...")

train_loader = DataLoader(
    train_tabular_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_tabular_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    f"[INFO] Tabular datasets created. "
    f"Train batches: {len(train_loader)}"
)


# ========== MODEL DEFINITIONS ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

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

class Tab2ImageProjector(nn.Module):

    def __init__(
        self,
        input_dim,
        tab_latent_size,
        num_classes,
        vif_values=None
    ):
        super(Tab2ImageProjector, self).__init__()

        # ==========================================
        # 1. Tabular representation
        # ==========================================
        self.mlp = SimpleMLP(
            input_dim=input_dim,
            latent_dim=tab_latent_size,
            num_classes=num_classes
        )

        # ==========================================
        # 2. VIF representation
        # ==========================================
        if vif_values is not None:

            self.vif_model = VIFInitialization(
                input_dim=input_dim,
                vif_values=vif_values
            )

        else:

            self.vif_model = None

        # ==========================================
        # 3. Tabular -> Image projector
        # ==========================================
        self.projector = nn.Sequential(

            nn.Linear(
                tab_latent_size + input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                28 * 28
            ),

            nn.Sigmoid()
        )

        # ==========================================
        # 4. CNN classifier
        # ==========================================
        self.final_classifier = SimpleCNN(
            num_classes=num_classes
        )

    def forward(self, tab_data):

        # ------------------------------------------
        # VIF branch
        # ------------------------------------------
        if self.vif_model is not None:

            vif_embedding = self.vif_model(
                tab_data
            )

        else:

            vif_embedding = tab_data

        # ------------------------------------------
        # MLP branch
        # ------------------------------------------
        tab_embedding, tab_pred = self.mlp(
            tab_data
        )

        # ------------------------------------------
        # Combine tabular representations
        # ------------------------------------------
        combined_embedding = torch.cat(
            [
                tab_embedding,
                vif_embedding
            ],
            dim=1
        )

        # ------------------------------------------
        # Generate pseudo-image
        # ------------------------------------------
        pseudo_image = self.projector(
            combined_embedding
        )

        # ------------------------------------------
        # Reshape: 784 -> 1 x 28 x 28
        # ------------------------------------------
        pseudo_image = pseudo_image.view(
            -1,
            1,
            28,
            28
        )

        # ------------------------------------------
        # CNN classification
        # ------------------------------------------
        img_pred = self.final_classifier(
            pseudo_image
        )

        return (
            pseudo_image,
            tab_pred,
            img_pred
        )

print("[INFO] Creating model...")

model = Tab2ImageProjector(
    input_dim=n_cont_features,
    tab_latent_size=tab_latent_size,
    num_classes=num_classes,
    vif_values=vif_values
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print(
    f"[INFO] Model created with "
    f"{sum(p.numel() for p in model.parameters())} parameters"
)
# ============================================================
# TABLE 2 – TRAINABLE PARAMETER COUNT (C = 2, N = 78)
# TRAINABLE PARAMETER COUNT
# ============================================================
num_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"[INFO] Model created with {num_params:,} trainable parameters")
print(f"       Configuration: C={num_classes} classes, N={n_cont_features} features")

print(
    f"       Architecture: "
    f"Tabular → MLP+VIF → 256 → 784 → 28×28 → CNN"
)
#############################################
def loss_function(
    tab_pred,
    tab_labels,
    img_pred,
    img_labels
):

    tab_loss = F.cross_entropy(
        tab_pred,
        tab_labels
    )

    img_loss = F.cross_entropy(
        img_pred,
        img_labels
    )

    return tab_loss + img_loss

def train(model, train_loader, optimizer, epoch):

    model.train()

    train_loss = 0.0

    for tab_data, tab_label in train_loader:

        tab_data = tab_data.to(DEVICE)
        tab_label = tab_label.to(DEVICE).long()

        optimizer.zero_grad()

        # Tabular -> pseudo-image -> CNN
        pseudo_image, tab_pred, img_pred = model(
            tab_data
        )

        # Classification-only objective
        loss = loss_function(
            tab_pred,
            tab_label,
            img_pred,
            tab_label
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def test(
    model,
    test_loader,
    epoch,
    best_accuracy,
    best_auc,
    best_epoch
):

    model.eval()

    test_loss = 0.0

    correct_tab_total = 0
    correct_img_total = 0

    total = 0

    all_tab_labels = []
    all_tab_preds = []

    all_img_labels = []
    all_img_preds = []

    with torch.no_grad():

        for tab_data, tab_label in test_loader:

            tab_data = tab_data.to(DEVICE)
            tab_label = tab_label.to(DEVICE).long()

            # Tabular -> pseudo-image -> CNN
            pseudo_image, tab_pred, img_pred = model(
                tab_data
            )

            loss = loss_function(
                tab_pred,
                tab_label,
                img_pred,
                tab_label
            )

            test_loss += loss.item()

            # Probabilities
            tab_probs = F.softmax(
                tab_pred,
                dim=1
            )

            img_probs = F.softmax(
                img_pred,
                dim=1
            )

            all_tab_labels.extend(
                tab_label.cpu().numpy()
            )

            all_tab_preds.extend(
                tab_probs.cpu().numpy()
            )

            all_img_labels.extend(
                tab_label.cpu().numpy()
            )

            all_img_preds.extend(
                img_probs.cpu().numpy()
            )

            # Predictions
            tab_predicted = torch.argmax(
                tab_pred,
                dim=1
            )

            img_predicted = torch.argmax(
                img_pred,
                dim=1
            )

            correct_tab_total += (
                tab_predicted == tab_label
            ).sum().item()

            correct_img_total += (
                img_predicted == tab_label
            ).sum().item()

            total += tab_label.size(0)

    test_loss /= len(test_loader)

    tab_accuracy_total = (
        100.0 *
        correct_tab_total /
        total
    )

    img_accuracy_total = (
        100.0 *
        correct_img_total /
        total
    )

    # Convert to numpy
    all_tab_preds_arr = np.array(
        all_tab_preds
    )

    all_img_preds_arr = np.array(
        all_img_preds
    )

    all_tab_labels_arr = np.array(
        all_tab_labels
    )

    all_img_labels_arr = np.array(
        all_img_labels
    )

    tab_auc = 0.0
    img_auc = 0.0

    try:

        if num_classes == 2:

            tab_auc = roc_auc_score(
                all_tab_labels_arr,
                all_tab_preds_arr[:, 1]
            )

            img_auc = roc_auc_score(
                all_img_labels_arr,
                all_img_preds_arr[:, 1]
            )

        else:

            tab_auc = roc_auc_score(
                all_tab_labels_arr,
                all_tab_preds_arr,
                multi_class="ovr",
                average="macro"
            )

            img_auc = roc_auc_score(
                all_img_labels_arr,
                all_img_preds_arr,
                multi_class="ovr",
                average="macro"
            )

    except Exception as e:

        print(
            f"[WARNING] AUC calculation failed: {e}"
        )

    if img_accuracy_total > best_accuracy:

        best_accuracy = img_accuracy_total
        best_epoch = epoch

        print(
            f"[INFO] New best accuracy: "
            f"{best_accuracy:.2f}% "
            f"at epoch {epoch}"
        )

    if img_auc > best_auc:

        best_auc = img_auc

    return (
        best_accuracy,
        best_auc,
        best_epoch,
        test_loss,
        tab_accuracy_total,
        img_accuracy_total
    )

# ========== IMAGE SAVING FUNCTION ==========

def save_sample_images(
    model,
    test_loader,
    dataset_name,
    num_classes,
    num_images=20
):

    model.eval()

    images = []
    labels = []
    predictions = []

    with torch.no_grad():

        for tab_data, tab_label in test_loader:

            tab_data = tab_data.to(DEVICE)

            pseudo_images, _, img_pred = model(
                tab_data
            )

            predicted = torch.argmax(
                img_pred,
                dim=1
            )

            for i in range(
                len(tab_label)
            ):

                if len(images) >= num_images:
                    break

                images.append(
                    pseudo_images[i]
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                labels.append(
                    tab_label[i].item()
                )

                predictions.append(
                    predicted[i].item()
                )

            if len(images) >= num_images:
                break

    images_base_dir = (
        "/home/gkianfar/scratch/Amin/AI/outputs/imageout"
    )

    images_dir = os.path.join(
        images_base_dir,
        dataset_name
    )

    os.makedirs(
        images_dir,
        exist_ok=True
    )

    num_cols = 5

    num_rows = int(
        np.ceil(
            len(images) / num_cols
        )
    )

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12, 2.5 * num_rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for i in range(len(images)):

        axes[i].imshow(
            images[i],
            cmap="gray"
        )

        axes[i].set_title(
            f"GT: {labels[i]} | "
            f"Pred: {predictions[i]}"
        )

        axes[i].axis("off")

    for i in range(
        len(images),
        len(axes)
    ):

        axes[i].axis("off")

    plt.tight_layout()

    grid_path = os.path.join(
        images_dir,
        "tabular_to_pseudo_images.png"
    )

    plt.savefig(
        grid_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[INFO] Saved pseudo-image visualization "
        f"to: {grid_path}"
    )

    return (
        len(images),
        images_dir
    )

# ========== TRAINING LOOP (NO MODEL SAVING) ==========
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

best_accuracy = 0
best_auc = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train_loss = train(
        model,
        train_loader,
        optimizer,
        epoch
    )

    best_accuracy, best_auc, best_epoch, test_loss, tab_acc, img_acc = test(
        model,
        test_loader,
        epoch,
        best_accuracy,
        best_auc,
        best_epoch
    )

    if epoch % 10 == 0 or epoch == 1:
        print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Tab Acc: {tab_acc:.2f}% | Img Acc: {img_acc:.2f}%")
        print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Tab Acc: {tab_acc:.2f}% | Img Acc: {img_acc:.2f}%")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print(f"Best Accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
print(f"Best AUC: {best_auc:.4f}")
print("="*70 + "\n")

################################################################
# AUTOMATIC # OF WINS TRACKER - ADD AFTER TRAINING
print("\n" + "="*60)
print("YOUR MODEL BENCHMARK RESULTS")
print("="*60)

# Load/save your results history
RESULTS_FILE = "/home/gkianfar/scratch/Amin/Sedo/output/my_model_wins.json"
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
    model,
    test_loader,
    file_name,
    num_classes,
    NUM_IMAGES_TO_SAVE
)




# Output results as JSON to stdout (for batch script to capture)
# Output results as JSON to stdout (for batch script to capture)
results = {
    'dataset': file_name,
    'num_samples': len(X_df),
    'num_features': n_cont_features,
    'num_classes': num_classes,
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
