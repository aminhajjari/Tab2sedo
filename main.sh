#!/bin/bash

#SBATCH --account=def-arashmoh
#SBATCH --job-name=Sedo_FULL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/Sedofull_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/Sedofull_%A.err


# ============================================================
# Paths
# ============================================================

PROJECT_DIR="/home/gkianfar/scratch/Amin/Sedo"
CODE_DIR="$PROJECT_DIR/Tab2sedo"

DATA="/home/gkianfar/scratch/Amin/ICC/Unzippeddata/CSV"

OUTPUT="$PROJECT_DIR/output"

VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc"

BATCH_SCRIPT="$CODE_DIR/run_all_datasets.py"
MAIN_SCRIPT="$CODE_DIR/main.py"

TIMEOUT=28800


# ============================================================
# Create output directories
# NOTE: SLURM opens --output/--error paths before the script body
# runs, so the logs/ directory must already exist at submit time.
# This mkdir only protects OUTPUT for later stages of the script.
# Run: mkdir -p /home/gkianfar/scratch/Amin/Sedo/output/logs
# once, from the shell, BEFORE you sbatch this script.
# ============================================================

mkdir -p "$OUTPUT"
mkdir -p "$OUTPUT/logs"


# ============================================================
# Load modules
# ============================================================

echo "=========================================="
echo "LOADING ENVIRONMENT"
echo "=========================================="

# Plain "module purge" leaves sticky/parent modules loaded
# (StdEnv, imkl, flexiblas, gcc, etc.). Those then conflict with
# the BLAS/MKL bundled in python/3.11's venv and cause the
# interpreter itself to crash with "Illegal instruction" before
# it can even print --version. --force clears everything.
module --force purge

module load StdEnv/2023
module load python/3.11
module load cuda/12.2


# ============================================================
# Activate virtual environment
# ============================================================

echo ""
echo "Activating virtual environment..."

source "$VENV_PATH/bin/activate"


# ============================================================
# Move to project directory
# ============================================================

cd "$CODE_DIR" || {
    echo "ERROR: Could not enter CODE_DIR:"
    echo "$CODE_DIR"
    exit 1
}


# ============================================================
# Basic information
# ============================================================

echo ""
echo "=========================================="
echo "SEDO FULL RUN - ALL DATASETS"
echo "=========================================="

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

echo "Project directory:"
echo "$PROJECT_DIR"

echo ""

echo "Code directory:"
echo "$CODE_DIR"

echo ""

echo "Dataset directory:"
echo "$DATA"

echo ""

echo "Output directory:"
echo "$OUTPUT"


# ============================================================
# Python environment check
# ============================================================

echo ""
echo "=========================================="
echo "PYTHON ENVIRONMENT CHECK"
echo "=========================================="

echo "Python executable:"
which python

echo ""

echo "Python version:"
python --version

echo ""

echo "Virtual environment:"
echo "$VIRTUAL_ENV"


# ============================================================
# PyTorch / CUDA check
# ============================================================

echo ""
echo "=========================================="
echo "PYTORCH / CUDA CHECK"
echo "=========================================="

python -c "
import torch

print('PyTorch version:', torch.__version__)
print('PyTorch CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print('WARNING: CUDA is NOT available')
"


# ============================================================
# GPU check
# ============================================================

echo ""
echo "=========================================="
echo "GPU CHECK"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader


# ============================================================
# Verify required files/directories
# ============================================================

echo ""
echo "=========================================="
echo "FILE CHECK"
echo "=========================================="

if [ ! -d "$DATA" ]; then
    echo "ERROR: Dataset directory does not exist:"
    echo "$DATA"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "ERROR: Batch script does not exist:"
    echo "$BATCH_SCRIPT"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "ERROR: Main script does not exist:"
    echo "$MAIN_SCRIPT"
    exit 1
fi

echo "Dataset directory: OK"
echo "Batch script: OK"
echo "Main script: OK"


# ============================================================
# List datasets
# ============================================================

echo ""
echo "=========================================="
echo "DATASETS"
echo "=========================================="

ls -1 "$DATA"


# ============================================================
# Final environment confirmation
# ============================================================

echo ""
echo "=========================================="
echo "FINAL ENVIRONMENT"
echo "=========================================="

echo "Python:"
which python

echo ""

echo "PyTorch:"
python -c "import torch; print(torch.__version__)"

echo ""

echo "CUDA:"
python -c "import torch; print(torch.cuda.is_available())"

echo ""

echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader


# ============================================================
# Run ALL datasets
# ============================================================

echo ""
echo "=========================================="
echo "STARTING FULL RUN"
echo "=========================================="

echo "Start time: $(date)"
echo ""

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATA" \
    --output_base "$OUTPUT" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT"

EXIT_CODE=$?

# ============================================================
# Aggregate results across all datasets
# ============================================================

echo "" >&2
echo "======================================================================" >&2
echo "AGGREGATING FINAL RESULTS" >&2
echo "======================================================================" >&2

# Find the output directory created for this SLURM job
RUN_OUTPUT_DIR=$(find "$OUTPUT" \
    -maxdepth 1 \
    -type d \
    -name "*_JOB${SLURM_JOB_ID}" \
    -print -quit)

if [ -z "$RUN_OUTPUT_DIR" ]; then
    echo "[WARNING] Could not find job-specific output directory." >&2
    echo "[WARNING] Searching all result JSON files under: $OUTPUT" >&2
    RUN_OUTPUT_DIR="$OUTPUT"
else
    echo "Job output directory:" >&2
    echo "$RUN_OUTPUT_DIR" >&2
fi

echo "" >&2
echo "Calculating average Accuracy and AUC..." >&2

python - "$RUN_OUTPUT_DIR" <<'PY' >&2
import sys
import json
from pathlib import Path
import numpy as np

output_dir = Path(sys.argv[1])

# Find all per-dataset result files
result_files = sorted(output_dir.rglob("*_results.json"))

print("=" * 70)
print("FINAL RESULTS ACROSS ALL DATASETS")
print("=" * 70)

if not result_files:
    print("[WARNING] No *_results.json files found.")
    print(f"Searched in: {output_dir}")
    sys.exit(0)

accuracies = []
aucs = []
datasets = []

for result_file in result_files:
    try:
        with open(result_file, "r") as f:
            result = json.load(f)

        accuracy = result.get("test_fused_accuracy")
        auc = result.get("test_fused_auc")
        dataset = result.get("dataset", result_file.stem)

        if accuracy is not None:
            accuracies.append(float(accuracy))

        if auc is not None:
            aucs.append(float(auc))

        if accuracy is not None or auc is not None:
            datasets.append(dataset)

    except Exception as e:
        print(f"[WARNING] Could not read {result_file}: {e}")

print(f"Result files found : {len(result_files)}")
print(f"Datasets included  : {len(datasets)}")
print()

if accuracies:
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
    median_accuracy = np.median(accuracies)

    print(f"Average Accuracy   : {avg_accuracy:.4f} ({avg_accuracy * 100:.2f}%)")
    print(f"Std Accuracy       : {std_accuracy:.4f} ({std_accuracy * 100:.2f}%)")
    print(f"Median Accuracy    : {median_accuracy:.4f} ({median_accuracy * 100:.2f}%)")
else:
    print("Average Accuracy   : N/A")

print()

if aucs:
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0
    median_auc = np.median(aucs)

    print(f"Average AUC        : {avg_auc:.4f}")
    print(f"Std AUC            : {std_auc:.4f}")
    print(f"Median AUC         : {median_auc:.4f}")
else:
    print("Average AUC        : N/A")

print("=" * 70)

# Save aggregated summary
summary = {
    "num_result_files": len(result_files),
    "num_datasets": len(datasets),
    "average_accuracy": float(np.mean(accuracies)) if accuracies else None,
    "std_accuracy": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None,
    "median_accuracy": float(np.median(accuracies)) if accuracies else None,
    "average_auc": float(np.mean(aucs)) if aucs else None,
    "std_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
    "median_auc": float(np.median(aucs)) if aucs else None,
}

summary_file = output_dir / "summary.json"

try:
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to   : {summary_file}")

except Exception as e:
    print(f"[WARNING] Could not save summary.json: {e}")

print("=" * 70)
PY
# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "FULL RUN COMPLETE"
echo "=========================================="

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"

echo ""
echo "Results:"
echo "$OUTPUT"

echo ""
echo "=========================================="


# ============================================================
# Exit
# ============================================================

exit $EXIT_CODE
