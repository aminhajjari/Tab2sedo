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
# ============================================================

mkdir -p "$OUTPUT"
mkdir -p "$OUTPUT/logs"


# ============================================================
# Load modules
# ============================================================

echo "=========================================="
echo "LOADING ENVIRONMENT"
echo "=========================================="

module purge

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
