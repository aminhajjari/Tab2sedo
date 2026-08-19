#!/bin/bash

#SBATCH --account=def-arashmoh
#SBATCH --job-name=Sedo_DEBUG
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/Sedodebug_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/Sedodebug_%A.err


# ============================================================
# Paths
# ============================================================

PROJECT_DIR="/home/gkianfar/scratch/Amin/Sedo"
CODE_DIR="$PROJECT_DIR/Tab2sedo"

DEBUG_DATA="/home/gkianfar/scratch/Amin/ICC/debug_data"

OUTPUT="$PROJECT_DIR/output"

VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate"

BATCH_SCRIPT="$CODE_DIR/run_all_datasets.py"
MAIN_SCRIPT="$CODE_DIR/main.py"

TIMEOUT=14400

# ============================================================
# Setup
# ============================================================

mkdir -p "$OUTPUT"
mkdir -p "$OUTPUT/logs"


# ============================================================
# Information
# ============================================================

echo "=========================================="
echo "SEDO DEBUG RUN - 5 DATASETS"
echo "=========================================="

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Dataset directory: $DEBUG_DATA"
echo ""

echo "Datasets:"
ls -1 "$DEBUG_DATA"

echo ""


# ============================================================
# Load environment
# ============================================================

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

source "$VENV_PATH"

cd "$CODE_DIR"


# ============================================================
# Environment check
# ============================================================

echo "=========================================="
echo "ENVIRONMENT CHECK"
echo "=========================================="

echo "Python:"
which python
python --version

echo ""

echo "PyTorch:"
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo ""

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""


# ============================================================
# Run
# ============================================================

echo "=========================================="
echo "STARTING DEBUG RUN"
echo "=========================================="

python "$BATCH_SCRIPT" \
    --datasets_dir "$DEBUG_DATA" \
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
echo "DEBUG RUN COMPLETE"
echo "=========================================="

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results: $OUTPUT"

echo "=========================================="

exit $EXIT_CODE
