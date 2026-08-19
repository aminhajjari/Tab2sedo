#!/bin/bash

#SBATCH --account=def-arashmoh
#SBATCH --job-name=Sedo_DEBUG
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/Sedodebug_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/Sedodebug_%A.err



# ============================================================
# Paths
# ============================================================

BASE="/home/gkianfar/scratch/Amin/ICC"

CODE="$BASE/main/V2I"
DATA="$BASE/Unzippeddata/CSV"
DEBUG_DATA="$BASE/debug_data"
OUTPUT="$BASE/output/debug"

VENV="$BASE/venvMsc/bin/activate"

MAIN="$CODE/main.py"
RUN="$CODE/run_all_datasets.py"

TIMEOUT=14400

# ============================================================
# Setup
# ============================================================

mkdir -p "$DEBUG_DATA"
mkdir -p "$OUTPUT"

# Clean previous debug dataset links
rm -rf "$DEBUG_DATA"/*

# ============================================================
# Select 5 datasets
# ============================================================

ln -s "$DATA/Bioresponse" "$DEBUG_DATA/Bioresponse"
ln -s "$DATA/cmc" "$DEBUG_DATA/cmc"
ln -s "$DATA/kc2" "$DEBUG_DATA/kc2"
ln -s "$DATA/phoneme" "$DEBUG_DATA/phoneme"
ln -s "$DATA/spambase" "$DEBUG_DATA/spambase"

echo "=========================================="
echo "V2I DEBUG RUN - 5 DATASETS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
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

source "$VENV"

cd "$CODE"

# ============================================================
# Environment check
# ============================================================

echo "Python:"
which python
python --version
echo ""

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo ""

# ============================================================
# Run 5 datasets
# ============================================================

echo "=========================================="
echo "STARTING DEBUG RUN"
echo "=========================================="

python "$RUN" \
    --datasets_dir "$DEBUG_DATA" \
    --output_base "$OUTPUT" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN" \
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
