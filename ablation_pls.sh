#!/bin/bash
#=======================================================================
# ABLATION: Supervised Decorrelated Projection (PLS) — 5 configurations
#   sbatch ablation_pls.sh           (runs array tasks 0-4 in parallel)
#=======================================================================
#SBATCH --account=def-arashmoh
#SBATCH --job-name=PLSabl
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/plsabl_%A_%a.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/plsabl_%A_%a.err

PROJECT_DIR="/home/gkianfar/scratch/Amin"
TAB2SEDO_DIR="$PROJECT_DIR/Sedo/Tab2sedo"
VENV_PATH="$PROJECT_DIR/ICC/venvMsc/bin/activate"
DATASETS_DIR="$PROJECT_DIR/ICC/Unzippeddata/CSV"
RESULTS_BASE="$PROJECT_DIR/Sedo/output/ablation_pls"

NAMES=(A0_none A1_pca A2_pls_frozen A3_pls_noVIF A4_pls_full)
ARGS=(
  "--proj none"                                  # original model
  "--proj pca --lambda_vif 1.0"                  # unsupervised projection baseline
  "--proj pls --freeze_proj"                     # PLS as plain preprocessing
  "--proj pls --lambda_vif 0"                    # end-to-end PLS, no VIF regulariser
  "--proj pls --lambda_vif 1.0"                  # proposed: end-to-end PLS + differentiable VIF
)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
MAIN_ARGS=${ARGS[$SLURM_ARRAY_TASK_ID]}
OUT="$RESULTS_BASE/$NAME"
mkdir -p "$OUT" "$PROJECT_DIR/Sedo/output/logs"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2
source "$VENV_PATH"

echo "Config: $NAME  ->  main.py $MAIN_ARGS"
cd "$TAB2SEDO_DIR"
python run_all_datasets.py \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$OUT" \
    --job_id "${SLURM_ARRAY_JOB_ID}_${NAME}" \
    --script_path "$TAB2SEDO_DIR/main.py" \
    --timeout 14400 \
    --main_args "$MAIN_ARGS"
