#!/bin/bash
# The tasks of functional.py, a share of the subjects each:
#   sbatch --array=0-39 functional.sbatch WORK HCP_DIR OUT 40
#SBATCH --job-name=hcp-fc
#SBATCH --partition=parietal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
WORK=$1 HCP=$2 OUT=$3 TASKS=$4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "$WORK/.venv/bin/python" \
  "$WORK/NeuroLang/tools/hcp_functional_connectivity/functional.py" \
  "$WORK" "$HCP" "$OUT" --task "$SLURM_ARRAY_TASK_ID" --tasks "$TASKS"
