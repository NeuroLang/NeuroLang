#!/bin/bash
# Workers of parcellate.py, which take the connectomes download.py brings
# into the staging directory (on scratch) as they come:
#   sbatch --array=1-3 parcellate.sbatch WORK STAGING OUT
#SBATCH --job-name=hcp-sc-parcellate
#SBATCH --partition=parietal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
WORK=$1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "$WORK/.venv/bin/python" \
  "$WORK/NeuroLang/tools/hcp_structural_connectivity/parcellate.py" "$@"
