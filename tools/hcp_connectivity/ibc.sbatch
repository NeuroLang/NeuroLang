#!/bin/bash
# The IBC functional connectivity on Margaret: python -m hcp_connectivity
# ibc over a Slurm array sharing the runs (README.md); then
# python -m hcp_connectivity combine ibc WORK RUNS DATASET.
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=28G
#SBATCH -t 4:00:00
#SBATCH --array=0-299%80
#SBATCH -o /scratch/dwasserm/ibc_fc/logs/%a.out
W=/data/parietal/store3/work/dwasserm/hcp_sc
export PYTHONPATH=$W/NeuroLang/tools OMP_NUM_THREADS=2
$W/.venv/bin/python -W ignore -m hcp_connectivity ibc $W /data/parietal/store2/data/ibc /scratch/dwasserm/ibc_fc/runs
