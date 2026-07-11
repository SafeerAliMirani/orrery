#!/bin/bash
#SBATCH -A EIRI_E_UNISA2
#SBATCH -p g100_usr_prod
#SBATCH -J orrery
#SBATCH --array=0-33
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/orrery_%a.out
#SBATCH -e logs/orrery_%a.err

# Orrery Phase 1: 34-task array, ~30k stars each, integrate + fit.
# Submit from the orrery working dir after pull_gaia.sh finished:
#   mkdir -p logs fits && sbatch job_array.sbatch
# CPU-only. Each task: ~9-15 min (Phase 0 laptop rate was 50 s per 3000
# stars). If g100_usr_prod queues badly, g100_usr_interactive also accepts
# sbatch but caps at ~30 min, which still fits.

source /g100_work/EIRI_E_UNISA2/smirani0/miniforge3/etc/profile.d/conda.sh
conda activate orrery

DATA=/g100_work/EIRI_E_UNISA2/smirani0/orrery/data
CSV=$DATA/gaia_1m.csv

NTOT=$(($(wc -l < "$CSV") - 1))
NTASK=34
PER=$(( (NTOT + NTASK - 1) / NTASK ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))
END=$(( START + PER )); [ $END -gt $NTOT ] && END=$NTOT
[ $START -ge $NTOT ] && exit 0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python integrate_fit.py --csv "$CSV" --start $START --end $END \
    --out fits/fit_$(printf '%03d' $SLURM_ARRAY_TASK_ID).h5
