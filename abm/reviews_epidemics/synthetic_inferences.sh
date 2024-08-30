#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=1-7

module load python/intel/3.8.6

RUNDIR=/scratch/jc12343/shaman_lab/AMR_ABM2/abm
cd $RUNDIR

PARRAY=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
## Execute the desired Stata do file script
python3 synthetic_inference.py --prev_idx $SLURM_ARRAY_TASK_ID