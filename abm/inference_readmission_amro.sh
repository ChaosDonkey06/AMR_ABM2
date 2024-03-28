#!/bin/bash

#SBATCH --nodes=6
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=0-6

module load python/intel/3.8.6

RUNDIR=$SCRATCH/jc12343/shaman_lab/AMR_ABM2/abm
cd $RUNDIR

PARRAY=(0 1 2 3 4 5 6)

## Execute the desired Stata do file script
python3 InferenceReadmission_HPC.py --amro_idx $(SLURM_ARRAY_TASK_ID)