#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=synthetic_inferences
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --output=out2/slurm_%j.out
#SBATCH --array=0-62

module load python/intel/3.8.6

RUNDIR=/scratch/jc12343/shaman_lab/AMR_ABM2/abm/reviews_epidemics
cd $RUNDIR

## Execute the desired Stata do file script
python3 covid_period_sensitivity1.py --idx_row $SLURM_ARRAY_TASK_ID