# Reviews Epidemics

## new inference experiments

ssh jc12343@greene.hpc.nyu.edu

1. what is the impact of a lower $\gamma$ in the inferred parameters for all pathogens (except MRSA)?
   - **experiment.** previous inferences were conducted with 3 values of $\gamma$ spanning the lowest to highest values found in literature, i.e. $\gamma \in \{\gamma_{\text{low}},\gamma_{\text{median}},\gamma_{\text{high}}\}$. the new inferences will be done with $\gamma=f\cdot \gamma_{\text{low}}$, where $f \in [0,1]$, varying it from $1$ to $0$.

         sbatch gamma_sensitivity.sh

      scp -r jc12343@greene.hpc.nyu.edu:/scratch/jc12343/shaman_lab/amr-hospitals/results/gamma_sensitivity/ /Users/chaosdonkey06/Dropbox/shaman-lab/amr-hospitals2/results

2. how well does the system identify the parameters with lower $\gamma$?
   - **experiment.** synthetic inferences with lower prevalence, i.e. $\gamma\in \{5, 10, 15 \}\%$. 4 inferences are run per *scenario*, and the nosocomial transmission rate was varied in $\beta \in \{0.01, 0.05, 0.1\}$ and the effective sensitivity $\rho \in \{1, 5, 10\}\%$.

         sbatch synthetic_inferences.sh

   copy results to local db directory.

         scp -r jc12343@greene.hpc.nyu.edu:/scratch/jc12343/shaman_lab/amr-hospitals/results/synthetic_inferences/ /Users/chaosdonkey06/Dropbox/shaman-lab/amr-hospitals2/results

   - **results and discussion.** comment 3, Figures R1-R4.

3. what is the impact of the COVID19 peak in admissions?
   - **experiment.** inferences by avoiding the increasing peak in admission and carriage at the beginning of the data as well as the COVID19 peak in admissions.

         sbatch synthetic_covid_period_sensitivity.sh

      copy results to local.

         scp -r jc12343@greene.hpc.nyu.edu:/scratch/jc12343/shaman_lab/amr-hospitals/results/amro_inferences/covid19_sensitivity/ /Users/chaosdonkey06/Dropbox/shaman-lab/amr-hospitals2/results