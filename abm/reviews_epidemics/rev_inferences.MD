### Reviews Epidemics

# New inference experiments

1. what is the impact of a lower \gamma in the inferred parameters for all pathogens (except MRSA)?
   - **experiment.** previous inferences were conducted with 3 values of $\gamma$ spanning the lowest to highest values found in literature, i.e. $\gamma\in \{\gamma_{\text{low}},\gamma_{\text{median}},\gamma_{\text{high}}\}$. the new inferences will be done with $\gamma=f\cdot \gamma_{\text{low}}$, where $f \in [0,1]$, varying it from $1$ to $0$.

    sbatch synthetic_inferences.sh

2. how well does the system identify the parameters with lower $\gamma$?
   - **experiment.** synthetic inferences with lower prevalence, maybe $\gamma\in \{10, 15 \}\%$.

3. what is the impact of the COVID19 peak in admissions?
   - **experiment.** inferences by avoiding the increasing peak in admission and carriage at the beginning of the data as well as the COVID19 peak in admissions.

