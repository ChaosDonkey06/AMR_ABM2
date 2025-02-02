Data and analysis code for transcriptional control in malaria parasites. Preprint forthcoming. Automated DNA design code is hosted on the [GeneTargeter repository](https://github.com/pablocarderam/genetargeter).

Archived on DOIs using Zenodo, current version at

[![DOI](https://zenodo.org/badge/817876760.svg)](https://zenodo.org/doi/10.5281/zenodo.12193161)





This repository has the code for reproducing the results in the **Epidemics** paper entitled [*Estimating nosocomial transmission of micro-organisms in hospital settings using patient records and culture data*.](https://www.sciencedirect.com/science/article/pii/S1755436525000052).

[![DOI](https://zenodo.org/badge/817876760.svg)](https://doi.org/10.1016/j.epidem.2025.100817)


**Authors:** Jaime Cascante Vega $^{1,4}$, Rami Yaari $^{1}$, Tal Robin $^{1}$, Lingsheng Wen $^{2}$, Jason Zucker $^{2}$, Anne-Catrin Uhlemann $^{2}$, Sen Pei $^{1}$, Jeffrey Shaman $^{1,3}$

**Affiliations:** 

1. Department of Environmental Health Sciences, Mailman School of Public Health, Columbia University, New York, NY, USA.
2. Division of Infectious Diseases, Department of Medicine, Columbia University, College of Physicians and Surgeons, New York, New York, USA.
3. Columbia Climate School, Columbia University, New York, NY, USA.
4. Current affiliation: Department of Biology, Center for Genomics and Systems Biology, New York University, New York, NY, USA

The repository tracked all the research process in the associated jupyter notebooks. Those were used to process the data, conduct parameter inference and produce the manuscript Figures. As a consequence it is not organized. In below I list what would you need to reproduce the results. The data however has individual patient information and thus is private. To have access to the data please email the corresponding authors: [Jeffrey Shaman](https://www.publichealth.columbia.edu/profile/jeffrey-shaman-phd) and [Sen Pei](https://www.publichealth.columbia.edu/profile/sen-pei).

We use an Iterated Ensemble Adjustment Kalman Filter to infer posterior parameter estimates. We developed another python package that code the iterated EAKF. To install the package run the code below.

    git clone -b numpy-version https://github.com/ChaosDonkey06/pompjax.git
    cd pompjax
    pip install .

We first present the data we used in Figure 1, the notebook associated is in [figure 1](manuscript/Fig1.ipynb). The associated Supplementary Material (SM) Figures are those presenting the data: the **Ward size distribution** ([SM Fig S1](manuscript/Fig1.ipynb)), the **Transfer matrices at the ward and building levels** ([SM Fig S2](manuscript/TransferWardAndBuildingLevel.ipynb)), the **Hospital traffic at the building level** ([SM Fig S3](manuscript/SI_Figures2.ipynb)), the **Hospital traffic at the ward level** ([SM Fig S4](manuscript/SI_Figures2.ipynb)), the **Relationship between hospitalizations and admissions** ([SM Fig S5](manuscript/SI_Figures2.ipynb)) and the **Length of stay distributions at hospital and building level** ([SM Fig S6](manuscript/AverageLOSInHospital2.ipynb)).

We then study the *"parameter identifiability"* of the agent based model. I use the *""* because people mean different things with the term and study it from different perspectives. We conducted inferences on simulated data and see if the method recovered the parameters. In the main text Figure 2 we present those inferences. The notebook to run the inferences on synthetic/simulated data are in [jupyter notebook](abm/WithoutShrinking_SyntheticLarge.ipynb). In the associated SM Figures we present **Convergence plots of inference for synthetic data** ([SM Fig S7](manuscript/ConvergencePlots.ipynb)), and investigate the **Goodness-of-fit of posterior inference on synthetic data, Monte Carlo and statistical error analysis** ([SM Fig S8](abm/MCE_synthetic.ipynb)). Lastly we present the **Hospital-level simulation of synthetic tests and calibration** ([SM Fig S9]()). In the main Figure we conducted synthetic inferences on importation rates of $\gamma\in${25\%, 50\%}. The reviewers were concerned on the ability of the system to infer parameters on lower importation rates, so we ran more inferences on $\gamma\in${5\%, 10\%, 15\%}. We present those posterior inferences in **Identifiability, parameter estimates on simulated data** ([SM Fig S10](abm/reviews_epidemics/synthetic_inference.ipynb)), and show **Convergence plots of inferences for synthetic data with low $\mathbf{\rho}$ and $\mathbf{\beta}$** ([SM Fig S11](abm/reviews_epidemics/synthetic_inference.ipynb)). Those inferences were conducted on a cluster the associated python and slurm files scripts are [inference.py](/abm/reviews_epidemics/synthetic_inference.py) and [inference.sh](/abm/reviews_epidemics/synthetic_inferences.sh)

Having confidence on the *identifiability* we went and conducted inferences on the real data for the 7 pathogens. In the main text we present the parameter inferences in Figure 3 ([Fig 3](manuscript/Fig3and4.ipynb)) and Simulated with those estimates and measured calibration at the hospital level ([Fig 4](manuscript/Fig3and4.ipynb)). The associated SM Figures include first the **Marginal posterior parameter estimates for each pathogen** ([Fig S12](manuscript/Fig3and4.ipynb)) and the associated **Building-level simulation of posterior parameter estimates and calibration** ([Fig S14]()). The reviewers were concerned with our parametrization of the importation rate $\gamma$. We extended inferences diminishing $\gamma$ until $f=30%$ of the highest value we found in the literature. This sensitivity analysis is presented in **Marginal posterior parameter estimates across different importation rate** ([Fig S13](abm/reviews_epidemics/gamma_sensitivity.ipynb)). The files for reproducing the results in a cluster are [python file](abm/reviews_epidemics/gamma_sensitivity.py) and [slurm file](abm/reviews_epidemics/gamma_sensitivity.sh).
Additionally, as our study period covered the first COVID-19 hospitalizations peak in NYC we conducted an additional sensitivity analysis: **Schematic for sensitivity analyses ignoring weeks at the beginning of the study period** ([Fig S16](abm/reviews_epidemics/covid_sense_schematic.ipynb)) and **Sensitivity analyses for the beginning of the time series, and the COVID-19 period** ([Fig S17](abm/reviews_epidemics/covid_sensitivity.ipynb)). The files for reproducing the results in a cluster are [python file](abm/reviews_epidemics/covid_period_sensitivity_no_readmission.py) and [slurm file](abm/reviews_epidemics/synthetic_covid_period_sensitivity1.sh).


The additional SM Figures are associated with a theoretical model for **Understanding the effective sensitivity** ([Fig S15](/manuscript/EffectiveSensitivity.ipynb)) and the **Total clinical cultures in each hospital and it's buildings** ([Fig S19](/manuscript/TestPerBuildingWard.ipynb)).
