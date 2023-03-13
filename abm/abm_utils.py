
from scipy.interpolate import UnivariateSpline

import pandas as pd
import numpy as np
import os

def empirical_prevalence(amro, path_to_prev="../data/amro_prevalence.csv"):
    amro_prev_df = pd.read_csv(path_to_prev)
    gamma        = amro_prev_df[amro_prev_df.amro==amro]["prevalence_mean1"].values[0]/100
    return gamma

def simulate_abm(f, f0, g, θ, model_settings):
    dates_simulation = model_settings["dates_simulation"]

    x = f0(θ)

    observations          = np.full((len(dates_simulation), model_settings["k"], model_settings["m"]), np.nan)
    observations[0, :, :] = g(0, x, θ)

    for t, date in enumerate(dates_simulation[1:]):
        x                       = f(t, x, θ)
        observations[t+1, :, :] = g(t, x, θ)
    return observations

def create_obs_infer(obs_sim, idx_infer, dates, model_settings, resample="W-Sun"):
    # obs_sim \in R^{[k x T x m]} as required by pompjax
    infer_df = pd.DataFrame(index=dates)
    for i in range(model_settings["k"]) :
        infer_df['y'+str(i+1)]   = obs_sim[i, :, idx_infer]
        infer_df['oev'+str(i+1)] = 1 +(0.2 * infer_df['y'+str(i+1)].values)**2
    infer_df                     = infer_df.resample(resample).sum()
    infer_df.index.values[-1]    = model_settings["dates"][-1]
    return infer_df


def return_score_cutoff(score, cut_off_prob=0.05):
    freq, score = np.histogram(score, bins=100, density=True)
    freq_cum    = np.cumsum(freq); freq_cum = freq_cum/freq_cum[-1]
    score       = score[1:]
    f_cum       = UnivariateSpline(score, freq_cum, s=0.001)
    sc_range    = np.linspace(np.min(score), np.max(score), 1000)
    score_cut   = sc_range[np.argmin(np.abs(f_cum(sc_range) * 100 - cut_off_prob*100))]
    return score_cut

def sample_scenarios(path_to_grid_search, cut_off=10/100):
    gs_df        = pd.read_csv( path_to_grid_search )
    sc_cutoff    = return_score_cutoff(gs_df.crps, cut_off_prob=cut_off)
    gs_df        = gs_df[gs_df.crps <= sc_cutoff].reset_index(drop=True)
    scenarios_df = gs_df.copy()
    scenarios_df = scenarios_df.sample(n=10); scenarios_df = scenarios_df[["rho", "beta", "crps", "calibration_score"]].reset_index(drop=True)
    return scenarios_df

### inference

from diagnostic_plots import convergence_plot
from utils import create_df_response
from ifeakf import ifeakf

def run_amro_synthetic(f, f0, g, fsim, model_settings, if_settings, id_run=0, path_to_save=None):
    dates        = pd.date_range(start=pd.to_datetime("2020-02-01"), end=pd.to_datetime("2021-02-28"), freq="D")

    θtruth  = np.array([model_settings["param_truth"]]).T * np.ones((model_settings["p"], model_settings["m"]))


    y_sim   = fsim(f               = f,
                    g              = g,
                    f0             = f0,
                    θ              = θtruth,
                    model_settings = model_settings)

    idx_infer = np.random.randint(model_settings["m"])
    obs_df    = create_obs_infer(y_sim.transpose(1, 0, 2), idx_infer, dates, model_settings, resample="W-Sun")

    ρmin              = 0.01/2 # test sensitivity minimum
    ρmax              = 0.2    # test sensitivity maximum
    βmin              = 0.00   # transmission rate minimum
    βmax              = 0.11   # transmission rate maximum

    state_space_range = np.array([0, 1])
    parameters_range  = np.array([[ρmin, ρmax],    [βmin, βmax]])
    σ_perturb         = np.array([(ρmax - ρmin)/4, (βmax - βmin)/4]) # (i hve the gut feeling that 0.25 is too large)

    θmle, θpost = ifeakf(process_model                = f,
                            state_space_initial_guess = f0,
                            observational_model       = g,
                            observations_df           = obs_df,
                            parameters_range          = parameters_range,
                            state_space_range         = state_space_range,
                            model_settings            = model_settings,
                            if_settings               = if_settings,
                            perturbation              = σ_perturb)

    np.savez_compressed(os.path.join(path_to_save, f"{str(id_run).zfill(3)}posterior.npz"),
                                    mle           = θmle,
                                    posterior     = θpost,
                                    observations  = y_sim,
                                    teta_truth    = θtruth,
                                    idx_infer     = idx_infer)

    ρ_df = create_df_response(θpost[0, :, :, :].mean(-2).T, time=if_settings["Nif"])
    β_df = create_df_response(θpost[1, :, :, :].mean(-2).T, time=if_settings["Nif"])

    p_dfs             = [ρ_df, β_df]
    param_label       = ["ρ", "β"]
    parameters_range  = np.array([[ρmin, ρmax], [βmin, βmax]])

    convergence_plot(θmle, p_dfs, parameters_range, param_label, param_truth=list(θtruth[:, 0]),
                        path_to_save=os.path.join(path_to_save, f"{str(id_run).zfill(3)}convergence.png"))
