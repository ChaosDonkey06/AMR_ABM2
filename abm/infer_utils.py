from diagnostic_plots import convergence_plot
from utils import create_df_response
from ifeakf import ifeakf
import numpy as  np
import os

def run_amro_inference(f, f0, g, obs_df, model_settings, if_settings, id_run=0, path_to_save=None):

    ρmin              = 0.01/2 # test sensitivity minimum
    ρmax              = 0.2    # test sensitivity maximum
    # - # - # - # - #
    βmin              = 0.00   # transmission rate minimum
    βmax              = 0.2    # transmission rate maximum

    state_space_range = np.array([0, 1])
    parameters_range  = np.array([[ρmin, ρmax],    [βmin, βmax]])
    σ_perturb         = np.array([(ρmax - ρmin)/10, (βmax - βmin)/10]) # (i hve the gut feeling that 0.25 is too large)

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
                                    posterior     = θpost)

    ρ_df = create_df_response(θpost[0, :, :, :].mean(-2).T, time=if_settings["Nif"])
    β_df = create_df_response(θpost[1, :, :, :].mean(-2).T, time=if_settings["Nif"])

    p_dfs             = [ρ_df, β_df]
    param_label       = ["ρ", "β"]
    parameters_range  = np.array([[ρmin, ρmax], [βmin, βmax]])

    convergence_plot(θmle, p_dfs, parameters_range, param_label, param_truth=None,
                        path_to_save=os.path.join(path_to_save, f"{str(id_run).zfill(3)}convergence.png"))
