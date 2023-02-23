from .infer_utils import sample_params_normal, sample_params_uniform, random_walk_perturbation, inflate_ensembles, checkbound_params, eakf_step_multi_obs, cooling
from tqdm import tqdm
import pandas as pd
import numpy as np

def IF2_eakf_ABM(model, pos_obs_df, neg_obs_df, movement_df, param_prior_dict, if2_settings, abm_settings, perturb_time=True):

    obs_w_chunk_df = pos_obs_df
    neg_w_chunk_df = neg_obs_df
    cooling_factor = cooling(if2_settings["num_iters_mif"], type_cool=if2_settings["type_cooling"], cooling_factor=if2_settings["alpha_mif"])

    param_range    = np.array([v for k, v in param_prior_dict.items()])
    std_param      = param_range[:,1] - param_range[:,0]
    SIG            = std_param ** 2 / 4; #  initial covariance of parameters

    # Perturbation is proportional to the prior range of search.
    perturbation     = np.array([std_param % list(np.round(std_param)+0.1)]).T
    num_steps        = len(obs_w_chunk_df)

    param_mean_iter  = np.full((if2_settings["num_params"],       if2_settings["num_iters_mif"]+1), np.nan)                                         # Array to store posterior parameters in iterations.
    para_post_all    = np.full((if2_settings["num_params"],       if2_settings["num_ensembles"], num_steps, if2_settings["num_iters_mif"]), np.nan) # Array to store posterior parameters.
    param_iter       = np.full((if2_settings["num_params"],       if2_settings["num_ensembles"], if2_settings["num_iters_mif"]), np.nan)

    obs_post_all_pos = np.full((if2_settings["num_observations"], if2_settings["num_ensembles"], num_steps, if2_settings["num_iters_mif"]), np.nan) # Array for store posterior observations
    obs_post_all_neg = np.full((if2_settings["num_observations"], if2_settings["num_ensembles"], num_steps, if2_settings["num_iters_mif"]), np.nan) # Array for store posterior observations
    p_priors_all     = np.full((if2_settings["num_params"],       if2_settings["num_ensembles"], if2_settings["num_iters_mif"]), np.nan)

    dates_assimilation     = obs_w_chunk_df.index.get_level_values(0).values
    dates_assimilation[-1] = abm_settings["dates"][-1]

    α            = np.random.uniform( 1/365, 1/175, size=(abm_settings["num_patients"], if2_settings["num_ensembles"]))
    perturb_time = True

    print(f"Running MIF  \n")
    for n in tqdm(range(if2_settings["num_iters_mif"])):
        if n==0: # Initial IF iteration
            p_prior               = sample_params_uniform(param_prior_dict, num_ensembles=if2_settings["num_ensembles"])
            param_mean_iter[:, n] = np.mean(p_prior, -1)
            p_priors_all[:,:,n]   = p_prior

        else:
            params_mean           = param_mean_iter[:,n]
            params_var            = SIG * cooling_factor[n]
            p_prior               = sample_params_normal(param_prior_dict, params_mean, params_var, num_ensembles=if2_settings["num_ensembles"])
            p_priors_all[:,:,n]   = p_prior

        patients_state    = np.zeros((abm_settings["num_patients"], if2_settings["num_ensembles"]))
        param_post_time   = np.full((if2_settings["num_params"], if2_settings["num_ensembles"], num_steps), np.nan)

        obs_post_time_pos = np.full((abm_settings["num_clusters"], if2_settings["num_ensembles"], num_steps), np.nan)
        obs_post_time_neg = np.full((po["num_clusters"], if2_settings["num_ensembles"], num_steps), np.nan)

        idx_date_update   = 0

        # Init observation arrays.
        chunk_pos_t = np.zeros((abm_settings["num_clusters"], if2_settings["num_ensembles"]))
        chunk_neg_t = np.zeros((abm_settings["num_clusters"], if2_settings["num_ensembles"]))

        for idx_date, date in enumerate(abm_settings["dates"]):
            # Integrate model
            γ = p_prior[0, :]
            β = p_prior[1, :]

            movement_date = movement_df.loc[date]
            patients_state, _, chunk_pos, _, chunk_neg = model(patients_state, γ, β, α, movement_date)

            chunk_pos_t += chunk_pos
            chunk_neg_t += chunk_neg

            if pd.to_datetime(date) == pd.to_datetime(dates_assimilation[idx_date_update]):
                # Perturb parameters according to the define mapping
                if perturb_time:
                    # Transform parameters for perturbation
                    std_params = perturbation * cooling_factor[n]
                    p_prior    = random_walk_perturbation(p_prior, std_params, if2_settings["num_params"], if2_settings["num_ensembles"])

                # Inflate parameters
                p_prior = inflate_ensembles(p_prior, inflation_value=if2_settings["lambda_inf"])
                p_prior = checkbound_params(param_prior_dict, p_prior)

                # first adjust using only positives
                oev_pos    = obs_w_chunk_df.loc[date][[f"oev_{idx_chunk}" for idx_chunk in range(abm_settings["num_clusters"])]].values
                pos_time   = obs_w_chunk_df.loc[date][[f"pos_{idx_chunk}" for idx_chunk in range(abm_settings["num_clusters"])]].values

                # then adjust using negatives
                oev_neg    = neg_w_chunk_df.loc[date][[f"oev_{idx_chunk}" for idx_chunk in range(abm_settings["num_clusters"])]].values
                neg_time   = neg_w_chunk_df.loc[date][[f"pos_{idx_chunk}" for idx_chunk in range(abm_settings["num_clusters"])]].values

                param_post = p_prior.copy()

                param_post, obs_post_pos = eakf_step_multi_obs(param_post, chunk_pos_t, np.expand_dims(pos_time, -1),  np.expand_dims(oev_pos, -1), param_prior_dict, int(if2_settings["num_observations"] )) # Use both positives to adjust
                param_post               = checkbound_params(param_prior_dict, params_ens=param_post)

                param_post, obs_post_neg = eakf_step_multi_obs(param_post, chunk_neg_t, np.expand_dims(neg_time, -1),  np.expand_dims(oev_neg, -1), param_prior_dict, int(if2_settings["num_observations"] )) # Use negatives to adjust
                param_post               = checkbound_params(param_prior_dict, params_ens=param_post)

                obs_post_time_pos[:, :, idx_date_update]    = obs_post_pos
                obs_post_time_neg[:, :, idx_date_update]    = obs_post_neg

                # Use posterior as next prior
                p_prior                              = param_post.copy()
                param_post_time[:,:,idx_date_update] = param_post
                idx_date_update += 1

                chunk_pos_t = np.zeros((abm_settings["num_clusters"], if2_settings["num_ensembles"]))
                chunk_neg_t = np.zeros((abm_settings["num_clusters"], if2_settings["num_ensembles"]))

        para_post_all[:,:,:,n]    = param_post_time
        param_mean_iter[:,n+1]    = param_post_time.mean(-1).mean(-1)
        obs_post_all_pos[:,:,:,n] = obs_post_time_pos
        obs_post_all_neg[:,:,:,n] = obs_post_time_neg

    return obs_post_all_pos, obs_post_all_neg, para_post_all, param_iter, param_mean_iter