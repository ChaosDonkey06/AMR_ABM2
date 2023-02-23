from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import datetime

def create_df_response(samples, time, date_init ='2020-02-01',  quantiles = [50, 80, 95], forecast_horizon=27, dates=None, use_future=False):
    """[summary]

    Args:
        samples ([type]): [description]
        time ([type]): [description]
        date_init (str, optional): [description]. Defaults to '2020-03-06'.
        forecast_horizon (int, optional): [description]. Defaults to 27.
        use_future (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if dates is not None:
        dates_fitted = dates
    else:
        dates_fitted   = pd.date_range(start=pd.to_datetime(date_init), periods=time)
        dates_forecast = pd.date_range(start=dates_fitted[-1]+datetime.timedelta(1), periods=forecast_horizon)

    dates = list(dates_fitted)
    types = ['estimate']*len(dates_fitted)
    if use_future:
        dates += list(dates_forecast)
        types  += ['forecast']*len(dates_forecast)

    results_df = pd.DataFrame(samples.T)
    df_response = pd.DataFrame(index=dates)
    # Calculate key statistics
    df_response['mean']        = results_df.mean(axis=1).values
    df_response['median']      = results_df.median(axis=1).values
    df_response['std']         = results_df.std(axis=1).values

    for quant in quantiles:
        low_q  = ((100-quant)/2)/100
        high_q = 1-low_q

        df_response[f'low_{quant}']  = results_df.quantile(q=low_q, axis=1).values
        df_response[f'high_{quant}'] = results_df.quantile(q=high_q, axis=1).values

    df_response['type']        =  types
    df_response.index.name = 'date'

    return df_response

def random_walk_perturbation(param, param_std, num_params, n_ens):
    return param + param_std * np.random.normal(size=(num_params, n_ens))

def inflate_ensembles(ens, inflation_value=1.2, n_ens=300):
    return np.mean(ens,1, keepdims=True)*np.ones((1,n_ens)) + inflation_value*(ens-np.mean(ens,1, keepdims=True)*np.ones((1,n_ens)))

def checkbound_params(params_range, params_ens):
    params_update = []
    for idx_p, p in enumerate(params_range.keys()):
        loww             = params_range[p][0]
        upp              = params_range[p][1]
        p_ens            = params_ens[idx_p, :].copy()
        idx_wrong        = np.where(np.logical_or(p_ens <loww, p_ens > upp))[0]
        idx_wrong_loww   = np.where(p_ens < loww)[0]
        idx_wrong_upp    = np.where(p_ens > upp)[0]
        idx_good         = np.where(np.logical_or(p_ens >=loww, p_ens <= upp))[0]
        p_ens[idx_wrong] = np.median(p_ens[idx_good])
        np.put(p_ens, idx_wrong_loww, loww * (1+0.2*np.random.rand( idx_wrong_loww.shape[0])) )
        np.put(p_ens, idx_wrong_upp, upp * (1-0.2*np.random.rand( idx_wrong_upp.shape[0])) )
        params_update.append(p_ens)

    return np.array(params_update)

def eakf_step_multi_obs(params_prior, obs_ens_time, obs_time, oev_time, params_range, num_obs=6):
    prior_mean_ct = obs_ens_time.mean(-1, keepdims=True) # Average over ensemble member
    prior_var_ct  = obs_ens_time.var(-1, keepdims=True)  # Compute variance over ensemble members

    idx_degenerate = np.where(prior_mean_ct==0)[0]
    prior_var_ct[idx_degenerate] =  1e-3

    post_var_ct  = prior_var_ct * oev_time / (prior_var_ct + oev_time)
    post_mean_ct = post_var_ct * (prior_mean_ct/prior_var_ct + obs_time / oev_time)
    alpha        = oev_time / (oev_time+prior_var_ct); alpha = alpha**0.5
    dy           = post_mean_ct + alpha*( obs_ens_time - prior_mean_ct ) - obs_ens_time

    # adjust parameters
    rr = np.full((len(params_range), num_obs), np.nan)
    dx = np.full((len(params_range) , obs_ens_time.shape[-1], num_obs), np.nan)

    for idx_obs in range(num_obs):
        for idx_p, p in enumerate(params_range.keys()):
            A = np.cov(params_prior[idx_p,:], obs_ens_time[idx_obs])
            rr[idx_p, idx_obs] =  A[1,0] / prior_var_ct[idx_obs]
        dx[:, :, idx_obs]      =  np.dot( np.expand_dims(rr[:, idx_obs],-1), np.expand_dims(dy[idx_obs,:], 0) )

    mean_dy    = dy.mean(0)  # Average over observation space
    mean_dx    = dx.mean(-1)
    param_post = params_prior + mean_dx
    obs_post   = obs_ens_time + mean_dy

    return param_post, obs_post

def geometric_cooling(num_iteration_if, cooling_factor=0.9):
    alphas = cooling_factor**np.arange(num_iteration_if)
    return alphas**2

def hyperbolic_cooling(num_iteration_if, cooling_factor=0.9):
    alphas = 1/(1+cooling_factor*np.arange(num_iteration_if))
    return alphas

def cooling(num_iteration_if, type_cool="geometric", cooling_factor=0.9):
    if type_cool=="geometric":
        return geometric_cooling(num_iteration_if, cooling_factor=cooling_factor)
    elif type_cool=="hyperbolic":
        return hyperbolic_cooling(num_iteration_if, cooling_factor=cooling_factor)

def sample_params_uniform(params_range, num_ensembles=100):
    param_ens_prior = []
    for p in params_range.keys():
        param_ens_prior.append( np.random.uniform( params_range[p][0], params_range[p][1]  , size=num_ensembles) )
    return np.array( param_ens_prior )

def sample_params_triangular(params_range, truth_dict, num_ensembles=100):
    param_ens_prior = []
    for p in params_range.keys():
        loww = params_range[p][0]
        upp  = params_range[p][1]
        param_ens_prior.append(  np.random.triangular(loww, np.minimum( truth_dict[p] + np.abs(np.random.rand())*(upp-loww)/2, upp) , upp,  size=num_ensembles) )
    return np.array( param_ens_prior )

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd )

def sample_params_normal(params_range, params_mean, params_var, num_ensembles=300):
    param_ens_prior = []
    for idx_p, p in enumerate(params_range.keys()):
        norm_gen = get_truncated_normal(mean=params_mean[idx_p], sd=params_var[idx_p]**(1/2), low=params_range[p][0], upp=params_range[p][1])
        param_ens_prior.append( norm_gen.rvs(num_ensembles) )
    return np.array( param_ens_prior )

def compute_oev(obs_vec, var_obs=0.2):
    return 1 + (var_obs*obs_vec)**2