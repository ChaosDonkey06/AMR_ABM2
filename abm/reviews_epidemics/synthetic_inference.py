from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import itertools
import datetime
import tqdm
import sys
import os

import matplotlib.pyplot as plt

def flatten_list(list_array):
    return list(itertools.chain(*list_array))

sys.path.insert(0, "../../pompjax/pompjax/")
sys.path.insert(0, "../..")
sys.path.insert(0, "../")

from global_config import config

results_dir          = config.get_property('results_dir')
results2_dir         = config.get_property('results2_dir')
data_dir             = config.get_property('data_dir')
paper_dir            = config.get_property('paper_dir')
data_db_dir          = config.get_property('data_db_dir')

data_cluster_dir     = config.get_property('data_cluster_dir')
results_cluster_dir  = config.get_property('results_cluster_dir')

feb_hosp_records_path = os.path.join(data_db_dir, 'long_files_8_25_2021')
path_to_save          = os.path.join(results_dir, "real_testing", "community")

COLOR_LIST1           = ["#F8AFA8", "#FDDDA0", "#F5CDB4", "#74A089"]

from utils_local.misc import amro2title, amro2cute
import matplotlib.ticker as mtick

####-####-####-####-####-####-####-####-####-####-####
import argparse

parser  = argparse.ArgumentParser(description='Create Configuration')
parser.add_argument('--idx_row', type=int, help='scenario row index', default=0)
idx_row = parser.parse_args().idx_row-1

print("running for row: ", idx_row)

####-####-####-####-####-####-####-####-####-####-####

def simulate_abm(f, f0, g, θ, model_settings):
    dates_simulation      = model_settings["dates_simulation"]
    x                     = f0(θ)
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

###-###-###-###

dates_simulation = pd.date_range(start="2020-02-01", end="2021-02-28", freq="D")

movement_df                  = pd.read_csv(os.path.join(data_cluster_dir, "long_files_8_25_2021", 'patient_movement_2022-Nov.csv'), parse_dates=['date']).drop_duplicates(subset=["date", "mrn"], keep="first")
movement_df["ward_total"]    = movement_df.apply(lambda x: x["ward"]+"-"+x["building"]+"-"+x["place"], axis=1)
movement_df                  = movement_df[movement_df["date"].isin(dates_simulation)]

mrd2id                       = {mrn: id for id, mrn in enumerate(movement_df.mrn.unique())}
ward2id                      = {ward_name: id for id, ward_name in enumerate(np.sort(movement_df.ward_total.unique()))}

movement_df["mrn_id"]        = movement_df.mrn.map(mrd2id)
movement_df["ward_id"]       = movement_df.ward_total.map(ward2id)

ward_size_df                 = movement_df.reset_index()
ward_size_df["ward_id"]      = ward_size_df["ward_total"].apply(lambda x: ward2id[x])
ward_size_df["num_patients"] = 1
ward_size_df                 = ward_size_df.groupby(["date", "ward", "ward_id"]).sum()[["num_patients"]].reset_index().drop(columns=["date"])
ward_size_df                 = ward_size_df.groupby(["ward", "ward_id"]).mean().reset_index().sort_values(by="num_patients")
ward2size                    = {r.ward_id: r.num_patients for idx_r, r in ward_size_df.iterrows()}

id2ward                      = dict((v, k) for k, v in ward2id.items())
###-###-###-###-###-###-###-###-###-###-###-###

selected_buildings = ['Allen Hospital-Allen', 'Harkness Pavilion-Columbia', 'Milstein Hospital-Columbia', 'Mschony-Chony', 'Presbyterian Hospital-Columbia']
building2id        = {selected_buildings[i]: i for i in range(len(selected_buildings))}

def building2observation(building):
    if building in selected_buildings:
        return building2id[building]
    else:
        return 5

ward_names                   = np.sort(list(movement_df.ward_total.unique()))
ward_names_df                = pd.DataFrame(ward_names, columns=["ward"])
ward_names_df                = pd.DataFrame(ward_names, columns=["ward"])
ward_names_df["building"]    = ward_names_df["ward"].apply(lambda x: "-".join(x.split("-")[1:]))
ward_names_df["buidling_id"] = ward_names_df["building"].apply(lambda x: building2observation(x) )
ward_names_df["ward_id"]     = ward_names_df.apply(lambda x: np.where(ward_names_df.ward == x.ward)[0][0], axis=1)

###-###-###-###-###-###-###-###-###-###-###-###

building2id            = {selected_buildings[i]: i for i in range(len(selected_buildings))}
wardid2buildingid      = {row.ward_id: row.buidling_id for i, row in ward_names_df.iterrows()}
ward2buildingid        = {row.ward: row.buidling_id for i, row in ward_names_df.iterrows()}
movement_df["cluster"] = movement_df.ward_id.map(wardid2buildingid)

###-###-###-###-###-###-###-###-###-###-###-###

from models import amr_abm, observe_cluster_individual

if_settings = {
        "Nif"                : 30,          # number of iterations of the IF
        "type_cooling"       : "geometric", # type of cooling schedule
        "shrinkage_factor"   : 0.9,         # shrinkage factor for the cooling schedule
        "inflation"          : 1.01         # inflation factor for spreading the variance after the EAKF step
        }

dates_simulation = pd.date_range(start=pd.to_datetime("2020-02-01"), end=pd.to_datetime("2021-02-28"), freq="D")
model_settings   = {
                    "m"                 : 300,
                    "p"                 : 2,
                    "n"                 : movement_df.mrn_id.unique().shape[0],
                    "k"                 : movement_df.cluster.unique().shape[0],
                    "dates"             : pd.date_range(start="2020-02-01", end="2021-02-28", freq="D"),
                    "dates_simulation"  : pd.date_range(start="2020-02-01", end="2021-02-28", freq="D"),
                    "T"                 : len(dates_simulation),  # time to run
                    "num_build"         : len(np.unique(list(wardid2buildingid.values()))),
                    "k"                 : len(np.unique(list(wardid2buildingid.values())))# observing at the building aggregation
                }

assim_dates                       = list(pd.date_range(start=pd.to_datetime("2020-02-01"), end=pd.to_datetime("2021-02-28"), freq="W-Sun"))
assim_dates[-1]                   = dates_simulation[-1]
if_settings["assimilation_dates"] = assim_dates

from data_utils import create_obs_building_amro
from infer_utils import run_amro_inference

###-###-###-###-###-###-###-###-###-###-###-###

gammas_search = [5/100, 10/100, 15/100]
betas_search  = [0.01, 0.05, 0.1]
rho_search    = [1/100, 5/100, 10/100, 15/100]

idx_sce = 0
scenarios_large_df = pd.DataFrame(columns=["scenario", "gamma", "beta", "rho"])

for g in gammas_search:
    for b in betas_search:
        for r in rho_search:
            df = pd.DataFrame({"scenario": f"scenario{int(idx_sce+1)}",
                                                            "gamma": g,
                                                            "beta": b,
                                                            "rho": r}, index=[0])
            scenarios_large_df = pd.concat([scenarios_large_df, df], axis=0).reset_index(drop=True)
            idx_sce += 1


from utils_local.misc import amro2title, amro2cute
from abm_utils import run_amro_synthetic
import matplotlib.pyplot as plt

#path_to_save = os.path.join(results_cluster_dir, "synthetic_inferences", "abm")
for id_run in range(4):

    row   = scenarios_large_df.iloc[idx_row]
    gamma = row["gamma"]

    path_to_save = os.path.join(results_cluster_dir, "synthetic_inferences", row["scenario"])

    print(f"\t Synthetic {idx_row+1}/{len(scenarios_large_df)}", end="\r")

    model_settings["param_truth"]     = [row["rho"], row["beta"]]
    if_settings["adjust_state_space"] = False
    if_settings["shrink_variance"]    = False

    os.makedirs(path_to_save, exist_ok=True)

    if os.path.isfile(os.path.join(path_to_save, f"{str(id_run).zfill(3)}posterior.npz")):
        continue
    else:

        model_settings["param_truth"]     = [row["rho"], row["beta"]]

        alpha               = 1/120
        init_state          = lambda θ: amr_abm(t = 0,
                                                agents_state   = np.zeros((model_settings["n"], model_settings["m"])),
                                                gamma          = gamma,
                                                beta           = θ[1, :],
                                                alpha          = alpha,
                                                movement       = movement_df[movement_df["date"]==dates_simulation[0]],
                                                ward2size      = ward2size,
                                                model_settings = model_settings)

        process       = lambda t, x, θ: amr_abm(t = t,
                                                agents_state   = x,
                                                gamma          = gamma,
                                                beta           = θ[1, :],
                                                alpha          = alpha,
                                                movement       = movement_df[movement_df["date"]==dates_simulation[t]],
                                                ward2size      = ward2size,
                                                model_settings = model_settings)

        obs_model = lambda t, x, θ: observe_cluster_individual(t = t,
                                                                agents_state   = x,
                                                                rho            = θ[0, :],
                                                                movement       = movement_df[movement_df["date"]==dates_simulation[t]],
                                                                model_settings = model_settings)

        run_amro_synthetic(f               = process,
                            f0             = init_state,
                            g              = obs_model,
                            fsim           = simulate_abm,
                            model_settings = model_settings,
                            if_settings    = if_settings,
                            id_run         = id_run,
                            path_to_save   = path_to_save,
                            use_mean       = False)