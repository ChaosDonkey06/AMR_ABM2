import pandas as pd

def create_obs_building_amro(amro, model_settings, ward2building_dict, path_to_data, resample="W-Sun", p_oev=0.2):
    k                     = model_settings["k"]
    amro_df               = pd.read_csv(path_to_data, parse_dates=["date"]).drop(columns=["Unnamed: 0"])
    amro_df               = amro_df[amro_df["amro"]==amro]

    amro_df["buildingid"] = amro_df["ward_total"].map(ward2building_dict)

    amro_df               = amro_df.groupby(["date", "buildingid"]).sum(numeric_only=True).unstack([1]).resample(resample).sum(numeric_only=True).stack().reset_index()
    amro_df["obs_name"]   = amro_df["buildingid"].map({i: f"y{i+1}" for i in range(k)})
    amro_df               = pd.pivot(amro_df, index="date", columns="obs_name", values="num_positives").reset_index()
    for i in range(k):
        amro_df['oev'+str(i+1)] = 1 +(p_oev * amro_df['y'+str(i+1)].values)**2
    amro_df = amro_df.set_index("date")
    return amro_df