
import pandas as pd
import os

def load_movement_df(path_to_movement_data, return_ward=False):
    date_min         = pd.to_datetime("2020-02-01")
    date_max         = pd.to_datetime("2021-02-28")
    dates_simulation = pd.date_range(start=date_min, end=date_max)

    movement_df  = pd.read_csv(os.path.join(path_to_movement_data,  'movement.csv'), parse_dates=["date"])#.drop(columns=["Unnamed: 0"])
    movement_df  = movement_df.dropna(subset=["date"])
    movement_df  = movement_df.set_index("date")
    movement_df  = movement_df.loc[dates_simulation]

    id2ward = pd.read_csv(  os.path.join(path_to_movement_data, "ward2id.csv"))
    id2ward = {row.ward_id: row.ward for idx, row in id2ward.iterrows()}

    id2room = pd.read_csv(  os.path.join(path_to_movement_data, "room2id.csv"))
    id2room = {row.room_id: row.room for idx, row in id2room.iterrows()}

    id2mrn = pd.read_csv(  os.path.join(path_to_movement_data, "mrn2id.csv"))
    id2mrn = {row.mrn_id: row.mrn for idx, row in id2mrn.iterrows()}

    movement_df["mrn"]  = movement_df["mrn_id"].map(id2mrn)
    movement_df["ward"] = movement_df["ward_id"].map(id2ward)
    movement_df["room"] = movement_df["room_id"].map(id2room)

    movement_df = movement_df.drop(columns=["ward_id", "room_id", "mrn_id"])

    ward2id     = {ward: idx for idx, ward in enumerate( movement_df.ward.unique())}
    room2id     = {room: idx for idx, room in enumerate( movement_df.room.unique())}
    mrn2id      = {mrn: idx for idx, mrn in enumerate( movement_df.mrn.unique())}

    movement_df["ward_id"] = movement_df["ward"].map(ward2id)
    movement_df["room_id"] = movement_df["room"].map(room2id)
    movement_df["mrn_id"]  = movement_df["mrn"].map(mrn2id)
    movement_df["test"]    = movement_df["test"].map(lambda x: int(x))

    communities_df         = pd.read_csv(os.path.join(path_to_movement_data,  "ward2cluster.csv"))

    ward2community              = communities_df[["ward_id", "community"]].copy()
    ward2community["community"][ward2community["community"]>=6] = 6

    ward2community["community"] = ward2community["community"]-1
    ward2community              = {r.ward_id: r.community for idx, r in ward2community.iterrows()}

    movement_df["cluster_id"] = movement_df["ward_id"].map(ward2community)

    if return_ward:
        movement_df               = movement_df[['ward', "first_day", "cluster_id", "ward_id", "mrn_id", "test", "specimen", "specimen_group", "organism_name", "organism_id"]]

    else:
        movement_df               = movement_df[["first_day", "cluster_id", "ward_id", "mrn_id", "test", "specimen", "specimen_group", "organism_name", "organism_id"]]

    return movement_df, ward2community


def ward2size(movement_dataframe):
    wrd_size_df = movement_dataframe.reset_index()
    wrd_size_df["num_patients"] = 1
    wrd_size_df = wrd_size_df.groupby(["date", "ward", "ward_id"]).sum()[["num_patients"]].reset_index().drop(columns=["date"])
    wrd_size_df = wrd_size_df.groupby(["ward", "ward_id"]).mean().reset_index().sort_values(by="num_patients")
    return wrd_size_df