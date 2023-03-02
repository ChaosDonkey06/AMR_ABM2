import pandas as pd
import numpy as np

def create_population_data(path_to_file, date_start=pd.to_datetime("2020-02-01"), date_end=pd.to_datetime("2021-02-28")):

    dates_simulation = pd.date_range(start=date_start, end=date_end, freq="D")

    data_df  = pd.read_csv( path_to_file, parse_dates=['date'])
    data_df  = data_df[data_df.date.isin(dates_simulation)]
    A_df     = pd.pivot(data_df, index='ward', columns='date', values='num_admitted')
    D_df     = pd.pivot(data_df, index='ward', columns='date', values='num_discharged')
    H_df     = pd.pivot(data_df, index='ward', columns='date', values='num_hospitalized')
    tests_df = pd.pivot(data_df, index='ward', columns='date', values='num_tested')
    Hmean_df = H_df.mean(axis=1)

    return A_df, D_df, H_df, tests_df, Hmean_df

def create_time_transfers(path_to_file, num_wards, ward_names, date_start=pd.to_datetime("2020-02-01"), date_end=pd.to_datetime("2021-02-28")):
    dates_simulation = pd.date_range(start=date_start, end=date_end, freq="D")
    transfers_df     = pd.read_csv(path_to_file, parse_dates=['date'])
    transfers_df     = transfers_df[transfers_df.date.isin(dates_simulation)]
    M_df             = np.zeros((num_wards, num_wards, len(dates_simulation)+1))
    for i in range(num_wards):
        ward_from = ward_names[i]
        for j in range(num_wards):
            ward_to      = ward_names[j]
            transfers_ij = transfers_df[(transfers_df.ward_from==ward_from) & (transfers_df.ward_to==ward_to)]

            if(transfers_ij.shape[0] > 0) :
                dates_ij                = transfers_ij.date.values
                dates_ind               = np.where(np.in1d(dates_ij, dates_simulation))[0]
                transfered              = transfers_ij.num_transfered.values
                M_df[i, j, dates_ind-1] = transfered
    return M_df
