#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
import gzip
import pandas as pd



def split_train_val(input_data, val_ratio=0.1):
    # split out the val set
    train_data, val_data = train_test_split(input_data, train_size=1.0-val_ratio, shuffle=True)
    return train_data, val_data


def write_dataframe_to_file(df, output_filename):
    df.to_parquet(output_filename)


def renormalize_weights(weights):
    import numpy as np
    return weights / np.sum(weights) * np.shape(weights)[0]


def renormalize_log_weights(log_weights):
    from scipy.special import logsumexp
    import numpy as np
    return log_weights - logsumexp(log_weights) + np.log(np.shape(log_weights)[0])


def read_traj(input_filelist, columns, stride, discard=0):
    data_list = []
    columns_to_read = columns.copy()
    columns_to_read.append('step')
    for filename in input_filelist:
        print(f'Load file: {filename}')
        # read the header to determine the columns available at first to save memory
        with gzip.open(filename, 'rt') as gz_input:
            available_columns = pd.read_csv(gz_input, nrows=0).columns.tolist()
            if not (set(columns_to_read) <= set(available_columns)):
                raise RuntimeError(f'Columns to read: {columns_to_read} '
                                   f'are not in available columns: {available_columns}')
            if 'weight' in available_columns:
                columns_to_read.append('weight')
        with gzip.open(filename, 'rt') as gz_input:
            data = pd.read_csv(
                gz_input, usecols=columns_to_read,
                skiprows=lambda x: (x > 0) and ((x-1) % stride != 0))
            if 'weight' not in data:
                data['weight'] = 1.0
            data_list.append(data[['step', 'weight'] + columns])
    all_data = pd.concat(data_list, ignore_index=True)
    max_step = all_data['step'].max()
    print(f'The maximum number of steps of the trajectories is {max_step}')
    if max_step <= discard:
        raise RuntimeError(f'Too many steps ({discard}) are going to be discarded.')
    all_data = all_data.loc[all_data['step'] >= discard]
    all_data.drop('step', axis=1, inplace=True)
    return all_data


def preprocess_traj(data, val_ratio=0.2, time_shift=2):
    data_length = data.shape[0]
    data_origin = data.iloc[:data_length - time_shift]
    data_target = data[time_shift:]
    new_names_origin = list()
    new_names_target = list()
    for column_name in data_origin.columns:
        new_names_origin.append(f'{column_name}_origin')
        new_names_target.append(f'{column_name}_target')
    data_origin.columns = new_names_origin
    data_target.columns = new_names_target
    data_origin.reset_index(drop=True, inplace=True)
    data_target.reset_index(drop=True, inplace=True)
    time_lagged_data = pd.concat([data_origin, data_target], axis=1)
    time_lagged_data['weight'] = time_lagged_data['weight_origin']* time_lagged_data['weight_target']
    train_data, val_data = split_train_val(time_lagged_data, val_ratio=val_ratio)
    return time_lagged_data, train_data, val_data