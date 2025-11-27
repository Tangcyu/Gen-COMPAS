#!/usr/bin/env python3
import pandas as pd
import torch


def write_numpy_to_file(data, header, filename):
    df = pd.DataFrame(data)
    df.columns = header
    df.to_parquet(filename)


def predict_datasets(files, variables, model, output_prefix, device='cpu'):
    datasets = [pd.read_parquet(x)[variables] for x in files]
    all_data = pd.concat(datasets, axis=0).to_numpy()
    pred_data = model(torch.tensor(all_data, dtype=torch.float32, device=device)).cpu().detach().numpy()
    num_variables = pred_data.shape[1]
    header = [f'CV{i}' for i in range(1, num_variables + 1)]
    write_numpy_to_file(data=pred_data, header=header,
                        filename=f'{output_prefix}_train_all_encoded.dat')
