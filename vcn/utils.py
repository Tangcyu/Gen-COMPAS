#!/usr/bin/env python3
import pandas as pd
import torch
import numpy as np


def load_trajs_to_dataset(traj_filenames: list[str], dataset_reader, **kwargs):
    traj_list = [pd.read_parquet(traj_filename) for traj_filename in traj_filenames]
    all_traj = pd.concat(traj_list, axis=0)
    return dataset_reader(data=all_traj, **kwargs)

def calc_committors_id(model,determine_AB,positions, device):
    # convert the input to pytorch tensor
    print(positions.shape)
    # input_tensor = torch.tensor(np.sin(positions[cv]*np.pi/180) for cv in positions)
    input_tensor = torch.tensor([np.sin(positions[:,0]*np.pi/180), np.sin(positions[:,1]*np.pi/180), np.sin(positions[:,2]*np.pi/180), np.cos(positions[:,0]*np.pi/180), np.cos(positions[:,1]*np.pi/180), np.cos(positions[:,2]*np.pi/180)], dtype=torch.float, device=device)
    # predict the output using the model
    print(input_tensor.T[0])
    exit()
    output_tensor = model.forward_id(input_tensor.T)

    nn_results = output_tensor.cpu().detach().numpy().flatten()
    states = np.apply_along_axis(determine_AB, 1, positions)
    #output_results = np.where(states == 'A', 0.0, nn_results)
    #output_results = np.where(states == 'B', 1.0, output_results)
    #output_tensor = torch.where(output_tensor<0.0, 0.0, output_tensor)
    #output_tensor = torch.where(output_tensor>1.0, 1.0, output_tensor)
    # convert the output tensor back to numpy
    nn_results = output_tensor.cpu().detach().numpy().flatten()
    # states = np.apply_along_axis(determine_AB, 1, positions)
    # output_results = np.where(states == 'A', 0.0, nn_results)
    # output_results = np.where(states == 'B', 1.0, output_results)
    output_results = nn_results
    return output_results
 
def calc_committors_sig(model,determine_AB,positions, device):
    # convert the input to pytorch tensor
    print(positions.shape)
    input_tensor = torch.tensor([np.sin(positions[:,0]*np.pi/180), np.sin(positions[:,1]*np.pi/180), np.sin(positions[:,2]*np.pi/180), np.cos(positions[:,0]*np.pi/180), np.cos(positions[:,1]*np.pi/180), np.cos(positions[:,2]*np.pi/180)], dtype=torch.float, device=device)
    # predict the output using the model
    print(input_tensor.shape)
    output_tensor = model(input_tensor.T)

    nn_results = output_tensor.cpu().detach().numpy().flatten()
    states = np.apply_along_axis(determine_AB, 1, positions)
    #output_results = np.where(states == 'A', 0.0, nn_results)
    #output_results = np.where(states == 'B', 1.0, output_results)
    #output_tensor = torch.where(output_tensor<0.0, 0.0, output_tensor)
    #output_tensor = torch.where(output_tensor>1.0, 1.0, output_tensor)
    # convert the output tensor back to numpy
    nn_results = output_tensor.cpu().detach().numpy().flatten()
    # states = np.apply_along_axis(determine_AB, 1, positions)
    # output_results = np.where(states == 'A', 0.0, nn_results)
    # output_results = np.where(states == 'B', 1.0, output_results)
    output_results = nn_results
    return output_results

def calc_committors_Z_matrix(model,cv_positions,device,layer="sig"):
    # convert the input to pytorch tensor
    print(cv_positions.shape)
    input_tensor = torch.tensor(cv_positions, dtype=torch.float, device=device)
    # predict the output using the model
    print(input_tensor.shape)
    if layer == 'sig':
        output_tensor = model(input_tensor.T)
    if layer == 'id':
        output_tensor = model.forward_id(input_tensor.T)

    nn_results = output_tensor.cpu().detach().numpy().flatten()

    output_results = nn_results
    return output_results