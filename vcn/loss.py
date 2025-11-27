#!/usr/bin/env python3
import torch


@torch.compile
def JAB(q_0, q_t, weights):
    L = torch.sum(weights * torch.square(q_0 - q_t))
    # L = torch.mean(torch.square(q_0 - q_t))
    return L / torch.sum(weights)
    # return L


@torch.compile
def loss_vcns_soft_endpoints(model, data, k_scale=100.0):
    data_0, data_t, weights, k_a0, k_at, k_b0, k_bt, center_0, center_t = data
    q_0 = model(data_0)
    q_t = model(data_t)
    loss = JAB(q_0, q_t, weights)
    # add harmonic restraint for basin A
    res_A = k_a0 * torch.square(q_0 - center_0) + k_at * torch.square(q_t - center_t)
    # add harmonic restraint for basin B
    res_B = k_b0 * torch.square(q_0 - center_0) + k_bt * torch.square(q_t - center_t)
    # weight the restraints
    # res = torch.sum(weights * (res_A + res_B)) / torch.sum(weights)
    res = torch.mean(res_A + res_B)
    return loss + k_scale*res
    # return res

