#!/usr/bin/env python3
import torch.nn as nn
import torch
import json


activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU()
}


class Dense(nn.Module):
    def __init__(self, name, in_features, out_features, activation, bias=True, device=None, dtype=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.sub_layers = nn.Sequential()
        self.sub_layers.append(nn.Linear(self.in_features, self.out_features, self.bias, self.device, self.dtype))
        if activation != 'linear':
            self.sub_layers.append(activations[activation])

    def forward(self, x):
        return self.sub_layers(x)


def parse_encoder_layers(json_file, num_input_features):
    encoder_layers = list()
    with open(json_file, 'r') as f_json:
        all_data = f_json.read()
        json_dict = json.loads(all_data)
        n_in = -1
        for index, layer_config in enumerate(json_dict['encoder_layers'], start=1):
            layer_name = f'encoder_layer_{index}'
            layer_type = 'Dense'
            if 'type' in layer_config:
                layer_type = layer_config['type']
            if layer_type == 'Dense':
                n_out = layer_config['units']
                activation_function = layer_config['activation_function']
                if n_in == -1:
                    n_in = num_input_features
                layer = Dense(layer_name, n_in, n_out, activation_function)
                encoder_layers.append(layer)
                # prepare next layer
                n_in = n_out
    return encoder_layers


class Encoder(nn.Module):
    def __init__(self, num_input_features=None):
        super(Encoder, self).__init__()
        self.in_features = num_input_features
        self.out_features = None
        self.model = None
        self.sig = nn.Sigmoid()

    def build_from_custom_json(self, json_file):
        if json_file is not None:
            encoder_layers = parse_encoder_layers(json_file, self.in_features)
            self.model = nn.Sequential()
            self.out_features = self.in_features
            for layer in encoder_layers:
                for sub_layer in layer.sub_layers:
                    if hasattr(sub_layer, 'out_features'):
                        self.out_features = sub_layer.out_features
                    self.model.append(sub_layer)

    def build(self, nodes_per_layer: list, activation_per_layer: list):
        self.model = nn.Sequential()
        self.out_features = self.in_features
        for num_nodes, activation in zip(nodes_per_layer, activation_per_layer):
            self.model.append(nn.Linear(in_features=self.out_features, out_features=num_nodes))
            self.model.append(activation)
            self.out_features = num_nodes 

    def forward(self, x):
        return self.sig(self.model(x))
        # return self.model(x)
    
    @torch.jit.export
    def forward_id(self, x):
        return self.model(x)

    def add_linear_layer_same_shape(self, w, b, device):
        layer = nn.Linear(self.out_features, self.out_features, device=device)
        layer.weight = nn.Parameter(w.type(torch.float32))
        layer.bias = nn.Parameter(b.type(torch.float32))
        self.model.append(layer)

    def dump_weights(self, output_prefix):
        # MLCV compatible code
        import numpy as np
        layers_info = dict()
        index = 1
        for layer in self.model:
            if isinstance(layer, nn.Linear) is False:
                activation_name = type(layer).__name__.lower()
                if activation_name != 'identity':
                    # this is an activation layer
                    layers_info[index-1]['Activation'] = type(layer).__name__.lower()
            else:
                # for linear layer
                name = f'encoder_layer_{index}'
                weights = layer.weight.cpu().detach().numpy()
                biases = layer.bias.cpu().detach().numpy()
                weights_file = f'{output_prefix}_{name}_weights.txt'
                biases_file = f'{output_prefix}_{name}_biases.txt'
                np.savetxt(weights_file, weights)
                np.savetxt(biases_file, biases)
                layers_info[index] = {'type': 'Dense',
                                      'WeightsFile': weights_file,
                                      'BiasesFile': biases_file,
                                      'Activation': 'linear'}
                index += 1
        with open(f'{output_prefix}_layer_info.json', 'w') as f_json:
            print(f'Save weights and biases configuration for MLCV to {f_json.name}')
            json.dump(layers_info, f_json, indent=4)

    def get_extra_state(self):
        return {'in_features': self.in_features,
                'out_features': self.out_features}

    def set_extra_state(self, state):
        if isinstance(state, dict):
            self.in_features = state['in_features']
            self.out_features = state['out_features']
