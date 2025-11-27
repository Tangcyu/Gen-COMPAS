#!/usr/bin/env python3
import argparse
import torch
from pprint import pprint
from vcn.model import Encoder
from vcn.train import train_model
from vcn.predict import predict_datasets
from vcn.utils import load_trajs_to_dataset
from vcn.custom_dataloader import MyDataLoader
from vcn.loss import loss_vcns_soft_endpoints


class CommittorDataset(torch.utils.data.Dataset):
    def __init__(self, data, variables, device):
        variables_origin = [f'{v}_origin' for v in variables]
        variables_target = [f'{v}_target' for v in variables]
        self.data_0 = torch.tensor(data[variables_origin].to_numpy(), dtype=torch.float32, device=device)
        self.data_t = torch.tensor(data[variables_target].to_numpy(), dtype=torch.float32, device=device)
        self.weights = torch.tensor(data[['weight']].to_numpy(), dtype=torch.float32, device=device)
        self.Ka_0 = torch.tensor(data[['Ka_origin']].to_numpy(), dtype=torch.int, device=device)
        self.Ka_t = torch.tensor(data[['Ka_target']].to_numpy(), dtype=torch.int, device=device)
        self.Kb_0 = torch.tensor(data[['Kb_origin']].to_numpy(), dtype=torch.float32, device=device)
        self.Kb_t = torch.tensor(data[['Kb_target']].to_numpy(), dtype=torch.float32, device=device)
        self.center_0 = torch.tensor(data[['center_origin']].to_numpy(), dtype=torch.float32, device=device)
        self.center_t = torch.tensor(data[['center_target']].to_numpy(), dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.data_0)

    def __getitem__(self, item):
        return self.data_0[item], self.data_t[item], self.weights[item], \
               self.Ka_0[item], self.Ka_t[item], \
               self.Kb_0[item], self.Kb_t[item], \
               self.center_0[item], self.center_t[item]


def main():
    parser = argparse.ArgumentParser(
        description='Train variational committor neural networks (VCNs) with soft restraints')
    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument('--variables', type=str, nargs='+', required=True,
                               help='number of variables in the training set')
    required_args.add_argument('--train_set', nargs='+', help='training dataset(s)', required=True)
    required_args.add_argument('--val_set', nargs='+', help='validation dataset(s)', required=True)
    required_args.add_argument('--output_prefix', type=str, help='output prefix', required=True)
    parser.add_argument('--device', type=str, default='cpu',
                        help='training device, available options are \'cpu\' and \'cuda\'')
    parser.add_argument('--previous_training', default=None, help='load previous training data from a checkpoint file')
    parser.add_argument('--previous_model_only',
                        action='store_true',
                        help='only load previous model without other parameters (such as the number of epochs and the '
                             'optimizer)')
    parser.add_argument('--previous_train_set', default=None, nargs='+',
                        help='previous training dataset (for checking only)')
    parser.add_argument('--model_config', default=None,
                        help='build model from a JSON file for the first iteration')
    parser.add_argument('--epochs', default=5000, type=int, help='training epochs')
    parser.add_argument('--patience', default=20, type=int, help='early stopping patience')
    parser.add_argument('--batch_size_factor', default=0.6, type=float,
                        help='a factor from 0 to 1 to determine the batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
    args = parser.parse_args()
    old_checkpoint = None
    num_inputs = len(args.variables)
    model_to_train = Encoder(num_input_features=num_inputs)
    model_to_train.build_from_custom_json(args.model_config)
    if args.previous_training is not None:
        # load model from previous round
        old_checkpoint = torch.load(args.previous_training)
        model_to_train.load_state_dict(old_checkpoint['model_state_dict'])
    model_to_train.to(device=args.device)
    print(model_to_train)
    # load data
    print('Load training dataset(s) from:')
    pprint(args.train_set)
    train_set = load_trajs_to_dataset(args.train_set, CommittorDataset, variables=args.variables, device=args.device)
    print(f'Total size of the training dataset: {len(train_set)}')
    print('Load validation dataset(s) from:')
    pprint(args.val_set)
    val_set = load_trajs_to_dataset(args.val_set, CommittorDataset, variables=args.variables, device=args.device)
    print(f'Total size of the validation dataset: {len(val_set)}')
    print(f'Use device: {args.device}')
    # use a more efficient custom dataloader for CPU training
    if args.device == 'cpu':
        dataloader = MyDataLoader
    else:
        # dataloader = torch.utils.data.DataLoader
        dataloader = MyDataLoader
    best_encoder_model = train_model(
        model_to_train=model_to_train, output_prefix=args.output_prefix,
        train_set=train_set, val_set=val_set, loss_function=loss_vcns_soft_endpoints, patience=args.patience,
        epochs=args.epochs, batch_size_factor=args.batch_size_factor, old_checkpoint=old_checkpoint,
        epoch_metrics_callback=None, dataloader=dataloader, load_old_model_only=args.previous_model_only)
    model_name = f'{args.output_prefix}_best_encoder_model.pt'
    torch.jit.script(best_encoder_model).save(model_name)
    best_encoder_model.dump_weights(args.output_prefix)
    # encoded all previous datasets
    if args.previous_train_set is not None:
        # the datasets have renamed variables
        variables_origin = [f'{v}_origin' for v in args.variables]
        if len(args.previous_train_set) > 0:
            predict_datasets(
                files=args.previous_train_set, variables=variables_origin,
                model=best_encoder_model, output_prefix=args.output_prefix,
                device=args.device)
    else:
        variables_origin = [f'{v}_origin' for v in args.variables]
        predict_datasets(
            files=args.train_set, variables=variables_origin,
            model=best_encoder_model, output_prefix=args.output_prefix,
            device=args.device)
