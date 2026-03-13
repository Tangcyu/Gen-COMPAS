from common.diffusion_train import train_diffusion_model
from common.diffusion_sample import run_diffusion_inference
from common.vcn_slice import run_committor_analysis
from common.vcn_train import train_committor_model
from tools.clustering import run_clustering
from tools.occupancy import add_occupancy
from tools.reweighting import run_reweighting
from tools.felestimate import run_fel_estimate
import yaml
import os
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Generate protein structures using SE(3) Diffusion Model")
    parser.add_argument('--step', type=str, required=True, help='Step to run: "train_diffusion", "sample_diffusion", "train_committor", "committor_analysis", "clustering", ""occupancy", "reweighting"')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.step == "train_diffusion":
        train_diffusion_model(config['Generative'])     
        
    elif args.step == "sample_diffusion":
        run_diffusion_inference(config['Generative'])     
        
    elif args.step == "train_committor":
        train_committor_model(config['VCN'])              
        
    elif args.step == "committor_analysis":
        run_committor_analysis(config['VCN'])             
        
    elif args.step == "clustering":
        run_clustering(config['Clustering'])         
        
    elif args.step == "occupancy":
        add_occupancy(config['Occupancy'])
        
    elif args.step == "reweighting":
        run_reweighting(config['Reweighting'])
    
    # elif args.step == "fel_estimate":
    #     run_fel_estimate(config['FEL_estimate'])
    
    else:
        raise ValueError(f"Unknown step: {args.step}. Choose from 'train_diffusion', 'sample_diffusion', 'train_committor', 'committor_analysis', 'clustering', 'occupancy', 'reweighting'.")       
