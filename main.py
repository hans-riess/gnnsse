import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
import json

from ssegnn.dataloader import DataLoader
from ssegnn.train import train
from ssegnn.config import load_config
from ssegnn.visualization import generate_experiment_plots


def parse_args():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description='Run SSE-GNN experiments')
    
    # Configuration files
    parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--hyper-config', type=str, default='configs/hyper.yaml',
                       help='Path to hyperparameter configuration file')
    
    parser.add_argument('--download', action='store_true',
                       help='Download data from Tilde API')
    parser.add_argument('--from-file', action='store_true',
                       help='Load data from file')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment (default: timestamp)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Directory to save experiment results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for training')
    
    # Model parameters (can override config)
    parser.add_argument('--sig-depth', type=int, default=None,
                       help='Signature depth (overrides hyper.yaml)')
    parser.add_argument('--hidden-features', type=int, default=None,
                       help='Hidden features (overrides hyper.yaml)')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay (overrides hyper.yaml)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides hyper.yaml)')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs (overrides hyper.yaml)')
    
    # Data parameters
    parser.add_argument('--split-seed', type=int, default=None,
                       help='Random seed for split')
    
    # Graph parameters
    parser.add_argument('--k', type=int, default=None,
                       help='K parameter for graph construction (overrides hyper.yaml)')
    parser.add_argument('--r', type=float, default=None,
                       help='R parameter for graph construction (overrides hyper.yaml)')
    
    # Training options
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    
    return parser.parse_args()


def setup_device(device_str):
    """Setup the device for training."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_str)

def update_config(data_config, hyper, args):
    """Update hyperparameters with command line arguments."""
    if args.hidden_features is not None:
        hyper['hidden_features'] = args.hidden_features
    if args.lr is not None:
        hyper['lr'] = args.lr
    if args.num_epochs is not None:
        hyper['num_epochs'] = args.num_epochs
    if args.split_seed is not None:
        hyper['split']['seed'] = args.split_seed
    if args.sig_depth is not None:
        hyper['sig']['depth'] = args.sig_depth
    if args.k is not None:
        hyper['graph']['k'] = args.k
    if args.r is not None:
        hyper['graph']['r'] = args.r
    if args.weight_decay is not None:
        hyper['weight_decay'] = args.weight_decay
    if args.download is not None:
        data_config['data']['download'] = args.download
    if args.from_file is not None:
        data_config['data']['from_file'] = args.from_file
    return hyper


def save_experiment_results(output_dir, experiment_name, data_config, hyper, results, static_graph=None):
    """Save experiment results and configuration."""
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(experiment_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    with open(experiment_dir / 'hyper.yaml', 'w') as f:
        yaml.dump(hyper, f, default_flow_style=False)
    
    # Save results
    with open(experiment_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save all visualization plots
    if static_graph is not None:
        try:
            generate_experiment_plots(results, static_graph, output_dir, experiment_name)
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
    
    print('\n')
    print(f"Experiment results saved to: {experiment_dir}")


def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Load configurations
    data_config = load_config(args.data_config)
    hyper = load_config(args.hyper_config)
    
    # Update with command line arguments
    hyper = update_config(data_config, hyper, args)
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    print(f"Starting experiment: {experiment_name}")
    
    try:
        # Run training
        results, static_graph = train(data_config, hyper, device=device, verbose=args.verbose, debug=args.debug)
        
        # Save results and generate plots
        save_experiment_results(args.output_dir, experiment_name, data_config, hyper, results, static_graph)
        
        print(f"Experiment completed successfully!")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        print(f"Final test F1 score (weighted): {results['test_f1_weighted']:.4f}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()