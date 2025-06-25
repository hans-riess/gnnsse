import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
import json

from src.dataloader import DataLoader
from src.train import train
from src.config import load_config


def parse_args():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description='Run SSE-GNN experiments')
    
    # Configuration files
    parser.add_argument('--config', type=str, default='configs/data.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--hyper', type=str, default='configs/hyper.yaml',
                       help='Path to hyperparameter configuration file')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment (default: timestamp)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save experiment results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Model parameters (can override config)
    parser.add_argument('--sig-depth', type=int, default=None,
                       help='Signature depth (overrides config)')
    parser.add_argument('--hidden-features', type=int, default=None,
                       help='Hidden features (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    # Data parameters
    parser.add_argument('--train-ratio', type=float, default=None,
                       help='Training split ratio (overrides config)')
    parser.add_argument('--test-ratio', type=float, default=None,
                       help='Test split ratio (overrides config)')
    parser.add_argument('--val-ratio', type=float, default=None,
                       help='Validation split ratio (overrides config)')
    
    # Graph parameters
    parser.add_argument('--k', type=int, default=None,
                       help='K parameter for graph construction (overrides config)')
    parser.add_argument('--r', type=float, default=None,
                       help='R parameter for graph construction (overrides config)')
    
    # Training options
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
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


def update_config(config, args):
    """Update configuration with command line arguments."""

    return config


def update_hyper(hyper, args):
    """Update hyperparameters with command line arguments."""
    if args.hidden_features is not None:
        hyper['hidden_features'] = args.hidden_features
    if args.lr is not None:
        hyper['lr'] = args.lr
    if args.num_epochs is not None:
        hyper['num_epochs'] = args.num_epochs
    if args.train_ratio is not None:
        hyper['split']['train_ratio'] = args.train_ratio
    if args.test_ratio is not None:
        hyper['split']['test_ratio'] = args.test_ratio
    if args.val_ratio is not None:
        hyper['split']['val_ratio'] = args.val_ratio
    if args.seed is not None:
        hyper['split']['seed'] = args.seed
    if args.sig_depth is not None:
        hyper['sig']['depth'] = args.sig_depth
    if args.k is not None:
        hyper['graph']['k'] = args.k
    if args.r is not None:
        hyper['graph']['r'] = args.r
    return hyper


def save_experiment_results(output_dir, experiment_name, config, hyper, results):
    """Save experiment results and configuration."""
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(experiment_dir / 'data.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    with open(experiment_dir / 'hyper.yaml', 'w') as f:
        yaml.dump(hyper, f, default_flow_style=False)
    
    # Save results
    with open(experiment_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Experiment results saved to: {experiment_dir}")


def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Load configurations
    config = load_config(args.config)
    hyper = load_config(args.hyper)
    
    # Update with command line arguments
    config = update_config(config, args)
    hyper = update_hyper(hyper, args)
    
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
    
    if args.verbose:
        print("Configuration:")
        print(f"  Signature depth: {hyper['sig']['depth']}")
        print(f"  Graph k: {hyper['graph']['k']}")
        print(f"  Graph r: {hyper['graph']['r']}")
        print(f"  Hidden features: {hyper['hidden_features']}")
        print(f"  Learning rate: {hyper['lr']}")
        print(f"  Epochs: {hyper['num_epochs']}")
        print(f"  Train/Test/Val split: {hyper['split']['train_ratio']}/{hyper['split']['test_ratio']}/{hyper['split']['val_ratio']}")
    
    try:
        # Run training
        results = train(config, hyper, device=device, verbose=args.verbose)
        
        # Save results
        save_experiment_results(args.output_dir, experiment_name, config, hyper, results)
        
        print(f"Experiment completed successfully!")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()











