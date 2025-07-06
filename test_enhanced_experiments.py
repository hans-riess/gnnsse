#!/usr/bin/env python3
"""
Test script for enhanced experiment functionality with synthetic data.
This creates a minimal test case to verify all visualization components work correctly.
"""

import numpy as np
import torch
import json
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import the experiment functionality
from ssegnn.dataloader import DataLoader
from ssegnn.train import train
from ssegnn.config import load_config
from ssegnn.visualization import generate_experiment_plots

def create_synthetic_data():
    """Create synthetic GPS data for testing."""
    print("Creating synthetic test data...")
    
    # Use only first 10 sites from labels for quick testing
    with open('datasets/labels.json', 'r') as f:
        all_labels = json.load(f)
    
    # Select first 10 sites for testing
    test_labels = all_labels[:10]
    site_ids = [entry['siteID'] for entry in test_labels]
    
    # Create synthetic GPS coordinates (New Zealand region)
    np.random.seed(42)
    lats = np.random.uniform(-47, -34, len(site_ids))  # NZ latitude range
    lons = np.random.uniform(166, 179, len(site_ids))  # NZ longitude range
    
    # Create synthetic time series data
    records = []
    timestamps = pd.date_range('2023-01-01', periods=30, freq='D')  # 30 days of data
    
    for i, site_id in enumerate(site_ids):
        lat, lon = lats[i], lons[i]
        
        # Create synthetic displacement data with some noise
        for ts in timestamps:
            # Add some realistic patterns - linear sites have different patterns
            label = next(entry['label'] for entry in test_labels if entry['siteID'] == site_id)
            if label == 1:  # Linear site - more systematic displacement
                e = 0.001 * np.sin(i * 0.1) + np.random.normal(0, 0.0005)
                n = 0.001 * np.cos(i * 0.1) + np.random.normal(0, 0.0005)
                u = 0.002 * np.sin(i * 0.15) + np.random.normal(0, 0.001)
            else:  # Non-linear site - more random
                e = np.random.normal(0, 0.001)
                n = np.random.normal(0, 0.001)
                u = np.random.normal(0, 0.002)
            
            records.append({
                'siteID': site_id,
                't': ts.isoformat(),
                'e': e,
                'n': n,
                'u': u,
                'geometry': Point(lon, lat)
            })
    
    # Create GeoDataFrame and save
    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs="EPSG:4326")
    gdf.to_file('datasets/test_signals.geojson', driver="GeoJSON")
    
    # Create test labels file
    with open('datasets/test_labels.json', 'w') as f:
        json.dump(test_labels, f)
    
    print(f"Created synthetic data with {len(site_ids)} sites and {len(timestamps)} time points")
    return len(site_ids), len(timestamps)

def create_test_configs():
    """Create test configuration files."""
    
    # Test data config
    test_data_config = {
        'paths': {
            'data': 'datasets/test_signals.geojson',
            'label': 'datasets/test_labels.json'
        },
        'data': {
            'start_date': '2023-01-01 00:00:00+0000',
            'end_date': '2023-01-31 23:59:59+0000',
            'download': False,
            'from_file': True
        }
    }
    
    # Test hyperparameters (reduced for quick testing)
    test_hyper_config = {
        'lr': 0.01,
        'num_epochs': 20,  # Reduced for testing
        'weight_decay': 0.01,
        'hidden_features': 16,  # Smaller for testing
        'torch_seed': 42,
        'split': {
            'seed': 31,
            'train_ratio': 0.6,
            'test_ratio': 0.4,
            'val_ratio': 0.0
        },
        'feat': 'sig',
        'graph': {
            'k': 3,  # Smaller k for limited nodes
            'r': None
        },
        'sig': {
            'depth': 2,  # Reduced for testing
            'normalize': False,
            'log_signature': False,
            'time_augment': False,
            'lead_lag': False
        },
        'rand': {
            'num_features': 50,
            'seed': 42
        }
    }
    
    return test_data_config, test_hyper_config

def test_enhanced_experiments():
    """Test the enhanced experiment functionality."""
    print("="*60)
    print("TESTING ENHANCED EXPERIMENT FUNCTIONALITY")
    print("="*60)
    
    # Create synthetic data
    num_sites, num_timestamps = create_synthetic_data()
    
    # Create test configurations
    data_config, hyper_config = create_test_configs()
    
    # Set up device
    device = torch.device('cpu')  # Use CPU for testing
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run training with enhanced functionality
    print("\nRunning training with enhanced metrics collection...")
    
    try:
        results, static_graph = train(
            data_config, 
            hyper_config, 
            device=device, 
            verbose=True, 
            debug=False
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Print enhanced results
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test F1 Score (weighted): {results['test_f1_weighted']:.4f}")
        print(f"Test F1 Score (macro): {results['test_f1_macro']:.4f}")
        print(f"Test Precision: {results['test_precision']:.4f}")
        print(f"Test Recall: {results['test_recall']:.4f}")
        print(f"True Positives: {results['test_true_positives']}")
        print(f"True Negatives: {results['test_true_negatives']}")
        print(f"False Positives: {results['test_false_positives']}")
        print(f"False Negatives: {results['test_false_negatives']}")
        
        # Test visualization generation
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS...")
        print("="*50)
        
        experiment_name = "test_enhanced_functionality"
        output_dir = "test_experiments"
        
        # Create experiment directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate all plots
        generate_experiment_plots(results, static_graph, output_dir, experiment_name)
        
        print("\n" + "="*50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"All plots and results saved to: {output_dir}/{experiment_name}/")
        print("\nGenerated files:")
        plot_dir = Path(output_dir) / experiment_name / "plots"
        if plot_dir.exists():
            for plot_file in plot_dir.glob("*"):
                print(f"  - {plot_file}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_experiments()
    if success:
        print("\n🎉 All enhanced experiment functionality is working correctly!")
    else:
        print("\n❌ Test failed. Please check the error messages above.")