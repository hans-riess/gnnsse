import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import networkx as nx
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from pathlib import Path
import pandas as pd

def plot_training_loss(train_losses, save_path=None, title="Training Loss"):
    """
    Plot training loss over epochs.
    
    Args:
        train_losses: List of training loss values
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2, alpha=0.8)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_validation_loss(train_losses, val_losses, save_path=None, title="Training vs Validation Loss"):
    """
    Plot training and validation loss together to assess generalization.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    epochs = range(len(train_losses))
    
    plt.plot(epochs, train_losses, 'b-', linewidth=2, alpha=0.8, label='Training Loss')
    
    # Only plot validation loss if we have valid values
    if val_losses and not all(np.isnan(val_losses)):
        plt.plot(epochs, val_losses, 'r-', linewidth=2, alpha=0.8, label='Validation Loss')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_metrics(train_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Plot training loss and accuracies in subplots.
    
    Args:
        train_losses: List of training loss values
        train_accuracies: List of training accuracy values
        val_accuracies: List of validation accuracy values
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss
    ax1.plot(train_losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Accuracies
    epochs = range(len(train_accuracies))
    ax2.plot(epochs, train_accuracies, 'g-', linewidth=2, alpha=0.8, label='Train Accuracy')
    if val_accuracies and not all(np.isnan(val_accuracies)):
        ax2.plot(epochs, val_accuracies, 'r-', linewidth=2, alpha=0.8, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comprehensive_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Plot comprehensive training metrics including both losses and accuracies.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values
        train_accuracies: List of training accuracy values
        val_accuracies: List of validation accuracy values
        save_path: Path to save the plot (optional)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    epochs = range(len(train_losses))
    
    # Training and validation loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, alpha=0.8, label='Training Loss')
    if val_losses and not all(np.isnan(val_losses)):
        ax1.plot(epochs, val_losses, 'r-', linewidth=2, alpha=0.8, label='Validation Loss')
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training and validation accuracy
    ax2.plot(epochs, train_accuracies, 'g-', linewidth=2, alpha=0.8, label='Training Accuracy')
    if val_accuracies and not all(np.isnan(val_accuracies)):
        ax2.plot(epochs, val_accuracies, 'orange', linewidth=2, alpha=0.8, label='Validation Accuracy')
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training loss only (zoomed view)
    ax3.plot(epochs, train_losses, 'b-', linewidth=2, alpha=0.8)
    ax3.set_title('Training Loss (Detailed)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Generalization gap (difference between validation and training loss)
    if val_losses and not all(np.isnan(val_losses)):
        generalization_gap = [val_losses[i] - train_losses[i] for i in range(len(epochs)) if not np.isnan(val_losses[i])]
        valid_epochs = [i for i in range(len(epochs)) if not np.isnan(val_losses[i])]
        ax4.plot(valid_epochs, generalization_gap, 'purple', linewidth=2, alpha=0.8)
        ax4.set_title('Generalization Gap (Val Loss - Train Loss)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss Difference', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    else:
        ax4.text(0.5, 0.5, 'No validation data available', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names (optional)
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    if labels is None:
        labels = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_network_predictions(edge_index, positions, y_true, y_pred, save_path=None, 
                           title="Network Visualization: Predictions vs Ground Truth"):
    """
    Plot network with nodes colored by prediction correctness.
    
    Args:
        edge_index: Edge connectivity in PyTorch Geometric format
        positions: Node positions (lat, lon)
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    # Convert to numpy if needed
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    if torch.is_tensor(positions):
        positions = positions.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Create NetworkX graph
    G = nx.Graph()
    num_nodes = len(y_true)
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    edges = [(edge_index[0][i], edge_index[1][i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    # Create position dictionary for NetworkX
    pos = {i: (positions[i][0], positions[i][1]) for i in range(num_nodes)}
    
    # Determine node colors based on prediction correctness
    node_colors = []
    node_labels = []
    for i in range(num_nodes):
        true_label = y_true[i]
        pred_label = y_pred[i]
        
        if true_label == 1 and pred_label == 1:
            # True Positive
            node_colors.append('green')
            node_labels.append('TP')
        elif true_label == 0 and pred_label == 0:
            # True Negative  
            node_colors.append('blue')
            node_labels.append('TN')
        elif true_label == 0 and pred_label == 1:
            # False Positive
            node_colors.append('red')
            node_labels.append('FP')
        else:  # true_label == 1 and pred_label == 0
            # False Negative
            node_colors.append('orange')
            node_labels.append('FN')
    
    plt.figure(figsize=(12, 10))
    
    # Draw the network
    nx.draw(G, pos, node_color=node_colors, node_size=100, alpha=0.8, 
            edge_color='gray', linewidths=0.5, width=0.5)
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='True Positive (TP)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='True Negative (TN)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='False Positive (FP)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=10, label='False Negative (FN)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'f1_weighted': f1,
        'f1_macro': f1_macro, 
        'f1_micro': f1_micro,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }

def save_classification_report(y_true, y_pred, save_path=None, class_names=None):
    """
    Generate and save a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the report (optional)
        class_names: Names for the classes (optional)
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    if class_names is None:
        class_names = [f'Class_{i}' for i in np.unique(y_true)]
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
    
    return report

def generate_experiment_plots(results, static_graph, output_dir, experiment_name):
    """
    Generate all plots for an experiment.
    
    Args:
        results: Results dictionary from training
        static_graph: The graph data with predictions
        output_dir: Directory to save plots
        experiment_name: Name of the experiment
    """
    plots_dir = Path(output_dir) / experiment_name / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training loss plot
    if 'train_losses' in results and results['train_losses']:
        plot_training_loss(
            results['train_losses'],
            save_path=plots_dir / 'training_loss.png',
            title=f"Training Loss - {experiment_name}"
        )
    
    # 2. Training vs Validation Loss plot
    if 'train_losses' in results and 'val_losses' in results and results['train_losses']:
        plot_training_validation_loss(
            results['train_losses'],
            results['val_losses'],
            save_path=plots_dir / 'training_validation_loss.png',
            title=f"Training vs Validation Loss - {experiment_name}"
        )
    
    # 3. Comprehensive training metrics plot
    if all(key in results for key in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']):
        plot_comprehensive_training_metrics(
            results['train_losses'],
            results['val_losses'],
            results['train_accuracies'], 
            results['val_accuracies'],
            save_path=plots_dir / 'comprehensive_training_metrics.png'
        )
    
    # 4. Original training metrics plot (for compatibility)
    if all(key in results for key in ['train_losses', 'train_accuracies', 'val_accuracies']):
        plot_training_metrics(
            results['train_losses'],
            results['train_accuracies'], 
            results['val_accuracies'],
            save_path=plots_dir / 'training_metrics.png'
        )
    
    # 5. Confusion matrix for test set
    if 'test_predictions' in results and 'test_labels' in results:
        plot_confusion_matrix(
            results['test_labels'],
            results['test_predictions'],
            labels=['Non-linear', 'Linear'],
            save_path=plots_dir / 'confusion_matrix_test.png',
            title=f"Test Set Confusion Matrix - {experiment_name}"
        )
    
    # 6. Network visualization
    if all(key in results for key in ['test_predictions', 'test_labels']) and hasattr(static_graph, 'edge_index'):
        # Use test mask to get positions and predictions for test nodes only
        test_mask = static_graph.test_mask.cpu().numpy()
        test_positions = static_graph.positions[test_mask]
        
        # Filter edges to only include edges between test nodes
        edge_index = static_graph.edge_index.cpu().numpy()
        test_node_indices = np.where(test_mask)[0]
        test_node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(test_node_indices)}
        
        # Create filtered edge index for test nodes only
        test_edges = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0][i], edge_index[1][i]
            if src in test_node_map and dst in test_node_map:
                test_edges.append([test_node_map[src], test_node_map[dst]])
        
        if test_edges:
            test_edge_index = np.array(test_edges).T
            plot_network_predictions(
                test_edge_index,
                test_positions,
                results['test_labels'],
                results['test_predictions'],
                save_path=plots_dir / 'network_predictions.png',
                title=f"Network Predictions vs Ground Truth - {experiment_name}"
            )
    
    # 7. Save classification report
    if 'test_predictions' in results and 'test_labels' in results:
        save_classification_report(
            results['test_labels'],
            results['test_predictions'],
            save_path=plots_dir / 'classification_report.txt',
            class_names=['Non-linear', 'Linear']
        )
    
    print(f"All plots saved to: {plots_dir}")
    
    # Print summary of generalization assessment
    if 'val_losses' in results and results['val_losses'] and not all(np.isnan(results['val_losses'])):
        final_train_loss = results['train_losses'][-1] if results['train_losses'] else 0
        final_val_loss = results['val_losses'][-1] if results['val_losses'] else 0
        if not np.isnan(final_val_loss):
            generalization_gap = final_val_loss - final_train_loss
            print(f"\nGeneralization Assessment:")
            print(f"Final Training Loss: {final_train_loss:.4f}")
            print(f"Final Validation Loss: {final_val_loss:.4f}")
            print(f"Generalization Gap: {generalization_gap:.4f}")
            if generalization_gap > 0.1:
                print("⚠️  Large generalization gap detected - model may be overfitting")
            elif generalization_gap < -0.05:
                print("⚠️  Negative generalization gap - possible validation set issues")
            else:
                print("✅ Reasonable generalization gap")
    else:
        print("\nℹ️  No validation loss available for generalization assessment")
        print("Consider increasing validation set size or checking data splits")