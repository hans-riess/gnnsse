# SSE-GNN Experiment Enhancements

## Overview

The SSE-GNN experiment framework has been enhanced to provide comprehensive visualization and metrics for model evaluation. The new functionality includes:

1. **Training Loss Plots** - Visual tracking of training loss over epochs
2. **Confusion Matrix Visualization** - Heatmap plots showing classification performance
3. **F1 Score Calculations** - Comprehensive classification metrics including weighted, macro, and micro F1 scores
4. **Network Visualizations** - Graph plots showing node predictions vs ground truth with color coding for TP/TN/FP/FN

## New Dependencies

The following packages have been added to support visualization:

- `matplotlib>=3.7.0` - Core plotting library
- `seaborn>=0.12.0` - Enhanced statistical visualization
- `scikit-learn>=1.3.0` - Classification metrics and confusion matrix
- `networkx>=3.1` - Network graph visualization

## Enhanced Functionality

### 1. Training Metrics Collection

The `train()` function now collects and returns:
- Test predictions and true labels
- Comprehensive classification metrics (F1 scores, precision, recall, specificity)
- Confusion matrix elements (TP, TN, FP, FN)

### 2. Visualization Module (`ssegnn/visualization.py`)

#### Core Functions:

- **`plot_training_loss()`** - Creates training loss plots over epochs
- **`plot_training_metrics()`** - Combined plots of loss and accuracy metrics
- **`plot_confusion_matrix()`** - Heatmap visualization of confusion matrices
- **`plot_network_predictions()`** - Network graph with color-coded predictions
- **`calculate_classification_metrics()`** - Computes comprehensive classification metrics
- **`generate_experiment_plots()`** - Main function that creates all plots for an experiment

#### Node Color Coding for Network Visualization:
- **Green** - True Positives (TP): Correctly predicted linear sites
- **Blue** - True Negatives (TN): Correctly predicted non-linear sites  
- **Red** - False Positives (FP): Incorrectly predicted as linear
- **Orange** - False Negatives (FN): Incorrectly predicted as non-linear

### 3. Experiment Output Structure

Each experiment now creates the following directory structure:

```
experiments/
├── experiment_YYYYMMDD_HHMMSS/
│   ├── data.yaml              # Data configuration
│   ├── hyper.yaml             # Hyperparameters
│   ├── results.json           # Numerical results including new metrics
│   └── plots/
│       ├── training_loss.png           # Training loss over epochs
│       ├── training_metrics.png        # Combined loss and accuracy plots
│       ├── confusion_matrix_test.png   # Test set confusion matrix
│       ├── network_predictions.png     # Network visualization with predictions
│       └── classification_report.txt   # Detailed classification report
```

### 4. Enhanced Results

The results JSON now includes:
- All original metrics (test_accuracy, train_accuracies, etc.)
- **New test metrics:**
  - `test_f1_weighted` - F1 score with class weights
  - `test_f1_macro` - Macro-averaged F1 score
  - `test_f1_micro` - Micro-averaged F1 score
  - `test_precision` - Precision score
  - `test_recall` - Recall score
  - `test_specificity` - Specificity score
  - `test_true_positives` - Count of true positives
  - `test_true_negatives` - Count of true negatives
  - `test_false_positives` - Count of false positives
  - `test_false_negatives` - Count of false negatives
  - `test_confusion_matrix` - Full confusion matrix
- Predictions and labels for visualization

## Usage

The enhanced functionality is automatically enabled when running experiments. Simply run:

```bash
python main.py --experiment-name my_experiment --verbose
```

All plots and enhanced metrics will be automatically generated and saved to the experiment directory.

### Example Output

After running an experiment, you'll see enhanced output like:

```
Experiment completed successfully!
Final test accuracy: 0.8542
Final test F1 score (weighted): 0.8456
All plots saved to: experiments/my_experiment/plots
```

## Key Features

1. **Automatic Plot Generation**: All visualizations are created automatically during experiment execution
2. **Comprehensive Metrics**: Beyond accuracy, get detailed classification performance metrics
3. **Error Analysis**: Network visualization makes it easy to identify spatial patterns in classification errors
4. **Publication Ready**: High-resolution plots (300 DPI) suitable for papers and presentations
5. **Robust Error Handling**: Visualization failures don't crash the experiment

## Interpretation Guide

### Network Visualization
- **Spatial Clustering**: Look for geographic clustering of false positives/negatives
- **Network Effects**: Identify if prediction errors correlate with graph connectivity
- **Class Distribution**: Visualize the spatial distribution of linear vs non-linear sites

### Confusion Matrix
- **Class Balance**: Assess performance across both classes
- **Error Patterns**: Identify systematic prediction biases
- **Threshold Effects**: Understand the trade-offs between precision and recall

### Training Metrics
- **Convergence**: Monitor training loss for convergence patterns
- **Overfitting**: Compare training vs validation accuracy to detect overfitting
- **Learning Dynamics**: Understand how the model learns over time

## Technical Notes

- The visualization module handles tensor/numpy conversion automatically
- Network plots are filtered to show only test nodes to avoid clutter
- All plots use consistent styling and color schemes
- Error handling ensures experiments continue even if plotting fails
- Memory efficient - plots are saved and closed to prevent memory leaks