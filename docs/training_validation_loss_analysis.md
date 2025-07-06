# Training and Validation Loss Analysis

## Overview

This document explains the enhanced training and validation loss tracking features implemented in the SSE-GNN framework. These improvements help assess model generalization and identify potential overfitting issues.

## Key Features

### 1. Enhanced Training Loop

The training loop now tracks both training and validation losses simultaneously:
- **Training Loss**: Computed during the forward pass on training data
- **Validation Loss**: Computed during evaluation on validation data
- **Training/Validation Accuracy**: Tracked for both sets

### 2. Comprehensive Visualization

Several new plotting functions provide detailed insights:

#### `plot_training_validation_loss()`
- Plots training and validation loss curves together
- Helps identify overfitting (validation loss increasing while training loss decreases)
- Essential for assessing model generalization

#### `plot_comprehensive_training_metrics()`
- 2x2 subplot layout showing:
  - Training vs Validation Loss
  - Training vs Validation Accuracy
  - Training Loss (detailed view)
  - Generalization Gap (Val Loss - Train Loss)

#### Generalization Gap Analysis
- Positive gap: Validation loss > Training loss (normal, indicates some overfitting)
- Large positive gap (>0.1): Significant overfitting
- Negative gap (<-0.05): Possible validation set issues

### 3. Automatic Generalization Assessment

The system now provides automatic assessment:
- ✅ **Reasonable generalization gap**: Gap between -0.05 and 0.1
- ⚠️ **Large generalization gap**: Gap > 0.1 (overfitting detected)
- ⚠️ **Negative generalization gap**: Gap < -0.05 (validation set issues)

## Low Sample Size Recommendations

### Problem
With low sample sizes, using only training/testing splits can lead to:
- Poor validation estimates
- Unreliable generalization assessment
- Difficulty in hyperparameter tuning

### Solutions

#### 1. Cross-Validation
```python
# Consider implementing k-fold cross-validation
# Instead of single train/val/test split
from sklearn.model_selection import StratifiedKFold

# Example: 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

#### 2. Bootstrap Sampling
```python
# Use bootstrap sampling for more robust estimates
# Particularly useful for very small datasets
import numpy as np

def bootstrap_sample(data, n_samples=1000):
    """Generate bootstrap samples"""
    indices = np.random.choice(len(data), size=n_samples, replace=True)
    return data[indices]
```

#### 3. Leave-One-Out (LOO) Cross-Validation
```python
# For very small datasets (< 100 samples)
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
```

#### 4. Stratified Sampling
```python
# Ensure balanced class distribution in all splits
from sklearn.model_selection import train_test_split

# Stratified split maintaining class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### 5. Data Augmentation
Consider data augmentation techniques appropriate for your domain:
- For time series: Adding noise, time warping, window shifting
- For graph data: Node/edge perturbation, subgraph sampling

### Configuration Recommendations

#### For Small Datasets (< 100 samples):
```yaml
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

#### For Very Small Datasets (< 50 samples):
```yaml
# Consider LOO or 5-fold CV instead of fixed splits
cross_validation:
  method: "stratified_kfold"
  n_splits: 5
  shuffle: true
  random_state: 42
```

## Usage Examples

### 1. Running with Enhanced Validation
```bash
python main.py --experiment-name generalization_test --verbose
```

### 2. Interpreting Results
Look for these patterns in the plots:
- **Good generalization**: Training and validation loss decrease together
- **Overfitting**: Training loss continues decreasing while validation loss increases
- **Underfitting**: Both losses remain high and plateau early

### 3. Addressing Issues
If overfitting is detected:
- Reduce model complexity
- Increase regularization (weight_decay)
- Early stopping based on validation loss
- Data augmentation

## Files Modified

1. `ssegnn/train.py`: Added validation loss tracking
2. `ssegnn/visualization.py`: Added comprehensive plotting functions
3. `main.py`: Enhanced experiment results saving

## Generated Plots

The enhanced system generates:
1. `training_loss.png`: Basic training loss curve
2. `training_validation_loss.png`: Training vs validation loss comparison
3. `comprehensive_training_metrics.png`: 2x2 subplot with full analysis
4. `training_metrics.png`: Training metrics overview
5. `confusion_matrix_test.png`: Test set confusion matrix
6. `network_predictions.png`: Network visualization with predictions
7. `classification_report.txt`: Detailed classification metrics

## Best Practices

1. **Always monitor validation loss** alongside training loss
2. **Use appropriate split ratios** for your dataset size
3. **Consider cross-validation** for small datasets
4. **Implement early stopping** based on validation loss
5. **Analyze generalization gap** to detect overfitting
6. **Save comprehensive metrics** for later analysis
7. **Use stratified sampling** to maintain class balance

## Troubleshooting

### No Validation Data Available
- Check if validation split ratio > 0
- Ensure sufficient samples for validation set
- Consider increasing dataset size or using cross-validation

### Poor Generalization
- Reduce model complexity
- Increase regularization
- Use dropout or other regularization techniques
- Implement data augmentation
- Consider ensemble methods

### Validation Loss is NaN
- Check for division by zero in loss computation
- Ensure validation set has samples from all classes
- Verify data preprocessing steps