# SSE-GNN

## Requirements

You will need the following Python packages:
- esig
- torch
- torch_geometric
- torch_geometric_temporal
- pandas
- (plus standard packages such as numpy, matplotlib, etc.)

Install the required packages individually:

```sh
pip install esig torch torch_geometric torch_geometric_temporal pandas
```

If you encounter issues with torch or torch_geometric, refer to their official installation guides for platform-specific wheels.

## How Everything Works

- **configs/**: Contains configuration files for data and hyperparameters.
- **datasets/**: Stores datasets, labels, and interfaces used for training and evaluation.
- **src/**: Main source code directory.
  - `config.py`: Loads and manages configuration files.
  - `dataloader.py`: Handles data loading and preprocessing.
  - `graph.py`: Contains graph construction utilities.
  - `model.py`: Defines the neural network models.
  - `signature.py`: Implements signature-related computations (e.g., using signatory).
  - `train.py`: Training loop and evaluation logic.
- **main.py**: Entry point for running experiments or training.
- **notebook.ipynb**: Example notebook for interactive exploration.
- **experiments/**, **logs/**, **results/**: Output directories for experiment tracking, logs, and results.

### Typical Workflow

1. Edit configuration files in `configs/` as needed.
2. Prepare your data in `datasets/`.
3. Run training or experiments via:
   ```sh
   python main.py
   ```
4. Check logs and results in the respective folders.

For more details on each module, refer to the docstrings in the source files. 