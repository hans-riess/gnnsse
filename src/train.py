import torch
from src.config import load_config
from signatory import signature_channels
from src.graph import NodeSplitMask, TemporalGraphNormalize
from src.model import ClassifierGCN
from src.dataloader import DataLoader
from src.signature import SignatureFeatures, RandomFeatures


def train(config, hyper, device='cpu', verbose=False):
    """
    Args:
        config: Base configuration dictionary
        hyper: Hyperparameter configuration dictionary
        device: Device to use for training ('cpu' or 'cuda')
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary containing training results
    """
    # Load the data
    dataloader = DataLoader(
        label_path=config['paths']['label'],
        data_path=config['paths']['data'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        download=config['data']['download'],
        load=config['data']['load'],
        debug=False
    )

    # Generate graph
    data = dataloader.get_graph(k=config['graph']['k'],
                                r=config['graph']['r'])
    
    # Apply normalization transform
    normalize_transform = TemporalGraphNormalize(normalize=True, fill_nan='both')
    data = normalize_transform(data)
    
    sig_transform = SignatureFeatures(
            sig_depth=hyper['sig']['depth'],
            normalize=hyper['sig']['normalize'],
            log_signature=hyper['sig']['log_signature'],
            time_augment=hyper['sig']['time_augment'],
            lead_lag=hyper['sig']['lead_lag']
        )
    
    random_transform = RandomFeatures(
            feature_dim=signature_channels(3, hyper['sig']['depth']),  # Number of random features per node
            normalize=True,  # Whether to normalize the features
            seed=hyper['rand']['seed']  # Random seed for reproducibility
        )

    if hyper['feat'] == 'sig':
        # Apply the transform to your temporal graph data
        static_graph = sig_transform(data)
    elif hyper['feat'] == 'rand':
        # Apply the transform to ignore temporal information
        static_graph = random_transform(data)
    else:
        raise ValueError(f"Unknown feature type: {hyper['feat']}. Must be 'sig' or 'rand'")

    # Split the nodes
    split_transform = NodeSplitMask(train_ratio=hyper['split']['train_ratio'],
                                    val_ratio=hyper['split']['val_ratio'],
                                    test_ratio=hyper['split']['test_ratio'],
                                    seed=hyper['split']['seed'])
    
    # Apply the transform to static graph
    static_graph = split_transform(static_graph)

    # Move data to device
    static_graph = static_graph.to(device)

    # GCN model
    model = ClassifierGCN(node_features=signature_channels(3, hyper['sig']['depth']),
                          hidden_features=hyper['hidden_features'],
                          num_classes=hyper['num_classes'])
    model = model.to(device)

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyper['lr'],
                                 weight_decay=hyper['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(int(hyper['num_epochs'])):
        model.train()
        optimizer.zero_grad()
        out = model(static_graph.x, static_graph.edge_index, static_graph.edge_attr)
        loss = criterion(out[static_graph.train_mask], static_graph.y[static_graph.train_mask].long())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            train_acc = (pred[static_graph.train_mask] == static_graph.y[static_graph.train_mask]).float().mean().item()
            val_acc = (pred[static_graph.val_mask] == static_graph.y[static_graph.val_mask]).float().mean().item() if static_graph.val_mask.sum() > 0 else float('nan')
            
            train_losses.append(loss.item())
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        out = model(static_graph.x, static_graph.edge_index, static_graph.edge_attr)
        pred = out.argmax(dim=1)
        test_acc = (pred[static_graph.test_mask] == static_graph.y[static_graph.test_mask]).float().mean().item() if static_graph.test_mask.sum() > 0 else float('nan')
    
    if verbose:
        print(f"Test Accuracy: {test_acc:.4f}")
    
    # Return results
    results = {
        'test_accuracy': test_acc,
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0.0,
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'num_epochs': len(train_losses)
    }
    
    return results

# For backward compatibility - if run directly, use default configs
if __name__ == "__main__":
    # Configuration files
    config = load_config('configs/base.yaml')
    hyper = load_config('configs/hyper.yaml')
    
    results = train(config, hyper, device='cpu', verbose=True)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")