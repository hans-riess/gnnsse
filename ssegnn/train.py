import torch
from ssegnn.config import load_config
from ssegnn.graph import NodeSplitMask, TemporalGraphNormalize
from ssegnn.model import ClassifierGCN
from ssegnn.dataloader import DataLoader
from ssegnn.signature import SignatureFeatures, RandomFeatures


def train(data_config, hyper_config, device='cpu', verbose=False, debug=False):
    """
    Args:
        data_config: Data configuration dictionary
        hyper_config: Hyperparameter configuration dictionary
        device: Device to use for training ('cpu' or 'cuda')
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary containing training results
    """
    # Load the data
    dataloader = DataLoader(
        label_path=data_config['paths']['label'],
        data_path=data_config['paths']['data'],
        start_date=data_config['data']['start_date'],
        end_date=data_config['data']['end_date'],
        download=data_config['data']['download'],
        from_file=data_config['data']['from_file'],
        verbose=verbose,
        debug=debug,
    )

    # Generate graph
    data = dataloader.get_graph(k=hyper_config['graph']['k'],
                                r=hyper_config['graph']['r'])
    
    # Apply normalization transform
    normalize_transform = TemporalGraphNormalize(normalize=True, fill_nan='both')
    data = normalize_transform(data)
    
    sig_transform = SignatureFeatures(
            sig_depth=hyper_config['sig']['depth'],
            normalize=hyper_config['sig']['normalize'],
            log_signature=hyper_config['sig']['log_signature'],
            time_augment=hyper_config['sig']['time_augment'],
            lead_lag=hyper_config['sig']['lead_lag']
        )
    
    random_transform = RandomFeatures(
            feature_dim=  hyper_config['rand']['num_features'],  # Number of random features per node
            normalize=True,  # Whether to normalize the features
            seed=hyper_config['rand']['seed']  # Random seed for reproducibility
        )

    # Apply the transform to your temporal graph data
    if hyper_config['feat'] == 'sig':
        static_graph = sig_transform(data)
    elif hyper_config['feat'] == 'rand':
        static_graph = random_transform(data)
    else:
        raise ValueError(f"Unknown feature type: {hyper_config['feat']}. Must be 'sig' or 'rand'")

    # Debug: Check for NaNs in features, labels, and edge attributes
    if debug:
        if torch.isnan(static_graph.x).any():
            print("[DEBUG] NaN detected in node features!")
            print(static_graph.x)
            raise ValueError("NaN in node features")
        if torch.isnan(static_graph.y).any():
            print("[DEBUG] NaN detected in labels!")
            print(static_graph.y)
            raise ValueError("NaN in labels")
        if hasattr(static_graph, 'edge_attr') and static_graph.edge_attr is not None and torch.isnan(static_graph.edge_attr).any():
            print("[DEBUG] NaN detected in edge attributes!")
            print(static_graph.edge_attr)
            raise ValueError("NaN in edge attributes")

    # Split the nodes
    split_transform = NodeSplitMask(train_ratio=hyper_config['split']['train_ratio'],
                                    val_ratio=hyper_config['split']['val_ratio'],
                                    test_ratio=hyper_config['split']['test_ratio'],
                                    seed=hyper_config['split']['seed'])
    
    # Apply the transform to static graph
    static_graph = split_transform(static_graph)

    # Move data to device
    static_graph = static_graph.to(device)

    # GCN model
    model = ClassifierGCN(node_features=static_graph.num_node_features,
                          hidden_features=hyper_config['hidden_features'],
                          num_classes=2)
    model = model.to(device)

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyper_config['lr'],
                                 weight_decay=hyper_config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(int(hyper_config['num_epochs'])):
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