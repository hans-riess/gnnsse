import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load YAML configuration file with inheritance support.
    
    Args:
        config_path (str): Path to the YAML config file
        
    Returns:
        dict: Merged configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if 'base_config' in config:
        base_path = Path(config_path).parent / config['base_config']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Remove base_config key
        del config['base_config']
        
        # Deep merge configurations
        merged_config = deep_merge(base_config, config)
        return merged_config
    
    return config

def deep_merge(base, override):
    """
    Recursively merge two dictionaries.
    
    Args:
        base (dict): Base configuration
        override (dict): Override configuration
        
    Returns:
        dict: Merged configuration
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
            
    return merged