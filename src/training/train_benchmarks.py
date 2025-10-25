"""
Training script for Benchmark models
"""

from ..models.benchmarks import BenchmarkModel
from ..data.s3_manager import get_s3_manager
import yaml


def train_benchmarks(preprocessed_data: dict, route: str, models: list = None):
    """
    Train benchmark models
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
        models: List of benchmark models to train. If None, trains all.
                Options: ['dcrnn', 'stgcn', 'gwnet', 'tgcn', 'mtgnn', 'stfgnn', 'st_resnet', 'st_gconv']
    
    Returns:
        Dictionary of trained benchmark models
    """
    print("\n" + "="*60)
    print("Training Benchmark Models")
    print("="*60)
    
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['benchmarks']
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    adjacency_matrix = preprocessed_data['adjacency_matrix']
    node_list = preprocessed_data['node_list']
    
    # Default to all models if not specified
    if models is None:
        models = ['dcrnn', 'stgcn', 'gwnet', 'tgcn']
    
    trained_models = {}
    s3_manager = get_s3_manager()
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print('='*60)
        
        try:
            # Get model-specific config
            model_config = config.get(model_name, {})
            
            # Initialize and train model
            model = BenchmarkModel(model_name, model_config)
            model.train(gps_data, adjacency_matrix, node_list)
            
            # Save model to S3
            model_params = model.get_model_params()
            
            metadata = {
                'model_type': f'Benchmark-{model_name.upper()}',
                'route': route,
                'num_nodes': len(node_list),
                'config': model_config
            }
            
            s3_manager.save_model(model_params, f'benchmark_{model_name}', route, metadata)
            
            trained_models[model_name] = model
            
            print(f"✓ {model_name.upper()} training complete and saved to S3")
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Benchmark training complete. Trained {len(trained_models)} models")
    print("="*60 + "\n")
    
    return trained_models


def train_single_benchmark(preprocessed_data: dict, route: str, model_name: str):
    """
    Train a single benchmark model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
        model_name: Name of benchmark model ('dcrnn', 'stgcn', 'gwnet', 'tgcn', etc.)
    
    Returns:
        Trained benchmark model
    """
    models = train_benchmarks(preprocessed_data, route, models=[model_name])
    return models.get(model_name)
