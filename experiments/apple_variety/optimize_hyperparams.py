"""Hyperparameter optimization for apple quality routing model."""

import itertools
import json
import yaml
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Tuple

# Define hyperparameter grid - reduced for faster testing
HYPERPARAMS = {
    'learning_rate': [0.001, 0.01],  # Skip 0.0001 as it's too slow
    'architecture': [[64, 48, 32], [128, 64, 32]],  # Current and larger
    'batch_size': [16, 32]  # Skip 64 as we have limited data
}

def create_temp_config(base_config_path: str, params: Dict) -> str:
    """Create temporary config file with specific hyperparameters."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update hyperparameters
    config['training']['learning_rate'] = params['learning_rate']
    config['training']['batch_size'] = params['batch_size']
    config['model']['architecture']['hidden_dims'] = params['architecture']
    
    # Save to temporary file
    temp_path = f"config_temp_{int(time.time())}.yaml"
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
    
    return temp_path

def run_experiment(config_path: str) -> Tuple[float, float, int]:
    """Run experiment and extract validation/test accuracy."""
    try:
        # Run the experiment from project root
        result = subprocess.run(
            ['./venv311/Scripts/python.exe', 'experiments/apple_variety/run_experiment.py', 
             '--config', f'experiments/apple_variety/{config_path}'],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd='../..'  # Run from project root
        )
        
        if result.returncode != 0:
            print(f"Experiment failed: {result.stderr}")
            return 0.0, 0.0, 0
        
        # Load results
        results_dir = Path('results/apple_variety_test')
        results_file = results_dir / 'apple_quality_routing_test_results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract metrics
            val_accuracies = results['results'].get('val_accuracies', [])
            test_accuracy = results['results'].get('test_accuracy', 0.0)
            early_stopped_epoch = results['results'].get('early_stopped_epoch', len(val_accuracies))
            
            # Get best validation accuracy
            best_val_acc = max(val_accuracies) if val_accuracies else 0.0
            
            return best_val_acc, test_accuracy, early_stopped_epoch
        else:
            print(f"Results file not found: {results_file}")
            return 0.0, 0.0, 0
            
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        return 0.0, 0.0, 0
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 0.0, 0.0, 0
    finally:
        # Clean up temp config
        Path(config_path).unlink(missing_ok=True)

def main():
    """Run hyperparameter optimization."""
    base_config = 'config_test.yaml'
    results = []
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        HYPERPARAMS['learning_rate'],
        HYPERPARAMS['architecture'],
        HYPERPARAMS['batch_size']
    ))
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations...")
    
    for i, (lr, arch, batch_size) in enumerate(param_combinations):
        print(f"\nCombination {i+1}/{len(param_combinations)}:")
        print(f"  Learning rate: {lr}")
        print(f"  Architecture: {arch}")
        print(f"  Batch size: {batch_size}")
        
        params = {
            'learning_rate': lr,
            'architecture': arch,
            'batch_size': batch_size
        }
        
        # Create temporary config
        temp_config = create_temp_config(base_config, params)
        
        # Run experiment
        val_acc, test_acc, epochs = run_experiment(temp_config)
        
        result = {
            'params': params,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'epochs': epochs
        }
        results.append(result)
        
        print(f"  Val accuracy: {val_acc:.2f}%")
        print(f"  Test accuracy: {test_acc:.2f}%")
        print(f"  Epochs: {epochs}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['val_accuracy'])
    
    print("\n" + "="*50)
    print("BEST CONFIGURATION:")
    print(f"  Learning rate: {best_result['params']['learning_rate']}")
    print(f"  Architecture: {best_result['params']['architecture']}")
    print(f"  Batch size: {best_result['params']['batch_size']}")
    print(f"  Val accuracy: {best_result['val_accuracy']:.2f}%")
    print(f"  Test accuracy: {best_result['test_accuracy']:.2f}%")
    print(f"  Epochs: {best_result['epochs']}")
    
    # Save all results
    output_file = 'hyperparameter_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'all_results': results,
            'best_result': best_result
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Create optimized config
    optimized_config = create_temp_config(base_config, best_result['params'])
    Path(optimized_config).rename('config_optimized.yaml')
    print("Created config_optimized.yaml with best hyperparameters")

if __name__ == '__main__':
    main()