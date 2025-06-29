"""
Check the status of the unified context experiment.
"""

import json
from pathlib import Path
import pickle

def check_experiment_status(results_dir="results_unified"):
    """Check the status of an experiment."""
    results_path = Path(results_dir)
    
    print(f"Checking experiment status in: {results_path}")
    print("-" * 50)
    
    # Check for checkpoint
    checkpoint_file = results_path / "checkpoints" / "latest_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"Checkpoint found:")
        print(f"  Last processed index: {checkpoint['last_processed_idx']}")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        
        # Check activations file size
        activations_file = results_path / "checkpoints" / "activations_partial.pkl"
        if activations_file.exists():
            size_mb = activations_file.stat().st_size / (1024 * 1024)
            print(f"  Activations file size: {size_mb:.1f} MB")
            
            # Load to check content
            with open(activations_file, 'rb') as f:
                activations = pickle.load(f)
            
            total_acts = 0
            for layer in activations:
                layer_acts = sum(len(acts) for acts in activations[layer].values())
                total_acts += layer_acts
                print(f"    Layer {layer}: {layer_acts} activations")
            print(f"  Total activations: {total_acts}")
    else:
        print("No checkpoint found - experiment hasn't started or saved yet")
    
    # Check for completed files
    completed_files = [
        "unified_activations.pkl",
        "unified_trajectories.json",
        "visualization_data.json"
    ]
    
    print("\nCompleted files:")
    for filename in completed_files:
        filepath = results_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")
        else:
            print(f"  {filename}: Not found")
    
    # Check cluster models
    cluster_dir = results_path / "cluster_models"
    if cluster_dir.exists():
        models = list(cluster_dir.glob("*.pkl"))
        assignments = list(cluster_dir.glob("*.json"))
        print(f"\nCluster models: {len(models)} saved")
        print(f"Cluster assignments: {len(assignments)} saved")
    
    return checkpoint_file.exists()

if __name__ == "__main__":
    # Check main experiment
    print("MAIN EXPERIMENT:")
    has_checkpoint_main = check_experiment_status("results_unified")
    
    print("\n" + "=" * 50 + "\n")
    
    # Check small experiment
    print("SMALL TEST EXPERIMENT:")
    has_checkpoint_small = check_experiment_status("results_unified_small")
    
    # Recommendation
    print("\n" + "=" * 50)
    print("\nRECOMMENDATION:")
    if has_checkpoint_main:
        print("Main experiment has a checkpoint - can resume from there")
    elif has_checkpoint_small:
        print("Small test has made progress - can analyze partial results")
    else:
        print("No progress found - need to start fresh")