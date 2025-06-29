"""
Build trajectories using the completed cluster models.
"""

import json
import pickle
import numpy as np
from pathlib import Path
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cluster_models(results_dir="results_unified"):
    """Load all cluster models."""
    cluster_dir = Path(results_dir) / "cluster_models"
    cluster_models = {}
    
    for layer in range(12):
        model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
        if model_path.exists():
            cluster_models[layer] = joblib.load(model_path)
            logger.info(f"Loaded model for layer {layer}")
    
    return cluster_models

def build_trajectories(results_dir="results_unified"):
    """Build trajectories from cluster assignments."""
    results_path = Path(results_dir)
    
    # Load cluster models
    cluster_models = load_cluster_models(results_dir)
    
    # Load activations
    activations_file = results_path / "unified_activations.pkl"
    logger.info(f"Loading activations from {activations_file}")
    
    with open(activations_file, 'rb') as f:
        activations = pickle.load(f)
    
    # Load test cases to get metadata
    test_cases_file = Path("test_cases_unified.json")
    with open(test_cases_file, 'r') as f:
        data = json.load(f)
        test_cases = data['test_cases']
        metadata = data['metadata']
    
    trajectories = {}
    
    # Build trajectories for all cases
    logger.info(f"Building trajectories for {len(test_cases)} test cases...")
    
    for case_idx, case in enumerate(test_cases):
        if case_idx % 1000 == 0:
            logger.info(f"  Processing case {case_idx}/{len(test_cases)}")
            
        trajectory = []
        
        for layer in range(12):
            if layer in activations and case_idx in activations[layer]:
                # Get the activation
                act_data = activations[layer][case_idx][0]  # Should only be one
                activation = act_data['activation'].reshape(1, -1)
                
                # Predict cluster
                if layer in cluster_models:
                    cluster = cluster_models[layer].predict(activation)[0]
                    trajectory.append(int(cluster))
                else:
                    trajectory.append(-1)
            else:
                trajectory.append(-1)
        
        # Store trajectory
        key = f"{case['token_idx']}_{case['context_frame']}"
        trajectories[key] = {
            'token_idx': case['token_idx'],
            'token_str': case['token_str'],
            'context_frame': case['context_frame'],
            'path': trajectory,
            'case_idx': case_idx
        }
    
    logger.info(f"Built {len(trajectories)} trajectories")
    
    # Save trajectories
    results = {
        'metadata': {
            'experiment': 'unified_context_effects',
            'timestamp': datetime.now().isoformat(),
            'num_tokens': metadata['num_tokens'],
            'num_contexts': metadata['num_contexts'],
            'num_test_cases': len(test_cases),
            'k_clusters': 20
        },
        'trajectories': trajectories
    }
    
    output_file = results_path / "unified_trajectories.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved trajectories to {output_file}")
    
    # Also save visualization data
    vis_data = {
        'trajectories': trajectories,
        'context_frames': metadata['context_frames']
    }
    
    vis_file = results_path / "visualization_data.json"
    with open(vis_file, 'w') as f:
        json.dump(vis_data, f, indent=2)
    
    logger.info(f"Saved visualization data to {vis_file}")
    
    # Print some sample trajectories
    print("\nSample trajectories:")
    for i, (key, traj) in enumerate(trajectories.items()):
        if i >= 10:
            break
        print(f"  {traj['token_str']:10s} ({traj['context_frame']:15s}): {traj['path']}")
    
    return trajectories

def main():
    """Build trajectories."""
    logger.info("Building trajectories for unified context experiment...")
    
    trajectories = build_trajectories()
    
    # Quick statistics
    contexts = set(t['context_frame'] for t in trajectories.values())
    tokens = set(t['token_idx'] for t in trajectories.values())
    
    print(f"\nStatistics:")
    print(f"- Total trajectories: {len(trajectories)}")
    print(f"- Unique tokens: {len(tokens)}")
    print(f"- Context frames: {len(contexts)}")

if __name__ == "__main__":
    main()