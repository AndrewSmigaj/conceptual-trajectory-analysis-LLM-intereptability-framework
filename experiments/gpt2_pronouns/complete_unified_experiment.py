"""
Complete the unified experiment using the saved activations.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import joblib
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def complete_clustering(results_dir="results_unified", k_clusters=20):
    """Complete the clustering phase using saved activations."""
    results_path = Path(results_dir)
    
    # Load activations
    activations_file = results_path / "unified_activations.pkl"
    logger.info(f"Loading activations from {activations_file}")
    
    with open(activations_file, 'rb') as f:
        activations = pickle.load(f)
    
    # Create cluster models directory
    cluster_dir = results_path / "cluster_models"
    cluster_dir.mkdir(exist_ok=True)
    
    # Perform clustering for each layer
    cluster_models = {}
    
    for layer in range(12):
        logger.info(f"Clustering layer {layer} with k={k_clusters}")
        
        # Collect all activations for this layer
        layer_activations = []
        activation_metadata = []
        
        if layer in activations:
            for case_idx in sorted(activations[layer].keys()):
                for act_data in activations[layer][case_idx]:
                    layer_activations.append(act_data['activation'])
                    activation_metadata.append({
                        'case_idx': act_data['case_idx'],
                        'token_idx': act_data['token_idx'],
                        'context': act_data['context']
                    })
        
        if not layer_activations:
            logger.warning(f"No activations for layer {layer}")
            continue
        
        # Convert to numpy array
        X = np.array(layer_activations)
        logger.info(f"  Layer {layer}: {X.shape[0]} activations, dim={X.shape[1]}")
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Save model
        model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
        joblib.dump(kmeans, model_path)
        logger.info(f"  Saved model to {model_path}")
        
        # Save cluster assignments
        assignments = []
        for i, (label, meta) in enumerate(zip(labels, activation_metadata)):
            assignments.append({
                'cluster': int(label),
                **meta
            })
        
        assignments_path = cluster_dir / f"assignments_layer_{layer}.json"
        with open(assignments_path, 'w') as f:
            json.dump(assignments, f)
        
        cluster_models[layer] = kmeans
        
        # Print cluster statistics
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"  Cluster sizes: {dict(zip(unique, counts))}")
    
    return cluster_models, activations

def build_trajectories(cluster_models, activations, results_dir="results_unified"):
    """Build trajectories from cluster assignments."""
    results_path = Path(results_dir)
    
    # Load test cases to get metadata
    test_cases_file = Path("test_cases_unified.json")
    with open(test_cases_file, 'r') as f:
        data = json.load(f)
        test_cases = data['test_cases']
        metadata = data['metadata']
    
    trajectories = {}
    
    # Build trajectories for all cases
    for case_idx, case in enumerate(test_cases):
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
            'k_clusters': k_clusters
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
    
    return trajectories

def main():
    """Complete the unified experiment."""
    logger.info("Completing unified context experiment...")
    
    # Complete clustering
    cluster_models, activations = complete_clustering()
    
    # Build trajectories
    trajectories = build_trajectories(cluster_models, activations)
    
    logger.info(f"Experiment completed with {len(trajectories)} trajectories")
    
    # Print summary
    print("\nSummary:")
    print(f"- Processed {len(trajectories)} test cases")
    print(f"- Created cluster models for {len(cluster_models)} layers")
    print(f"- Each trajectory has {len(trajectories[list(trajectories.keys())[0]]['path'])} steps")
    
    # Sample trajectories
    print("\nSample trajectories:")
    for i, (key, traj) in enumerate(trajectories.items()):
        if i >= 5:
            break
        print(f"  {traj['token_str']} ({traj['context_frame']}): {traj['path']}")

if __name__ == "__main__":
    main()