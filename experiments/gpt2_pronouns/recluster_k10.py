"""
Re-cluster with k=10 instead of k=20.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recluster_with_k10(results_dir="results_unified", k_clusters=10):
    """Re-cluster all layers with k=10."""
    results_path = Path(results_dir)
    
    # Load activations
    activations_file = results_path / "unified_activations.pkl"
    logger.info(f"Loading activations from {activations_file}")
    
    with open(activations_file, 'rb') as f:
        activations = pickle.load(f)
    
    # Create cluster models directory
    cluster_dir = results_path / "cluster_models_k10"
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
    
    return cluster_models

def build_trajectories_k10(results_dir="results_unified"):
    """Build trajectories using k=10 cluster models."""
    results_path = Path(results_dir)
    
    # Load cluster models
    cluster_dir = results_path / "cluster_models_k10"
    cluster_models = {}
    
    for layer in range(12):
        model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
        if model_path.exists():
            cluster_models[layer] = joblib.load(model_path)
    
    # Load activations
    activations_file = results_path / "unified_activations.pkl"
    with open(activations_file, 'rb') as f:
        activations = pickle.load(f)
    
    # Load test cases
    test_cases_file = Path("test_cases_unified.json")
    with open(test_cases_file, 'r') as f:
        data = json.load(f)
        test_cases = data['test_cases']
        metadata = data['metadata']
    
    trajectories = {}
    
    # Build trajectories
    logger.info(f"Building trajectories for {len(test_cases)} test cases...")
    
    for case_idx, case in enumerate(test_cases):
        if case_idx % 1000 == 0:
            logger.info(f"  Processing case {case_idx}/{len(test_cases)}")
            
        trajectory = []
        
        for layer in range(12):
            if layer in activations and case_idx in activations[layer]:
                # Get the activation
                act_data = activations[layer][case_idx][0]
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
    
    # Save trajectories
    results = {
        'metadata': {
            'experiment': 'unified_context_effects_k10',
            'timestamp': datetime.now().isoformat(),
            'num_tokens': metadata['num_tokens'],
            'num_contexts': metadata['num_contexts'],
            'num_test_cases': len(test_cases),
            'k_clusters': 10
        },
        'trajectories': trajectories
    }
    
    output_file = results_path / "unified_trajectories_k10.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved trajectories to {output_file}")
    
    # Print sample trajectories
    print("\nSample trajectories (k=10):")
    for i, (key, traj) in enumerate(trajectories.items()):
        if i >= 10:
            break
        print(f"  {traj['token_str']:10s} ({traj['context_frame']:15s}): {traj['path']}")
    
    return trajectories

def main():
    """Re-cluster with k=10."""
    logger.info("Re-clustering with k=10...")
    
    # Re-cluster
    cluster_models = recluster_with_k10()
    
    # Build trajectories
    trajectories = build_trajectories_k10()
    
    logger.info(f"Complete! Built {len(trajectories)} trajectories with k=10")

if __name__ == "__main__":
    main()