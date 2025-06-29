"""
Continue k=10 clustering from where it left off.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_progress(results_dir="results_unified"):
    """Check which layers have been clustered."""
    cluster_dir = Path(results_dir) / "cluster_models_k10"
    completed_layers = []
    
    if cluster_dir.exists():
        for layer in range(12):
            model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
            if model_path.exists():
                completed_layers.append(layer)
    
    return completed_layers

def continue_clustering(results_dir="results_unified", k_clusters=10):
    """Continue clustering from where it left off."""
    results_path = Path(results_dir)
    
    # Check what's already done
    completed_layers = check_progress(results_dir)
    logger.info(f"Already completed layers: {completed_layers}")
    
    if len(completed_layers) == 12:
        logger.info("All layers already clustered!")
        return
    
    # Load activations
    activations_file = results_path / "unified_activations.pkl"
    logger.info(f"Loading activations from {activations_file}")
    
    with open(activations_file, 'rb') as f:
        activations = pickle.load(f)
    
    # Create cluster models directory
    cluster_dir = results_path / "cluster_models_k10"
    cluster_dir.mkdir(exist_ok=True)
    
    # Continue clustering from next layer
    for layer in range(12):
        if layer in completed_layers:
            logger.info(f"Skipping layer {layer} (already done)")
            continue
            
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
        
        # Print cluster statistics
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"  Cluster sizes: {dict(zip(unique, counts))}")

def main():
    """Continue clustering where it left off."""
    logger.info("Continuing k=10 clustering...")
    
    # Check initial status
    completed = check_progress()
    logger.info(f"Starting with {len(completed)}/12 layers completed")
    
    # Continue clustering
    continue_clustering()
    
    # Check final status
    completed = check_progress()
    logger.info(f"Finished with {len(completed)}/12 layers completed")

if __name__ == "__main__":
    main()