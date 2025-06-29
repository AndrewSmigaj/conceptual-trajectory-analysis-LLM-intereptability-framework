"""
Build trajectories with k=10 and properly analyze divergence.
"""

import json
import pickle
import numpy as np
from pathlib import Path
import joblib
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cluster_models(results_dir="results_unified"):
    """Load all k=10 cluster models."""
    cluster_dir = Path(results_dir) / "cluster_models_k10"
    cluster_models = {}
    
    for layer in range(12):
        model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
        if model_path.exists():
            cluster_models[layer] = joblib.load(model_path)
    
    return cluster_models

def build_trajectories(results_dir="results_unified"):
    """Build trajectories using k=10 models."""
    results_path = Path(results_dir)
    
    # Load cluster models
    cluster_models = load_cluster_models(results_dir)
    
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
    
    return trajectories

def calculate_proper_divergence(trajectories):
    """Calculate divergence scores properly."""
    # Group by token
    token_groups = defaultdict(dict)
    for key, traj_data in trajectories.items():
        token_idx = traj_data['token_idx']
        context = traj_data['context_frame']
        token_groups[token_idx][context] = traj_data
    
    divergence_results = {
        'per_token': {},
        'by_context': defaultdict(list),
        'summary': {}
    }
    
    # Calculate divergences
    for token_idx, contexts in token_groups.items():
        if 'baseline' not in contexts:
            continue
            
        baseline_path = contexts['baseline']['path']
        token_str = contexts['baseline']['token_str']
        
        token_divergences = {}
        
        for context_name, traj_data in contexts.items():
            if context_name == 'baseline':
                continue
                
            context_path = traj_data['path']
            
            # Calculate layer-wise divergence
            layer_diffs = []
            for i in range(len(baseline_path)):
                if baseline_path[i] != -1 and context_path[i] != -1:
                    layer_diffs.append(1 if baseline_path[i] != context_path[i] else 0)
            
            # Calculate divergence metrics
            if layer_diffs:
                full_divergence = sum(layer_diffs) / len(layer_diffs)
                early_divergence = sum(layer_diffs[:4]) / min(4, len(layer_diffs)) if layer_diffs else 0
                
                # Find first divergence point
                bifurcation_layer = -1
                for i, diff in enumerate(layer_diffs):
                    if diff == 1:
                        bifurcation_layer = i
                        break
            else:
                full_divergence = 0
                early_divergence = 0
                bifurcation_layer = -1
            
            divergence = {
                'full_divergence': full_divergence,
                'early_divergence': early_divergence,
                'bifurcation_layer': bifurcation_layer,
                'divergent_layers': sum(layer_diffs) if layer_diffs else 0
            }
            
            token_divergences[context_name] = divergence
            divergence_results['by_context'][context_name].append({
                'token_idx': token_idx,
                'token_str': token_str,
                **divergence
            })
        
        # Store per-token results
        if token_divergences:
            divergence_results['per_token'][token_idx] = {
                'token_str': token_str,
                'divergences': token_divergences,
                'max_divergence': max(d['full_divergence'] for d in token_divergences.values()),
                'mean_divergence': np.mean([d['full_divergence'] for d in token_divergences.values()])
            }
    
    # Calculate summary statistics
    all_divergences = []
    for token_data in divergence_results['per_token'].values():
        for div in token_data['divergences'].values():
            all_divergences.append(div['full_divergence'])
    
    divergence_results['summary'] = {
        'mean_divergence': np.mean(all_divergences) if all_divergences else 0,
        'std_divergence': np.std(all_divergences) if all_divergences else 0,
        'max_divergence': max(all_divergences) if all_divergences else 0,
        'min_divergence': min(all_divergences) if all_divergences else 0,
        'tokens_affected': sum(1 for t in divergence_results['per_token'].values() if t['max_divergence'] > 0),
        'total_tokens': len(divergence_results['per_token']),
        'total_comparisons': len(all_divergences)
    }
    
    return divergence_results

def main():
    """Build trajectories and analyze with k=10."""
    logger.info("Building trajectories with k=10...")
    
    # Build trajectories
    trajectories = build_trajectories()
    
    # Print sample trajectories
    print("\nSample trajectories (k=10):")
    count = 0
    for key, traj in trajectories.items():
        if count >= 10:
            break
        print(f"  {traj['token_str']:10s} ({traj['context_frame']:15s}): {traj['path']}")
        count += 1
    
    # Calculate proper divergences
    logger.info("Calculating divergence scores...")
    divergence_results = calculate_proper_divergence(trajectories)
    
    # Save analysis
    output_dir = Path("results_unified/analysis_k10")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "divergence_analysis_k10.json", 'w') as f:
        json.dump(divergence_results, f, indent=2)
    
    # Print summary
    summary = divergence_results['summary']
    print(f"\nDivergence Analysis (k=10):")
    print(f"  Mean divergence: {summary['mean_divergence']:.3f}")
    print(f"  Std divergence: {summary['std_divergence']:.3f}")
    print(f"  Min divergence: {summary['min_divergence']:.3f}")
    print(f"  Max divergence: {summary['max_divergence']:.3f}")
    print(f"  Tokens affected: {summary['tokens_affected']}/{summary['total_tokens']}")
    
    # Context effect summary
    print(f"\nContext Effects:")
    for context_name, tokens in divergence_results['by_context'].items():
        if tokens:
            mean_div = np.mean([t['full_divergence'] for t in tokens])
            print(f"  {context_name}: {mean_div:.3f} mean divergence")

if __name__ == "__main__":
    main()