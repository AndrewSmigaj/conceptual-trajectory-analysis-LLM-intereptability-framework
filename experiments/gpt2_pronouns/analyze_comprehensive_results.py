#!/usr/bin/env python3
"""
Analyze comprehensive results with activation-based distance metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple
import pickle

def load_trajectories(trajectory_file: str) -> Dict:
    """Load trajectory data."""
    print(f"Loading trajectories from {trajectory_file}...")
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    return data

def compute_trajectory_distance_matrix(trajectories: Dict, contexts: List[str]) -> np.ndarray:
    """
    Compute distance matrix based on actual trajectory paths.
    
    Instead of using transition matrices, this computes distances based on:
    1. How many layers have different cluster assignments
    2. How large the cluster differences are
    3. The overall trajectory shape
    """
    n_contexts = len(contexts)
    distance_matrix = np.zeros((n_contexts, n_contexts))
    
    # Get all token IDs that appear in all contexts
    token_sets = {}
    for ctx in contexts:
        token_sets[ctx] = set()
        for key, traj_data in trajectories.items():
            if ctx in key or (ctx == "baseline" and "_baseline" in key):
                token_id = traj_data.get('token_idx', key.split('_')[0])
                token_sets[ctx].add(token_id)
    
    # Find common tokens across all contexts
    common_tokens = set.intersection(*token_sets.values())
    print(f"Found {len(common_tokens)} tokens common to all contexts")
    
    # Compute pairwise distances
    for i, ctx1 in enumerate(contexts):
        for j, ctx2 in enumerate(contexts):
            if i >= j:
                continue
                
            distances = []
            
            for token_id in common_tokens:
                # Get trajectories for this token in both contexts
                traj1 = None
                traj2 = None
                
                for key, traj_data in trajectories.items():
                    if str(token_id) in key:
                        if ctx1 in key or (ctx1 == "baseline" and "_baseline" in key):
                            traj1 = traj_data.get('path', [])
                        if ctx2 in key or (ctx2 == "baseline" and "_baseline" in key):
                            traj2 = traj_data.get('path', [])
                
                if traj1 and traj2 and len(traj1) == len(traj2):
                    # Compute trajectory distance
                    # 1. Hamming distance (proportion of different clusters)
                    hamming = sum(c1 != c2 for c1, c2 in zip(traj1, traj2)) / len(traj1)
                    
                    # 2. Cluster distance (how far apart the clusters are)
                    cluster_dist = sum(abs(c1 - c2) for c1, c2 in zip(traj1, traj2)) / len(traj1)
                    
                    # 3. Shape distance (correlation between trajectories)
                    if len(set(traj1)) > 1 and len(set(traj2)) > 1:
                        corr = np.corrcoef(traj1, traj2)[0, 1]
                        shape_dist = 1 - max(0, corr)  # Convert correlation to distance
                    else:
                        shape_dist = 0.5  # Default if no variation
                    
                    # Combine metrics
                    total_dist = 0.4 * hamming + 0.4 * cluster_dist / 20 + 0.2 * shape_dist
                    distances.append(total_dist)
            
            if distances:
                avg_distance = np.mean(distances)
                distance_matrix[i, j] = avg_distance
                distance_matrix[j, i] = avg_distance
            else:
                # If no common tokens, use maximum distance
                distance_matrix[i, j] = 1.0
                distance_matrix[j, i] = 1.0
    
    return distance_matrix

def compute_activation_distance_matrix(activations_file: str, contexts: List[str]) -> np.ndarray:
    """
    Compute distance matrix based on activation patterns.
    
    This uses the actual activation vectors rather than cluster assignments.
    """
    print(f"Loading activations from {activations_file}...")
    
    # Load activations
    with open(activations_file, 'rb') as f:
        all_activations = pickle.load(f)
    
    n_contexts = len(contexts)
    distance_matrix = np.zeros((n_contexts, n_contexts))
    
    # Compute average activation pattern per context
    context_patterns = {}
    
    for ctx in contexts:
        patterns = []
        
        # Collect activations for this context across all tokens and layers
        for layer in range(12):  # GPT-2 layers
            layer_key = str(layer)
            if layer_key in all_activations:
                for position_key, acts in all_activations[layer_key].items():
                    # Filter activations for this context
                    for act_data in acts:
                        if isinstance(act_data, dict) and 'context' in act_data:
                            if act_data['context'] == ctx:
                                if 'activation' in act_data:
                                    patterns.append(act_data['activation'])
        
        if patterns:
            # Average activation pattern for this context
            context_patterns[ctx] = np.mean(patterns, axis=0)
        else:
            print(f"Warning: No activations found for context '{ctx}'")
    
    # Compute pairwise distances
    for i, ctx1 in enumerate(contexts):
        for j, ctx2 in enumerate(contexts):
            if i >= j:
                continue
            
            if ctx1 in context_patterns and ctx2 in context_patterns:
                # Cosine distance
                v1 = context_patterns[ctx1]
                v2 = context_patterns[ctx2]
                
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_dist = 1 - cos_sim
                
                # Euclidean distance (normalized)
                eucl_dist = np.linalg.norm(v1 - v2) / np.sqrt(len(v1))
                
                # Combine
                distance_matrix[i, j] = 0.7 * cos_dist + 0.3 * eucl_dist
                distance_matrix[j, i] = distance_matrix[i, j]
            else:
                distance_matrix[i, j] = 1.0
                distance_matrix[j, i] = 1.0
    
    return distance_matrix

def create_improved_dendrogram(distance_matrix: np.ndarray, contexts: List[str], 
                             title: str = "Context Similarity Based on Activation Patterns") -> plt.Figure:
    """Create dendrogram with improved visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Ensure valid distance matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Apply square root transform for better spread
    distance_matrix = np.sqrt(distance_matrix)
    
    # Normalize to [0, 1]
    if distance_matrix.max() > 0:
        distance_matrix = distance_matrix / distance_matrix.max()
    
    # Convert to condensed form
    condensed_dist = squareform(distance_matrix)
    
    # Perform hierarchical clustering with Ward linkage
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # Create dendrogram
    dendro = dendrogram(
        linkage_matrix,
        labels=[c.replace('_', ' ').title() for c in contexts],
        ax=ax,
        leaf_font_size=8,
        color_threshold=None
    )
    
    # Styling
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Context Type')
    ax.set_ylabel('Distance')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Rotate labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def analyze_comprehensive_results(results_dir: str = "results_comprehensive"):
    """Main analysis function."""
    results_path = Path(results_dir)
    
    # Load trajectories
    traj_file = results_path / "unified_trajectories_k20.json"
    if not traj_file.exists():
        print(f"Trajectory file not found: {traj_file}")
        return
    
    trajectories = load_trajectories(traj_file)
    
    # Define contexts to analyze
    primary_contexts = [
        "baseline", "the", "a", "he", "she", "is", "was",
        "in", "on", "and", "but", "not", "said", "good", ".", "sentence_start"
    ]
    
    # 1. Compute trajectory-based distance matrix
    print("\nComputing trajectory-based distances...")
    traj_dist_matrix = compute_trajectory_distance_matrix(
        trajectories.get('trajectories', {}), 
        primary_contexts
    )
    
    # Create trajectory-based dendrogram
    fig1 = create_improved_dendrogram(
        traj_dist_matrix,
        primary_contexts,
        "Context Similarity Based on Trajectory Patterns"
    )
    fig1.savefig(results_path / "dendrogram_trajectory_based.pdf", dpi=300, bbox_inches='tight')
    fig1.savefig(results_path / "dendrogram_trajectory_based.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. If activations are available, compute activation-based distances
    act_file = results_path / "unified_activations.pkl"
    if act_file.exists():
        print("\nComputing activation-based distances...")
        act_dist_matrix = compute_activation_distance_matrix(str(act_file), primary_contexts)
        
        fig2 = create_improved_dendrogram(
            act_dist_matrix,
            primary_contexts,
            "Context Similarity Based on Activation Patterns"
        )
        fig2.savefig(results_path / "dendrogram_activation_based.pdf", dpi=300, bbox_inches='tight')
        fig2.savefig(results_path / "dendrogram_activation_based.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    analyze_comprehensive_results()