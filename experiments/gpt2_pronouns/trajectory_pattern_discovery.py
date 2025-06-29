"""
Trajectory Pattern Discovery

Systematic identification of trajectory patterns across the full vocabulary.
Finds archetypal paths, groups similar trajectories, and discovers relationships.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, Counter
import logging
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import hamming
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryPatternDiscovery:
    """Discover and analyze trajectory patterns across vocabulary."""
    
    def __init__(self, trajectories_path: str, token_info_path: str = None):
        """Initialize with trajectory data."""
        # Load trajectories
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
            
        if 'trajectories' in data:
            self.trajectories = data['trajectories']
        else:
            self.trajectories = data
            
        # Load token information
        self.token_info = {}
        if token_info_path:
            token_path = Path(token_info_path)
            if not token_path.is_absolute():
                token_path = Path(trajectories_path).parent.parent / "gpt2/all_tokens/top_10k_tokens_full.json"
                
            if token_path.exists():
                with open(token_path, 'r') as f:
                    tokens = json.load(f)
                    self.token_info = {i: t for i, t in enumerate(tokens)}
                    
        logger.info(f"Loaded {len(self.trajectories)} trajectories for pattern discovery")
        
    def extract_archetypal_paths(self, layer_range: Tuple[int, int] = (0, 4)) -> Dict[str, Any]:
        """Extract archetypal trajectory patterns."""
        # Count trajectory patterns
        path_counter = Counter()
        path_examples = defaultdict(list)
        
        for key, traj_data in self.trajectories.items():
            # Focus on specified layer range
            path = tuple(traj_data['path'][layer_range[0]:layer_range[1]])
            
            # Skip paths with missing data
            if -1 not in path:
                path_counter[path] += 1
                
                token_idx = traj_data['token_idx']
                token_str = self.token_info.get(token_idx, {}).get('token_str', f'token_{token_idx}')
                path_examples[path].append({
                    'token': token_str,
                    'context': traj_data['context_frame']
                })
                
        # Identify archetypal paths (most common)
        archetypal_paths = []
        total_valid_paths = sum(path_counter.values())
        
        for path, count in path_counter.most_common(50):  # Top 50 patterns
            frequency = count / total_valid_paths
            
            # Get diverse examples
            examples = path_examples[path]
            diverse_examples = self._get_diverse_examples(examples, n=5)
            
            archetypal_paths.append({
                'path': list(path),
                'frequency': frequency,
                'count': count,
                'examples': diverse_examples
            })
            
        return {
            'layer_range': layer_range,
            'total_unique_paths': len(path_counter),
            'total_observations': total_valid_paths,
            'archetypal_paths': archetypal_paths
        }
        
    def _get_diverse_examples(self, examples: List[Dict], n: int = 5) -> List[Dict]:
        """Get diverse examples from a list."""
        if len(examples) <= n:
            return examples
            
        # Try to get examples from different contexts
        by_context = defaultdict(list)
        for ex in examples:
            by_context[ex['context']].append(ex)
            
        diverse = []
        # Round-robin selection from different contexts
        context_iters = {ctx: iter(exs) for ctx, exs in by_context.items()}
        
        while len(diverse) < n:
            for ctx, ctx_iter in context_iters.items():
                try:
                    diverse.append(next(ctx_iter))
                    if len(diverse) >= n:
                        break
                except StopIteration:
                    continue
                    
        return diverse[:n]
        
    def cluster_trajectory_patterns(self, eps: float = 0.3, min_samples: int = 5) -> Dict[str, Any]:
        """Cluster similar trajectory patterns using DBSCAN."""
        # Extract unique trajectories
        unique_trajectories = {}
        trajectory_tokens = defaultdict(list)
        
        for key, traj_data in self.trajectories.items():
            path = tuple(traj_data['path'][:4])  # First 4 layers
            if -1 not in path:  # Valid path
                unique_trajectories[path] = True
                trajectory_tokens[path].append({
                    'token_idx': traj_data['token_idx'],
                    'context': traj_data['context_frame']
                })
                
        # Convert to matrix for clustering
        trajectory_list = list(unique_trajectories.keys())
        n_trajectories = len(trajectory_list)
        
        # Compute pairwise distances
        distance_matrix = np.zeros((n_trajectories, n_trajectories))
        for i in range(n_trajectories):
            for j in range(i+1, n_trajectories):
                # Hamming distance normalized by length
                dist = sum(a != b for a, b in zip(trajectory_list[i], trajectory_list[j])) / 4
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Organize results
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                trajectory = trajectory_list[i]
                clusters[label].append({
                    'trajectory': list(trajectory),
                    'tokens': trajectory_tokens[trajectory]
                })
                
        # Characterize each cluster
        cluster_info = {}
        for cluster_id, members in clusters.items():
            # Find centroid (most representative trajectory)
            if members:
                cluster_trajectories = [m['trajectory'] for m in members]
                centroid_idx = self._find_centroid(cluster_trajectories)
                centroid = cluster_trajectories[centroid_idx]
                
                # Get example tokens
                all_tokens = []
                for member in members:
                    for token_data in member['tokens']:
                        token_idx = token_data['token_idx']
                        token_str = self.token_info.get(token_idx, {}).get('token_str', '')
                        if token_str:
                            all_tokens.append(token_str)
                            
                # Get unique examples
                unique_tokens = list(set(all_tokens))[:10]
                
                cluster_info[f'cluster_{cluster_id}'] = {
                    'size': len(members),
                    'centroid': centroid,
                    'example_tokens': unique_tokens,
                    'trajectory_diversity': self._calculate_diversity(cluster_trajectories)
                }
                
        return {
            'n_clusters': len(clusters),
            'n_noise_points': sum(1 for l in cluster_labels if l == -1),
            'clusters': cluster_info,
            'clustering_params': {'eps': eps, 'min_samples': min_samples}
        }
        
    def _find_centroid(self, trajectories: List[List[int]]) -> int:
        """Find the most representative trajectory in a set."""
        min_total_dist = float('inf')
        centroid_idx = 0
        
        for i, traj_i in enumerate(trajectories):
            total_dist = 0
            for j, traj_j in enumerate(trajectories):
                if i != j:
                    dist = sum(a != b for a, b in zip(traj_i, traj_j))
                    total_dist += dist
                    
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                centroid_idx = i
                
        return centroid_idx
        
    def _calculate_diversity(self, trajectories: List[List[int]]) -> float:
        """Calculate diversity of trajectories in a set."""
        if len(trajectories) < 2:
            return 0.0
            
        total_dist = 0
        comparisons = 0
        
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                dist = sum(a != b for a, b in zip(trajectories[i], trajectories[j])) / 4
                total_dist += dist
                comparisons += 1
                
        return total_dist / comparisons if comparisons > 0 else 0.0
        
    def analyze_context_trajectory_relationships(self) -> Dict[str, Any]:
        """Analyze how different contexts affect trajectory selection."""
        # Group by token and analyze context effects
        token_context_patterns = defaultdict(lambda: defaultdict(list))
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            path = tuple(traj_data['path'][:4])
            
            if -1 not in path:
                token_context_patterns[context][path].append(token_idx)
                
        # Analyze patterns per context
        context_analysis = {}
        
        for context, path_tokens in token_context_patterns.items():
            # Find dominant paths for this context
            path_counts = {path: len(tokens) for path, tokens in path_tokens.items()}
            total_tokens = sum(path_counts.values())
            
            # Get top paths
            top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            context_analysis[context] = {
                'total_tokens': total_tokens,
                'unique_paths': len(path_counts),
                'dominant_paths': [
                    {
                        'path': list(path),
                        'frequency': count / total_tokens,
                        'count': count
                    }
                    for path, count in top_paths
                ],
                'path_entropy': self._calculate_entropy(list(path_counts.values()))
            }
            
        return context_analysis
        
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy of a distribution."""
        total = sum(counts)
        if total == 0:
            return 0.0
            
        probs = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
        
    def find_trajectory_transitions(self) -> Dict[str, Any]:
        """Find common trajectory transitions between contexts."""
        # Group trajectories by token
        token_trajectories = defaultdict(dict)
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data['path'][:4]
            
        # Analyze transitions
        transitions = defaultdict(int)
        transition_examples = defaultdict(list)
        
        for token_idx, contexts in token_trajectories.items():
            if 'baseline' not in contexts:
                continue
                
            baseline_path = tuple(contexts['baseline'])
            if -1 in baseline_path:
                continue
                
            for context_name, context_path in contexts.items():
                if context_name != 'baseline' and -1 not in context_path:
                    context_path_tuple = tuple(context_path)
                    
                    if baseline_path != context_path_tuple:
                        transition = (baseline_path, context_path_tuple, context_name)
                        transitions[transition] += 1
                        
                        token_str = self.token_info.get(token_idx, {}).get('token_str', '')
                        if token_str:
                            transition_examples[transition].append(token_str)
                            
        # Organize results
        common_transitions = []
        for transition, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:50]:
            from_path, to_path, context = transition
            
            common_transitions.append({
                'from_path': list(from_path),
                'to_path': list(to_path),
                'context': context,
                'count': count,
                'examples': list(set(transition_examples[transition]))[:5]
            })
            
        return {
            'total_transitions': len(transitions),
            'common_transitions': common_transitions
        }
        
    def generate_trajectory_similarity_matrix(self, n_tokens: int = 100) -> np.ndarray:
        """Generate similarity matrix for token trajectories."""
        # Get a subset of tokens
        token_indices = sorted(set(t['token_idx'] for t in self.trajectories.values()))[:n_tokens]
        
        # Extract baseline trajectories
        baseline_trajectories = {}
        for token_idx in token_indices:
            key = f"{token_idx}_baseline"
            if key in self.trajectories:
                path = self.trajectories[key]['path'][:4]
                if -1 not in path:
                    baseline_trajectories[token_idx] = path
                    
        # Compute similarity matrix
        tokens = sorted(baseline_trajectories.keys())
        n = len(tokens)
        similarity_matrix = np.zeros((n, n))
        
        for i, token_i in enumerate(tokens):
            for j, token_j in enumerate(tokens):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    traj_i = baseline_trajectories[token_i]
                    traj_j = baseline_trajectories[token_j]
                    
                    # Similarity = 1 - normalized hamming distance
                    distance = sum(a != b for a, b in zip(traj_i, traj_j)) / len(traj_i)
                    similarity_matrix[i, j] = 1 - distance
                    
        return similarity_matrix, tokens
        
    def save_analysis(self, output_dir: Path) -> None:
        """Run all pattern discovery analyses and save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Extract archetypal paths
        logger.info("Extracting archetypal paths...")
        archetypal = self.extract_archetypal_paths()
        
        with open(output_dir / "archetypal_paths.json", 'w') as f:
            json.dump(archetypal, f, indent=2)
            
        # Cluster trajectory patterns
        logger.info("Clustering trajectory patterns...")
        clusters = self.cluster_trajectory_patterns()
        
        with open(output_dir / "trajectory_clusters.json", 'w') as f:
            json.dump(clusters, f, indent=2)
            
        # Analyze context relationships
        logger.info("Analyzing context-trajectory relationships...")
        context_patterns = self.analyze_context_trajectory_relationships()
        
        with open(output_dir / "context_trajectory_patterns.json", 'w') as f:
            json.dump(context_patterns, f, indent=2)
            
        # Find trajectory transitions
        logger.info("Finding trajectory transitions...")
        transitions = self.find_trajectory_transitions()
        
        with open(output_dir / "trajectory_transitions.json", 'w') as f:
            json.dump(transitions, f, indent=2)
            
        # Generate similarity matrix
        logger.info("Generating trajectory similarity matrix...")
        similarity_matrix, tokens = self.generate_trajectory_similarity_matrix()
        
        np.save(output_dir / "trajectory_similarity_matrix.npy", similarity_matrix)
        
        with open(output_dir / "similarity_matrix_tokens.json", 'w') as f:
            json.dump([int(t) for t in tokens], f)
            
        # Create summary
        summary = {
            'archetypal_paths_found': len(archetypal['archetypal_paths']),
            'trajectory_clusters': clusters['n_clusters'],
            'unique_transitions': transitions['total_transitions'],
            'contexts_analyzed': len(context_patterns),
            'similarity_matrix_size': similarity_matrix.shape
        }
        
        with open(output_dir / "pattern_discovery_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Pattern discovery complete. Results saved to {output_dir}")
        

def main():
    """Run trajectory pattern discovery."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str,
                       default='results/visualization_data.json',
                       help='Path to trajectory data')
    parser.add_argument('--output', type=str,
                       default='results/pattern_discovery/',
                       help='Output directory')
    args = parser.parse_args()
    
    # Run analysis
    discoverer = TrajectoryPatternDiscovery(args.trajectories)
    discoverer.save_analysis(Path(args.output))
    

if __name__ == "__main__":
    main()