"""
Comprehensive Clustering Analysis for Context Effects

Maps all token activations (with and without context) to the existing k10 clusters.
Uses the baseline token assignments as ground truth and compares context-influenced
activations to understand trajectory changes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
from tqdm import tqdm
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveClusteringAnalysis:
    """Map all activations to existing k10 clusters and analyze trajectories."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with paths to existing data."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load existing cluster assignments
        labels_path = Path(self.config['analysis']['clustering']['labels_path'])
        if not labels_path.is_absolute():
            labels_path = Path(config_path).parent / labels_path
            
        with open(labels_path, 'r') as f:
            self.baseline_clusters = json.load(f)
            
        # Convert string keys to int
        self.baseline_clusters = {int(k): v for k, v in self.baseline_clusters.items()}
        
        logger.info(f"Loaded baseline clusters for {len(self.baseline_clusters)} layers")
        
    def load_experiment_results(self, results_path: str) -> Dict[str, Any]:
        """Load trajectories from vocabulary experiment."""
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data.get('trajectories', {}))} trajectories")
        return data
        
    def analyze_trajectory_consistency(self, trajectories: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how consistent trajectories are within token groups."""
        # Group trajectories by token
        token_trajectories = defaultdict(dict)
        
        for key, traj_data in trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data['path']
            
        consistency_analysis = {
            'per_token_consistency': {},
            'aggregate_stats': {},
            'context_effects': defaultdict(list)
        }
        
        # Analyze each token
        for token_idx, contexts in token_trajectories.items():
            if 'baseline' not in contexts:
                continue
                
            baseline = contexts['baseline']
            
            # Check if baseline matches original clustering
            baseline_match = self._check_baseline_match(token_idx, baseline)
            
            # Compare each context to baseline
            divergences = {}
            for context_name, trajectory in contexts.items():
                if context_name == 'baseline':
                    continue
                    
                divergence = self._calculate_trajectory_divergence(baseline, trajectory)
                divergences[context_name] = divergence
                consistency_analysis['context_effects'][context_name].append(divergence)
                
            consistency_analysis['per_token_consistency'][token_idx] = {
                'baseline_matches_original': baseline_match,
                'context_divergences': divergences,
                'max_divergence': max(divergences.values()) if divergences else 0,
                'avg_divergence': np.mean(list(divergences.values())) if divergences else 0
            }
            
        # Compute aggregate statistics
        all_divergences = []
        for token_data in consistency_analysis['per_token_consistency'].values():
            all_divergences.extend(token_data['context_divergences'].values())
            
        consistency_analysis['aggregate_stats'] = {
            'mean_divergence': np.mean(all_divergences) if all_divergences else 0,
            'std_divergence': np.std(all_divergences) if all_divergences else 0,
            'max_divergence': max(all_divergences) if all_divergences else 0,
            'tokens_with_context_effects': sum(1 for t in consistency_analysis['per_token_consistency'].values()
                                              if t['max_divergence'] > 0)
        }
        
        return consistency_analysis
        
    def _check_baseline_match(self, token_idx: int, trajectory: List[int]) -> float:
        """Check if baseline trajectory matches original k10 clustering."""
        matches = 0
        total = 0
        
        for layer, cluster in enumerate(trajectory):
            if cluster == -1:  # Skip missing data
                continue
                
            if layer in self.baseline_clusters and token_idx < len(self.baseline_clusters[layer]):
                original_cluster = self.baseline_clusters[layer][token_idx]
                if cluster == original_cluster:
                    matches += 1
                total += 1
                
        return matches / total if total > 0 else 0
        
    def _calculate_trajectory_divergence(self, traj1: List[int], traj2: List[int]) -> float:
        """Calculate normalized divergence between two trajectories."""
        divergence = 0
        valid_positions = 0
        
        for i in range(min(len(traj1), len(traj2))):
            if traj1[i] != -1 and traj2[i] != -1:
                if traj1[i] != traj2[i]:
                    divergence += 1
                valid_positions += 1
                
        return divergence / valid_positions if valid_positions > 0 else 0
        
    def identify_context_sensitive_tokens(self, consistency_analysis: Dict[str, Any], 
                                        threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Identify tokens whose trajectories are most affected by context."""
        sensitive_tokens = []
        
        for token_idx, data in consistency_analysis['per_token_consistency'].items():
            if data['max_divergence'] >= threshold:
                sensitive_tokens.append({
                    'token_idx': int(token_idx),
                    'max_divergence': data['max_divergence'],
                    'avg_divergence': data['avg_divergence'],
                    'most_affecting_contexts': sorted(
                        data['context_divergences'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                })
                
        # Sort by maximum divergence
        sensitive_tokens.sort(key=lambda x: x['max_divergence'], reverse=True)
        
        return sensitive_tokens
        
    def analyze_context_patterns(self, consistency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which contexts have the strongest effects."""
        context_effects = consistency_analysis['context_effects']
        
        pattern_analysis = {}
        
        for context, divergences in context_effects.items():
            pattern_analysis[context] = {
                'mean_effect': np.mean(divergences),
                'std_effect': np.std(divergences),
                'max_effect': max(divergences) if divergences else 0,
                'tokens_affected': sum(1 for d in divergences if d > 0),
                'strong_effects': sum(1 for d in divergences if d > 0.5)
            }
            
        # Rank contexts by average effect
        ranked_contexts = sorted(
            pattern_analysis.items(), 
            key=lambda x: x[1]['mean_effect'], 
            reverse=True
        )
        
        return {
            'context_rankings': ranked_contexts,
            'strongest_context': ranked_contexts[0] if ranked_contexts else None,
            'weakest_context': ranked_contexts[-1] if ranked_contexts else None
        }
        
    def generate_trajectory_heatmap_data(self, trajectories: Dict[str, Any], 
                                       num_tokens: int = 100) -> np.ndarray:
        """Generate data for trajectory heatmap visualization."""
        # Group by token
        token_trajectories = defaultdict(dict)
        
        for key, traj_data in trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data['path']
            
        # Create matrix: tokens × contexts × layers
        context_types = ['baseline', 'determiner_the', 'determiner_a', 'pronoun_i', 
                        'pronoun_they', 'preposition_with', 'preposition_of',
                        'sentence_start_is', 'sentence_start_are']
        
        # Select first num_tokens tokens
        selected_tokens = sorted(list(token_trajectories.keys()))[:num_tokens]
        
        heatmap_data = []
        
        for token_idx in selected_tokens:
            token_data = []
            for context in context_types:
                if context in token_trajectories[token_idx]:
                    trajectory = token_trajectories[token_idx][context]
                    # Use first 4 layers for visualization
                    token_data.append(trajectory[:4])
                else:
                    token_data.append([-1, -1, -1, -1])
            heatmap_data.append(token_data)
            
        return np.array(heatmap_data)
        
    def save_analysis(self, output_dir: Path, trajectories: Dict[str, Any]) -> None:
        """Run all analyses and save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Run consistency analysis
        logger.info("Analyzing trajectory consistency...")
        consistency = self.analyze_trajectory_consistency(trajectories)
        
        with open(output_dir / "trajectory_consistency_analysis.json", 'w') as f:
            json.dump(consistency, f, indent=2)
            
        # Identify context-sensitive tokens
        logger.info("Identifying context-sensitive tokens...")
        sensitive_tokens = self.identify_context_sensitive_tokens(consistency)
        
        with open(output_dir / "context_sensitive_tokens.json", 'w') as f:
            json.dump(sensitive_tokens[:100], f, indent=2)  # Top 100
            
        # Analyze context patterns
        logger.info("Analyzing context patterns...")
        context_patterns = self.analyze_context_patterns(consistency)
        
        with open(output_dir / "context_pattern_analysis.json", 'w') as f:
            json.dump(context_patterns, f, indent=2)
            
        # Generate heatmap data
        logger.info("Generating visualization data...")
        heatmap_data = self.generate_trajectory_heatmap_data(trajectories)
        
        np.save(output_dir / "trajectory_heatmap_data.npy", heatmap_data)
        
        # Summary report
        summary = {
            'total_tokens_analyzed': len(consistency['per_token_consistency']),
            'tokens_with_context_effects': consistency['aggregate_stats']['tokens_with_context_effects'],
            'mean_divergence': consistency['aggregate_stats']['mean_divergence'],
            'top_5_sensitive_tokens': sensitive_tokens[:5] if sensitive_tokens else [],
            'strongest_context_effect': context_patterns['strongest_context'],
            'percentage_affected': (consistency['aggregate_stats']['tokens_with_context_effects'] / 
                                  len(consistency['per_token_consistency']) * 100)
        }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_dir}")
        

def main():
    """Run comprehensive clustering analysis."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str, 
                       default='results/visualization_data.json',
                       help='Path to trajectory data from experiment')
    parser.add_argument('--output', type=str,
                       default='results/clustering_analysis/',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComprehensiveClusteringAnalysis()
    
    # Load trajectories
    data = analyzer.load_experiment_results(args.trajectories)
    trajectories = data.get('trajectories', {})
    
    # Run analysis
    analyzer.save_analysis(Path(args.output), trajectories)
    

if __name__ == "__main__":
    main()