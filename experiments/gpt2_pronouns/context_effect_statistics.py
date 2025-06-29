"""
Context Effect Statistics

Comprehensive statistical analysis of how context affects token trajectories.
Includes significance testing, effect size calculations, and distribution analysis.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextEffectStatistics:
    """Statistical analysis of context effects on token trajectories."""
    
    def __init__(self, trajectories_path: str, token_info_path: str = None):
        """Initialize with trajectory data."""
        # Load trajectories
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
            
        if 'trajectories' in data:
            self.trajectories = data['trajectories']
        else:
            self.trajectories = data
            
        # Load token information if provided
        self.token_info = {}
        if token_info_path:
            token_path = Path(token_info_path)
            if not token_path.is_absolute():
                token_path = Path(trajectories_path).parent.parent / "gpt2/all_tokens/top_10k_tokens_full.json"
                
            if token_path.exists():
                with open(token_path, 'r') as f:
                    tokens = json.load(f)
                    self.token_info = {i: t for i, t in enumerate(tokens)}
                    
        logger.info(f"Loaded {len(self.trajectories)} trajectories")
        
    def compute_effect_sizes(self) -> Dict[str, Any]:
        """Compute Cohen's d effect sizes for each context type."""
        # Group trajectories by token and context
        token_groups = defaultdict(dict)
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        # Compute effect sizes for each context type
        context_effects = defaultdict(list)
        
        for token_idx, contexts in token_groups.items():
            if 'baseline' not in contexts:
                continue
                
            baseline = contexts['baseline']
            
            for context_name, trajectory in contexts.items():
                if context_name == 'baseline':
                    continue
                    
                # Compute trajectory distance as effect
                effect = self._compute_trajectory_distance(baseline, trajectory)
                context_effects[context_name].append(effect)
                
        # Calculate Cohen's d for each context
        effect_sizes = {}
        
        for context, effects in context_effects.items():
            if effects:
                # Compare to zero (no effect)
                d = np.mean(effects) / (np.std(effects) + 1e-10)
                
                effect_sizes[context] = {
                    'cohens_d': d,
                    'mean_effect': np.mean(effects),
                    'std_effect': np.std(effects),
                    'n_tokens': len(effects),
                    'interpretation': self._interpret_cohens_d(d)
                }
                
        return effect_sizes
        
    def _compute_trajectory_distance(self, traj1: List[int], traj2: List[int]) -> float:
        """Compute normalized Hamming distance between trajectories."""
        distance = 0
        valid_positions = 0
        
        # Focus on early layers where effects are strongest
        for i in range(min(4, len(traj1), len(traj2))):
            if traj1[i] != -1 and traj2[i] != -1:
                if traj1[i] != traj2[i]:
                    distance += 1
                valid_positions += 1
                
        return distance / valid_positions if valid_positions > 0 else 0
        
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
            
    def test_context_independence(self) -> Dict[str, Any]:
        """Test if context effects are independent using chi-squared tests."""
        # Create contingency tables for cluster assignments
        results = {}
        
        # For each layer, test if cluster distribution depends on context
        for layer in range(4):  # Focus on early layers
            contingency_table = self._build_contingency_table(layer)
            
            if contingency_table is not None:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Cramér's V for effect size
                n = contingency_table.sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                results[f'layer_{layer}'] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'cramers_v': cramers_v,
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_cramers_v(cramers_v)
                }
                
        return results
        
    def _build_contingency_table(self, layer: int) -> Optional[np.ndarray]:
        """Build contingency table for cluster assignments at a layer."""
        # Group trajectories
        context_clusters = defaultdict(list)
        
        for traj_data in self.trajectories.values():
            context = traj_data['context_frame']
            path = traj_data['path']
            
            if layer < len(path) and path[layer] != -1:
                context_clusters[context].append(path[layer])
                
        if len(context_clusters) < 2:
            return None
            
        # Build table
        contexts = sorted(context_clusters.keys())
        max_cluster = max(max(clusters) for clusters in context_clusters.values())
        
        table = np.zeros((len(contexts), max_cluster + 1))
        
        for i, context in enumerate(contexts):
            for cluster in context_clusters[context]:
                table[i, cluster] += 1
                
        # Remove empty columns
        table = table[:, table.sum(axis=0) > 0]
        
        return table if table.size > 0 else None
        
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
            
    def analyze_token_type_effects(self) -> Dict[str, Any]:
        """Analyze if certain token types are more affected by context."""
        if not self.token_info:
            logger.warning("No token information available for type analysis")
            return {}
            
        # Group effects by token type
        type_effects = defaultdict(list)
        
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        for token_idx, contexts in token_groups.items():
            if 'baseline' not in contexts or token_idx not in self.token_info:
                continue
                
            token_type = self.token_info[token_idx].get('token_type', 'unknown')
            baseline = contexts['baseline']
            
            # Compute maximum effect across all contexts
            max_effect = 0
            for context_name, trajectory in contexts.items():
                if context_name != 'baseline':
                    effect = self._compute_trajectory_distance(baseline, trajectory)
                    max_effect = max(max_effect, effect)
                    
            type_effects[token_type].append(max_effect)
            
        # Compute statistics per type
        type_stats = {}
        for token_type, effects in type_effects.items():
            if effects:
                type_stats[token_type] = {
                    'mean_effect': np.mean(effects),
                    'std_effect': np.std(effects),
                    'median_effect': np.median(effects),
                    'n_tokens': len(effects),
                    'highly_affected': sum(1 for e in effects if e > 0.5)
                }
                
        # ANOVA test for differences between token types
        if len(type_effects) > 2:
            f_stat, p_value = stats.f_oneway(*type_effects.values())
            type_stats['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        return type_stats
        
    def compute_layer_wise_statistics(self) -> Dict[str, Any]:
        """Compute statistics for each layer separately."""
        layer_stats = {}
        
        for layer in range(12):
            layer_divergences = []
            
            token_groups = defaultdict(dict)
            for key, traj_data in self.trajectories.items():
                token_idx = traj_data['token_idx']
                context = traj_data['context_frame']
                token_groups[token_idx][context] = traj_data['path']
                
            for token_idx, contexts in token_groups.items():
                if 'baseline' not in contexts:
                    continue
                    
                baseline = contexts['baseline']
                
                for context_name, trajectory in contexts.items():
                    if context_name == 'baseline':
                        continue
                        
                    if (layer < len(baseline) and layer < len(trajectory) and
                        baseline[layer] != -1 and trajectory[layer] != -1):
                        if baseline[layer] != trajectory[layer]:
                            layer_divergences.append(1)
                        else:
                            layer_divergences.append(0)
                            
            if layer_divergences:
                layer_stats[f'layer_{layer}'] = {
                    'divergence_rate': np.mean(layer_divergences),
                    'n_comparisons': len(layer_divergences),
                    'divergent_pairs': sum(layer_divergences)
                }
                
        return layer_stats
        
    def identify_outliers(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Identify tokens with unusually high context sensitivity."""
        token_effects = {}
        
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        for token_idx, contexts in token_groups.items():
            if 'baseline' not in contexts:
                continue
                
            baseline = contexts['baseline']
            effects = []
            
            for context_name, trajectory in contexts.items():
                if context_name != 'baseline':
                    effect = self._compute_trajectory_distance(baseline, trajectory)
                    effects.append(effect)
                    
            if effects:
                token_effects[token_idx] = {
                    'mean_effect': np.mean(effects),
                    'max_effect': max(effects),
                    'token_str': self.token_info.get(token_idx, {}).get('token_str', 'unknown')
                }
                
        # Calculate z-scores
        all_effects = [t['mean_effect'] for t in token_effects.values()]
        mean_effect = np.mean(all_effects)
        std_effect = np.std(all_effects)
        
        outliers = []
        for token_idx, data in token_effects.items():
            z_score = (data['mean_effect'] - mean_effect) / (std_effect + 1e-10)
            if abs(z_score) > threshold:
                outliers.append({
                    'token_idx': token_idx,
                    'token_str': data['token_str'],
                    'mean_effect': data['mean_effect'],
                    'z_score': z_score
                })
                
        # Sort by z-score
        outliers.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return outliers
        
    def generate_report(self, output_path: str) -> None:
        """Generate comprehensive statistical report."""
        report = {
            'effect_sizes': self.compute_effect_sizes(),
            'independence_tests': self.test_context_independence(),
            'token_type_analysis': self.analyze_token_type_effects(),
            'layer_statistics': self.compute_layer_wise_statistics(),
            'outliers': self.identify_outliers()[:20]  # Top 20 outliers
        }
        
        # Add summary statistics
        all_effects = []
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        for token_idx, contexts in token_groups.items():
            if 'baseline' in contexts:
                baseline = contexts['baseline']
                for context_name, trajectory in contexts.items():
                    if context_name != 'baseline':
                        effect = self._compute_trajectory_distance(baseline, trajectory)
                        all_effects.append(effect)
                        
        report['summary'] = {
            'total_comparisons': len(all_effects),
            'mean_effect': np.mean(all_effects) if all_effects else 0,
            'median_effect': np.median(all_effects) if all_effects else 0,
            'std_effect': np.std(all_effects) if all_effects else 0,
            'tokens_with_any_effect': sum(1 for e in all_effects if e > 0),
            'tokens_with_large_effect': sum(1 for e in all_effects if e > 0.5)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Statistical report saved to {output_path}")
        

def main():
    """Run statistical analysis."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str,
                       default='results/visualization_data.json',
                       help='Path to trajectory data')
    parser.add_argument('--output', type=str,
                       default='results/statistical_report.json',
                       help='Output path for report')
    args = parser.parse_args()
    
    # Run analysis
    stats = ContextEffectStatistics(args.trajectories)
    stats.generate_report(args.output)
    

if __name__ == "__main__":
    main()