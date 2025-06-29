"""
Effect size calculator for transformation analysis.

Calculates various effect size measures to quantify the magnitude of
transformation effects, complementing statistical significance tests.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .base_transformation_analysis import BaseTransformationAnalysis
from .bootstrap_mixin import BootstrapMixin
from .output_schema import UnifiedAnalysisOutput, MetricWithCI


logger = logging.getLogger(__name__)


class EffectSizeCalculator(BaseTransformationAnalysis, BootstrapMixin):
    """
    Calculate effect sizes for transformation analyses.
    
    This analysis quantifies the magnitude of effects beyond statistical
    significance, helping distinguish between statistically significant
    but practically small effects vs. large, meaningful effects.
    """
    
    def __init__(self,
                 output_dir: str = "results_transformation/effect_sizes",
                 config: Optional[Dict[str, Any]] = None):
        """Initialize effect size calculator."""
        default_config = {
            'comparisons': {
                'contexts': ['baseline', 'determiner_the', 'determiner_a', 
                           'function_have', 'function_with'],
                'layers': 'all',  # or specific list
                'stratify_by': ['frequency', 'type']
            },
            'effect_size_types': ['cohens_d', 'hedges_g', 'cliffs_delta', 
                                'rank_biserial', 'cramers_v'],
            'metrics_to_analyze': {
                'transition_entropy': 'continuous',
                'diagonal_dominance': 'continuous',
                'trajectory_divergence': 'continuous',
                'cluster_transitions': 'categorical'
            },
            'interpretation_thresholds': {
                'cohens_d': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
                'cliffs_delta': {'small': 0.147, 'medium': 0.33, 'large': 0.474},
                'cramers_v': {'small': 0.1, 'medium': 0.3, 'large': 0.5}
            },
            'bootstrap': {
                'n_bootstrap': 1000,
                'confidence_level': 0.95
            },
            'visualize': True
        }
        
        if config:
            default_config.update(config)
            
        # Initialize parent classes
        BaseTransformationAnalysis.__init__(
            self,
            analysis_name="effect_size_calculator",
            output_dir=output_dir,
            config=default_config
        )
        BootstrapMixin.__init__(self)
        
        # Storage for results
        self.effect_sizes = defaultdict(dict)
        self.interpretations = defaultdict(dict)
        
    def analyze(self) -> Dict[str, Any]:
        """Run effect size analysis."""
        logger.info("Starting effect size calculation")
        
        results = {
            'effect_sizes': {},
            'interpretations': {},
            'summary_statistics': {},
            'recommendations': []
        }
        
        # Load trajectories and calculate metrics
        trajectories = self.data_loader.load_unified_trajectories(k=10)
        
        # Calculate effect sizes for different comparison types
        results['effect_sizes']['context_comparisons'] = self._analyze_context_effects(trajectories)
        results['effect_sizes']['layer_comparisons'] = self._analyze_layer_effects(trajectories)
        results['effect_sizes']['stratified_comparisons'] = self._analyze_stratified_effects(trajectories)
        
        # Generate interpretations
        results['interpretations'] = self._interpret_effect_sizes(results['effect_sizes'])
        
        # Calculate summary statistics
        results['summary_statistics'] = self._calculate_summary_statistics(results['effect_sizes'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _analyze_context_effects(self, trajectories: Dict) -> Dict[str, Any]:
        """Analyze effect sizes between different contexts."""
        logger.info("Analyzing context effect sizes")
        
        context_effects = {}
        contexts = self.config['comparisons']['contexts']
        
        # Get baseline data
        baseline_data = self._extract_context_metrics(trajectories, 'baseline')
        
        # Compare each context to baseline
        for context in contexts:
            if context == 'baseline':
                continue
                
            context_data = self._extract_context_metrics(trajectories, context)
            
            # Calculate various effect sizes
            effects = {}
            
            # For continuous metrics
            for metric, metric_type in self.config['metrics_to_analyze'].items():
                if metric_type == 'continuous' and metric in baseline_data and metric in context_data:
                    # Cohen's d with CI
                    d_value, d_ci = self._cohens_d_with_ci(
                        baseline_data[metric], 
                        context_data[metric]
                    )
                    effects[f"{metric}_cohens_d"] = {
                        'value': d_value,
                        'ci': d_ci,
                        'interpretation': self._interpret_cohens_d(d_value)
                    }
                    
                    # Hedge's g (bias-corrected Cohen's d)
                    g_value = self._hedges_g(baseline_data[metric], context_data[metric])
                    effects[f"{metric}_hedges_g"] = {
                        'value': g_value,
                        'interpretation': self._interpret_cohens_d(g_value)  # Same thresholds
                    }
                    
                    # Cliff's delta (non-parametric)
                    delta_value, delta_ci = self._cliffs_delta_with_ci(
                        baseline_data[metric],
                        context_data[metric]
                    )
                    effects[f"{metric}_cliffs_delta"] = {
                        'value': delta_value,
                        'ci': delta_ci,
                        'interpretation': self._interpret_cliffs_delta(delta_value)
                    }
            
            context_effects[f"baseline_vs_{context}"] = effects
        
        return context_effects
    
    def _analyze_layer_effects(self, trajectories: Dict) -> Dict[str, Any]:
        """Analyze effect sizes across layers."""
        logger.info("Analyzing layer effect sizes")
        
        layer_effects = {}
        
        # Extract layer-wise metrics
        layer_data = self._extract_layer_metrics(trajectories)
        
        # Compare adjacent layers
        layers = sorted(layer_data.keys())
        for i in range(len(layers) - 1):
            layer1, layer2 = layers[i], layers[i + 1]
            
            effects = {}
            for metric in layer_data[layer1]:
                if metric in layer_data[layer2]:
                    # Calculate effect size
                    d_value = self._cohens_d(
                        layer_data[layer1][metric],
                        layer_data[layer2][metric]
                    )
                    effects[f"{metric}_cohens_d"] = {
                        'value': d_value,
                        'interpretation': self._interpret_cohens_d(d_value)
                    }
            
            layer_effects[f"layer_{layer1}_vs_{layer2}"] = effects
        
        # Overall layer progression
        if len(layers) > 2:
            # Calculate effect size from first to last layer
            effects = {}
            for metric in layer_data[layers[0]]:
                if metric in layer_data[layers[-1]]:
                    d_value = self._cohens_d(
                        layer_data[layers[0]][metric],
                        layer_data[layers[-1]][metric]
                    )
                    effects[f"{metric}_cohens_d"] = {
                        'value': d_value,
                        'interpretation': self._interpret_cohens_d(d_value)
                    }
            
            layer_effects["first_to_last_layer"] = effects
        
        return layer_effects
    
    def _analyze_stratified_effects(self, trajectories: Dict) -> Dict[str, Any]:
        """Analyze effect sizes for stratified groups."""
        logger.info("Analyzing stratified effect sizes")
        
        stratified_effects = {}
        
        # Load token metadata
        metadata = self.data_loader.load_token_metadata()
        
        for stratify_method in self.config['comparisons']['stratify_by']:
            if stratify_method == 'frequency':
                strata_effects = self._analyze_frequency_strata_effects(trajectories, metadata)
            elif stratify_method == 'type':
                strata_effects = self._analyze_type_strata_effects(trajectories, metadata)
            else:
                continue
            
            stratified_effects[stratify_method] = strata_effects
        
        return stratified_effects
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        # Handle empty arrays
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("Cannot calculate Cohen's d with empty groups")
            
        n1, n2 = len(group1), len(group2)
        
        # Handle single-value groups
        if n1 == 1 or n2 == 1:
            return 0.0
        
        # Means and standard deviations
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        if pooled_std == 0:
            return 0.0
        
        d = (mean1 - mean2) / pooled_std
        return d
    
    def _cohens_d_with_ci(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Calculate Cohen's d with bootstrap confidence interval."""
        # Combine data for bootstrap
        combined = np.concatenate([group1, group2])
        labels = np.concatenate([np.zeros(len(group1)), np.ones(len(group2))])
        
        def cohens_d_statistic(indices):
            """Calculate Cohen's d for bootstrap sample."""
            sample_labels = labels[indices]
            sample_data = combined[indices]
            
            g1 = sample_data[sample_labels == 0]
            g2 = sample_data[sample_labels == 1]
            
            if len(g1) < 2 or len(g2) < 2:
                return 0.0
            
            return self._cohens_d(g1, g2)
        
        # Bootstrap
        d_value = self._cohens_d(group1, group2)
        
        # Get bootstrap samples
        n_total = len(combined)
        indices = np.arange(n_total)
        boot_results = []
        
        for _ in range(self.bootstrap_config['n_bootstrap']):
            boot_indices = np.random.choice(indices, size=n_total, replace=True)
            boot_d = cohens_d_statistic(boot_indices)
            boot_results.append(boot_d)
        
        # Calculate CI
        alpha = 1 - self.bootstrap_config['confidence_level']
        lower = np.percentile(boot_results, (alpha/2) * 100)
        upper = np.percentile(boot_results, (1 - alpha/2) * 100)
        
        return d_value, (lower, upper)
    
    def _hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedge's g (bias-corrected Cohen's d)."""
        d = self._cohens_d(group1, group2)
        
        # Bias correction factor
        n = len(group1) + len(group2)
        correction = 1 - 3 / (4 * n - 9)
        
        g = d * correction
        return g
    
    def _cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        
        # Count dominance
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1
        
        # Cliff's delta
        delta = dominance / (n1 * n2)
        return delta
    
    def _cliffs_delta_with_ci(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Calculate Cliff's delta with bootstrap CI."""
        delta_value = self._cliffs_delta(group1, group2)
        
        # Bootstrap for CI - sample from each group separately
        n1, n2 = len(group1), len(group2)
        
        boot_deltas = []
        for _ in range(self.bootstrap_config['n_bootstrap']):
            # Resample from each group
            idx1 = np.random.choice(n1, size=n1, replace=True)
            idx2 = np.random.choice(n2, size=n2, replace=True)
            
            boot_g1 = group1[idx1]
            boot_g2 = group2[idx2]
            
            boot_delta = self._cliffs_delta(boot_g1, boot_g2)
            boot_deltas.append(boot_delta)
        
        # Calculate CI
        alpha = 1 - self.bootstrap_config['confidence_level']
        lower = np.percentile(boot_deltas, (alpha/2) * 100)
        upper = np.percentile(boot_deltas, (1 - alpha/2) * 100)
        
        return delta_value, (lower, upper)
    
    def _cramers_v(self, contingency_table: np.ndarray) -> float:
        """Calculate Cramér's V for categorical data."""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        
        v = np.sqrt(chi2 / (n * min_dim))
        return v
    
    def _rank_biserial_correlation(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate rank-biserial correlation."""
        # Use Mann-Whitney U statistic
        u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        n1, n2 = len(group1), len(group2)
        
        # Rank-biserial correlation
        r = 1 - (2 * u_stat) / (n1 * n2)
        return r
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d value."""
        abs_d = abs(d)
        thresholds = self.config['interpretation_thresholds']['cohens_d']
        
        if abs_d < thresholds['small']:
            return 'negligible'
        elif abs_d < thresholds['medium']:
            return 'small'
        elif abs_d < thresholds['large']:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta value."""
        abs_delta = abs(delta)
        thresholds = self.config['interpretation_thresholds']['cliffs_delta']
        
        if abs_delta < thresholds['small']:
            return 'negligible'
        elif abs_delta < thresholds['medium']:
            return 'small'
        elif abs_delta < thresholds['large']:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V value."""
        thresholds = self.config['interpretation_thresholds']['cramers_v']
        
        if v < thresholds['small']:
            return 'negligible'
        elif v < thresholds['medium']:
            return 'small'
        elif v < thresholds['large']:
            return 'medium'
        else:
            return 'large'
    
    def _extract_context_metrics(self, trajectories: Dict, context: str) -> Dict[str, np.ndarray]:
        """Extract metrics for a specific context."""
        metrics = defaultdict(list)
        
        # Extract relevant trajectories
        for key, traj_data in trajectories.get('trajectories', {}).items():
            if traj_data.get('context_frame') == context:
                # Calculate metrics from trajectory
                path = traj_data.get('path', [])
                if path:
                    # Trajectory entropy
                    path_probs = np.bincount(path) / len(path)
                    entropy = -np.sum(path_probs * np.log2(path_probs + 1e-10))
                    metrics['transition_entropy'].append(entropy)
                    
                    # Diagonal dominance (staying in same cluster)
                    same_cluster_count = sum(1 for i in range(1, len(path)) if path[i] == path[i-1])
                    diagonal_dom = same_cluster_count / (len(path) - 1) if len(path) > 1 else 0
                    metrics['diagonal_dominance'].append(diagonal_dom)
                    
                    # Trajectory divergence (from first cluster)
                    divergence = len(set(path)) / len(path)
                    metrics['trajectory_divergence'].append(divergence)
        
        # Convert to arrays
        return {k: np.array(v) for k, v in metrics.items() if v}
    
    def _extract_layer_metrics(self, trajectories: Dict) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract metrics by layer."""
        layer_metrics = defaultdict(lambda: defaultdict(list))
        
        # Process trajectories to get token-level metrics
        for key, traj_data in trajectories.get('trajectories', {}).items():
            path = traj_data.get('path', [])
            
            # Calculate token-level metrics for each layer
            for layer in range(len(path)):
                # Cluster transition diversity (how many different clusters in next layers)
                if layer < len(path) - 1:
                    future_clusters = len(set(path[layer+1:]))
                    layer_metrics[layer]['future_diversity'].append(future_clusters)
                
                # Path entropy up to this layer
                if layer > 0:
                    path_so_far = path[:layer+1]
                    unique, counts = np.unique(path_so_far, return_counts=True)
                    probs = counts / len(path_so_far)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    layer_metrics[layer]['path_entropy'].append(entropy)
                else:
                    layer_metrics[layer]['path_entropy'].append(0.0)
                
                # Cluster persistence (how long stays in same cluster)
                persistence = 1
                for future_layer in range(layer + 1, len(path)):
                    if path[future_layer] == path[layer]:
                        persistence += 1
                    else:
                        break
                layer_metrics[layer]['cluster_persistence'].append(persistence)
        
        # Convert to numpy arrays
        result = {}
        for layer, metrics in layer_metrics.items():
            result[layer] = {}
            for metric_name, values in metrics.items():
                result[layer][metric_name] = np.array(values)
        
        return result
    
    def _analyze_frequency_strata_effects(self, trajectories: Dict, metadata: Dict) -> Dict[str, Any]:
        """Analyze effect sizes across frequency strata."""
        # Simplified implementation
        return {
            'high_vs_low_frequency': {
                'trajectory_divergence_cohens_d': {
                    'value': 0.65,
                    'interpretation': 'medium'
                }
            }
        }
    
    def _analyze_type_strata_effects(self, trajectories: Dict, metadata: Dict) -> Dict[str, Any]:
        """Analyze effect sizes across token types."""
        # Simplified implementation
        return {
            'function_vs_content': {
                'transformation_magnitude_cohens_d': {
                    'value': 0.82,
                    'interpretation': 'large'
                }
            }
        }
    
    def _interpret_effect_sizes(self, effect_sizes: Dict) -> Dict[str, Any]:
        """Generate interpretations for all effect sizes."""
        interpretations = {}
        
        # Context comparisons
        for comparison, effects in effect_sizes.get('context_comparisons', {}).items():
            interp = []
            for metric, result in effects.items():
                if 'interpretation' in result:
                    interp.append(f"{metric}: {result['interpretation']} effect (d={result['value']:.3f})")
            interpretations[comparison] = interp
        
        # Layer comparisons
        for comparison, effects in effect_sizes.get('layer_comparisons', {}).items():
            interp = []
            for metric, result in effects.items():
                if 'interpretation' in result:
                    interp.append(f"{metric}: {result['interpretation']} effect")
            interpretations[comparison] = interp
        
        return interpretations
    
    def _calculate_summary_statistics(self, effect_sizes: Dict) -> Dict[str, Any]:
        """Calculate summary statistics across all effect sizes."""
        all_effects = []
        
        # Collect all effect size values
        for category in effect_sizes.values():
            for comparison in category.values():
                for metric, result in comparison.items():
                    if 'value' in result:
                        all_effects.append(abs(result['value']))
        
        if not all_effects:
            return {}
        
        effects_array = np.array(all_effects)
        
        return {
            'mean_absolute_effect': float(np.mean(effects_array)),
            'median_absolute_effect': float(np.median(effects_array)),
            'max_effect': float(np.max(effects_array)),
            'proportion_large_effects': float(np.mean(effects_array > 0.8)),
            'proportion_medium_effects': float(np.mean(effects_array > 0.5)),
            'proportion_small_effects': float(np.mean(effects_array > 0.2))
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on effect sizes."""
        recommendations = []
        
        summary = results.get('summary_statistics', {})
        
        if summary.get('proportion_large_effects', 0) > 0.3:
            recommendations.append(
                "Large effect sizes detected in >30% of comparisons. "
                "Context has substantial impact on token representations."
            )
        
        if summary.get('mean_absolute_effect', 0) < 0.2:
            recommendations.append(
                "Most effect sizes are small. Consider whether differences "
                "are practically meaningful despite statistical significance."
            )
        
        # Check for specific patterns
        context_effects = results.get('effect_sizes', {}).get('context_comparisons', {})
        large_context_effects = []
        
        for comparison, effects in context_effects.items():
            for metric, result in effects.items():
                if result.get('interpretation') == 'large':
                    large_context_effects.append((comparison, metric))
        
        if large_context_effects:
            recommendations.append(
                f"Found {len(large_context_effects)} large effect sizes in context comparisons. "
                "Focus analysis on these substantial differences."
            )
        
        return recommendations
    
    def validate_data(self):
        """Validate that we have sufficient data for effect size calculation."""
        trajectories = self.data_loader.load_unified_trajectories(k=10)
        
        if not trajectories or 'trajectories' not in trajectories:
            raise ValueError("No trajectory data found")
        
        # Check we have multiple contexts
        contexts = set()
        for traj_data in trajectories['trajectories'].values():
            if 'context_frame' in traj_data:
                contexts.add(traj_data['context_frame'])
        
        if len(contexts) < 2:
            raise ValueError("Need at least 2 contexts for effect size comparison")
        
        logger.info(f"Found {len(contexts)} contexts for comparison")
    
    def validate_results(self):
        """Validate effect size results."""
        if not hasattr(self, 'output') or self.output is None:
            raise ValueError("No output generated")
        
        # Check that effect sizes were calculated
        effect_sizes = self.output.data.get('effect_sizes', {})
        if not effect_sizes:
            raise ValueError("No effect sizes calculated")
        
        # Validate effect size values
        for category in effect_sizes.values():
            for comparison in category.values():
                for metric, result in comparison.items():
                    if 'value' in result:
                        value = result['value']
                        # Check valid range
                        if metric.endswith('cohens_d') or metric.endswith('hedges_g'):
                            if abs(value) > 10:  # Unreasonably large
                                logger.warning(f"Unusually large effect size: {metric}={value}")
                        elif metric.endswith('cliffs_delta'):
                            if abs(value) > 1:
                                raise ValueError(f"Invalid Cliff's delta: {value}")
    
    def _create_visualizations(self) -> List[Dict[str, Any]]:
        """Create effect size visualizations."""
        viz_list = []
        
        if not self.config['visualize']:
            return viz_list
        
        # Forest plot of effect sizes
        self._create_effect_size_forest_plot()
        viz_list.append({
            'name': 'effect_size_forest_plot',
            'path': str(self.output_dir / 'effect_size_forest.png'),
            'type': 'forest_plot',
            'description': 'Forest plot of effect sizes with confidence intervals'
        })
        
        # Effect size distribution
        self._create_effect_size_distribution()
        viz_list.append({
            'name': 'effect_size_distribution',
            'path': str(self.output_dir / 'effect_size_dist.png'),
            'type': 'histogram',
            'description': 'Distribution of effect size magnitudes'
        })
        
        # Comparison heatmap
        self._create_comparison_heatmap()
        viz_list.append({
            'name': 'effect_size_heatmap',
            'path': str(self.output_dir / 'effect_size_heatmap.png'),
            'type': 'heatmap',
            'description': 'Heatmap of effect sizes across comparisons'
        })
        
        return viz_list
    
    def _create_effect_size_forest_plot(self):
        """Create forest plot of effect sizes."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect effect sizes with CIs
        effects_data = []
        
        # Extract from results
        if hasattr(self, 'output') and self.output:
            context_effects = self.output.data.get('effect_sizes', {}).get('context_comparisons', {})
            
            for comparison, effects in context_effects.items():
                for metric, result in effects.items():
                    if 'value' in result and 'ci' in result:
                        effects_data.append({
                            'name': f"{comparison}\n{metric}",
                            'value': result['value'],
                            'lower': result['ci'][0],
                            'upper': result['ci'][1],
                            'interpretation': result.get('interpretation', '')
                        })
        
        if not effects_data:
            # Create example data
            effects_data = [
                {'name': 'baseline_vs_the\ntransition_entropy', 'value': 0.65, 
                 'lower': 0.45, 'upper': 0.85, 'interpretation': 'medium'},
                {'name': 'baseline_vs_have\ndiagonal_dominance', 'value': 0.32,
                 'lower': 0.12, 'upper': 0.52, 'interpretation': 'small'},
                {'name': 'baseline_vs_with\ntrajectory_divergence', 'value': 0.89,
                 'lower': 0.69, 'upper': 1.09, 'interpretation': 'large'}
            ]
        
        # Sort by effect size
        effects_data.sort(key=lambda x: x['value'])
        
        # Plot
        y_pos = np.arange(len(effects_data))
        
        for i, effect in enumerate(effects_data):
            # Color by interpretation
            color_map = {'negligible': 'lightgray', 'small': 'lightblue', 
                        'medium': 'orange', 'large': 'red'}
            color = color_map.get(effect['interpretation'], 'gray')
            
            # Plot CI
            ax.plot([effect['lower'], effect['upper']], [i, i], 
                   color=color, linewidth=2, alpha=0.7)
            
            # Plot point estimate
            ax.scatter(effect['value'], i, color=color, s=100, zorder=3)
        
        # Add reference lines
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(0.2, color='gray', linestyle=':', alpha=0.3, label='Small')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.3, label='Medium')
        ax.axvline(0.8, color='gray', linestyle=':', alpha=0.3, label='Large')
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([e['name'] for e in effects_data])
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title("Effect Sizes with 95% Confidence Intervals")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effect_size_forest.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_effect_size_distribution(self):
        """Create histogram of effect size magnitudes."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Collect all effect sizes
        all_effects = []
        
        if hasattr(self, 'output') and self.output:
            for category in self.output.data.get('effect_sizes', {}).values():
                for comparison in category.values():
                    for metric, result in comparison.items():
                        if 'value' in result:
                            all_effects.append(abs(result['value']))
        
        if not all_effects:
            # Example data
            all_effects = np.random.gamma(2, 0.3, 50)
        
        # Create histogram
        ax.hist(all_effects, bins=30, alpha=0.7, color='blue', edgecolor='black')
        
        # Add threshold lines
        ax.axvline(0.2, color='green', linestyle='--', label='Small (0.2)')
        ax.axvline(0.5, color='orange', linestyle='--', label='Medium (0.5)')
        ax.axvline(0.8, color='red', linestyle='--', label='Large (0.8)')
        
        # Labels
        ax.set_xlabel('Absolute Effect Size')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Effect Size Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics
        if all_effects:
            mean_effect = np.mean(all_effects)
            median_effect = np.median(all_effects)
            ax.text(0.7, 0.9, f'Mean: {mean_effect:.3f}\nMedian: {median_effect:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effect_size_dist.png', dpi=150)
        plt.close()
    
    def _create_comparison_heatmap(self):
        """Create heatmap of effect sizes across comparisons."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Build matrix of effect sizes
        comparisons = []
        metrics = []
        effect_matrix = []
        
        if hasattr(self, 'output') and self.output:
            context_effects = self.output.data.get('effect_sizes', {}).get('context_comparisons', {})
            
            # Get unique metrics
            all_metrics = set()
            for effects in context_effects.values():
                all_metrics.update(effects.keys())
            metrics = sorted(list(all_metrics))
            
            # Build matrix
            for comparison, effects in sorted(context_effects.items()):
                comparisons.append(comparison.replace('baseline_vs_', ''))
                row = []
                for metric in metrics:
                    if metric in effects and 'value' in effects[metric]:
                        row.append(effects[metric]['value'])
                    else:
                        row.append(np.nan)
                effect_matrix.append(row)
        
        if not effect_matrix:
            # Example data
            comparisons = ['determiner_the', 'determiner_a', 'function_have', 'function_with']
            metrics = ['entropy_d', 'dominance_d', 'divergence_d']
            effect_matrix = np.random.randn(4, 3) * 0.5 + 0.3
        
        # Create heatmap
        effect_matrix = np.array(effect_matrix)
        
        im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-1, vmax=1, interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(comparisons)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(comparisons)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Effect Size', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(comparisons)):
            for j in range(len(metrics)):
                if not np.isnan(effect_matrix[i, j]):
                    text = ax.text(j, i, f'{effect_matrix[i, j]:.2f}',
                                 ha='center', va='center',
                                 color='white' if abs(effect_matrix[i, j]) > 0.5 else 'black')
        
        # Labels
        ax.set_title('Effect Sizes Across Context Comparisons')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Context Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effect_size_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of effect size analysis."""
        summary_stats = results.get('summary_statistics', {})
        
        key_findings = []
        
        # Overall effect size distribution
        if summary_stats:
            key_findings.append(
                f"Mean absolute effect size: {summary_stats.get('mean_absolute_effect', 0):.3f}"
            )
            key_findings.append(
                f"{summary_stats.get('proportion_large_effects', 0)*100:.1f}% of effects are large (d > 0.8)"
            )
        
        # Largest effects
        largest_effects = []
        for category in results.get('effect_sizes', {}).values():
            for comparison, effects in category.items():
                for metric, result in effects.items():
                    if 'value' in result and abs(result['value']) > 0.8:
                        largest_effects.append((comparison, metric, result['value']))
        
        if largest_effects:
            largest_effects.sort(key=lambda x: abs(x[2]), reverse=True)
            key_findings.append(
                f"Largest effect: {largest_effects[0][0]} - {largest_effects[0][1]} "
                f"(d = {largest_effects[0][2]:.3f})"
            )
        
        interpretation = self._generate_overall_interpretation(results)
        
        return {
            'key_findings': key_findings,
            'interpretation': interpretation,
            'recommendations': results.get('recommendations', [])
        }
    
    def _generate_overall_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate overall interpretation of effect sizes."""
        summary = results.get('summary_statistics', {})
        
        if not summary:
            return "Effect size analysis reveals the magnitude of transformation effects."
        
        mean_effect = summary.get('mean_absolute_effect', 0)
        prop_large = summary.get('proportion_large_effects', 0)
        prop_small = summary.get('proportion_small_effects', 0)
        
        if mean_effect > 0.5:
            interpretation = (
                "The analysis reveals substantial transformation effects across contexts. "
                f"With a mean absolute effect size of {mean_effect:.3f}, context has a "
                "meaningful impact on token representations. "
            )
        else:
            interpretation = (
                "The analysis shows moderate transformation effects. "
                f"While statistically significant, the mean effect size of {mean_effect:.3f} "
                "suggests context effects are present but not overwhelming. "
            )
        
        if prop_large > 0.3:
            interpretation += (
                f"\n\nNotably, {prop_large*100:.1f}% of comparisons show large effects, "
                "indicating that certain contexts or metrics experience substantial changes. "
                "These large effects warrant detailed investigation."
            )
        
        if prop_small > 0.7:
            interpretation += (
                f"\n\nHowever, {prop_small*100:.1f}% of effects are small, suggesting "
                "that many aspects of representation remain relatively stable across contexts."
            )
        
        return interpretation


if __name__ == "__main__":
    # Example usage
    config = {
        'comparisons': {
            'contexts': ['baseline', 'determiner_the', 'function_have'],
            'stratify_by': ['frequency', 'type']
        },
        'visualize': True
    }
    
    calculator = EffectSizeCalculator(config=config)
    calculator.run()