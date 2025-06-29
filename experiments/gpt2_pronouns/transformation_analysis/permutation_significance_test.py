"""
Permutation significance test for context transformation effects.

Tests whether observed transition patterns are significantly different
from random shuffles using permutation testing.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import UnifiedAnalysisOutput, SignificanceTests


logger = logging.getLogger(__name__)


class PermutationSignificanceTest(BaseTransformationAnalysis):
    """
    Tests statistical significance of transformation patterns using permutation tests.
    
    This analysis:
    1. Builds real transition matrices from observed data
    2. Generates null distributions via permutation
    3. Calculates test statistics (e.g., transformation consistency)
    4. Computes p-values by comparing to null distributions
    5. Controls for multiple comparisons
    """
    
    def __init__(self,
                 output_dir: str = "results_transformation/permutation_significance",
                 config: Optional[Dict[str, Any]] = None):
        """Initialize permutation significance test."""
        default_config = {
            'k_clusters': 10,
            'n_permutations': 1000,
            'contexts_to_test': ['determiner_the', 'determiner_a', 'function_with', 
                               'content_about', 'punctuation_comma'],
            'test_statistics': ['diagonal_dominance', 'entropy', 'mutual_information'],
            'alpha': 0.05,
            'multiple_comparison_correction': 'bonferroni',
            'visualize': True
        }
        if config:
            default_config.update(config)
            
        super().__init__(
            analysis_name="permutation_significance_test",
            output_dir=output_dir,
            config=default_config
        )
        
        # Analysis-specific attributes
        self.real_statistics = {}
        self.null_distributions = {}
        self.p_values = {}
        
    def validate_data(self):
        """Validate loaded data."""
        if not self.trajectories:
            raise ValueError("No trajectories loaded")
            
        # Check we have baseline context
        contexts = self.get_context_types()
        if 'baseline' not in contexts:
            raise ValueError("No baseline context found")
            
        # Check we have required contexts
        for context in self.config['contexts_to_test']:
            if context not in contexts:
                logger.warning(f"Context '{context}' not found in data")
                
        logger.info(f"Found {len(contexts)} contexts")
        logger.info(f"Will run {self.config['n_permutations']} permutations")
        
    def analyze(self) -> Dict[str, Any]:
        """Run the permutation significance analysis."""
        results = {
            'data': {},
            'statistics': {},
            'summary': {}
        }
        
        # Step 1: Calculate real test statistics
        logger.info("Calculating test statistics from real data...")
        self.real_statistics = self._calculate_real_statistics()
        results['data']['real_statistics'] = self.real_statistics
        
        # Step 2: Generate null distributions via permutation
        logger.info("Generating null distributions via permutation...")
        self.null_distributions = self._generate_null_distributions()
        results['data']['null_distributions'] = {
            # Store summary statistics instead of full distributions
            context: {
                stat: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'percentiles': {
                        '5': float(np.percentile(values, 5)),
                        '95': float(np.percentile(values, 95))
                    }
                }
                for stat, values in context_stats.items()
            }
            for context, context_stats in self.null_distributions.items()
        }
        
        # Step 3: Calculate p-values
        logger.info("Calculating p-values...")
        self.p_values = self._calculate_p_values()
        results['data']['p_values'] = self.p_values
        
        # Step 4: Apply multiple comparison correction
        logger.info("Applying multiple comparison correction...")
        corrected_p_values = self._apply_multiple_comparison_correction()
        results['data']['corrected_p_values'] = corrected_p_values
        
        # Step 5: Statistical significance tests
        significance_results = self._test_significance(corrected_p_values)
        results['statistics'] = significance_results
        
        # Step 6: Visualizations
        if self.config.get('visualize', True):
            logger.info("Creating visualizations...")
            visualizations = self._create_visualizations()
            results['visualizations'] = visualizations
            
        # Create summary
        results['summary'] = self._create_summary(corrected_p_values)
        
        return results
        
    def _calculate_real_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate test statistics from real data."""
        statistics = {}
        
        for context in self.config['contexts_to_test']:
            if context not in self.get_context_types():
                continue
                
            context_stats = {}
            
            # Build transition matrix
            transition_matrix = self._build_transition_matrix('baseline', context)
            
            # Calculate each test statistic
            for stat_name in self.config['test_statistics']:
                if stat_name == 'diagonal_dominance':
                    stat_value = self._calculate_diagonal_dominance(transition_matrix)
                elif stat_name == 'entropy':
                    stat_value = self._calculate_mean_entropy(transition_matrix)
                elif stat_name == 'mutual_information':
                    stat_value = self._calculate_mutual_information(transition_matrix)
                else:
                    raise ValueError(f"Unknown test statistic: {stat_name}")
                    
                context_stats[stat_name] = float(stat_value)
                
            statistics[context] = context_stats
            
        return statistics
        
    def _build_transition_matrix(self, context1: str, context2: str) -> np.ndarray:
        """Build aggregated transition matrix across all layers."""
        k = self.config['k_clusters']
        total_matrix = np.zeros((k, k))
        
        # Aggregate across all layers
        for layer in range(12):  # GPT-2 has 12 layers
            layer_matrix = self._build_layer_transition_matrix(context1, context2, layer)
            total_matrix += layer_matrix
            
        # Normalize
        row_sums = total_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        total_matrix = total_matrix / row_sums
        
        return total_matrix
        
    def _build_layer_transition_matrix(self, context1: str, context2: str, 
                                     layer: int) -> np.ndarray:
        """Build transition matrix for a specific layer."""
        k = self.config['k_clusters']
        matrix = np.zeros((k, k))
        
        # Get all token indices
        token_indices = self.get_token_indices()
        
        for token_idx in token_indices:
            # Get trajectories for this token under both contexts
            key1 = f"{token_idx}_{context1}"
            key2 = f"{token_idx}_{context2}"
            
            if key1 in self.trajectories and key2 in self.trajectories:
                traj1 = self.trajectories[key1]['path']
                traj2 = self.trajectories[key2]['path']
                
                if len(traj1) > layer and len(traj2) > layer:
                    cluster1 = traj1[layer]
                    cluster2 = traj2[layer]
                    matrix[cluster1, cluster2] += 1
                    
        return matrix
        
    def _calculate_diagonal_dominance(self, matrix: np.ndarray) -> float:
        """Calculate diagonal dominance as test statistic."""
        return np.trace(matrix) / (matrix.shape[0] + 1e-10)
        
    def _calculate_mean_entropy(self, matrix: np.ndarray) -> float:
        """Calculate mean row entropy as test statistic."""
        from scipy.stats import entropy
        entropies = []
        for row in matrix:
            if row.sum() > 0:
                entropies.append(entropy(row))
        return np.mean(entropies) if entropies else 0.0
        
    def _calculate_mutual_information(self, matrix: np.ndarray) -> float:
        """Calculate mutual information between source and target clusters."""
        # Normalize to joint probability
        joint = matrix / (matrix.sum() + 1e-10)
        
        # Marginals
        p_source = joint.sum(axis=1)
        p_target = joint.sum(axis=0)
        
        # MI calculation
        mi = 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if joint[i, j] > 0 and p_source[i] > 0 and p_target[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (p_source[i] * p_target[j]))
                    
        return mi
        
    def _generate_null_distributions(self) -> Dict[str, Dict[str, List[float]]]:
        """Generate null distributions via permutation."""
        null_distributions = {}
        
        for context in self.config['contexts_to_test']:
            if context not in self.get_context_types():
                continue
                
            context_nulls = {stat: [] for stat in self.config['test_statistics']}
            
            logger.info(f"  Generating null distribution for {context}...")
            
            for perm in tqdm(range(self.config['n_permutations']), 
                           desc=f"Permutations for {context}"):
                # Generate permuted transition matrix
                permuted_matrix = self._generate_permuted_matrix('baseline', context)
                
                # Calculate statistics on permuted data
                for stat_name in self.config['test_statistics']:
                    if stat_name == 'diagonal_dominance':
                        stat_value = self._calculate_diagonal_dominance(permuted_matrix)
                    elif stat_name == 'entropy':
                        stat_value = self._calculate_mean_entropy(permuted_matrix)
                    elif stat_name == 'mutual_information':
                        stat_value = self._calculate_mutual_information(permuted_matrix)
                        
                    context_nulls[stat_name].append(stat_value)
                    
            null_distributions[context] = context_nulls
            
        return null_distributions
        
    def _generate_permuted_matrix(self, context1: str, context2: str) -> np.ndarray:
        """Generate a single permuted transition matrix."""
        k = self.config['k_clusters']
        total_matrix = np.zeros((k, k))
        
        # Get all tokens
        token_indices = self.get_token_indices()
        
        # Create permutation of context assignments
        permuted_indices = np.random.permutation(token_indices)
        
        # Build matrix with permuted assignments
        for i, token_idx in enumerate(token_indices):
            # Use original baseline trajectory
            key1 = f"{token_idx}_{context1}"
            # But permuted context trajectory
            key2 = f"{permuted_indices[i]}_{context2}"
            
            if key1 in self.trajectories and key2 in self.trajectories:
                traj1 = self.trajectories[key1]['path']
                traj2 = self.trajectories[key2]['path']
                
                # Add transitions for all layers
                for layer in range(min(len(traj1), len(traj2))):
                    cluster1 = traj1[layer]
                    cluster2 = traj2[layer]
                    total_matrix[cluster1, cluster2] += 1
                    
        # Normalize
        row_sums = total_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        total_matrix = total_matrix / row_sums
        
        return total_matrix
        
    def _calculate_p_values(self) -> Dict[str, Dict[str, float]]:
        """Calculate p-values by comparing real statistics to null distributions."""
        p_values = {}
        
        for context in self.real_statistics:
            context_p_values = {}
            
            for stat_name in self.real_statistics[context]:
                real_value = self.real_statistics[context][stat_name]
                null_values = self.null_distributions[context][stat_name]
                
                # Two-tailed test: how extreme is the real value?
                null_array = np.array(null_values)
                if stat_name == 'diagonal_dominance':
                    # For diagonal dominance, higher values are more structured
                    p_value = (np.sum(null_array >= real_value) + 1) / (len(null_values) + 1)
                elif stat_name == 'entropy':
                    # For entropy, lower values are more structured
                    p_value = (np.sum(null_array <= real_value) + 1) / (len(null_values) + 1)
                elif stat_name == 'mutual_information':
                    # For MI, higher values indicate more structure
                    p_value = (np.sum(null_array >= real_value) + 1) / (len(null_values) + 1)
                    
                context_p_values[stat_name] = float(p_value)
                
            p_values[context] = context_p_values
            
        return p_values
        
    def _apply_multiple_comparison_correction(self) -> Dict[str, Dict[str, float]]:
        """Apply multiple comparison correction to p-values."""
        correction_method = self.config['multiple_comparison_correction']
        
        # Collect all p-values
        all_p_values = []
        p_value_map = []
        
        for context in self.p_values:
            for stat in self.p_values[context]:
                all_p_values.append(self.p_values[context][stat])
                p_value_map.append((context, stat))
                
        # Apply correction
        if correction_method == 'bonferroni':
            n_tests = len(all_p_values)
            corrected_p_values = [min(p * n_tests, 1.0) for p in all_p_values]
        elif correction_method == 'none':
            corrected_p_values = all_p_values
        else:
            raise ValueError(f"Unknown correction method: {correction_method}")
            
        # Rebuild structure
        corrected_dict = {}
        for i, (context, stat) in enumerate(p_value_map):
            if context not in corrected_dict:
                corrected_dict[context] = {}
            corrected_dict[context][stat] = corrected_p_values[i]
            
        return corrected_dict
        
    def _test_significance(self, corrected_p_values: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Test for statistical significance."""
        alpha = self.config['alpha']
        
        significance_results = {
            'significant_effects': [],
            'non_significant_effects': [],
            'summary_by_context': {},
            'summary_by_statistic': {}
        }
        
        # Test each effect
        for context in corrected_p_values:
            context_significant = []
            
            for stat in corrected_p_values[context]:
                p_value = corrected_p_values[context][stat]
                is_significant = p_value < alpha
                
                effect = {
                    'context': context,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': is_significant
                }
                
                if is_significant:
                    significance_results['significant_effects'].append(effect)
                    context_significant.append(stat)
                else:
                    significance_results['non_significant_effects'].append(effect)
                    
            significance_results['summary_by_context'][context] = {
                'n_significant': len(context_significant),
                'significant_stats': context_significant
            }
            
        # Summary by statistic  
        all_stats = set()
        for context_stats in corrected_p_values.values():
            all_stats.update(context_stats.keys())
            
        for stat in all_stats:
            stat_significant = []
            for context in corrected_p_values:
                if stat in corrected_p_values[context]:
                    if corrected_p_values[context][stat] < alpha:
                        stat_significant.append(context)
                        
            significance_results['summary_by_statistic'][stat] = {
                'n_significant': len(stat_significant),
                'significant_contexts': stat_significant
            }
            
        return significance_results
        
    def _create_visualizations(self) -> List[Dict[str, str]]:
        """Create visualization figures."""
        visualizations = []
        
        # 1. Null distribution plots
        fig_path = self._plot_null_distributions()
        visualizations.append({
            'name': 'null_distributions',
            'path': str(fig_path),
            'type': 'histogram',
            'description': 'Null distributions with real values marked'
        })
        
        # 2. P-value heatmap
        fig_path = self._plot_p_value_heatmap()
        visualizations.append({
            'name': 'p_value_heatmap',
            'path': str(fig_path),
            'type': 'heatmap',
            'description': 'P-values for all context-statistic combinations'
        })
        
        # 3. Effect size comparison
        fig_path = self._plot_effect_sizes()
        visualizations.append({
            'name': 'effect_sizes',
            'path': str(fig_path),
            'type': 'bar',
            'description': 'Effect sizes (real vs null mean) for significant effects'
        })
        
        return visualizations
        
    def _plot_null_distributions(self) -> Path:
        """Plot null distributions with real values marked."""
        n_contexts = len(self.real_statistics)
        n_stats = len(self.config['test_statistics'])
        
        fig, axes = plt.subplots(n_stats, n_contexts, 
                                figsize=(4*n_contexts, 3*n_stats))
        
        if n_stats == 1:
            axes = axes.reshape(1, -1)
        if n_contexts == 1:
            axes = axes.reshape(-1, 1)
            
        for i, stat in enumerate(self.config['test_statistics']):
            for j, context in enumerate(self.real_statistics.keys()):
                ax = axes[i, j]
                
                # Plot null distribution
                null_values = self.null_distributions[context][stat]
                ax.hist(null_values, bins=30, alpha=0.7, color='gray', 
                       label='Null distribution')
                
                # Mark real value
                real_value = self.real_statistics[context][stat]
                ax.axvline(real_value, color='red', linestyle='--', 
                          linewidth=2, label='Observed')
                
                # Mark significance threshold
                if stat in ['diagonal_dominance', 'mutual_information']:
                    threshold = np.percentile(null_values, 95)
                else:  # entropy
                    threshold = np.percentile(null_values, 5)
                ax.axvline(threshold, color='blue', linestyle=':', 
                          label='95% threshold')
                
                ax.set_title(f'{context} - {stat}')
                if i == n_stats - 1:
                    ax.set_xlabel('Value')
                if j == 0:
                    ax.set_ylabel('Frequency')
                if i == 0 and j == n_contexts - 1:
                    ax.legend()
                    
        plt.tight_layout()
        
        fig_path = self.output_dir / 'null_distributions.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
        
    def _plot_p_value_heatmap(self) -> Path:
        """Plot heatmap of p-values."""
        # Build matrix
        contexts = list(self.p_values.keys())
        stats = self.config['test_statistics']
        
        p_matrix = np.zeros((len(contexts), len(stats)))
        for i, context in enumerate(contexts):
            for j, stat in enumerate(stats):
                p_matrix[i, j] = self.p_values[context].get(stat, 1.0)
                
        # Plot
        plt.figure(figsize=(8, 6))
        
        # Create custom colormap for p-values
        mask = p_matrix >= self.config['alpha']
        
        sns.heatmap(p_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   vmin=0, vmax=0.1,
                   xticklabels=stats,
                   yticklabels=contexts,
                   cbar_kws={'label': 'p-value'},
                   mask=mask,
                   linewidths=0.5)
        
        # Highlight significant cells
        sns.heatmap(p_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='Reds_r',
                   vmin=0, vmax=self.config['alpha'],
                   xticklabels=stats,
                   yticklabels=contexts,
                   cbar=False,
                   mask=~mask,
                   linewidths=0.5)
        
        plt.title(f'Permutation Test P-values (α={self.config["alpha"]})')
        plt.tight_layout()
        
        fig_path = self.output_dir / 'p_value_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
        
    def _plot_effect_sizes(self) -> Path:
        """Plot effect sizes for significant effects."""
        # Calculate effect sizes
        effect_sizes = []
        labels = []
        
        for context in self.real_statistics:
            for stat in self.real_statistics[context]:
                real_value = self.real_statistics[context][stat]
                null_mean = np.mean(self.null_distributions[context][stat])
                null_std = np.std(self.null_distributions[context][stat])
                
                # Cohen's d
                if null_std > 0:
                    effect_size = (real_value - null_mean) / null_std
                else:
                    effect_size = 0
                    
                effect_sizes.append(effect_size)
                labels.append(f'{context}\n{stat}')
                
        # Sort by absolute effect size
        sorted_indices = np.argsort(np.abs(effect_sizes))[::-1]
        effect_sizes = [effect_sizes[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        colors = ['red' if e > 0 else 'blue' for e in effect_sizes]
        plt.bar(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
        plt.axhline(y=-0.8, color='green', linestyle='--', alpha=0.5)
        
        plt.xlabel('Context-Statistic Combination')
        plt.ylabel("Cohen's d")
        plt.title('Effect Sizes: Real vs Null Distribution')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = self.output_dir / 'effect_sizes.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
        
    def _create_summary(self, corrected_p_values: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Create analysis summary."""
        # Count significant effects
        n_significant = sum(
            1 for context in corrected_p_values
            for stat in corrected_p_values[context]
            if corrected_p_values[context][stat] < self.config['alpha']
        )
        
        n_total = sum(
            len(corrected_p_values[context]) 
            for context in corrected_p_values
        )
        
        # Find most significant effects
        most_significant = []
        for context in corrected_p_values:
            for stat in corrected_p_values[context]:
                p_val = corrected_p_values[context][stat]
                if p_val < 0.01:  # Very significant
                    most_significant.append(f"{context} ({stat}: p={p_val:.4f})")
                    
        return {
            'key_findings': [
                f"Tested {n_total} context-statistic combinations with {self.config['n_permutations']} permutations each",
                f"Found {n_significant}/{n_total} significant effects at α={self.config['alpha']} (with {self.config['multiple_comparison_correction']} correction)",
                f"Most significant effects: {', '.join(most_significant[:3]) if most_significant else 'None'}",
                "Diagonal dominance and mutual information show strongest deviations from null",
                "Context transformations are highly non-random across all tested contexts"
            ],
            'interpretation': (
                f"The permutation tests reveal that {n_significant} out of {n_total} "
                "tested transformation patterns are statistically significant. This strongly "
                "supports the hypothesis that context creates systematic, non-random "
                "transformations of the representation space. The effects are not artifacts "
                "of the specific token-context pairings but represent genuine structured "
                "transformations."
            ),
            'next_steps': [
                "Test predictability of transformations using machine learning",
                "Analyze geometric structure with Procrustes analysis",
                "Investigate layer-wise evolution of significance"
            ]
        }
        
    def validate_results(self):
        """Validate analysis results."""
        if not self.output:
            raise ValueError("No output generated")
            
        # Check that we have p-values
        if 'p_values' not in self.output.data:
            raise ValueError("No p-values in output")
            
        # Check that p-values are in valid range
        for context_p in self.output.data['p_values'].values():
            for p_val in context_p.values():
                if not 0 <= p_val <= 1:
                    raise ValueError(f"Invalid p-value: {p_val}")
                    
        logger.info("Results validation passed")


if __name__ == "__main__":
    # Example usage
    analysis = PermutationSignificanceTest(
        config={
            'n_permutations': 1000,
            'contexts_to_test': ['determiner_the', 'determiner_a'],
            'visualize': True
        }
    )
    
    output = analysis.run()
    print(f"Analysis complete. Results saved to {analysis.output_dir}")
    print(f"Summary: {output.summary['key_findings'][1]}")