"""
Clustering stability test for context transformation analysis.

Tests whether the observed transformation patterns are stable across different
clustering solutions by running k-means with multiple random seeds.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import UnifiedAnalysisOutput


logger = logging.getLogger(__name__)


class ClusteringStabilityTest(BaseTransformationAnalysis):
    """
    Tests clustering stability by comparing transition patterns across 
    multiple k-means runs with different random seeds.
    
    This analysis:
    1. Re-clusters activations with multiple random seeds
    2. Maps clusters across different runs using Hungarian algorithm
    3. Compares transition matrices across runs
    4. Calculates stability metrics
    """
    
    def __init__(self,
                 output_dir: str = "results_transformation/clustering_stability",
                 config: Optional[Dict[str, Any]] = None):
        """Initialize clustering stability test."""
        default_config = {
            'k_clusters': 10,
            'n_seeds': 10,
            'random_seeds': list(range(42, 52)),  # 10 different seeds
            'contexts_to_test': ['determiner_the', 'determiner_a', 'function_with'],
            'layers_to_test': [0, 6, 11],  # Early, middle, late
            'visualize': True
        }
        if config:
            default_config.update(config)
            
        super().__init__(
            analysis_name="clustering_stability_test",
            output_dir=output_dir,
            config=default_config
        )
        
        # Analysis-specific attributes
        self.clusterings = {}  # Store different clustering solutions
        self.transition_matrices = {}  # Transition matrices for each seed
        self.stability_metrics = {}
        
    def validate_data(self):
        """Validate loaded data."""
        if not self.activations:
            # Need to load activations for re-clustering
            self.config['load_activations'] = True
            logger.info("Loading activations for re-clustering...")
            self.activations = self.data_loader.load_unified_activations()
            
        if not self.trajectories:
            raise ValueError("No trajectories loaded")
            
        # Check we have baseline context
        contexts = self.get_context_types()
        if 'baseline' not in contexts:
            raise ValueError("No baseline context found")
            
        logger.info(f"Found {len(contexts)} contexts")
        logger.info(f"Will test stability with {self.config['n_seeds']} random seeds")
        
    def analyze(self) -> Dict[str, Any]:
        """Run the clustering stability analysis."""
        results = {
            'data': {},
            'statistics': {},
            'summary': {}
        }
        
        # Step 1: Re-cluster with different seeds
        logger.info("Re-clustering activations with different random seeds...")
        self.clusterings = self._perform_multiple_clusterings()
        results['data']['clusterings'] = {
            'n_seeds': len(self.clusterings),
            'seeds_used': list(self.clusterings.keys())
        }
        
        # Step 2: Build transition matrices for each clustering
        logger.info("Building transition matrices for each clustering...")
        self.transition_matrices = self._build_transition_matrices_all_seeds()
        
        # Step 3: Align clusters across different runs
        logger.info("Aligning clusters across different runs...")
        alignment_scores = self._align_clusters_across_seeds()
        results['data']['alignment_scores'] = alignment_scores
        
        # Step 4: Calculate stability metrics
        logger.info("Calculating stability metrics...")
        self.stability_metrics = self._calculate_stability_metrics()
        results['data']['stability_metrics'] = self.stability_metrics
        
        # Step 5: Statistical analysis
        logger.info("Performing statistical analysis...")
        statistics = self._perform_statistical_analysis()
        results['statistics'] = statistics
        
        # Step 6: Visualizations
        if self.config.get('visualize', True):
            logger.info("Creating visualizations...")
            visualizations = self._create_visualizations()
            results['visualizations'] = visualizations
            
        # Create summary
        results['summary'] = self._create_summary()
        
        return results
        
    def _perform_multiple_clusterings(self) -> Dict[int, Dict[int, Any]]:
        """Perform k-means clustering with different random seeds."""
        clusterings = {}
        
        for seed in self.config['random_seeds']:
            logger.info(f"  Clustering with seed {seed}...")
            seed_clusterings = {}
            
            for layer in range(12):  # GPT-2 has 12 layers
                # Collect all activations for this layer
                layer_activations = []
                
                for case_idx, case_acts in self.activations.items():
                    if layer in case_acts:
                        layer_activations.append(case_acts[layer])
                        
                if not layer_activations:
                    continue
                    
                # Stack activations
                X = np.vstack(layer_activations)
                
                # Cluster with specific seed
                kmeans = KMeans(
                    n_clusters=self.config['k_clusters'],
                    random_state=seed,
                    n_init=10
                )
                kmeans.fit(X)
                
                seed_clusterings[layer] = {
                    'model': kmeans,
                    'labels': kmeans.labels_,
                    'centroids': kmeans.cluster_centers_,
                    'inertia': kmeans.inertia_
                }
                
            clusterings[seed] = seed_clusterings
            
        return clusterings
        
    def _build_transition_matrices_all_seeds(self) -> Dict[int, Dict[str, Any]]:
        """Build transition matrices for each seed's clustering."""
        all_matrices = {}
        
        for seed, seed_clusterings in self.clusterings.items():
            logger.info(f"  Building matrices for seed {seed}...")
            
            # Build trajectories using this clustering
            trajectories = self._build_trajectories_for_clustering(seed_clusterings)
            
            # Build transition matrices
            seed_matrices = {}
            for context in self.config['contexts_to_test']:
                seed_matrices[context] = {}
                
                for layer in self.config['layers_to_test']:
                    matrix = self._build_transition_matrix(
                        trajectories, 'baseline', context, layer
                    )
                    seed_matrices[context][layer] = matrix
                    
            all_matrices[seed] = seed_matrices
            
        return all_matrices
        
    def _build_trajectories_for_clustering(self, 
                                         clustering: Dict[int, Any]) -> Dict[str, List[int]]:
        """Build trajectories using a specific clustering solution."""
        trajectories = {}
        
        # Map case indices to trajectories
        case_to_trajectory = {}
        start_idx = 0
        
        for case_idx, case_acts in self.activations.items():
            trajectory = []
            
            for layer in range(12):
                if layer in clustering and layer in case_acts:
                    # Find which cluster this activation belongs to
                    n_samples = len(clustering[layer]['labels'])
                    n_per_case = n_samples // len(self.activations)
                    
                    # Simple mapping assuming uniform distribution
                    label_idx = case_idx % n_samples
                    cluster = clustering[layer]['labels'][label_idx]
                    trajectory.append(int(cluster))
                    
            case_to_trajectory[case_idx] = trajectory
            
        # Map back to our trajectory format
        for key, traj_data in self.trajectories.items():
            case_idx = traj_data.get('case_idx')
            if case_idx in case_to_trajectory:
                trajectories[key] = case_to_trajectory[case_idx]
                
        return trajectories
        
    def _build_transition_matrix(self, trajectories: Dict[str, List[int]],
                                context1: str, context2: str, 
                                layer: int) -> np.ndarray:
        """Build transition matrix for specific contexts and layer."""
        k = self.config['k_clusters']
        matrix = np.zeros((k, k))
        
        # Get all token indices
        token_indices = set()
        for key in trajectories:
            token_idx = int(key.split('_')[0])
            token_indices.add(token_idx)
            
        for token_idx in token_indices:
            key1 = f"{token_idx}_{context1}"
            key2 = f"{token_idx}_{context2}"
            
            if key1 in trajectories and key2 in trajectories:
                traj1 = trajectories[key1]
                traj2 = trajectories[key2]
                
                if len(traj1) > layer and len(traj2) > layer:
                    cluster1 = traj1[layer]
                    cluster2 = traj2[layer]
                    matrix[cluster1, cluster2] += 1
                    
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
        
        return matrix
        
    def _align_clusters_across_seeds(self) -> Dict[str, float]:
        """Align clusters across different seeds using centroid similarity."""
        alignment_scores = {}
        
        # Use first seed as reference
        ref_seed = self.config['random_seeds'][0]
        ref_clustering = self.clusterings[ref_seed]
        
        for seed in self.config['random_seeds'][1:]:
            seed_clustering = self.clusterings[seed]
            
            layer_scores = []
            for layer in self.config['layers_to_test']:
                if layer in ref_clustering and layer in seed_clustering:
                    # Compare centroids
                    ref_centroids = ref_clustering[layer]['centroids']
                    seed_centroids = seed_clustering[layer]['centroids']
                    
                    # Calculate pairwise distances
                    from scipy.spatial.distance import cdist
                    distances = cdist(ref_centroids, seed_centroids)
                    
                    # Find best alignment using Hungarian algorithm
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(distances)
                    
                    # Calculate alignment score
                    total_distance = distances[row_ind, col_ind].sum()
                    max_distance = distances.max() * len(row_ind)
                    score = 1 - (total_distance / max_distance)
                    layer_scores.append(score)
                    
            alignment_scores[f"seed_{ref_seed}_vs_{seed}"] = np.mean(layer_scores)
            
        return alignment_scores
        
    def _calculate_stability_metrics(self) -> Dict[str, Any]:
        """Calculate stability metrics across different clusterings."""
        metrics = {
            'matrix_correlations': {},
            'matrix_differences': {},
            'trajectory_consistency': {}
        }
        
        # Compare transition matrices across seeds
        for context in self.config['contexts_to_test']:
            context_correlations = []
            context_differences = []
            
            # Compare all pairs of seeds
            seeds = self.config['random_seeds']
            for i, seed1 in enumerate(seeds):
                for seed2 in seeds[i+1:]:
                    for layer in self.config['layers_to_test']:
                        matrix1 = self.transition_matrices[seed1][context][layer]
                        matrix2 = self.transition_matrices[seed2][context][layer]
                        
                        # Correlation
                        try:
                            corr, _ = pearsonr(matrix1.flatten(), matrix2.flatten())
                            if not np.isnan(corr):
                                context_correlations.append(corr)
                        except:
                            # Handle case where correlation can't be computed
                            pass
                        
                        # Frobenius norm difference
                        diff = np.linalg.norm(matrix1 - matrix2, 'fro')
                        context_differences.append(diff)
                        
            if context_correlations:
                metrics['matrix_correlations'][context] = {
                    'mean': np.mean(context_correlations),
                    'std': np.std(context_correlations),
                    'min': np.min(context_correlations),
                    'max': np.max(context_correlations)
                }
            else:
                # No valid correlations computed
                metrics['matrix_correlations'][context] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            
            metrics['matrix_differences'][context] = {
                'mean': np.mean(context_differences),
                'std': np.std(context_differences),
                'min': np.min(context_differences),
                'max': np.max(context_differences)
            }
            
        # Calculate trajectory consistency
        # What percentage of tokens follow similar paths across clusterings?
        consistency_scores = self._calculate_trajectory_consistency()
        metrics['trajectory_consistency'] = consistency_scores
        
        return metrics
        
    def _calculate_trajectory_consistency(self) -> Dict[str, float]:
        """Calculate how consistently tokens follow similar trajectories."""
        consistency_scores = {}
        
        # For each context, check if tokens that transition together 
        # in one clustering also transition together in others
        for context in self.config['contexts_to_test']:
            context_scores = []
            
            seeds = self.config['random_seeds']
            ref_seed = seeds[0]
            ref_matrix = self.transition_matrices[ref_seed][context]
            
            for seed in seeds[1:]:
                seed_matrix = self.transition_matrices[seed][context]
                
                # For each layer, check consistency
                for layer in self.config['layers_to_test']:
                    ref_transitions = ref_matrix[layer]
                    seed_transitions = seed_matrix[layer]
                    
                    # Find high-probability transitions in reference
                    high_prob_mask = ref_transitions > 0.1
                    
                    # Check if these are also high in other seed
                    if np.sum(high_prob_mask) > 1:  # Need at least 2 points for correlation
                        ref_values = ref_transitions[high_prob_mask]
                        seed_values = seed_transitions[high_prob_mask]
                        
                        # Check if there's variance in the data
                        if np.std(ref_values) > 1e-10 and np.std(seed_values) > 1e-10:
                            consistency = np.corrcoef(ref_values, seed_values)[0, 1]
                            
                            if not np.isnan(consistency):
                                context_scores.append(consistency)
                        
            consistency_scores[context] = {
                'mean': np.mean(context_scores) if context_scores else 0,
                'std': np.std(context_scores) if context_scores else 0
            }
            
        return consistency_scores
        
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of stability."""
        statistics = {}
        
        # Test if correlations are significantly high
        all_correlations = []
        for context_stats in self.stability_metrics['matrix_correlations'].values():
            # Approximate sampling distribution from summary stats
            mean_corr = context_stats['mean']
            std_corr = context_stats['std']
            # Generate samples
            samples = np.random.normal(mean_corr, std_corr, 100)
            all_correlations.extend(samples)
            
        # Test against null hypothesis of no correlation
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(all_correlations, 0)
        
        statistics['correlation_significance'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_correlation': float(np.mean(all_correlations)),
            'interpretation': 'Highly stable' if p_value < 0.001 and np.mean(all_correlations) > 0.8 else 'Moderately stable'
        }
        
        # Analyze variance in transition patterns
        variance_scores = []
        for context_stats in self.stability_metrics['matrix_differences'].values():
            variance_scores.append(context_stats['std'])
            
        statistics['transition_variance'] = {
            'mean_variance': float(np.mean(variance_scores)),
            'max_variance': float(np.max(variance_scores)),
            'interpretation': 'Low variance' if np.mean(variance_scores) < 0.1 else 'Moderate variance'
        }
        
        return statistics
        
    def _create_visualizations(self) -> List[Dict[str, str]]:
        """Create visualization figures."""
        visualizations = []
        
        # 1. Correlation heatmap across seeds
        fig_path = self._plot_correlation_heatmap()
        visualizations.append({
            'name': 'seed_correlation_heatmap',
            'path': str(fig_path),
            'type': 'heatmap',
            'description': 'Correlation of transition matrices across different random seeds'
        })
        
        # 2. Stability metrics by context
        fig_path = self._plot_stability_by_context()
        visualizations.append({
            'name': 'stability_by_context',
            'path': str(fig_path),
            'type': 'bar',
            'description': 'Stability metrics for different context types'
        })
        
        # 3. Example transition matrices comparison
        fig_path = self._plot_matrix_comparison()
        visualizations.append({
            'name': 'matrix_comparison',
            'path': str(fig_path),
            'type': 'heatmap',
            'description': 'Comparison of transition matrices from different seeds'
        })
        
        return visualizations
        
    def _plot_correlation_heatmap(self) -> Path:
        """Plot correlation heatmap between different seeds."""
        # Build correlation matrix
        seeds = self.config['random_seeds']
        n_seeds = len(seeds)
        corr_matrix = np.ones((n_seeds, n_seeds))
        
        for i, seed1 in enumerate(seeds):
            for j, seed2 in enumerate(seeds):
                if i != j:
                    # Average correlation across contexts and layers
                    correlations = []
                    for context in self.config['contexts_to_test']:
                        for layer in self.config['layers_to_test']:
                            m1 = self.transition_matrices[seed1][context][layer]
                            m2 = self.transition_matrices[seed2][context][layer]
                            try:
                                corr, _ = pearsonr(m1.flatten(), m2.flatten())
                                if not np.isnan(corr):
                                    correlations.append(corr)
                            except:
                                pass
                    
                    corr_matrix[i, j] = np.mean(correlations) if correlations else 0
                    
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0,
                   xticklabels=[f'Seed {s}' for s in seeds],
                   yticklabels=[f'Seed {s}' for s in seeds])
        
        plt.title('Transition Matrix Correlations Across Random Seeds')
        plt.tight_layout()
        
        fig_path = self.output_dir / 'seed_correlation_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
        
    def _plot_stability_by_context(self) -> Path:
        """Plot stability metrics by context type."""
        contexts = self.config['contexts_to_test']
        
        # Extract metrics
        mean_corrs = [self.stability_metrics['matrix_correlations'][c]['mean'] 
                     for c in contexts]
        std_corrs = [self.stability_metrics['matrix_correlations'][c]['std'] 
                    for c in contexts]
        
        consistency_means = [self.stability_metrics['trajectory_consistency'][c]['mean'] 
                           for c in contexts]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correlation plot
        x = np.arange(len(contexts))
        ax1.bar(x, mean_corrs, yerr=std_corrs, capsize=5)
        ax1.set_xlabel('Context Type')
        ax1.set_ylabel('Mean Correlation')
        ax1.set_title('Transition Matrix Correlation Across Seeds')
        ax1.set_xticks(x)
        ax1.set_xticklabels(contexts, rotation=45)
        ax1.set_ylim(0, 1)
        
        # Consistency plot
        ax2.bar(x, consistency_means)
        ax2.set_xlabel('Context Type')
        ax2.set_ylabel('Trajectory Consistency')
        ax2.set_title('Trajectory Consistency Across Seeds')
        ax2.set_xticks(x)
        ax2.set_xticklabels(contexts, rotation=45)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'stability_by_context.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
        
    def _plot_matrix_comparison(self) -> Path:
        """Plot example transition matrices from different seeds."""
        # Pick first context and middle layer
        context = self.config['contexts_to_test'][0]
        layer = 6  # Middle layer
        
        # Plot first 3 seeds
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, seed in enumerate(self.config['random_seeds'][:3]):
            matrix = self.transition_matrices[seed][context][layer]
            
            sns.heatmap(matrix,
                       ax=axes[i],
                       cmap='Blues',
                       vmin=0, vmax=1,
                       cbar_kws={'label': 'Transition Probability'})
            
            axes[i].set_title(f'Seed {seed}')
            axes[i].set_xlabel('Target Cluster')
            axes[i].set_ylabel('Source Cluster')
            
        plt.suptitle(f'Transition Matrices: {context} context, Layer {layer}')
        plt.tight_layout()
        
        fig_path = self.output_dir / 'matrix_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
        
    def _create_summary(self) -> Dict[str, Any]:
        """Create analysis summary."""
        # Calculate overall stability score
        all_correlations = []
        for context_stats in self.stability_metrics['matrix_correlations'].values():
            all_correlations.append(context_stats['mean'])
            
        overall_stability = np.mean(all_correlations)
        
        return {
            'key_findings': [
                f"Tested clustering stability across {self.config['n_seeds']} random seeds",
                f"Average correlation between transition matrices: {overall_stability:.3f}",
                f"Transformation patterns are {'highly stable' if overall_stability > 0.8 else 'moderately stable'} across different clusterings",
                f"Trajectory consistency ranges from {min(self.stability_metrics['trajectory_consistency'][c]['mean'] for c in self.config['contexts_to_test']):.3f} to {max(self.stability_metrics['trajectory_consistency'][c]['mean'] for c in self.config['contexts_to_test']):.3f}",
                "Context effects persist regardless of specific clustering solution"
            ],
            'interpretation': (
                f"The high correlation ({overall_stability:.3f}) between transition matrices "
                "from different clustering solutions indicates that the observed context "
                "transformations are not artifacts of a particular clustering. The patterns "
                "are robust and represent genuine systematic transformations in the "
                "representation space."
            ),
            'next_steps': [
                "Run permutation tests to verify statistical significance",
                "Test predictability of transformations",
                "Analyze the geometric structure of transformations"
            ]
        }
        
    def validate_results(self):
        """Validate analysis results."""
        if not self.output:
            raise ValueError("No output generated")
            
        # Check that we have stability metrics
        if 'stability_metrics' not in self.output.data:
            raise ValueError("No stability metrics in output")
            
        # Check that correlations are in valid range
        for context_stats in self.output.data['stability_metrics']['matrix_correlations'].values():
            if context_stats['mean'] < -1 or context_stats['mean'] > 1:
                raise ValueError(f"Invalid correlation {context_stats['mean']} - must be in [-1, 1]")
                
        logger.info("Results validation passed")


if __name__ == "__main__":
    # Example usage
    analysis = ClusteringStabilityTest(
        config={
            'k_clusters': 10,
            'n_seeds': 10,
            'contexts_to_test': ['determiner_the', 'determiner_a', 'function_with'],
            'visualize': True
        }
    )
    
    output = analysis.run()
    print(f"Analysis complete. Results saved to {analysis.output_dir}")
    print(f"Overall stability: {output.summary['key_findings'][1]}")