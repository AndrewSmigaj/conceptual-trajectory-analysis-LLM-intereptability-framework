"""
Comprehensive validation suite for transformation analysis.

This analysis validates key methodological choices:
1. Choice of k=10 clusters
2. Normalization methods
3. Clustering algorithms
4. Stability across different parameters
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import UnifiedAnalysisOutput

logger = logging.getLogger(__name__)


class ComprehensiveValidationSuite(BaseTransformationAnalysis):
    """
    Validates methodological choices for transformation analysis.
    
    This analysis:
    1. Tests different k values using elbow method and silhouette analysis
    2. Compares clustering algorithms (KMeans, DBSCAN, Hierarchical)
    3. Evaluates normalization strategies
    4. Assesses stability of findings across parameter variations
    """
    
    def __init__(self, 
                 output_dir: str = "results_transformation/comprehensive_validation",
                 config: Optional[Dict[str, Any]] = None):
        """Initialize validation suite."""
        default_config = {
            'k_range': [5, 7, 10, 15, 20, 25],
            'algorithms': ['kmeans', 'hierarchical', 'dbscan'],
            'normalizations': ['none', 'standard', 'minmax', 'robust'],
            'n_samples': 5000,  # Subsample for efficiency
            'random_state': 42,
            'visualize': True,
            'layers_to_test': [0, 5, 11]  # Early, middle, late
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            analysis_name="comprehensive_validation_suite",
            output_dir=output_dir,
            config=default_config
        )
        
        self.validation_results = {
            'k_selection': {},
            'algorithm_comparison': {},
            'normalization_comparison': {},
            'stability_analysis': {}
        }
        
    def analyze(self) -> Dict[str, Any]:
        """Run comprehensive validation analyses."""
        logger.info("Starting comprehensive validation suite")
        
        # Load activation data
        activations = self.data_loader.load_activations()
        if activations is None:
            raise ValueError("No activation data found")
            
        # Run validation analyses
        self._validate_k_selection(activations)
        self._compare_algorithms(activations)
        self._compare_normalizations(activations)
        self._analyze_stability(activations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            'validation_results': self.validation_results,
            'recommendations': recommendations
        }
        
    def _validate_k_selection(self, activations: Dict) -> None:
        """Validate choice of k using multiple methods."""
        logger.info("Validating k selection...")
        
        for layer in self.config['layers_to_test']:
            logger.info(f"  Testing layer {layer}")
            
            # Get activations for this layer
            layer_acts = self._extract_layer_activations(activations, layer)
            
            if layer_acts is None or len(layer_acts) == 0:
                logger.warning(f"No activations for layer {layer}")
                continue
                
            # Subsample for efficiency
            if len(layer_acts) > self.config['n_samples']:
                indices = np.random.choice(len(layer_acts), self.config['n_samples'], replace=False)
                layer_acts = layer_acts[indices]
                
            # Normalize
            scaler = StandardScaler()
            layer_acts_norm = scaler.fit_transform(layer_acts)
            
            # Test different k values
            k_results = {
                'inertias': [],
                'silhouette_scores': [],
                'calinski_harabasz_scores': [],
                'davies_bouldin_scores': []
            }
            
            for k in self.config['k_range']:
                logger.info(f"    Testing k={k}")
                
                # Fit KMeans
                kmeans = KMeans(n_clusters=k, random_state=self.config['random_state'], n_init=10)
                labels = kmeans.fit_predict(layer_acts_norm)
                
                # Calculate metrics
                k_results['inertias'].append(kmeans.inertia_)
                
                if k > 1:  # Silhouette requires at least 2 clusters
                    sil_score = silhouette_score(layer_acts_norm, labels, sample_size=min(1000, len(labels)))
                    k_results['silhouette_scores'].append(sil_score)
                    
                    ch_score = calinski_harabasz_score(layer_acts_norm, labels)
                    k_results['calinski_harabasz_scores'].append(ch_score)
                    
                    db_score = davies_bouldin_score(layer_acts_norm, labels)
                    k_results['davies_bouldin_scores'].append(db_score)
                    
            self.validation_results['k_selection'][f'layer_{layer}'] = k_results
            
        # Find optimal k using elbow method
        self._find_optimal_k()
        
    def _find_optimal_k(self) -> None:
        """Find optimal k using elbow method and other criteria."""
        logger.info("Finding optimal k...")
        
        for layer_key, results in self.validation_results['k_selection'].items():
            if not results['inertias']:
                continue
                
            # Elbow method - find point of maximum curvature
            inertias = np.array(results['inertias'])
            k_values = self.config['k_range'][:len(inertias)]
            
            # Calculate second derivative
            if len(inertias) > 2:
                second_diff = np.diff(np.diff(inertias))
                # Find elbow point (maximum second derivative)
                elbow_idx = np.argmax(second_diff) + 2  # +2 because of double diff
                optimal_k_elbow = k_values[min(elbow_idx, len(k_values)-1)]
            else:
                optimal_k_elbow = k_values[0]
                
            # Best silhouette score
            if results['silhouette_scores']:
                sil_scores = results['silhouette_scores']
                optimal_k_silhouette = k_values[1:][np.argmax(sil_scores)]  # Skip k=1
            else:
                optimal_k_silhouette = None
                
            results['optimal_k'] = {
                'elbow_method': optimal_k_elbow,
                'silhouette_method': optimal_k_silhouette,
                'recommended': 10  # Our current choice
            }
            
    def _compare_algorithms(self, activations: Dict) -> None:
        """Compare different clustering algorithms."""
        logger.info("Comparing clustering algorithms...")
        
        # Use middle layer for comparison
        layer = self.config['layers_to_test'][1]  # Middle layer
        layer_acts = self._extract_layer_activations(activations, layer)
        
        if layer_acts is None or len(layer_acts) == 0:
            logger.warning("No activations for algorithm comparison")
            return
            
        # Subsample
        if len(layer_acts) > self.config['n_samples']:
            indices = np.random.choice(len(layer_acts), self.config['n_samples'], replace=False)
            layer_acts = layer_acts[indices]
            
        # Normalize
        scaler = StandardScaler()
        layer_acts_norm = scaler.fit_transform(layer_acts)
        
        algorithm_results = {}
        
        # KMeans
        if 'kmeans' in self.config['algorithms']:
            logger.info("  Testing KMeans...")
            kmeans = KMeans(n_clusters=10, random_state=self.config['random_state'], n_init=10)
            kmeans_labels = kmeans.fit_predict(layer_acts_norm)
            
            algorithm_results['kmeans'] = {
                'silhouette_score': silhouette_score(layer_acts_norm, kmeans_labels, sample_size=min(1000, len(kmeans_labels))),
                'calinski_harabasz_score': calinski_harabasz_score(layer_acts_norm, kmeans_labels),
                'davies_bouldin_score': davies_bouldin_score(layer_acts_norm, kmeans_labels),
                'n_clusters': len(np.unique(kmeans_labels))
            }
            
        # Hierarchical
        if 'hierarchical' in self.config['algorithms']:
            logger.info("  Testing Hierarchical...")
            hierarchical = AgglomerativeClustering(n_clusters=10)
            hier_labels = hierarchical.fit_predict(layer_acts_norm)
            
            algorithm_results['hierarchical'] = {
                'silhouette_score': silhouette_score(layer_acts_norm, hier_labels, sample_size=min(1000, len(hier_labels))),
                'calinski_harabasz_score': calinski_harabasz_score(layer_acts_norm, hier_labels),
                'davies_bouldin_score': davies_bouldin_score(layer_acts_norm, hier_labels),
                'n_clusters': len(np.unique(hier_labels))
            }
            
        # DBSCAN
        if 'dbscan' in self.config['algorithms']:
            logger.info("  Testing DBSCAN...")
            # Find appropriate eps value
            eps = self._find_dbscan_eps(layer_acts_norm)
            dbscan = DBSCAN(eps=eps, min_samples=5)
            dbscan_labels = dbscan.fit_predict(layer_acts_norm)
            
            # Filter out noise points for metrics
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:
                algorithm_results['dbscan'] = {
                    'silhouette_score': silhouette_score(layer_acts_norm[non_noise_mask], 
                                                        dbscan_labels[non_noise_mask], 
                                                        sample_size=min(1000, np.sum(non_noise_mask))),
                    'n_clusters': len(np.unique(dbscan_labels[non_noise_mask])),
                    'noise_ratio': np.sum(dbscan_labels == -1) / len(dbscan_labels),
                    'eps': eps
                }
            else:
                algorithm_results['dbscan'] = {
                    'error': 'Too many noise points',
                    'noise_ratio': np.sum(dbscan_labels == -1) / len(dbscan_labels)
                }
                
        self.validation_results['algorithm_comparison'] = algorithm_results
        
    def _find_dbscan_eps(self, data: np.ndarray, k: int = 5) -> float:
        """Find appropriate eps value for DBSCAN using k-nearest neighbors."""
        # Calculate distances to k-th nearest neighbor
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(data)
        distances, _ = neighbors.kneighbors(data)
        
        # Sort distances
        distances = np.sort(distances[:, k-1])
        
        # Find elbow point (simplified - use 90th percentile)
        eps = np.percentile(distances, 90)
        
        return eps
        
    def _compare_normalizations(self, activations: Dict) -> None:
        """Compare different normalization strategies."""
        logger.info("Comparing normalization methods...")
        
        # Use middle layer
        layer = self.config['layers_to_test'][1]
        layer_acts = self._extract_layer_activations(activations, layer)
        
        if layer_acts is None or len(layer_acts) == 0:
            logger.warning("No activations for normalization comparison")
            return
            
        # Subsample
        if len(layer_acts) > self.config['n_samples']:
            indices = np.random.choice(len(layer_acts), self.config['n_samples'], replace=False)
            layer_acts = layer_acts[indices]
            
        norm_results = {}
        
        for norm_type in self.config['normalizations']:
            logger.info(f"  Testing {norm_type} normalization...")
            
            # Apply normalization
            if norm_type == 'none':
                data_norm = layer_acts
            elif norm_type == 'standard':
                scaler = StandardScaler()
                data_norm = scaler.fit_transform(layer_acts)
            elif norm_type == 'minmax':
                scaler = MinMaxScaler()
                data_norm = scaler.fit_transform(layer_acts)
            elif norm_type == 'robust':
                scaler = RobustScaler()
                data_norm = scaler.fit_transform(layer_acts)
            else:
                continue
                
            # Cluster with normalized data
            kmeans = KMeans(n_clusters=10, random_state=self.config['random_state'], n_init=10)
            labels = kmeans.fit_predict(data_norm)
            
            # Calculate metrics
            norm_results[norm_type] = {
                'silhouette_score': silhouette_score(data_norm, labels, sample_size=min(1000, len(labels))),
                'calinski_harabasz_score': calinski_harabasz_score(data_norm, labels),
                'davies_bouldin_score': davies_bouldin_score(data_norm, labels),
                'inertia': kmeans.inertia_,
                'data_stats': {
                    'mean': float(np.mean(data_norm)),
                    'std': float(np.std(data_norm)),
                    'min': float(np.min(data_norm)),
                    'max': float(np.max(data_norm))
                }
            }
            
        self.validation_results['normalization_comparison'] = norm_results
        
    def _analyze_stability(self, activations: Dict) -> None:
        """Analyze stability of clustering across random seeds."""
        logger.info("Analyzing clustering stability...")
        
        # Use middle layer
        layer = self.config['layers_to_test'][1]
        layer_acts = self._extract_layer_activations(activations, layer)
        
        if layer_acts is None or len(layer_acts) == 0:
            logger.warning("No activations for stability analysis")
            return
            
        # Subsample
        if len(layer_acts) > self.config['n_samples']:
            indices = np.random.choice(len(layer_acts), self.config['n_samples'], replace=False)
            layer_acts = layer_acts[indices]
            
        # Normalize
        scaler = StandardScaler()
        layer_acts_norm = scaler.fit_transform(layer_acts)
        
        # Run clustering with different seeds
        n_runs = 10
        all_labels = []
        all_scores = []
        
        for seed in range(n_runs):
            kmeans = KMeans(n_clusters=10, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(layer_acts_norm)
            all_labels.append(labels)
            
            sil_score = silhouette_score(layer_acts_norm, labels, sample_size=min(1000, len(labels)))
            all_scores.append(sil_score)
            
        # Calculate stability metrics
        # Use adjusted rand index between different runs
        from sklearn.metrics import adjusted_rand_score
        
        ari_scores = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)
                
        self.validation_results['stability_analysis'] = {
            'silhouette_scores': {
                'mean': float(np.mean(all_scores)),
                'std': float(np.std(all_scores)),
                'min': float(np.min(all_scores)),
                'max': float(np.max(all_scores))
            },
            'adjusted_rand_scores': {
                'mean': float(np.mean(ari_scores)),
                'std': float(np.std(ari_scores)),
                'min': float(np.min(ari_scores)),
                'max': float(np.max(ari_scores))
            },
            'n_runs': n_runs
        }
        
    def _extract_layer_activations(self, activations: Dict, layer: int) -> Optional[np.ndarray]:
        """Extract activations for a specific layer."""
        if 'unified_activations' in activations:
            # Unified format
            layer_key = f'layer_{layer}'
            if layer_key in activations['unified_activations']:
                acts = activations['unified_activations'][layer_key]
                if isinstance(acts, list):
                    return np.array(acts)
                return acts
        elif 'activations' in activations:
            # Standard format
            all_acts = []
            for token_data in activations['activations'].values():
                if 'layers' in token_data and layer < len(token_data['layers']):
                    all_acts.append(token_data['layers'][layer])
            if all_acts:
                return np.vstack(all_acts)
                
        return None
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # K selection
        if 'k_selection' in self.validation_results:
            for layer_key, results in self.validation_results['k_selection'].items():
                if 'optimal_k' in results:
                    opt_k = results['optimal_k']
                    if opt_k.get('elbow_method') == 10 or opt_k.get('silhouette_method') == 10:
                        recommendations.append(f"k=10 is validated as optimal for {layer_key}")
                    else:
                        recommendations.append(f"Consider k={opt_k.get('elbow_method')} based on elbow method for {layer_key}")
                        
        # Algorithm comparison
        if 'algorithm_comparison' in self.validation_results:
            alg_results = self.validation_results['algorithm_comparison']
            # Find best algorithm by silhouette score
            best_alg = None
            best_score = -1
            for alg, metrics in alg_results.items():
                if 'silhouette_score' in metrics and metrics['silhouette_score'] > best_score:
                    best_score = metrics['silhouette_score']
                    best_alg = alg
                    
            if best_alg:
                recommendations.append(f"{best_alg.capitalize()} clustering shows best performance (silhouette={best_score:.3f})")
                
        # Normalization
        if 'normalization_comparison' in self.validation_results:
            norm_results = self.validation_results['normalization_comparison']
            best_norm = None
            best_score = -1
            for norm, metrics in norm_results.items():
                if metrics['silhouette_score'] > best_score:
                    best_score = metrics['silhouette_score']
                    best_norm = norm
                    
            if best_norm:
                recommendations.append(f"{best_norm.capitalize()} normalization recommended (silhouette={best_score:.3f})")
                
        # Stability
        if 'stability_analysis' in self.validation_results:
            stability = self.validation_results['stability_analysis']
            ari_mean = stability['adjusted_rand_scores']['mean']
            if ari_mean > 0.8:
                recommendations.append(f"Clustering is highly stable across random seeds (ARI={ari_mean:.3f})")
            elif ari_mean > 0.6:
                recommendations.append(f"Clustering shows moderate stability (ARI={ari_mean:.3f})")
            else:
                recommendations.append(f"WARNING: Low clustering stability (ARI={ari_mean:.3f})")
                
        return recommendations
        
    def validate_data(self) -> None:
        """Validate input data."""
        trajectories = self.data_loader.load_trajectories()
        
        if not trajectories or 'trajectories' not in trajectories:
            raise ValueError("No trajectory data found")
            
        # Check if we have activation data
        activations = self.data_loader.load_activations()
        if activations is None:
            raise ValueError("No activation data found for validation")
            
        logger.info(f"Found trajectory and activation data")
        
    def validate_results(self) -> None:
        """Validate analysis results."""
        if not self.output:
            raise ValueError("No output generated")
            
        # Check that we have validation results
        if not hasattr(self.output, 'data') or 'validation_results' not in self.output.data:
            raise ValueError("No validation results in output")
            
        val_results = self.output.data['validation_results']
        
        # Check key analyses were performed
        required_analyses = ['k_selection', 'algorithm_comparison', 
                           'normalization_comparison', 'stability_analysis']
        for analysis in required_analyses:
            if analysis not in val_results or not val_results[analysis]:
                raise ValueError(f"Missing or empty {analysis} results")
                
        logger.info("Validation results verified")
        
    def _create_visualizations(self) -> List[Dict[str, Any]]:
        """Create validation visualizations."""
        visualizations = []
        
        if not self.config.get('visualize', True):
            return visualizations
            
        # K selection plots
        if 'k_selection' in self.validation_results:
            for layer_key, results in self.validation_results['k_selection'].items():
                if 'inertias' in results and results['inertias']:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    k_values = self.config['k_range'][:len(results['inertias'])]
                    
                    # Elbow plot
                    axes[0, 0].plot(k_values, results['inertias'], 'bo-')
                    axes[0, 0].set_xlabel('Number of clusters (k)')
                    axes[0, 0].set_ylabel('Inertia')
                    axes[0, 0].set_title(f'Elbow Method - {layer_key}')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Silhouette scores
                    if results.get('silhouette_scores'):
                        k_values_sil = k_values[1:]  # Skip k=1
                        axes[0, 1].plot(k_values_sil, results['silhouette_scores'], 'ro-')
                        axes[0, 1].set_xlabel('Number of clusters (k)')
                        axes[0, 1].set_ylabel('Silhouette Score')
                        axes[0, 1].set_title(f'Silhouette Analysis - {layer_key}')
                        axes[0, 1].grid(True, alpha=0.3)
                        
                    # Calinski-Harabasz scores
                    if results.get('calinski_harabasz_scores'):
                        axes[1, 0].plot(k_values_sil, results['calinski_harabasz_scores'], 'go-')
                        axes[1, 0].set_xlabel('Number of clusters (k)')
                        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
                        axes[1, 0].set_title(f'Calinski-Harabasz Index - {layer_key}')
                        axes[1, 0].grid(True, alpha=0.3)
                        
                    # Davies-Bouldin scores (lower is better)
                    if results.get('davies_bouldin_scores'):
                        axes[1, 1].plot(k_values_sil, results['davies_bouldin_scores'], 'mo-')
                        axes[1, 1].set_xlabel('Number of clusters (k)')
                        axes[1, 1].set_ylabel('Davies-Bouldin Score')
                        axes[1, 1].set_title(f'Davies-Bouldin Index - {layer_key}')
                        axes[1, 1].grid(True, alpha=0.3)
                        
                    plt.tight_layout()
                    
                    # Save figure
                    fig_path = self.output_dir / f'k_selection_{layer_key}.png'
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    visualizations.append({
                        'name': f'k_selection_{layer_key}',
                        'type': 'plot',
                        'path': str(fig_path)
                    })
                    
        # Algorithm comparison plot
        if 'algorithm_comparison' in self.validation_results:
            alg_results = self.validation_results['algorithm_comparison']
            
            algorithms = list(alg_results.keys())
            metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, metric in enumerate(metrics):
                values = []
                valid_algs = []
                
                for alg in algorithms:
                    if metric in alg_results[alg]:
                        values.append(alg_results[alg][metric])
                        valid_algs.append(alg)
                        
                if values:
                    bars = axes[idx].bar(valid_algs, values)
                    axes[idx].set_title(metric.replace('_', ' ').title())
                    axes[idx].set_ylabel('Score')
                    axes[idx].tick_params(axis='x', rotation=45)
                    
                    # Color bars
                    if 'davies_bouldin' in metric:
                        # Lower is better for Davies-Bouldin
                        best_idx = np.argmin(values)
                    else:
                        # Higher is better for others
                        best_idx = np.argmax(values)
                        
                    for i, bar in enumerate(bars):
                        if i == best_idx:
                            bar.set_color('green')
                        else:
                            bar.set_color('steelblue')
                            
            plt.tight_layout()
            
            fig_path = self.output_dir / 'algorithm_comparison.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                'name': 'algorithm_comparison',
                'type': 'plot',
                'path': str(fig_path)
            })
            
        return visualizations


if __name__ == "__main__":
    # Run validation suite
    validator = ComprehensiveValidationSuite()
    output = validator.run()
    
    print("\nValidation Suite Results:")
    print("=" * 50)
    
    if hasattr(output, 'data') and 'recommendations' in output.data:
        print("\nRecommendations:")
        for rec in output.data['recommendations']:
            print(f"  - {rec}")
            
    print("\nValidation complete!")