"""
Subspace Alignment Analysis

Uses PCA to understand the principal components of context transformations
and measures subspace angles between different contexts.

This analysis:
1. Applies PCA to transformation vectors to find principal directions
2. Calculates canonical angles between subspaces of different contexts
3. Analyzes explained variance to identify key transformation dimensions
4. Tracks evolution of principal components across layers
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from scipy.linalg import svd, qr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import (
    UnifiedAnalysisOutput, AnalysisMetadata, Visualization
)


class SubspaceAlignmentAnalysis(BaseTransformationAnalysis):
    """
    Analyzes context transformations using PCA and subspace alignment techniques.
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.pca_cache = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Run subspace alignment analysis"""
        self.logger.info("Starting subspace alignment analysis")
        
        results = {
            'pca_results': {},
            'subspace_angles': {},
            'explained_variance': {},
            'principal_directions': {},
            'layer_evolution': {},
            'dimensionality_analysis': {}
        }
        
        # Get context types to analyze
        context_types = self.config.get('context_types', ['determiner_the', 'function_have'])
        
        # Prepare transformation vectors for each context
        transformation_data = self._prepare_transformation_vectors(context_types)
        
        if not transformation_data:
            self.logger.warning("No transformation data available")
            return results
        
        # Perform PCA analysis for each context and layer
        for context_type in context_types:
            if context_type not in transformation_data:
                continue
                
            self.logger.info(f"Analyzing context: {context_type}")
            
            context_pca = {}
            context_variance = {}
            context_directions = {}
            
            for layer in self.config['layers']:
                layer_key = f'layer_{layer}'
                if layer_key not in transformation_data[context_type]:
                    continue
                
                # Get transformation vectors for this layer
                X = transformation_data[context_type][layer_key]
                
                if X.shape[0] < 2:  # Need at least 2 samples for PCA
                    continue
                
                # Perform PCA
                pca_result = self._perform_pca_analysis(X, layer, context_type)
                context_pca[layer_key] = pca_result
                
                # Analyze explained variance
                variance_analysis = self._analyze_explained_variance(pca_result)
                context_variance[layer_key] = variance_analysis
                
                # Store principal directions
                context_directions[layer_key] = {
                    'components': pca_result['components'],
                    'explained_variance_ratio': pca_result['explained_variance_ratio']
                }
            
            results['pca_results'][context_type] = context_pca
            results['explained_variance'][context_type] = context_variance
            results['principal_directions'][context_type] = context_directions
            
            # Analyze layer evolution for this context
            evolution = self._analyze_layer_evolution(context_pca)
            results['layer_evolution'][context_type] = evolution
        
        # Calculate subspace angles between contexts
        if len(context_types) > 1:
            angles = self._calculate_subspace_angles(results['principal_directions'])
            results['subspace_angles'] = angles
        
        # Overall dimensionality analysis
        results['dimensionality_analysis'] = self._analyze_overall_dimensionality(results)
        
        return results
    
    def _prepare_transformation_vectors(self, context_types: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare transformation vectors from baseline to context activations"""
        transformation_data = {}
        
        # Load activations
        all_activations = self.data_loader.load_unified_activations()
        if not all_activations:
            self.logger.warning("No activation data available")
            return transformation_data
        
        # Get trajectories to match tokens
        trajectories = self.data_loader.load_unified_trajectories(k=self.config['k'])
        if not trajectories or 'trajectories' not in trajectories:
            self.logger.warning("No trajectory data available")
            return transformation_data
        
        # Build transformation vectors for each context
        for context_type in context_types:
            context_transforms = {}
            
            for layer in self.config['layers']:
                layer_transforms = []
                
                # Find tokens that exist in both baseline and this context
                for token in self.data_loader.get_all_tokens()[:self.config.get('max_tokens', 1000)]:
                    baseline_case = None
                    context_case = None
                    
                    # Find case indices from trajectories
                    for key, traj_data in trajectories['trajectories'].items():
                        if (traj_data.get('token_str') == token and 
                            traj_data.get('context_frame') == 'baseline'):
                            baseline_case = traj_data.get('case_idx')
                        elif (traj_data.get('token_str') == token and 
                              traj_data.get('context_frame') == context_type):
                            context_case = traj_data.get('case_idx')
                    
                    if (baseline_case is not None and context_case is not None and
                        baseline_case in all_activations and context_case in all_activations and
                        layer in all_activations[baseline_case] and layer in all_activations[context_case]):
                        
                        # Calculate transformation vector
                        baseline_activation = all_activations[baseline_case][layer]
                        context_activation = all_activations[context_case][layer]
                        
                        # Transformation vector is the difference
                        transform_vector = context_activation - baseline_activation
                        layer_transforms.append(transform_vector)
                
                if layer_transforms:
                    context_transforms[f'layer_{layer}'] = np.array(layer_transforms)
                    self.logger.info(f"Context {context_type}, Layer {layer}: {len(layer_transforms)} transformation vectors")
            
            if context_transforms:
                transformation_data[context_type] = context_transforms
        
        return transformation_data
    
    def _perform_pca_analysis(self, X: np.ndarray, layer: int, context: str) -> Dict[str, Any]:
        """Perform PCA analysis on transformation vectors"""
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # Determine number of components (min of samples-1 and features)
        n_components = min(X.shape[0] - 1, X.shape[1], 50)  # Cap at 50 for efficiency
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_centered)
        
        # Calculate additional metrics
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find effective dimensionality (95% variance)
        effective_dim = np.argmax(cumulative_variance >= 0.95) + 1
        
        return {
            'pca': pca,
            'components': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'transformed_data': X_transformed,
            'effective_dimensionality': effective_dim,
            'total_variance': np.sum(pca.explained_variance_),
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
    
    def _analyze_explained_variance(self, pca_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze explained variance from PCA results"""
        variance_ratios = pca_result['explained_variance_ratio']
        cumulative = pca_result['cumulative_variance']
        
        analysis = {
            'first_component_variance': float(variance_ratios[0]),
            'top_3_components_variance': float(np.sum(variance_ratios[:3])),
            'top_5_components_variance': float(np.sum(variance_ratios[:5])),
            'effective_dimensionality': int(pca_result['effective_dimensionality']),
            'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(pca_result),
            'variance_concentration': self._calculate_variance_concentration(variance_ratios)
        }
        
        return analysis
    
    def _estimate_intrinsic_dimensionality(self, pca_result: Dict[str, Any]) -> int:
        """Estimate intrinsic dimensionality using broken stick model"""
        eigenvalues = pca_result['explained_variance']
        n = len(eigenvalues)
        
        # Broken stick model: expected value for k-th largest eigenvalue
        broken_stick = []
        for k in range(n):
            expected = np.sum([1/j for j in range(k+1, n+1)])
            broken_stick.append(expected / n)
        
        # Find where actual eigenvalues drop below broken stick expectation
        normalized_eigenvals = eigenvalues / np.sum(eigenvalues)
        intrinsic_dim = np.sum(normalized_eigenvals > np.array(broken_stick))
        
        return int(intrinsic_dim)
    
    def _calculate_variance_concentration(self, variance_ratios: np.ndarray) -> float:
        """Calculate how concentrated variance is in top components"""
        # Use Shannon entropy normalized
        # Higher concentration = lower entropy
        probs = variance_ratios / np.sum(variance_ratios)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        
        # Concentration score (0 = uniform, 1 = all in first component)
        concentration = 1.0 - (entropy / max_entropy)
        return float(concentration)
    
    def _calculate_subspace_angles(self, principal_directions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate canonical angles between subspaces of different contexts"""
        angles_results = {}
        
        context_types = list(principal_directions.keys())
        
        for i, ctx1 in enumerate(context_types):
            for j, ctx2 in enumerate(context_types[i+1:], i+1):
                comparison_key = f'{ctx1}_vs_{ctx2}'
                layer_angles = {}
                
                for layer in self.config['layers']:
                    layer_key = f'layer_{layer}'
                    
                    if (layer_key in principal_directions[ctx1] and 
                        layer_key in principal_directions[ctx2]):
                        
                        # Get principal components (top k)
                        k_components = min(5, 
                                         principal_directions[ctx1][layer_key]['components'].shape[0],
                                         principal_directions[ctx2][layer_key]['components'].shape[0])
                        
                        U1 = principal_directions[ctx1][layer_key]['components'][:k_components].T
                        U2 = principal_directions[ctx2][layer_key]['components'][:k_components].T
                        
                        # Calculate canonical angles
                        angles = self._compute_canonical_angles(U1, U2)
                        
                        layer_angles[layer_key] = {
                            'canonical_angles': angles.tolist(),
                            'mean_angle': float(np.mean(angles)),
                            'max_angle': float(np.max(angles)),
                            'min_angle': float(np.min(angles)),
                            'subspace_similarity': float(np.cos(np.mean(angles)))
                        }
                
                angles_results[comparison_key] = layer_angles
        
        return angles_results
    
    def _compute_canonical_angles(self, U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
        """Compute canonical angles between two subspaces"""
        # Ensure matrices are orthonormal
        U1, _ = qr(U1, mode='economic')
        U2, _ = qr(U2, mode='economic')
        
        # SVD of U1.T @ U2
        _, s, _ = svd(U1.T @ U2)
        
        # Canonical angles are arccos of singular values (clamped to [0,1])
        s = np.clip(s, 0, 1)
        angles = np.arccos(s)
        
        return angles
    
    def _analyze_layer_evolution(self, layer_pca_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how principal components evolve across layers"""
        evolution = {
            'variance_trends': {},
            'dimensionality_trends': {},
            'stability_analysis': {}
        }
        
        layers = sorted([int(k.split('_')[1]) for k in layer_pca_results.keys()])
        
        # Track variance concentration across layers
        concentration_trend = []
        effective_dim_trend = []
        first_pc_variance_trend = []
        
        for layer in layers:
            layer_key = f'layer_{layer}'
            if layer_key in layer_pca_results:
                result = layer_pca_results[layer_key]
                
                # Calculate concentration for this layer
                variance_ratios = result['explained_variance_ratio']
                concentration = self._calculate_variance_concentration(variance_ratios)
                concentration_trend.append(concentration)
                
                effective_dim_trend.append(result['effective_dimensionality'])
                first_pc_variance_trend.append(variance_ratios[0])
        
        evolution['variance_trends'] = {
            'concentration_by_layer': concentration_trend,
            'effective_dimensionality_by_layer': effective_dim_trend,
            'first_pc_variance_by_layer': first_pc_variance_trend,
            'layers': layers
        }
        
        # Calculate trends
        if len(concentration_trend) > 1:
            concentration_slope = np.polyfit(layers[:len(concentration_trend)], concentration_trend, 1)[0]
            dim_slope = np.polyfit(layers[:len(effective_dim_trend)], effective_dim_trend, 1)[0]
            
            evolution['dimensionality_trends'] = {
                'concentration_slope': float(concentration_slope),
                'dimensionality_slope': float(dim_slope),
                'interpretation': self._interpret_trends(concentration_slope, dim_slope)
            }
        
        return evolution
    
    def _interpret_trends(self, concentration_slope: float, dim_slope: float) -> str:
        """Interpret the trends in variance concentration and dimensionality"""
        if concentration_slope > 0.01:
            concentration_desc = "increasingly concentrated"
        elif concentration_slope < -0.01:
            concentration_desc = "increasingly distributed"
        else:
            concentration_desc = "stable"
        
        if dim_slope > 0.1:
            dim_desc = "increasing dimensionality"
        elif dim_slope < -0.1:
            dim_desc = "decreasing dimensionality"
        else:
            dim_desc = "stable dimensionality"
        
        return f"Variance is {concentration_desc} with {dim_desc} across layers"
    
    def _analyze_overall_dimensionality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall dimensionality patterns across contexts and layers"""
        analysis = {
            'context_comparison': {},
            'layer_patterns': {},
            'key_insights': []
        }
        
        # Compare effective dimensionality across contexts
        context_dims = {}
        for context, variance_data in results['explained_variance'].items():
            dims_by_layer = []
            for layer_key, variance_info in variance_data.items():
                dims_by_layer.append(variance_info['effective_dimensionality'])
            
            if dims_by_layer:
                context_dims[context] = {
                    'mean_dimensionality': float(np.mean(dims_by_layer)),
                    'std_dimensionality': float(np.std(dims_by_layer)),
                    'dims_by_layer': dims_by_layer
                }
        
        analysis['context_comparison'] = context_dims
        
        # Identify patterns
        if context_dims:
            mean_dims = [info['mean_dimensionality'] for info in context_dims.values()]
            
            if np.std(mean_dims) < 2:
                analysis['key_insights'].append("Similar dimensionality across contexts")
            else:
                highest_dim_context = max(context_dims.keys(), 
                                        key=lambda x: context_dims[x]['mean_dimensionality'])
                analysis['key_insights'].append(f"{highest_dim_context} shows highest dimensionality transformations")
        
        return analysis
    
    def validate_data(self) -> None:
        """Validate loaded data"""
        if not hasattr(self, 'data_loader') or self.data_loader is None:
            raise ValueError("Data loader not initialized")
        
        # Check we have activation data
        activations = self.data_loader.load_unified_activations()
        if not activations:
            raise ValueError("No activation data found")
        
        # Check we have trajectory data
        trajectories = self.data_loader.load_unified_trajectories(k=self.config['k'])
        if not trajectories:
            raise ValueError("No trajectory data found")
        
        self.logger.info("Data validation passed")
    
    def validate_results(self) -> None:
        """Validate analysis results"""
        if not hasattr(self, 'output') or self.output is None:
            raise ValueError("No output generated")
        
        # Check required fields
        if 'pca_results' not in self.output.data:
            raise ValueError("Missing PCA results")
        
        if 'explained_variance' not in self.output.data:
            raise ValueError("Missing explained variance analysis")
        
        self.logger.info("Results validation passed")
    
    def _create_visualizations(self) -> List[Dict[str, Any]]:
        """Create visualizations for subspace analysis"""
        viz_list = []
        
        # Explained variance plot
        viz_list.append({
            'name': 'explained_variance',
            'path': str(self.output_dir / 'explained_variance.png'),
            'type': 'line_plot',
            'description': 'Explained variance ratios by principal component'
        })
        
        # Subspace angles heatmap
        viz_list.append({
            'name': 'subspace_angles',
            'path': str(self.output_dir / 'subspace_angles.png'),
            'type': 'heatmap',
            'description': 'Canonical angles between context subspaces'
        })
        
        # Dimensionality evolution
        viz_list.append({
            'name': 'dimensionality_evolution',
            'path': str(self.output_dir / 'dimensionality_evolution.png'),
            'type': 'line_plot',
            'description': 'Effective dimensionality evolution across layers'
        })
        
        return viz_list
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        key_findings = []
        
        # Analyze dimensionality
        dim_analysis = results.get('dimensionality_analysis', {})
        context_dims = dim_analysis.get('context_comparison', {})
        
        if context_dims:
            dims = [info['mean_dimensionality'] for info in context_dims.values()]
            mean_dim = np.mean(dims)
            
            if mean_dim < 5:
                key_findings.append(f"Transformations are low-dimensional (mean ~{mean_dim:.1f} dimensions)")
            elif mean_dim < 20:
                key_findings.append(f"Transformations use moderate dimensionality (~{mean_dim:.1f} dimensions)")
            else:
                key_findings.append(f"Transformations are high-dimensional (~{mean_dim:.1f} dimensions)")
        
        # Analyze subspace similarities
        if 'subspace_angles' in results:
            similarities = []
            for comparison, layers in results['subspace_angles'].items():
                for layer, angles in layers.items():
                    similarities.append(angles['subspace_similarity'])
            
            if similarities:
                mean_sim = np.mean(similarities)
                if mean_sim > 0.8:
                    key_findings.append("High subspace similarity between contexts")
                elif mean_sim > 0.5:
                    key_findings.append("Moderate subspace similarity between contexts")
                else:
                    key_findings.append("Low subspace similarity between contexts")
        
        # Analyze variance concentration
        for context, evolution in results.get('layer_evolution', {}).items():
            trends = evolution.get('dimensionality_trends', {})
            if 'interpretation' in trends:
                key_findings.append(f"{context}: {trends['interpretation']}")
        
        return {
            'key_findings': key_findings,
            'interpretation': self._generate_interpretation(results),
            'next_steps': [
                "Investigate specific principal component directions",
                "Analyze token-specific subspace projections",
                "Compare subspace patterns across different model layers"
            ]
        }
    
    def _generate_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate interpretation of results"""
        dim_analysis = results.get('dimensionality_analysis', {})
        context_dims = dim_analysis.get('context_comparison', {})
        
        if not context_dims:
            return "Insufficient data for interpretation."
        
        dims = [info['mean_dimensionality'] for info in context_dims.values()]
        mean_dim = np.mean(dims)
        
        if mean_dim < 10:
            interpretation = (
                "The low dimensionality of transformations suggests that context effects "
                "operate through a small number of principal directions in the activation space. "
                "This indicates that transformers may use structured, low-dimensional manifolds "
                "to represent context-dependent modifications to token meanings."
            )
        else:
            interpretation = (
                "The high dimensionality of transformations indicates that context effects "
                "involve complex, distributed changes across many dimensions. This suggests "
                "that context processing in transformers involves rich, high-dimensional "
                "representations rather than simple low-dimensional mappings."
            )
        
        # Add subspace similarity information
        if 'subspace_angles' in results:
            similarities = []
            for comparison, layers in results['subspace_angles'].items():
                for layer, angles in layers.items():
                    similarities.append(angles['subspace_similarity'])
            
            if similarities:
                mean_sim = np.mean(similarities)
                if mean_sim > 0.7:
                    interpretation += (
                        " The high similarity between context subspaces suggests that "
                        "different contexts use overlapping transformation directions."
                    )
                else:
                    interpretation += (
                        " The low similarity between context subspaces indicates that "
                        "different contexts employ distinct transformation strategies."
                    )
        
        return interpretation


if __name__ == "__main__":
    # Example usage
    analysis = SubspaceAlignmentAnalysis("config_unified.yaml")
    analysis.run()