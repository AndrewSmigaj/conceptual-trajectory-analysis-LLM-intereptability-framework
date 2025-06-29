"""
Procrustes Cross-Validation Analysis

Finds optimal linear transformations (rotation, scaling, translation) between
baseline and context activations using Procrustes analysis with cross-validation.

This analysis:
1. Aligns activation spaces using Procrustes superimposition
2. Extracts transformation parameters (rotation, scale, translation)
3. Uses cross-validation to ensure transformations generalize
4. Provides quality metrics and confidence intervals
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes, norm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import (
    UnifiedAnalysisOutput, TransformationMatrix, QualityMetrics, 
    AnalysisMetadata, Visualization
)


class ProcrustesAnalysis(BaseTransformationAnalysis):
    """
    Analyzes context effects as geometric transformations using Procrustes analysis.
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.transformation_cache = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Run Procrustes analysis with cross-validation"""
        self.logger.info("Starting Procrustes cross-validation analysis")
        
        results = {
            'transformation_matrices': {},
            'transformation_properties': {},
            'cross_validation_scores': {},
            'layer_evolution': {},
            'context_comparison': {}
        }
        
        # Get context types to analyze
        context_types = self.config.get('context_types', ['determiner_the', 'function_have'])
        
        # Analyze each context type
        for context_type in context_types:
            self.logger.info(f"Analyzing context: {context_type}")
            
            # Get activation pairs
            activation_pairs = self._prepare_activation_pairs('baseline', context_type)
            
            if not activation_pairs:
                self.logger.warning(f"No activation pairs found for {context_type}")
                continue
            
            # Analyze each layer
            layer_results = {}
            for layer in self.config['layers']:
                self.logger.info(f"Processing layer {layer}")
                
                # Extract activations for this layer
                X_baseline = np.array([pair['baseline'][layer] for pair in activation_pairs])
                X_context = np.array([pair['context'][layer] for pair in activation_pairs])
                
                # Find transformation with cross-validation
                transform_results = self._cross_validate_transforms(X_baseline, X_context)
                layer_results[f'layer_{layer}'] = transform_results
                
                # Analyze transformation properties
                properties = self._analyze_transformation_properties(
                    transform_results['best_transform']
                )
                
                if 'transformation_properties' not in results:
                    results['transformation_properties'] = {}
                if context_type not in results['transformation_properties']:
                    results['transformation_properties'][context_type] = {}
                results['transformation_properties'][context_type][f'layer_{layer}'] = properties
            
            results['transformation_matrices'][context_type] = layer_results
            
            # Calculate layer evolution
            evolution = self._analyze_layer_evolution(layer_results)
            results['layer_evolution'][context_type] = evolution
        
        # Compare transformations across contexts
        if len(context_types) > 1:
            results['context_comparison'] = self._compare_context_transformations(
                results['transformation_matrices']
            )
        
        # Calculate overall statistics
        results['overall_statistics'] = self._calculate_overall_statistics(results)
        
        return results
    
    def _prepare_activation_pairs(self, baseline_context: str, 
                                 target_context: str) -> List[Dict[str, Any]]:
        """Prepare paired activations for Procrustes analysis"""
        pairs = []
        
        # Load activations
        all_activations = self.data_loader.load_unified_activations()
        if not all_activations:
            self.logger.warning("No activation data available")
            return pairs
        
        # Get trajectories to match tokens
        trajectories = self.data_loader.load_unified_trajectories(k=self.config['k'])
        if not trajectories or 'trajectories' not in trajectories:
            self.logger.warning("No trajectory data available")
            return pairs
        
        # Build pairs
        for token in self.data_loader.get_all_tokens()[:self.config.get('max_tokens', 1000)]:
            # Check if token exists in both contexts
            baseline_key = f"{token}_{baseline_context}"
            context_key = f"{token}_{target_context}"
            
            baseline_case = None
            context_case = None
            
            # Find case indices from trajectories
            for key, traj_data in trajectories.get('trajectories', {}).items():
                if (traj_data.get('token_str') == token and 
                    traj_data.get('context_frame') == baseline_context):
                    baseline_case = traj_data.get('case_idx')
                elif (traj_data.get('token_str') == token and 
                      traj_data.get('context_frame') == target_context):
                    context_case = traj_data.get('case_idx')
            
            if baseline_case is not None and context_case is not None:
                if baseline_case in all_activations and context_case in all_activations:
                    pairs.append({
                        'token': token,
                        'baseline': all_activations[baseline_case],
                        'context': all_activations[context_case]
                    })
        
        self.logger.info(f"Prepared {len(pairs)} activation pairs")
        return pairs
    
    def _cross_validate_transforms(self, X_baseline: np.ndarray, 
                                 X_context: np.ndarray) -> Dict[str, Any]:
        """Find transformation with cross-validation"""
        # Validate inputs
        if X_baseline.shape != X_context.shape:
            raise ValueError(f"Shape mismatch: baseline {X_baseline.shape} vs context {X_context.shape}")
        
        # Ensure we have enough samples for CV
        min_samples = 10
        if len(X_baseline) < min_samples:
            self.logger.warning(f"Too few samples ({len(X_baseline)}) for cross-validation. Using single split.")
            # Fall back to single train/test split
            n_train = int(0.8 * len(X_baseline))
            train_idx = np.arange(n_train)
            test_idx = np.arange(n_train, len(X_baseline))
            
            # Find transformation
            transform = self._find_procrustes_transform(X_baseline[train_idx], X_context[train_idx])
            X_test_pred = self._apply_transform(X_baseline[test_idx], transform)
            
            # Calculate R² per sample
            r2_scores_per_sample = []
            for i in range(len(X_context[test_idx])):
                if np.var(X_context[test_idx][i]) > 1e-10:
                    r2_i = r2_score(X_context[test_idx][i], X_test_pred[i])
                    r2_scores_per_sample.append(r2_i)
            
            r2 = np.mean(r2_scores_per_sample) if r2_scores_per_sample else 0.0
            mse = mean_squared_error(X_context[test_idx].flatten(), X_test_pred.flatten())
            cos_sim = self._calculate_cosine_similarity(X_context[test_idx], X_test_pred)
            
            best_transform = self._find_procrustes_transform(X_baseline, X_context)
            
            return {
                'best_transform': best_transform,
                'cv_scores': [{'r2': r2, 'mse': mse, 'cosine_similarity': cos_sim}],
                'mean_r2': float(r2),
                'std_r2': 0.0,
                'mean_mse': float(mse),
                'std_mse': 0.0,
                'mean_cosine_similarity': float(cos_sim),
                'std_cosine_similarity': 0.0,
                'n_folds': 1
            }
        
        n_folds = min(5, len(X_baseline) // 3)  # Ensure at least 3 samples per fold
        n_folds = max(2, n_folds)  # At least 2 folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_transforms = []
        
        for train_idx, test_idx in kf.split(X_baseline):
            # Split data
            X_train_base = X_baseline[train_idx]
            X_train_ctx = X_context[train_idx]
            X_test_base = X_baseline[test_idx]
            X_test_ctx = X_context[test_idx]
            
            # Find transformation on training set
            transform = self._find_procrustes_transform(X_train_base, X_train_ctx)
            cv_transforms.append(transform)
            
            # Test on validation set
            X_test_pred = self._apply_transform(X_test_base, transform)
            
            # Calculate quality metrics
            # For multi-output regression, calculate R² for each sample and average
            r2_scores_per_sample = []
            for i in range(len(X_test_ctx)):
                # Avoid division by zero for constant vectors
                if np.var(X_test_ctx[i]) > 1e-10:
                    r2_i = r2_score(X_test_ctx[i], X_test_pred[i])
                    r2_scores_per_sample.append(r2_i)
            
            r2 = np.mean(r2_scores_per_sample) if r2_scores_per_sample else 0.0
            mse = mean_squared_error(X_test_ctx.flatten(), X_test_pred.flatten())
            cos_sim = self._calculate_cosine_similarity(X_test_ctx, X_test_pred)
            
            cv_scores.append({
                'r2': r2,
                'mse': mse,
                'cosine_similarity': cos_sim
            })
        
        # Find best transformation on full data
        best_transform = self._find_procrustes_transform(X_baseline, X_context)
        
        # Calculate confidence intervals
        r2_scores = [s['r2'] for s in cv_scores]
        mse_scores = [s['mse'] for s in cv_scores]
        cos_scores = [s['cosine_similarity'] for s in cv_scores]
        
        return {
            'best_transform': best_transform,
            'cv_scores': cv_scores,
            'mean_r2': float(np.mean(r2_scores)),
            'std_r2': float(np.std(r2_scores)),
            'mean_mse': float(np.mean(mse_scores)),
            'std_mse': float(np.std(mse_scores)),
            'mean_cosine_similarity': float(np.mean(cos_scores)),
            'std_cosine_similarity': float(np.std(cos_scores)),
            'n_folds': n_folds
        }
    
    def _find_procrustes_transform(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Find optimal transformation using Procrustes analysis"""
        # Validate inputs
        if X.shape != Y.shape:
            raise ValueError(f"Shape mismatch in Procrustes: {X.shape} vs {Y.shape}")
        
        # Center the data
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        
        # Use orthogonal Procrustes to find optimal rotation
        # Note: orthogonal_procrustes returns (R, ss) where ss is sum of squared differences
        R, ss = orthogonal_procrustes(X_centered, Y_centered)
        
        # Calculate the actual scale factor
        # After rotation, find scale that minimizes ||Y_c - scale * (X_c @ R)||
        X_rotated = X_centered @ R
        scale = np.sum(Y_centered * X_rotated) / np.sum(X_rotated * X_rotated)
        
        # Calculate translation
        # Y = scale * X @ R + translation
        # translation = Y_mean - scale * X_mean @ R
        translation = Y_mean - scale * (X_mean @ R)
        
        # Calculate disparity (Procrustes distance)
        Y_pred_centered = scale * (X_centered @ R)
        disparity = np.sqrt(np.sum((Y_centered - Y_pred_centered) ** 2))
        
        # Create affine transformation matrix
        # [Y 1] = [X 1] @ T where T = [[scale*R, 0], [translation, 1]]
        d = X.shape[1]
        T = np.eye(d + 1)
        T[:d, :d] = scale * R
        T[:d, -1] = translation
        
        return {
            'rotation': R,
            'scale': float(scale),
            'translation': translation,
            'affine_matrix': T,
            'disparity': float(disparity),
            'centered_X': X_centered,
            'centered_Y': Y_centered
        }
    
    def _apply_transform(self, X: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """Apply transformation to data"""
        R = transform['rotation']
        scale = transform['scale']
        translation = transform['translation']
        
        return scale * (X @ R) + translation
    
    def _analyze_transformation_properties(self, transform: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze transformation properties"""
        R = transform['rotation']
        scale = transform['scale']
        translation = transform['translation']
        
        # Analyze rotation
        # For 2D/3D, we can extract rotation angles
        # For high-D, we analyze eigenvalues of R
        eigenvals = np.linalg.eigvals(R)
        rotation_angle = np.angle(eigenvals)  # Complex eigenvalues give rotation info
        
        # Calculate rotation magnitude (Frobenius norm of (R - I))
        rotation_magnitude = norm(R - np.eye(R.shape[0]), 'fro')
        
        # Analyze scale
        scale_change = scale - 1.0  # Relative scale change
        
        # Analyze translation
        translation_magnitude = norm(translation)
        
        return {
            'rotation_magnitude': float(rotation_magnitude),
            'scale_factor': float(scale),
            'scale_change_percent': float(scale_change * 100),
            'translation_magnitude': float(translation_magnitude),
            'disparity': float(transform['disparity']),
            'is_pure_rotation': bool(abs(scale - 1.0) < 0.01 and translation_magnitude < 0.01),
            'is_isometry': bool(abs(scale - 1.0) < 0.01),  # Preserves distances
            'primary_rotation_angle': float(np.max(np.abs(rotation_angle))) if len(rotation_angle) > 0 else 0.0
        }
    
    def _analyze_layer_evolution(self, layer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how transformations evolve across layers"""
        evolution = {
            'rotation_progression': [],
            'scale_progression': [],
            'translation_progression': [],
            'quality_progression': []
        }
        
        for layer_idx in sorted(self.config['layers']):
            layer_key = f'layer_{layer_idx}'
            if layer_key in layer_results:
                transform = layer_results[layer_key]['best_transform']
                props = self._analyze_transformation_properties(transform)
                
                evolution['rotation_progression'].append(props['rotation_magnitude'])
                evolution['scale_progression'].append(props['scale_factor'])
                evolution['translation_progression'].append(props['translation_magnitude'])
                evolution['quality_progression'].append(layer_results[layer_key]['mean_r2'])
        
        # Calculate trends
        layers = list(range(len(evolution['rotation_progression'])))
        
        evolution['rotation_trend'] = float(np.polyfit(layers, evolution['rotation_progression'], 1)[0])
        evolution['scale_trend'] = float(np.polyfit(layers, evolution['scale_progression'], 1)[0])
        evolution['translation_trend'] = float(np.polyfit(layers, evolution['translation_progression'], 1)[0])
        
        # Identify phase transitions
        evolution['peak_rotation_layer'] = int(np.argmax(evolution['rotation_progression']))
        evolution['peak_scale_layer'] = int(np.argmax(np.abs(np.array(evolution['scale_progression']) - 1.0)))
        
        return evolution
    
    def _compare_context_transformations(self, all_transforms: Dict[str, Any]) -> Dict[str, Any]:
        """Compare transformations across different context types"""
        comparison = {
            'similarity_matrix': {},
            'consistent_properties': [],
            'divergent_properties': []
        }
        
        context_types = list(all_transforms.keys())
        
        # Compare each pair of contexts
        for i, ctx1 in enumerate(context_types):
            for j, ctx2 in enumerate(context_types[i+1:], i+1):
                similarity_scores = []
                
                for layer_idx in self.config['layers']:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in all_transforms[ctx1] and layer_key in all_transforms[ctx2]:
                        R1 = all_transforms[ctx1][layer_key]['best_transform']['rotation']
                        R2 = all_transforms[ctx2][layer_key]['best_transform']['rotation']
                        
                        # Compare rotation matrices using trace of R1.T @ R2
                        # For orthogonal matrices: -d <= trace(R1.T @ R2) <= d
                        # Normalize to [0, 1] where 1 is identical
                        d = R1.shape[0]
                        trace_val = np.trace(R1.T @ R2)
                        # Simple normalization: trace/d gives [-1, 1], shift to [0, 1]
                        rotation_similarity = (trace_val / d + 1.0) / 2.0
                        similarity_scores.append(rotation_similarity)
                
                comparison['similarity_matrix'][f'{ctx1}_vs_{ctx2}'] = {
                    'mean_similarity': float(np.mean(similarity_scores)),
                    'std_similarity': float(np.std(similarity_scores))
                }
        
        return comparison
    
    def _calculate_cosine_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate average cosine similarity between vectors"""
        similarities = []
        for x, y in zip(X, Y):
            x_norm = x / (norm(x) + 1e-10)
            y_norm = y / (norm(y) + 1e-10)
            similarities.append(np.dot(x_norm, y_norm))
        return float(np.mean(similarities))
    
    def _calculate_overall_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall statistics across all analyses"""
        stats = {
            'mean_quality_scores': {},
            'transformation_consistency': {},
            'key_insights': []
        }
        
        # Aggregate quality scores
        all_r2_scores = []
        all_scales = []
        all_rotations = []
        
        for context, layers in results['transformation_matrices'].items():
            for layer, transform_data in layers.items():
                all_r2_scores.append(transform_data['mean_r2'])
                
                props = results['transformation_properties'][context][layer]
                all_scales.append(props['scale_factor'])
                all_rotations.append(props['rotation_magnitude'])
        
        stats['mean_quality_scores'] = {
            'overall_r2': float(np.mean(all_r2_scores)),
            'overall_r2_std': float(np.std(all_r2_scores))
        }
        
        # Analyze consistency
        stats['transformation_consistency'] = {
            'scale_variance': float(np.var(all_scales)),
            'rotation_variance': float(np.var(all_rotations)),
            'primarily_rotational': bool(np.mean(all_rotations) > 0.5 and np.var(all_scales) < 0.1)
        }
        
        # Generate insights
        if stats['transformation_consistency']['primarily_rotational']:
            stats['key_insights'].append("Transformations are primarily rotational")
        
        if np.mean(all_r2_scores) > 0.8:
            stats['key_insights'].append("High-quality linear approximations (R² > 0.8)")
        elif np.mean(all_r2_scores) > 0.6:
            stats['key_insights'].append("Moderate linear approximations (R² > 0.6)")
        else:
            stats['key_insights'].append("Transformations are not well-approximated by linear maps")
        
        return stats
    
    def validate_data(self) -> None:
        """Validate loaded data"""
        if not hasattr(self, 'data_loader') or self.data_loader is None:
            raise ValueError("Data loader not initialized")
        
        # Check we have activation data
        activations = self.data_loader.load_unified_activations()
        if not activations:
            raise ValueError("No activation data found")
        
        # Check we have trajectory data for matching
        trajectories = self.data_loader.load_unified_trajectories(k=self.config['k'])
        if not trajectories:
            raise ValueError("No trajectory data found")
        
        self.logger.info("Data validation passed")
    
    def validate_results(self) -> None:
        """Validate analysis results"""
        if not hasattr(self, 'output') or self.output is None:
            raise ValueError("No output generated")
        
        # Check required fields
        if 'transformation_matrices' not in self.output.data:
            raise ValueError("Missing transformation matrices")
        
        if 'transformation_properties' not in self.output.data:
            raise ValueError("Missing transformation properties")
        
        self.logger.info("Results validation passed")
    
    def _create_visualizations(self) -> List[Dict[str, Any]]:
        """Create visualizations for Procrustes analysis"""
        viz_list = []
        
        # Transformation evolution plot
        viz_list.append({
            'name': 'transformation_evolution',
            'path': str(self.output_dir / 'transformation_evolution.png'),
            'type': 'line_plot',
            'description': 'Evolution of transformation properties across layers'
        })
        
        # Quality scores heatmap
        viz_list.append({
            'name': 'quality_heatmap',
            'path': str(self.output_dir / 'quality_heatmap.png'),
            'type': 'heatmap',
            'description': 'R² scores across contexts and layers'
        })
        
        # Rotation matrix visualization
        viz_list.append({
            'name': 'rotation_matrices',
            'path': str(self.output_dir / 'rotation_matrices.png'),
            'type': 'matrix_plot',
            'description': 'Visualization of rotation matrices'
        })
        
        return viz_list
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        stats = results.get('overall_statistics', {})
        
        # Key findings
        key_findings = stats.get('key_insights', [])
        
        # Add specific findings
        mean_r2 = stats.get('mean_quality_scores', {}).get('overall_r2', 0)
        if mean_r2 > 0.7:
            key_findings.append(f"Context effects are well-approximated by linear transformations (mean R²={mean_r2:.3f})")
        
        # Check evolution patterns
        for context, evolution in results.get('layer_evolution', {}).items():
            if evolution.get('rotation_trend', 0) > 0.1:
                key_findings.append(f"{context}: Rotation magnitude increases through layers")
            elif evolution.get('rotation_trend', 0) < -0.1:
                key_findings.append(f"{context}: Rotation magnitude decreases through layers")
        
        return {
            'key_findings': key_findings,
            'interpretation': self._generate_interpretation(results),
            'next_steps': [
                "Analyze specific rotation patterns in embedding space",
                "Test if transformations are consistent across token types",
                "Investigate non-linear transformation components"
            ]
        }
    
    def _generate_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate interpretation of results"""
        stats = results.get('overall_statistics', {})
        mean_r2 = stats.get('mean_quality_scores', {}).get('overall_r2', 0)
        
        if mean_r2 > 0.8:
            interpretation = (
                "The high quality of linear approximations suggests that context effects "
                "can be largely understood as geometric transformations in activation space. "
                "This supports the hypothesis that transformers learn systematic mappings "
                "for how context modifies representations through rotation, scaling, and translation."
            )
        elif mean_r2 > 0.6:
            interpretation = (
                "Moderate linear approximation quality indicates that while geometric "
                "transformations capture significant aspects of context effects, there are "
                "also non-linear components. The linear transformations provide a useful "
                "but incomplete picture of how context modifies representations."
            )
        else:
            interpretation = (
                "Low linear approximation quality suggests that context effects involve "
                "complex, non-linear transformations that cannot be reduced to simple "
                "geometric operations. This indicates sophisticated context-dependent "
                "processing beyond rotation and scaling."
            )
        
        # Add consistency information
        if stats.get('transformation_consistency', {}).get('primarily_rotational', False):
            interpretation += (
                " The transformations are primarily rotational, suggesting that "
                "context mainly reorients representations rather than rescaling them."
            )
        
        return interpretation


if __name__ == "__main__":
    # Example usage
    analysis = ProcrustesAnalysis("config_unified.yaml")
    analysis.run()