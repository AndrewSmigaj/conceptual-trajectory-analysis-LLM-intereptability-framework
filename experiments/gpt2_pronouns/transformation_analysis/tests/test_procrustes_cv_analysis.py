"""
Tests for Procrustes Cross-Validation Analysis
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
from scipy.spatial import procrustes

from ..procrustes_cv_analysis import ProcrustesAnalysis
from ..output_schema import UnifiedAnalysisOutput


class TestProcrustesAnalysis:
    """Test Procrustes cross-validation analysis"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'experiment_name': 'test_procrustes',
            'output_dir': 'test_output',
            'k': 10,
            'layers': [0, 1, 2],
            'context_types': ['determiner_the', 'function_have'],
            'max_tokens': 50,
            'enable_logging': False
        }
    
    @pytest.fixture
    def analysis(self, sample_config):
        """Create analysis instance"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_config, f)
            config_path = f.name
        
        analysis = ProcrustesAnalysis(config_path)
        analysis.config = sample_config
        analysis.output_dir = Path(tempfile.mkdtemp())
        
        # Mock logger
        analysis.logger = Mock()
        
        # Mock data loader
        analysis.data_loader = Mock()
        
        yield analysis
        
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
    
    def test_initialization(self, analysis):
        """Test proper initialization"""
        assert hasattr(analysis, 'transformation_cache')
        assert analysis.transformation_cache == {}
    
    def test_prepare_activation_pairs(self, analysis):
        """Test activation pair preparation"""
        # Mock data
        analysis.data_loader.load_unified_activations.return_value = {
            0: {0: np.random.randn(768), 1: np.random.randn(768), 2: np.random.randn(768)},
            1: {0: np.random.randn(768), 1: np.random.randn(768), 2: np.random.randn(768)},
            2: {0: np.random.randn(768), 1: np.random.randn(768), 2: np.random.randn(768)},
            3: {0: np.random.randn(768), 1: np.random.randn(768), 2: np.random.randn(768)}
        }
        
        analysis.data_loader.load_unified_trajectories.return_value = {
            'trajectories': {
                'token1_baseline': {
                    'token_str': 'token1',
                    'context_frame': 'baseline',
                    'case_idx': 0
                },
                'token1_determiner_the': {
                    'token_str': 'token1',
                    'context_frame': 'determiner_the',
                    'case_idx': 1
                },
                'token2_baseline': {
                    'token_str': 'token2',
                    'context_frame': 'baseline',
                    'case_idx': 2
                },
                'token2_determiner_the': {
                    'token_str': 'token2',
                    'context_frame': 'determiner_the',
                    'case_idx': 3
                }
            }
        }
        
        analysis.data_loader.get_all_tokens.return_value = ['token1', 'token2']
        
        # Prepare pairs
        pairs = analysis._prepare_activation_pairs('baseline', 'determiner_the')
        
        assert len(pairs) == 2
        assert pairs[0]['token'] == 'token1'
        assert pairs[1]['token'] == 'token2'
        assert 0 in pairs[0]['baseline']  # Layer 0 exists
        assert 0 in pairs[0]['context']
    
    def test_find_procrustes_transform(self, analysis):
        """Test Procrustes transformation finding"""
        np.random.seed(42)
        n_points = 50
        n_dims = 10
        
        # Create synthetic transformation
        X = np.random.randn(n_points, n_dims)
        
        # Apply known transformation
        true_R = np.linalg.qr(np.random.randn(n_dims, n_dims))[0]  # Random rotation
        true_scale = 1.5
        true_translation = np.random.randn(n_dims) * 0.1
        
        Y = true_scale * (X @ true_R) + true_translation
        
        # Find transformation
        transform = analysis._find_procrustes_transform(X, Y)
        
        assert 'rotation' in transform
        assert 'scale' in transform
        assert 'translation' in transform
        assert 'affine_matrix' in transform
        
        # Check dimensions
        assert transform['rotation'].shape == (n_dims, n_dims)
        assert transform['translation'].shape == (n_dims,)
        assert transform['affine_matrix'].shape == (n_dims + 1, n_dims + 1)
        
        # Check orthogonality of rotation
        R = transform['rotation']
        assert np.allclose(R @ R.T, np.eye(n_dims), atol=1e-10)
        
        # Check scale is positive
        assert transform['scale'] > 0
    
    def test_apply_transform(self, analysis):
        """Test transformation application"""
        np.random.seed(42)
        X = np.random.randn(10, 5)
        
        # Create transform
        transform = {
            'rotation': np.eye(5),  # Identity rotation
            'scale': 2.0,
            'translation': np.ones(5)
        }
        
        # Apply transform
        Y_pred = analysis._apply_transform(X, transform)
        
        # Check result
        Y_expected = 2.0 * X + np.ones(5)
        assert np.allclose(Y_pred, Y_expected)
    
    def test_cross_validate_transforms(self, analysis):
        """Test cross-validation of transformations"""
        np.random.seed(42)
        n_points = 100
        n_dims = 20
        
        # Create data with known linear relationship
        X_baseline = np.random.randn(n_points, n_dims)
        R = np.linalg.qr(np.random.randn(n_dims, n_dims))[0]  # Random rotation
        scale = 1.2
        translation = np.random.randn(n_dims) * 0.1
        
        # Apply known transformation with small noise
        X_context = scale * (X_baseline @ R) + translation + 0.01 * np.random.randn(n_points, n_dims)
        
        # Run cross-validation
        results = analysis._cross_validate_transforms(X_baseline, X_context)
        
        assert 'best_transform' in results
        assert 'cv_scores' in results
        assert 'mean_r2' in results
        assert 'mean_cosine_similarity' in results
        
        # Check structure
        assert len(results['cv_scores']) == results['n_folds']
        assert results['n_folds'] >= 2
        
        # Check that transform recovers the known parameters approximately
        transform = results['best_transform']
        assert abs(transform['scale'] - scale) < 0.1  # Scale should be close to 1.2
        
        # Apply transform and check fit
        Y_pred = analysis._apply_transform(X_baseline, transform)
        error = np.mean((Y_pred - X_context) ** 2)
        assert error < 0.1  # Should have low reconstruction error
    
    def test_analyze_transformation_properties(self, analysis):
        """Test transformation property analysis"""
        # Pure rotation
        R = np.array([[0, -1], [1, 0]])  # 90 degree rotation in 2D
        transform = {
            'rotation': R,
            'scale': 1.0,
            'translation': np.zeros(2),
            'disparity': 0.1
        }
        
        props = analysis._analyze_transformation_properties(transform)
        
        assert 'rotation_magnitude' in props
        assert 'scale_factor' in props
        assert 'translation_magnitude' in props
        assert 'is_pure_rotation' in props
        assert 'is_isometry' in props
        
        # Should identify as pure rotation and isometry
        assert props['is_pure_rotation'] == True
        assert props['is_isometry'] == True
        assert props['scale_factor'] == 1.0
        assert props['translation_magnitude'] < 0.01
    
    def test_analyze_layer_evolution(self, analysis):
        """Test layer evolution analysis"""
        # Create mock layer results
        layer_results = {}
        for i in range(3):
            transform = {
                'rotation': np.eye(10) + 0.1 * i * np.random.randn(10, 10),
                'scale': 1.0 + 0.1 * i,
                'translation': np.zeros(10) + 0.05 * i,
                'disparity': 0.1
            }
            
            layer_results[f'layer_{i}'] = {
                'best_transform': transform,
                'mean_r2': 0.9 - 0.1 * i
            }
        
        evolution = analysis._analyze_layer_evolution(layer_results)
        
        assert 'rotation_progression' in evolution
        assert 'scale_progression' in evolution
        assert 'translation_progression' in evolution
        assert 'quality_progression' in evolution
        assert 'rotation_trend' in evolution
        assert 'peak_rotation_layer' in evolution
        
        # Check progressions have correct length
        assert len(evolution['rotation_progression']) == 3
        assert len(evolution['scale_progression']) == 3
    
    def test_compare_context_transformations(self, analysis):
        """Test context transformation comparison"""
        # Create mock transformations for two contexts
        all_transforms = {
            'context1': {},
            'context2': {}
        }
        
        np.random.seed(42)
        n_dims = 10
        
        for layer in range(3):
            # R1 is identity
            R1 = np.eye(n_dims)
            
            # Create a small rotation using rotation matrix formula
            # This ensures we get a proper rotation, not a reflection
            theta = 0.1  # Small rotation angle
            # Create rotation in the first two dimensions
            R2 = np.eye(n_dims)
            R2[0, 0] = np.cos(theta)
            R2[0, 1] = -np.sin(theta)
            R2[1, 0] = np.sin(theta)
            R2[1, 1] = np.cos(theta)
            
            all_transforms['context1'][f'layer_{layer}'] = {
                'best_transform': {'rotation': R1}
            }
            all_transforms['context2'][f'layer_{layer}'] = {
                'best_transform': {'rotation': R2}
            }
        
        comparison = analysis._compare_context_transformations(all_transforms)
        
        assert 'similarity_matrix' in comparison
        assert 'context1_vs_context2' in comparison['similarity_matrix']
        assert 'mean_similarity' in comparison['similarity_matrix']['context1_vs_context2']
        
        # With small perturbations, similarity should be high
        sim = comparison['similarity_matrix']['context1_vs_context2']['mean_similarity']
        assert sim > 0.8  # Should be high similarity
        assert sim < 1.0  # But not perfect
    
    def test_calculate_cosine_similarity(self, analysis):
        """Test cosine similarity calculation"""
        # Test with identical vectors
        X = np.random.randn(10, 5)
        sim = analysis._calculate_cosine_similarity(X, X)
        assert np.isclose(sim, 1.0)
        
        # Test with orthogonal vectors
        X = np.array([[1, 0], [0, 1]])
        Y = np.array([[0, 1], [-1, 0]])
        sim = analysis._calculate_cosine_similarity(X, Y)
        assert np.isclose(sim, 0.0, atol=1e-10)
    
    def test_full_analysis_pipeline(self, analysis):
        """Test full analysis pipeline"""
        # Mock comprehensive data
        n_tokens = 20
        n_layers = 3
        n_dims = 768
        
        # Create activations
        activations = {}
        for i in range(n_tokens * 2):  # baseline + context
            activations[i] = {}
            for layer in range(n_layers):
                activations[i][layer] = np.random.randn(n_dims)
        
        analysis.data_loader.load_unified_activations.return_value = activations
        
        # Create trajectories
        trajectories = {'trajectories': {}}
        for i in range(n_tokens):
            trajectories['trajectories'][f'token{i}_baseline'] = {
                'token_str': f'token{i}',
                'context_frame': 'baseline',
                'case_idx': i * 2
            }
            trajectories['trajectories'][f'token{i}_determiner_the'] = {
                'token_str': f'token{i}',
                'context_frame': 'determiner_the',
                'case_idx': i * 2 + 1
            }
        
        analysis.data_loader.load_unified_trajectories.return_value = trajectories
        analysis.data_loader.get_all_tokens.return_value = [f'token{i}' for i in range(n_tokens)]
        
        analysis.config['context_types'] = ['determiner_the']
        analysis.config['layers'] = list(range(n_layers))
        
        # Run analysis
        results = analysis.analyze()
        
        assert 'transformation_matrices' in results
        assert 'transformation_properties' in results
        assert 'layer_evolution' in results
        assert 'overall_statistics' in results
        
        # Check structure
        assert 'determiner_the' in results['transformation_matrices']
        assert len(results['transformation_matrices']['determiner_the']) == n_layers
    
    def test_validation_methods(self, analysis):
        """Test data and results validation"""
        # Test data validation failure
        analysis.data_loader.load_unified_activations.return_value = None
        
        with pytest.raises(ValueError, match="No activation data"):
            analysis.validate_data()
        
        # Test successful validation
        analysis.data_loader.load_unified_activations.return_value = {0: {}}
        analysis.data_loader.load_unified_trajectories.return_value = {'trajectories': {}}
        
        analysis.validate_data()  # Should not raise
        
        # Test results validation
        analysis.output = None
        with pytest.raises(ValueError, match="No output generated"):
            analysis.validate_results()
    
    def test_generate_summary(self, analysis):
        """Test summary generation"""
        results = {
            'overall_statistics': {
                'mean_quality_scores': {'overall_r2': 0.85},
                'transformation_consistency': {'primarily_rotational': True},
                'key_insights': ['High-quality linear approximations (R² > 0.8)']
            },
            'layer_evolution': {
                'context1': {'rotation_trend': 0.15}
            }
        }
        
        summary = analysis._generate_summary(results)
        
        assert 'key_findings' in summary
        assert 'interpretation' in summary
        assert 'next_steps' in summary
        
        # Check findings
        assert any('well-approximated' in f for f in summary['key_findings'])
        assert any('increases through layers' in f for f in summary['key_findings'])
        
        # Check interpretation mentions geometric transformations
        assert 'geometric transformations' in summary['interpretation']