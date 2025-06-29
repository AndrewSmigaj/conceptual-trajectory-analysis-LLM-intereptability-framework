"""
Tests for Subspace Alignment Analysis
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock
from sklearn.decomposition import PCA

from ..subspace_alignment_analysis import SubspaceAlignmentAnalysis
from ..output_schema import UnifiedAnalysisOutput


class TestSubspaceAlignmentAnalysis:
    """Test subspace alignment analysis"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'experiment_name': 'test_subspace',
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
        
        analysis = SubspaceAlignmentAnalysis(config_path)
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
        assert hasattr(analysis, 'pca_cache')
        assert analysis.pca_cache == {}
    
    def test_prepare_transformation_vectors(self, analysis):
        """Test transformation vector preparation"""
        # Mock comprehensive data
        n_tokens = 10
        n_layers = 3
        n_dims = 768
        
        # Create activations
        activations = {}
        for i in range(n_tokens * 3):  # baseline + 2 contexts
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
                'case_idx': i * 3
            }
            trajectories['trajectories'][f'token{i}_determiner_the'] = {
                'token_str': f'token{i}',
                'context_frame': 'determiner_the',
                'case_idx': i * 3 + 1
            }
            trajectories['trajectories'][f'token{i}_function_have'] = {
                'token_str': f'token{i}',
                'context_frame': 'function_have',
                'case_idx': i * 3 + 2
            }
        
        analysis.data_loader.load_unified_trajectories.return_value = trajectories
        analysis.data_loader.get_all_tokens.return_value = [f'token{i}' for i in range(n_tokens)]
        
        # Prepare transformation vectors
        transform_data = analysis._prepare_transformation_vectors(['determiner_the', 'function_have'])
        
        assert 'determiner_the' in transform_data
        assert 'function_have' in transform_data
        
        # Check each layer has data
        for layer in range(n_layers):
            layer_key = f'layer_{layer}'
            assert layer_key in transform_data['determiner_the']
            assert layer_key in transform_data['function_have']
            
            # Check shape
            assert transform_data['determiner_the'][layer_key].shape == (n_tokens, n_dims)
            assert transform_data['function_have'][layer_key].shape == (n_tokens, n_dims)
    
    def test_perform_pca_analysis(self, analysis):
        """Test PCA analysis performance"""
        np.random.seed(42)
        n_samples = 50
        n_features = 100
        
        # Create synthetic data with known structure
        # First 10 dimensions have high variance, rest are noise
        X = np.random.randn(n_samples, n_features)
        X[:, :10] *= 5  # Amplify first 10 dimensions
        
        # Perform PCA
        result = analysis._perform_pca_analysis(X, layer=0, context='test')
        
        assert 'pca' in result
        assert 'components' in result
        assert 'explained_variance' in result
        assert 'explained_variance_ratio' in result
        assert 'effective_dimensionality' in result
        
        # Check dimensions
        assert result['components'].shape[1] == n_features
        assert len(result['explained_variance_ratio']) > 0
        
        # Check that first components explain more variance
        ratios = result['explained_variance_ratio']
        assert ratios[0] > ratios[-1]  # First PC should explain more than last
        
        # Check effective dimensionality is reasonable
        assert 1 <= result['effective_dimensionality'] <= n_samples
    
    def test_analyze_explained_variance(self, analysis):
        """Test explained variance analysis"""
        # Create mock PCA result
        np.random.seed(42)
        n_components = 20
        
        # Create decreasing variance ratios
        variance_ratios = np.exp(-np.arange(n_components) * 0.5)
        variance_ratios = variance_ratios / np.sum(variance_ratios)
        
        pca_result = {
            'explained_variance_ratio': variance_ratios,
            'cumulative_variance': np.cumsum(variance_ratios),
            'effective_dimensionality': 5,
            'explained_variance': variance_ratios * 100  # Scale for broken stick
        }
        
        # Analyze variance
        analysis_result = analysis._analyze_explained_variance(pca_result)
        
        assert 'first_component_variance' in analysis_result
        assert 'top_3_components_variance' in analysis_result
        assert 'effective_dimensionality' in analysis_result
        assert 'intrinsic_dimensionality' in analysis_result
        assert 'variance_concentration' in analysis_result
        
        # Check values are reasonable
        assert analysis_result['first_component_variance'] == variance_ratios[0]
        assert analysis_result['top_3_components_variance'] == np.sum(variance_ratios[:3])
        assert 0 <= analysis_result['variance_concentration'] <= 1
    
    def test_calculate_variance_concentration(self, analysis):
        """Test variance concentration calculation"""
        # Test uniform distribution (low concentration)
        uniform_ratios = np.ones(10) / 10
        conc_uniform = analysis._calculate_variance_concentration(uniform_ratios)
        
        # Test concentrated distribution (high concentration)
        concentrated_ratios = np.array([0.9] + [0.01] * 10)
        conc_concentrated = analysis._calculate_variance_concentration(concentrated_ratios)
        
        assert 0 <= conc_uniform <= 1
        assert 0 <= conc_concentrated <= 1
        assert conc_concentrated > conc_uniform  # Concentrated should be higher
    
    def test_estimate_intrinsic_dimensionality(self, analysis):
        """Test intrinsic dimensionality estimation"""
        # Create eigenvalues with clear break
        eigenvalues = np.array([10, 8, 6, 1, 0.5, 0.3, 0.1, 0.05])
        
        pca_result = {
            'explained_variance': eigenvalues
        }
        
        intrinsic_dim = analysis._estimate_intrinsic_dimensionality(pca_result)
        
        assert isinstance(intrinsic_dim, int)
        assert 1 <= intrinsic_dim <= len(eigenvalues)
    
    def test_compute_canonical_angles(self, analysis):
        """Test canonical angle computation"""
        np.random.seed(42)
        
        # Create two subspaces
        n_dims = 20
        k = 5
        
        # First subspace (random)
        U1 = np.random.randn(n_dims, k)
        U1, _ = np.linalg.qr(U1)
        
        # Second subspace (slight rotation of first)
        theta = 0.1
        rotation = np.eye(n_dims)
        rotation[:2, :2] = [[np.cos(theta), -np.sin(theta)], 
                           [np.sin(theta), np.cos(theta)]]
        U2 = rotation @ U1
        
        # Compute angles
        angles = analysis._compute_canonical_angles(U1, U2)
        
        assert len(angles) == k
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi/2)
        
        # Angles should be small for similar subspaces
        assert np.max(angles) < np.pi/4  # Less than 45 degrees
    
    def test_calculate_subspace_angles(self, analysis):
        """Test subspace angle calculation between contexts"""
        # Create mock principal directions
        np.random.seed(42)
        n_dims = 100
        n_components = 5
        
        principal_directions = {
            'context1': {},
            'context2': {}
        }
        
        for layer in [0, 1, 2]:
            layer_key = f'layer_{layer}'
            
            # Create random orthogonal components
            components1 = np.random.randn(n_components, n_dims)
            components1 = np.linalg.qr(components1.T)[0].T[:n_components]
            
            components2 = np.random.randn(n_components, n_dims)
            components2 = np.linalg.qr(components2.T)[0].T[:n_components]
            
            principal_directions['context1'][layer_key] = {
                'components': components1,
                'explained_variance_ratio': np.exp(-np.arange(n_components))
            }
            principal_directions['context2'][layer_key] = {
                'components': components2,
                'explained_variance_ratio': np.exp(-np.arange(n_components))
            }
        
        analysis.config['layers'] = [0, 1, 2]
        
        # Calculate angles
        angles_result = analysis._calculate_subspace_angles(principal_directions)
        
        assert 'context1_vs_context2' in angles_result
        
        for layer in [0, 1, 2]:
            layer_key = f'layer_{layer}'
            assert layer_key in angles_result['context1_vs_context2']
            
            layer_data = angles_result['context1_vs_context2'][layer_key]
            assert 'canonical_angles' in layer_data
            assert 'mean_angle' in layer_data
            assert 'subspace_similarity' in layer_data
            
            # Check angle bounds
            assert 0 <= layer_data['mean_angle'] <= np.pi/2
            assert 0 <= layer_data['subspace_similarity'] <= 1
    
    def test_analyze_layer_evolution(self, analysis):
        """Test layer evolution analysis"""
        # Create mock PCA results for multiple layers
        layer_pca_results = {}
        
        for layer in [0, 1, 2]:
            layer_key = f'layer_{layer}'
            
            # Create PCA result with decreasing concentration
            n_components = 20
            base_variance = 0.8 - layer * 0.2  # Decreasing concentration
            variance_ratios = np.exp(-np.arange(n_components) * (1 + layer * 0.5))
            variance_ratios = variance_ratios / np.sum(variance_ratios)
            variance_ratios[0] = base_variance  # Set first component
            variance_ratios = variance_ratios / np.sum(variance_ratios)  # Renormalize
            
            layer_pca_results[layer_key] = {
                'explained_variance_ratio': variance_ratios,
                'effective_dimensionality': 5 + layer  # Increasing dimensionality
            }
        
        # Analyze evolution
        evolution = analysis._analyze_layer_evolution(layer_pca_results)
        
        assert 'variance_trends' in evolution
        assert 'dimensionality_trends' in evolution
        
        trends = evolution['variance_trends']
        assert 'concentration_by_layer' in trends
        assert 'effective_dimensionality_by_layer' in trends
        assert 'layers' in trends
        
        # Check that we have data for all layers
        assert len(trends['concentration_by_layer']) == 3
        assert len(trends['effective_dimensionality_by_layer']) == 3
    
    def test_interpret_trends(self, analysis):
        """Test trend interpretation"""
        # Test increasing concentration
        interpretation = analysis._interpret_trends(0.05, 0.5)
        assert "increasingly concentrated" in interpretation
        assert "increasing dimensionality" in interpretation
        
        # Test decreasing concentration
        interpretation = analysis._interpret_trends(-0.05, -0.5)
        assert "increasingly distributed" in interpretation
        assert "decreasing dimensionality" in interpretation
        
        # Test stable trends
        interpretation = analysis._interpret_trends(0.001, 0.05)
        assert "stable" in interpretation
    
    def test_full_analysis_pipeline(self, analysis):
        """Test full analysis pipeline"""
        # Mock comprehensive data
        n_tokens = 20
        n_layers = 3
        n_dims = 100  # Smaller for faster testing
        
        # Create activations with structure
        activations = {}
        np.random.seed(42)
        
        for i in range(n_tokens * 3):  # baseline + 2 contexts
            activations[i] = {}
            for layer in range(n_layers):
                # Create structured activations
                base_activation = np.random.randn(n_dims)
                
                # Add context-specific transformation
                if i % 3 == 1:  # determiner_the context
                    transform = np.zeros(n_dims)
                    transform[:10] = np.random.randn(10) * 2  # Transform first 10 dims
                    base_activation += transform
                elif i % 3 == 2:  # function_have context
                    transform = np.zeros(n_dims)
                    transform[10:20] = np.random.randn(10) * 2  # Transform dims 10-20
                    base_activation += transform
                
                activations[i][layer] = base_activation
        
        analysis.data_loader.load_unified_activations.return_value = activations
        
        # Create trajectories
        trajectories = {'trajectories': {}}
        for i in range(n_tokens):
            trajectories['trajectories'][f'token{i}_baseline'] = {
                'token_str': f'token{i}',
                'context_frame': 'baseline',
                'case_idx': i * 3
            }
            trajectories['trajectories'][f'token{i}_determiner_the'] = {
                'token_str': f'token{i}',
                'context_frame': 'determiner_the',
                'case_idx': i * 3 + 1
            }
            trajectories['trajectories'][f'token{i}_function_have'] = {
                'token_str': f'token{i}',
                'context_frame': 'function_have',
                'case_idx': i * 3 + 2
            }
        
        analysis.data_loader.load_unified_trajectories.return_value = trajectories
        analysis.data_loader.get_all_tokens.return_value = [f'token{i}' for i in range(n_tokens)]
        
        # Run analysis
        results = analysis.analyze()
        
        assert 'pca_results' in results
        assert 'subspace_angles' in results
        assert 'explained_variance' in results
        assert 'layer_evolution' in results
        assert 'dimensionality_analysis' in results
        
        # Check structure
        assert 'determiner_the' in results['pca_results']
        assert 'function_have' in results['pca_results']
        
        # Check subspace angles
        assert 'determiner_the_vs_function_have' in results['subspace_angles']
    
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
            'dimensionality_analysis': {
                'context_comparison': {
                    'context1': {'mean_dimensionality': 3.5},
                    'context2': {'mean_dimensionality': 4.2}
                }
            },
            'subspace_angles': {
                'context1_vs_context2': {
                    'layer_0': {'subspace_similarity': 0.85},
                    'layer_1': {'subspace_similarity': 0.90}
                }
            },
            'layer_evolution': {
                'context1': {
                    'dimensionality_trends': {
                        'interpretation': 'Variance is stable with increasing dimensionality'
                    }
                }
            }
        }
        
        summary = analysis._generate_summary(results)
        
        assert 'key_findings' in summary
        assert 'interpretation' in summary
        assert 'next_steps' in summary
        
        # Check findings include dimensionality info
        assert any('low-dimensional' in f or 'dimensions' in f for f in summary['key_findings'])
        
        # Check interpretation mentions context effects
        assert 'context effects' in summary['interpretation']