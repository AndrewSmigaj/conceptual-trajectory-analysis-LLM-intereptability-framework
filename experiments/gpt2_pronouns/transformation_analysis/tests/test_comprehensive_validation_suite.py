"""
Tests for comprehensive validation suite.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from ..comprehensive_validation_suite import ComprehensiveValidationSuite
from ..output_schema import UnifiedAnalysisOutput


class TestComprehensiveValidationSuite:
    """Test comprehensive validation functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'k_range': [3, 5, 7],
            'algorithms': ['kmeans', 'hierarchical'],
            'normalizations': ['none', 'standard'],
            'n_samples': 100,
            'random_state': 42,
            'visualize': False,
            'layers_to_test': [0, 5],
            'enable_logging': False
        }
    
    @pytest.fixture
    def validator(self, sample_config, tmp_path):
        """Create validator instance."""
        validator = ComprehensiveValidationSuite(
            output_dir=str(tmp_path),
            config=sample_config
        )
        
        # Mock data loader
        validator.data_loader = Mock()
        
        # Mock trajectory data
        validator.data_loader.load_trajectories.return_value = {
            'trajectories': {
                'token1_baseline': {'path': [0, 1, 2, 1, 0]},
                'token2_baseline': {'path': [1, 1, 2, 2, 1]}
            }
        }
        
        # Mock activation data
        np.random.seed(42)
        n_samples = 200
        n_features = 768
        
        # Create synthetic activations with some structure
        # Layer 0: 3 clear clusters
        layer0_acts = np.vstack([
            np.random.normal(0, 1, (70, n_features)),
            np.random.normal(5, 1, (70, n_features)),
            np.random.normal(10, 1, (60, n_features))
        ])
        
        # Layer 5: 5 clusters
        layer5_acts = np.vstack([
            np.random.normal(0, 1, (40, n_features)),
            np.random.normal(3, 1, (40, n_features)),
            np.random.normal(6, 1, (40, n_features)),
            np.random.normal(9, 1, (40, n_features)),
            np.random.normal(12, 1, (40, n_features))
        ])
        
        validator.data_loader.load_activations.return_value = {
            'unified_activations': {
                'layer_0': layer0_acts,
                'layer_5': layer5_acts
            }
        }
        
        return validator
    
    def test_initialization(self, validator):
        """Test proper initialization."""
        assert validator.analysis_name == 'comprehensive_validation_suite'
        assert hasattr(validator, 'validation_results')
        assert len(validator.validation_results) == 4
    
    def test_k_selection_validation(self, validator):
        """Test k selection validation."""
        # Get mock activations
        activations = validator.data_loader.load_activations()
        
        # Run k selection validation
        validator._validate_k_selection(activations)
        
        # Check results structure
        assert 'k_selection' in validator.validation_results
        k_results = validator.validation_results['k_selection']
        
        # Should have results for each test layer
        assert len(k_results) == len(validator.config['layers_to_test'])
        
        # Check layer 0 results
        if 'layer_0' in k_results:
            layer0_results = k_results['layer_0']
            assert 'inertias' in layer0_results
            assert 'silhouette_scores' in layer0_results
            
            # Should have decreasing inertia with increasing k
            inertias = layer0_results['inertias']
            assert len(inertias) == len(validator.config['k_range'])
            assert all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1))
            
            # Should have found optimal k
            assert 'optimal_k' in layer0_results
            assert 'elbow_method' in layer0_results['optimal_k']
    
    def test_algorithm_comparison(self, validator):
        """Test algorithm comparison."""
        activations = validator.data_loader.load_activations()
        
        # Run algorithm comparison
        validator._compare_algorithms(activations)
        
        # Check results
        assert 'algorithm_comparison' in validator.validation_results
        alg_results = validator.validation_results['algorithm_comparison']
        
        # Should have results for each algorithm
        assert 'kmeans' in alg_results
        assert 'hierarchical' in alg_results
        
        # Check kmeans results
        kmeans_results = alg_results['kmeans']
        assert 'silhouette_score' in kmeans_results
        assert 'calinski_harabasz_score' in kmeans_results
        assert 'davies_bouldin_score' in kmeans_results
        assert 'n_clusters' in kmeans_results
        
        # Scores should be reasonable
        assert -1 <= kmeans_results['silhouette_score'] <= 1
        assert kmeans_results['calinski_harabasz_score'] > 0
        assert kmeans_results['davies_bouldin_score'] > 0
    
    def test_normalization_comparison(self, validator):
        """Test normalization comparison."""
        activations = validator.data_loader.load_activations()
        
        # Run normalization comparison
        validator._compare_normalizations(activations)
        
        # Check results
        assert 'normalization_comparison' in validator.validation_results
        norm_results = validator.validation_results['normalization_comparison']
        
        # Should have results for each normalization
        assert 'none' in norm_results
        assert 'standard' in norm_results
        
        # Check standard normalization results
        std_results = norm_results['standard']
        assert 'silhouette_score' in std_results
        assert 'data_stats' in std_results
        
        # Normalized data should have mean ~0, std ~1
        stats = std_results['data_stats']
        assert abs(stats['mean']) < 0.1
        assert 0.9 < stats['std'] < 1.1
    
    def test_stability_analysis(self, validator):
        """Test stability analysis."""
        activations = validator.data_loader.load_activations()
        
        # Run stability analysis
        validator._analyze_stability(activations)
        
        # Check results
        assert 'stability_analysis' in validator.validation_results
        stability = validator.validation_results['stability_analysis']
        
        assert 'silhouette_scores' in stability
        assert 'adjusted_rand_scores' in stability
        assert 'n_runs' in stability
        
        # Check score statistics
        sil_stats = stability['silhouette_scores']
        assert 'mean' in sil_stats
        assert 'std' in sil_stats
        assert sil_stats['std'] >= 0  # Non-negative std
        
        # ARI should be between 0 and 1 for stable clustering
        ari_stats = stability['adjusted_rand_scores']
        assert 0 <= ari_stats['mean'] <= 1
        assert ari_stats['std'] >= 0
    
    def test_find_dbscan_eps(self, validator):
        """Test DBSCAN eps finding."""
        # Create simple test data
        np.random.seed(42)
        data = np.random.randn(100, 10)
        
        eps = validator._find_dbscan_eps(data)
        
        # Should return positive eps
        assert eps > 0
        assert isinstance(eps, float)
    
    def test_generate_recommendations(self, validator):
        """Test recommendation generation."""
        # Set up some validation results
        validator.validation_results = {
            'k_selection': {
                'layer_0': {
                    'optimal_k': {
                        'elbow_method': 10,
                        'silhouette_method': 10
                    }
                }
            },
            'algorithm_comparison': {
                'kmeans': {'silhouette_score': 0.6},
                'hierarchical': {'silhouette_score': 0.5}
            },
            'normalization_comparison': {
                'standard': {'silhouette_score': 0.65},
                'none': {'silhouette_score': 0.55}
            },
            'stability_analysis': {
                'adjusted_rand_scores': {'mean': 0.85}
            }
        }
        
        recommendations = validator._generate_recommendations()
        
        # Should generate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation content
        assert any('k=10' in rec for rec in recommendations)
        assert any('Kmeans' in rec for rec in recommendations)
        assert any('Standard normalization' in rec for rec in recommendations)
        assert any('highly stable' in rec for rec in recommendations)
    
    def test_full_analysis_pipeline(self, validator):
        """Test full analysis pipeline."""
        # Run full analysis
        results = validator.analyze()
        
        # Check results structure
        assert 'validation_results' in results
        assert 'recommendations' in results
        
        val_results = results['validation_results']
        
        # Should have all validation types
        assert 'k_selection' in val_results
        assert 'algorithm_comparison' in val_results
        assert 'normalization_comparison' in val_results
        assert 'stability_analysis' in val_results
        
        # Should have recommendations
        assert isinstance(results['recommendations'], list)
        assert len(results['recommendations']) > 0
    
    def test_data_validation(self, validator):
        """Test data validation."""
        # Valid data should pass
        validator.validate_data()
        
        # Missing trajectories should fail
        validator.data_loader.load_trajectories.return_value = None
        with pytest.raises(ValueError, match="No trajectory data"):
            validator.validate_data()
        
        # Missing activations should fail
        validator.data_loader.load_trajectories.return_value = {'trajectories': {}}
        validator.data_loader.load_activations.return_value = None
        with pytest.raises(ValueError, match="No activation data"):
            validator.validate_data()
    
    def test_results_validation(self, validator):
        """Test results validation."""
        # No output should fail
        validator.output = None
        with pytest.raises(ValueError, match="No output generated"):
            validator.validate_results()
        
        # Missing validation results should fail
        validator.output = Mock()
        validator.output.data = {}
        with pytest.raises(ValueError, match="No validation results"):
            validator.validate_results()
        
        # Complete results should pass
        validator.output.data = {
            'validation_results': {
                'k_selection': {'layer_0': {}},
                'algorithm_comparison': {'kmeans': {}},
                'normalization_comparison': {'standard': {}},
                'stability_analysis': {'mean': 0.8}
            }
        }
        validator.validate_results()  # Should not raise
    
    def test_extract_layer_activations(self, validator):
        """Test layer activation extraction."""
        activations = validator.data_loader.load_activations()
        
        # Extract layer 0
        layer0 = validator._extract_layer_activations(activations, 0)
        assert layer0 is not None
        assert layer0.shape[0] == 200  # n_samples
        assert layer0.shape[1] == 768  # n_features
        
        # Non-existent layer should return None
        layer99 = validator._extract_layer_activations(activations, 99)
        assert layer99 is None
        
        # Test with different format
        alt_activations = {
            'activations': {
                'token1': {'layers': [np.random.randn(768) for _ in range(12)]},
                'token2': {'layers': [np.random.randn(768) for _ in range(12)]}
            }
        }
        
        layer0_alt = validator._extract_layer_activations(alt_activations, 0)
        assert layer0_alt is not None
        assert layer0_alt.shape[0] == 2  # 2 tokens
        assert layer0_alt.shape[1] == 768