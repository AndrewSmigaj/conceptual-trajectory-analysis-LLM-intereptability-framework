"""
Tests for Information Theory Metrics analysis.
"""

import pytest
import numpy as np
from pathlib import Path

from ..information_theory_metrics import InformationTheoryMetrics
from ..output_schema import MetricWithCI


class TestInformationTheoryMetrics:
    """Test information theory calculations."""
    
    @pytest.mark.unit
    def test_initialization(self, temp_dir):
        """Test initialization."""
        analysis = InformationTheoryMetrics(
            output_dir=str(temp_dir),
            config={'n_bootstrap': 100, 'enable_logging': False}
        )
        
        assert analysis.analysis_name == "information_theory_metrics"
        assert analysis.config['n_bootstrap'] == 100
        assert analysis.config['k_clusters'] == 10
        
    @pytest.mark.unit
    def test_entropy_calculation(self):
        """Test discrete entropy calculation."""
        analysis = InformationTheoryMetrics(config={'enable_logging': False})
        
        # Test uniform distribution (max entropy)
        data = [0, 1, 2, 3] * 25  # 100 samples, 4 categories
        entropy = analysis._calculate_discrete_entropy(data)
        expected = 2.0  # log2(4) = 2 bits
        assert abs(entropy - expected) < 0.01
        
        # Test deterministic distribution (zero entropy)
        data = [0] * 100
        entropy = analysis._calculate_discrete_entropy(data)
        assert entropy < 0.01  # Should be ~0
        
        # Test binary distribution
        data = [0, 1] * 50
        entropy = analysis._calculate_discrete_entropy(data)
        expected = 1.0  # log2(2) = 1 bit
        assert abs(entropy - expected) < 0.01
        
    @pytest.mark.unit
    def test_transition_matrix_building(self, mock_data_loader):
        """Test transition matrix construction."""
        analysis = InformationTheoryMetrics(config={'k_clusters': 10, 'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Build transition matrix for baseline
        matrix = analysis._build_transition_matrix('baseline')
        
        # Check properties
        assert matrix.shape == (10, 10)
        assert np.all(matrix >= 0)  # Non-negative
        assert np.all(matrix <= 1)  # Probabilities
        
        # Check normalization (rows should sum to 1 or 0)
        row_sums = matrix.sum(axis=1)
        assert np.all((row_sums == 0) | (np.abs(row_sums - 1) < 1e-6))
        
    @pytest.mark.unit
    def test_mutual_information_calculation(self, mock_data_loader):
        """Test mutual information calculation."""
        analysis = InformationTheoryMetrics(config={'k_clusters': 10, 'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        contexts = ['baseline', 'determiner_the']
        mi_results = analysis._calculate_mutual_information(contexts)
        
        # Check structure
        assert 'layer_mi' in mi_results
        assert len(mi_results['layer_mi']) == 12  # GPT-2 layers
        
        # MI should be non-negative
        for mi in mi_results['layer_mi']:
            if hasattr(mi, 'value'):
                assert mi.value >= 0
            else:
                assert mi >= 0
                
    @pytest.mark.unit
    def test_kl_divergence_calculation(self, mock_data_loader):
        """Test KL divergence calculation."""
        analysis = InformationTheoryMetrics(config={'k_clusters': 10, 'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        contexts = ['baseline', 'determiner_the']
        kl_results = analysis._calculate_kl_divergence(contexts)
        
        # Check structure
        assert 'context_kl' in kl_results
        assert 'layer_evolution' in kl_results
        
        # KL divergence should be non-negative
        for context, metric in kl_results['context_kl'].items():
            assert isinstance(metric, MetricWithCI)
            assert metric.value >= 0
            
    @pytest.mark.unit
    def test_jensen_shannon_calculation(self, mock_data_loader):
        """Test Jensen-Shannon divergence calculation."""
        analysis = InformationTheoryMetrics(config={'k_clusters': 10, 'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        contexts = ['baseline', 'determiner_the', 'determiner_a']
        js_results = analysis._calculate_jensen_shannon(contexts)
        
        # Check structure
        assert 'divergence_matrix' in js_results
        assert 'most_similar' in js_results
        assert 'most_dissimilar' in js_results
        
        # Check matrix properties
        matrix = js_results['divergence_matrix']
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, matrix.T)  # Symmetric
        assert np.all(matrix >= 0)  # Non-negative
        assert np.all(matrix <= 1)  # JS divergence bounded by 1
        assert np.allclose(np.diag(matrix), 0)  # Zero on diagonal
        
    @pytest.mark.unit
    def test_layer_evolution(self, mock_data_loader):
        """Test layer evolution analysis."""
        analysis = InformationTheoryMetrics(config={'k_clusters': 10, 'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        contexts = ['baseline', 'determiner_the']
        evolution = analysis._analyze_layer_evolution(contexts)
        
        # Check structure
        assert 'entropy_by_layer' in evolution
        assert 'divergence_by_layer' in evolution
        assert 'information_gain' in evolution
        
        # Check dimensions
        assert len(evolution['entropy_by_layer']) == 12
        assert len(evolution['divergence_by_layer']) == 12
        assert len(evolution['information_gain']) == 11  # One less than layers
        
        # Entropy should be non-negative
        assert all(e >= 0 for e in evolution['entropy_by_layer'])
        
    @pytest.mark.unit
    def test_stratified_analysis(self, mock_data_loader):
        """Test stratified analysis."""
        analysis = InformationTheoryMetrics(config={'k_clusters': 10, 'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        contexts = ['baseline', 'determiner_the']
        stratified = analysis._stratified_analysis(contexts)
        
        # Check structure
        assert 'by_frequency' in stratified
        assert 'by_type' in stratified
        
        # Check frequency strata
        for stratum, data in stratified['by_frequency'].items():
            assert 'n_tokens' in data
            assert 'avg_divergence' in data
            assert data['n_tokens'] > 0
            assert data['avg_divergence'] >= 0
            
    @pytest.mark.integration
    def test_full_analysis(self, mock_data_loader, temp_dir):
        """Test complete analysis pipeline."""
        analysis = InformationTheoryMetrics(
            output_dir=str(temp_dir),
            config={
                'k_clusters': 10,
                'n_bootstrap': 10,  # Small for testing
                'contexts_to_analyze': ['baseline', 'determiner_the'],
                'enable_logging': False
            }
        )
        
        analysis.data_loader = mock_data_loader
        output = analysis.run()
        
        # Check output structure
        assert hasattr(output, 'metadata')
        assert hasattr(output, 'data')
        assert hasattr(output, 'statistics')
        assert hasattr(output, 'summary')
        
        # Check all components present
        data = output.data
        assert 'mutual_information' in data
        assert 'kl_divergence' in data
        assert 'entropy_metrics' in data
        assert 'jensen_shannon' in data
        
        # Check files created
        assert (temp_dir / "information_theory_metrics_results.json").exists()
        
    @pytest.mark.unit
    def test_validation(self, mock_data_loader):
        """Test data validation."""
        analysis = InformationTheoryMetrics(config={'enable_logging': False})
        
        # Should fail without data
        with pytest.raises(ValueError, match="No trajectory data"):
            analysis.validate_data()
            
        # Load data and validate
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        analysis.validate_data()  # Should pass
        
    @pytest.mark.unit
    def test_edge_cases(self):
        """Test edge cases in calculations."""
        analysis = InformationTheoryMetrics(config={'enable_logging': False})
        
        # Empty distribution
        dist = analysis._get_context_distribution('nonexistent')
        assert len(dist) > 0  # Should return zeros, not empty
        assert np.all(dist >= 0)
        
        # Single token trajectory
        analysis.trajectories = {'0_baseline': {'path': [0]}}
        matrix = analysis._build_transition_matrix('baseline')
        assert matrix.shape[0] == matrix.shape[1]
        
    @pytest.mark.unit 
    def test_confidence_intervals(self, mock_data_loader):
        """Test bootstrap confidence intervals."""
        analysis = InformationTheoryMetrics(
            config={'n_bootstrap': 50, 'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Calculate a metric with CI
        contexts = ['baseline', 'determiner_the']
        kl_results = analysis._calculate_kl_divergence(contexts)
        
        # Check CIs are present (or None for now)
        for context, metric in kl_results['context_kl'].items():
            assert hasattr(metric, 'ci')
            ci = metric.ci
            # TODO: Implement proper bootstrap CIs
            # For now, just check the structure is correct
            if ci is not None:
                assert ci.lower <= metric.value <= ci.upper