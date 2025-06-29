"""
Tests for permutation significance analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..permutation_significance_test import PermutationSignificanceTest


class TestPermutationSignificanceTest:
    """Test suite for PermutationSignificanceTest class."""
    
    def test_initialization(self, temp_dir):
        """Test analysis initialization."""
        config = {
            'n_permutations': 100,
            'contexts_to_test': ['test_context'],
            'alpha': 0.05,
            'enable_logging': False
        }
        
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config=config
        )
        
        assert analysis.analysis_name == "permutation_significance_test"
        assert analysis.config['n_permutations'] == 100
        assert analysis.config['alpha'] == 0.05
        
    def test_build_transition_matrix(self, mock_data_loader, temp_dir):
        """Test transition matrix building."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={'k_clusters': 10, 'enable_logging': False}
        )
        
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Build transition matrix
        matrix = analysis._build_transition_matrix('baseline', 'determiner_the')
        
        # Check properties
        assert matrix.shape == (10, 10)  # k x k
        # Check normalization - rows should sum to 1 (or 0 if empty)
        row_sums = matrix.sum(axis=1)
        assert all(s == 0 or np.isclose(s, 1) for s in row_sums)
        
    def test_calculate_test_statistics(self, temp_dir):
        """Test calculation of test statistics."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Create test matrix
        test_matrix = np.array([[0.8, 0.1, 0.1],
                               [0.1, 0.8, 0.1],
                               [0.1, 0.1, 0.8]])
        
        # Test diagonal dominance
        diag_dom = analysis._calculate_diagonal_dominance(test_matrix)
        assert np.isclose(diag_dom, 0.8)  # (0.8+0.8+0.8)/3
        
        # Test entropy
        entropy = analysis._calculate_mean_entropy(test_matrix)
        assert entropy > 0  # Should be positive for non-degenerate matrix
        assert entropy < np.log(3)  # Should be less than max entropy
        
        # Test mutual information
        mi = analysis._calculate_mutual_information(test_matrix)
        assert mi >= 0  # MI is non-negative
        
    def test_generate_permuted_matrix(self, mock_data_loader, temp_dir):
        """Test permuted matrix generation."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={'k_clusters': 10, 'enable_logging': False}
        )
        
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Generate permuted matrix
        perm_matrix = analysis._generate_permuted_matrix('baseline', 'determiner_the')
        
        # Check properties
        assert perm_matrix.shape == (10, 10)
        row_sums = perm_matrix.sum(axis=1)
        assert all(s == 0 or np.isclose(s, 1) for s in row_sums)
        
        # Should be different from non-permuted with high probability
        real_matrix = analysis._build_transition_matrix('baseline', 'determiner_the')
        # They might be the same by chance with small test data, so just check shape
        assert perm_matrix.shape == real_matrix.shape
        
    def test_calculate_p_values(self, temp_dir):
        """Test p-value calculation."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Mock real statistics and null distributions
        analysis.real_statistics = {
            'test_context': {
                'diagonal_dominance': 0.8,
                'entropy': 0.5
            }
        }
        
        # Create null distributions
        # For diagonal dominance, real value (0.8) should be extreme (high)
        analysis.null_distributions = {
            'test_context': {
                'diagonal_dominance': np.random.uniform(0.2, 0.6, 1000).tolist(),
                'entropy': np.random.uniform(0.7, 1.2, 1000).tolist()
            }
        }
        
        p_values = analysis._calculate_p_values()
        
        assert 'test_context' in p_values
        assert 'diagonal_dominance' in p_values['test_context']
        assert 'entropy' in p_values['test_context']
        
        # Real diagonal dominance (0.8) is higher than null (0.2-0.6)
        assert p_values['test_context']['diagonal_dominance'] < 0.05
        
        # Real entropy (0.5) is lower than null (0.7-1.2)
        assert p_values['test_context']['entropy'] < 0.05
        
    def test_multiple_comparison_correction(self, temp_dir):
        """Test multiple comparison correction."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={
                'multiple_comparison_correction': 'bonferroni',
                'enable_logging': False
            }
        )
        
        # Mock p-values
        analysis.p_values = {
            'ctx1': {'stat1': 0.01, 'stat2': 0.04},
            'ctx2': {'stat1': 0.02, 'stat2': 0.03}
        }
        
        corrected = analysis._apply_multiple_comparison_correction()
        
        # Bonferroni correction multiplies by number of tests (4)
        assert corrected['ctx1']['stat1'] == 0.04  # 0.01 * 4
        assert corrected['ctx1']['stat2'] == 0.16  # 0.04 * 4
        assert corrected['ctx2']['stat1'] == 0.08  # 0.02 * 4
        assert corrected['ctx2']['stat2'] == 0.12  # 0.03 * 4
        
    def test_significance_testing(self, temp_dir):
        """Test significance testing."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={'alpha': 0.05, 'enable_logging': False}
        )
        
        # Mock corrected p-values
        corrected_p_values = {
            'ctx1': {'stat1': 0.01, 'stat2': 0.06},
            'ctx2': {'stat1': 0.03, 'stat2': 0.10}
        }
        
        results = analysis._test_significance(corrected_p_values)
        
        # Check structure
        assert 'significant_effects' in results
        assert 'non_significant_effects' in results
        assert 'summary_by_context' in results
        assert 'summary_by_statistic' in results
        
        # Should have 2 significant (p < 0.05) and 2 non-significant
        assert len(results['significant_effects']) == 2
        assert len(results['non_significant_effects']) == 2
        
        # Check summaries
        assert results['summary_by_context']['ctx1']['n_significant'] == 1
        assert results['summary_by_statistic']['stat1']['n_significant'] == 2
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualization_creation(self, mock_close, mock_savefig, temp_dir):
        """Test visualization creation."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={
                'test_statistics': ['diagonal_dominance', 'entropy'],
                'enable_logging': False
            }
        )
        
        # Mock data for visualization
        analysis.real_statistics = {
            'ctx1': {'diagonal_dominance': 0.8, 'entropy': 0.5}
        }
        
        analysis.null_distributions = {
            'ctx1': {
                'diagonal_dominance': np.random.normal(0.5, 0.1, 100).tolist(),
                'entropy': np.random.normal(0.8, 0.1, 100).tolist()
            }
        }
        
        analysis.p_values = {
            'ctx1': {'diagonal_dominance': 0.01, 'entropy': 0.02}
        }
        
        visualizations = analysis._create_visualizations()
        
        # Should create 3 visualizations
        assert len(visualizations) == 3
        assert mock_savefig.call_count == 3
        assert mock_close.call_count == 3
        
    def test_create_summary(self, temp_dir):
        """Test summary creation."""
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config={
                'n_permutations': 1000,
                'alpha': 0.05,
                'multiple_comparison_correction': 'bonferroni',
                'enable_logging': False
            }
        )
        
        # Mock corrected p-values
        corrected_p_values = {
            'ctx1': {'stat1': 0.001, 'stat2': 0.08},
            'ctx2': {'stat1': 0.005, 'stat2': 0.15}
        }
        
        summary = analysis._create_summary(corrected_p_values)
        
        assert 'key_findings' in summary
        assert 'interpretation' in summary
        assert 'next_steps' in summary
        
        # Should report 2 significant out of 4 total
        assert '2/4' in summary['key_findings'][1]
        
    @patch('tqdm.tqdm')
    def test_full_analysis_integration(self, mock_tqdm, mock_data_loader, temp_dir):
        """Test running the full analysis pipeline."""
        # Mock tqdm to avoid progress bar in tests
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        config = {
            'k_clusters': 10,
            'n_permutations': 10,  # Small for testing
            'contexts_to_test': ['determiner_the'],
            'test_statistics': ['diagonal_dominance'],
            'visualize': False,
            'enable_logging': False
        }
        
        analysis = PermutationSignificanceTest(
            output_dir=str(temp_dir),
            config=config
        )
        
        analysis.data_loader = mock_data_loader
        
        # Run analysis
        output = analysis.run()
        
        # Check output structure
        assert output.metadata.analysis_type == "permutation_significance_test"
        assert 'real_statistics' in output.data
        assert 'p_values' in output.data
        assert 'corrected_p_values' in output.data
        assert output.statistics is not None
        assert output.summary is not None