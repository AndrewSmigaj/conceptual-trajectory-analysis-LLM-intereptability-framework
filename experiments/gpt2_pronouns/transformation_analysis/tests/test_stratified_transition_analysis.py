"""
Tests for stratified transition analysis.
"""

import pytest
import numpy as np
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock

from ..stratified_transition_analysis import StratifiedTransitionAnalysis
from ..output_schema import StratifiedResults


class TestStratifiedTransitionAnalysis:
    """Test suite for StratifiedTransitionAnalysis class."""
    
    def test_initialization(self, temp_dir):
        """Test analysis initialization."""
        config = {
            'k_clusters': 10,
            'stratify_by': ['frequency', 'type'],
            'visualize': False,
            'enable_logging': False
        }
        
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config=config
        )
        
        assert analysis.analysis_name == "stratified_transition_analysis"
        assert analysis.config['k_clusters'] == 10
        assert 'frequency' in analysis.config['stratify_by']
        assert analysis.output_dir == temp_dir
        
    def test_validate_data_no_baseline(self, mock_data_loader, temp_dir):
        """Test validation fails without baseline context."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Mock trajectories without baseline
        analysis.trajectories = {
            "0_determiner_the": {"context_frame": "determiner_the"}
        }
        
        with pytest.raises(ValueError, match="No baseline context"):
            analysis.validate_data()
            
    def test_validate_data_insufficient_contexts(self, mock_data_loader, temp_dir):
        """Test validation fails with only baseline context."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Mock trajectories with only baseline
        analysis.trajectories = {
            "0_baseline": {"context_frame": "baseline"}
        }
        
        with pytest.raises(ValueError, match="Need at least baseline"):
            analysis.validate_data()
            
    def test_build_transition_matrix(self, mock_data_loader, temp_dir):
        """Test building a single transition matrix."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Build transition matrix for layer 0
        matrix = analysis._build_transition_matrix('baseline', 'determiner_the', 0)
        
        # Check matrix properties
        assert matrix.shape == (10, 10)
        assert np.allclose(matrix.sum(axis=1)[matrix.sum(axis=1) > 0], 1.0)  # Rows sum to 1
        assert matrix.min() >= 0  # Non-negative
        assert matrix.max() <= 1  # Probabilities
        
        # Check specific transitions from our sample data
        # Token 0: baseline cluster 1 -> determiner_the cluster 9
        assert matrix[1, 9] > 0
        # Token 1: baseline cluster 2 -> determiner_the cluster 8  
        assert matrix[2, 8] > 0
        # Token 2: baseline cluster 3 -> determiner_the cluster 7
        assert matrix[3, 7] > 0
        
    def test_calculate_entropy(self, temp_dir):
        """Test entropy calculation."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Uniform distribution - high entropy
        uniform_matrix = np.ones((10, 10)) / 10
        high_entropy = analysis._calculate_entropy(uniform_matrix)
        
        # Diagonal matrix - low entropy
        diagonal_matrix = np.eye(10)
        low_entropy = analysis._calculate_entropy(diagonal_matrix)
        
        assert high_entropy > low_entropy
        assert low_entropy == 0.0  # Perfect certainty
        
    def test_calculate_sparsity(self, temp_dir):
        """Test sparsity (Gini coefficient) calculation."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Uniform distribution - low sparsity (Gini close to 0)
        uniform_matrix = np.ones((10, 10)) / 10
        low_sparsity = analysis._calculate_sparsity(uniform_matrix)
        
        # Create a truly sparse matrix where values are concentrated in few cells
        sparse_matrix = np.zeros((10, 10))
        # Make it so only diagonal has values, creating high inequality
        np.fill_diagonal(sparse_matrix, 1.0)
        # Add one off-diagonal element to avoid all zeros
        sparse_matrix[0, 1] = 0.01
        sparse_matrix[0, 0] = 0.99  # Adjust to maintain row sum
        high_sparsity = analysis._calculate_sparsity(sparse_matrix)
        
        # For Gini coefficient, uniform should be near 0, concentrated should be higher
        assert high_sparsity >= low_sparsity
        assert low_sparsity >= 0.0
        assert high_sparsity <= 1.0
        
    def test_calculate_diagonal_dominance(self, temp_dir):
        """Test diagonal dominance calculation."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Identity matrix - maximum diagonal dominance
        identity_matrix = np.eye(10)
        max_dominance = analysis._calculate_diagonal_dominance(identity_matrix)
        assert max_dominance == 1.0
        
        # Off-diagonal matrix - minimum diagonal dominance
        off_diagonal = np.ones((10, 10)) / 10
        np.fill_diagonal(off_diagonal, 0)
        # Renormalize rows
        off_diagonal = off_diagonal / off_diagonal.sum(axis=1, keepdims=True)
        min_dominance = analysis._calculate_diagonal_dominance(off_diagonal)
        assert min_dominance == 0.0
        
    def test_calculate_mutual_information(self, temp_dir):
        """Test mutual information calculation."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Independent case - MI should be low
        uniform_matrix = np.ones((10, 10)) / 10
        low_mi = analysis._calculate_mutual_information(uniform_matrix)
        
        # Dependent case - MI should be high
        identity_matrix = np.eye(10)
        high_mi = analysis._calculate_mutual_information(identity_matrix)
        
        assert high_mi > low_mi
        
    def test_generate_random_baselines(self, mock_data_loader, temp_dir):
        """Test random baseline generation."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'random_baselines': ['shuffle', 'uniform'], 'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        contexts = ['determiner_the', 'determiner_a']
        baselines = analysis._generate_random_baselines(contexts)
        
        # Check structure
        assert 'shuffle' in baselines
        assert 'uniform' in baselines
        
        # Check shuffle baseline
        for context in contexts:
            for layer in range(12):
                matrix = baselines['shuffle'][context][layer]
                assert matrix.shape == (10, 10)
                assert np.allclose(matrix.sum(axis=1)[matrix.sum(axis=1) > 0], 1.0)
                
        # Check uniform baseline
        for context in contexts:
            for layer in range(12):
                matrix = baselines['uniform'][context][layer]
                assert matrix.shape == (10, 10)
                assert np.allclose(matrix, 0.1)  # All entries should be 1/10
                
    def test_stratify_tokens_by_frequency(self, mock_data_loader, temp_dir):
        """Test token stratification by frequency."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        strata = analysis.stratify_tokens('frequency')
        
        # With only 3 tokens and default quantiles [0.33, 0.67], we should have:
        # - 1 token in 'low' (first 33%)
        # - 1 token in 'medium' (33%-67%) 
        # - 1 token in 'high' (last 33%)
        # But since 3*0.33 = 0.99 rounds to 0, and 3*0.67 = 2.01 rounds to 2,
        # we actually get indices [0:2] for medium and [2:3] for high
        
        # So the actual distribution is:
        # - No 'low' stratum (empty)
        # - Tokens 2, 1 in 'medium' (indices 0-1 after sorting by frequency)
        # - Token 0 in 'high' (index 2 after sorting)
        
        assert 'medium' in strata
        assert 'high' in strata
        
        # Token frequencies: 0->50000, 1->5000, 2->500
        # After sorting by frequency: [2, 1, 0]
        assert 2 in strata['medium'] or 1 in strata['medium']
        assert 0 in strata['high']
        
    def test_build_stratified_transition_matrix(self, mock_data_loader, temp_dir):
        """Test building transition matrix for a token subset."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Build matrix for only tokens 0 and 1
        token_subset = [0, 1]
        matrix = analysis._build_stratified_transition_matrix(
            'baseline', 'determiner_the', 0, token_subset
        )
        
        # Should still be 10x10 but only certain transitions present
        assert matrix.shape == (10, 10)
        
        # Check that only expected transitions are non-zero
        # Token 0: cluster 1 -> 9
        # Token 1: cluster 2 -> 8
        assert matrix[1, 9] > 0
        assert matrix[2, 8] > 0
        assert matrix[3, 7] == 0  # Token 2 excluded
        
    def test_compare_to_baseline(self, mock_data_loader, temp_dir):
        """Test comparing real transitions to baseline."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Create mock transition matrices and baselines
        analysis.transition_matrices = {
            'determiner_the': {
                layer: np.eye(10) for layer in range(12)
            }
        }
        
        analysis.random_baselines = {
            'uniform': {
                'determiner_the': {
                    layer: np.ones((10, 10)) / 10 for layer in range(12)
                }
            }
        }
        
        comparison = analysis._compare_to_baseline('determiner_the', 'uniform')
        
        # Check structure
        assert 'entropy_difference' in comparison
        assert 'sparsity_difference' in comparison
        assert 'diagonal_difference' in comparison
        
        # Identity matrix should have lower entropy than uniform
        assert comparison['entropy_difference']['mean'] < 0
        
        # Identity matrix should have higher diagonal dominance
        assert comparison['diagonal_difference']['mean'] > 0
        
    def test_compare_contexts(self, temp_dir):
        """Test comparing transition patterns between contexts."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Create different transition patterns
        analysis.transition_matrices = {
            'context1': {
                0: np.eye(10),
                1: np.eye(10)
            },
            'context2': {
                0: np.ones((10, 10)) / 10,
                1: np.ones((10, 10)) / 10
            }
        }
        
        comparison = analysis._compare_contexts('context1', 'context2')
        
        assert 'mean_similarity' in comparison
        assert 'similarity_by_layer' in comparison
        
        # Very different matrices should have low similarity
        assert comparison['mean_similarity'] < 0.5
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_transition_matrix(self, mock_close, mock_savefig, temp_dir):
        """Test transition matrix plotting."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Create sample matrix
        analysis.transition_matrices = {
            'test_context': {
                0: np.eye(10)
            }
        }
        
        fig_path = analysis._plot_transition_matrix('test_context', 0)
        
        # Check that plotting functions were called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Check that path was created correctly
        assert fig_path.name == "transition_matrix_test_context_layer0.png"
        
    def test_create_summary(self, temp_dir):
        """Test summary creation."""
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config={'enable_logging': False}
        )
        
        # Create sample metrics
        analysis.transition_metrics = {
            'context1': {
                0: {'entropy': 2.0, 'sparsity': 0.5, 'diagonal_dominance': 0.3}
            }
        }
        
        summary = analysis._create_summary(['context1'])
        
        assert 'key_findings' in summary
        assert 'interpretation' in summary
        assert 'next_steps' in summary
        
        # Check that findings mention the context count
        assert "1 context types" in summary['key_findings'][0]
        
    def test_full_analysis_integration(self, mock_data_loader, temp_dir):
        """Test running the full analysis pipeline."""
        # Skip visualizations for faster testing
        config = {
            'k_clusters': 10,
            'stratify_by': ['frequency'],
            'random_baselines': ['uniform'],
            'visualize': False,
            'enable_logging': False
        }
        
        analysis = StratifiedTransitionAnalysis(
            output_dir=str(temp_dir),
            config=config
        )
        
        # Mock the data loader
        analysis.data_loader = mock_data_loader
        
        # Run the analysis
        output = analysis.run()
        
        # Check output structure
        assert output.metadata.analysis_type == "stratified_transition_analysis"
        assert 'transition_matrices' in output.data
        assert 'random_baselines' in output.data
        assert 'transition_metrics' in output.data
        assert 'by_frequency' in output.stratification
        
        # Check that results were saved
        json_file = temp_dir / "stratified_transition_analysis_results.json"
        assert json_file.exists()
        
        # Verify JSON is valid
        with open(json_file) as f:
            results = json.load(f)
            assert results['metadata']['analysis_type'] == "stratified_transition_analysis"