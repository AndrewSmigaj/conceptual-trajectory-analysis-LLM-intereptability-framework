"""
Tests for bootstrap utility functions.
"""

import pytest
import numpy as np
from scipy import stats

from ..bootstrap_utils import (
    bootstrap_sample,
    calculate_bootstrap_statistic,
    percentile_method,
    bca_method,
    basic_method,
    bootstrap_confidence_interval,
    bootstrap_matrix_ci,
    BootstrapMethod
)


class TestBootstrapUtils:
    """Test bootstrap utility functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=100)
    
    @pytest.fixture
    def known_distribution_data(self):
        """Create data from known distribution for CI testing"""
        np.random.seed(42)
        # Normal distribution with known mean=50, std=10
        return np.random.normal(loc=50, scale=10, size=1000)
    
    def test_bootstrap_sample(self, sample_data):
        """Test bootstrap sample generation"""
        n_bootstrap = 100
        indices = bootstrap_sample(sample_data, n_bootstrap, random_state=42)
        
        assert indices.shape == (n_bootstrap, len(sample_data))
        assert np.all(indices >= 0)
        assert np.all(indices < len(sample_data))
        
        # Check reproducibility
        indices2 = bootstrap_sample(sample_data, n_bootstrap, random_state=42)
        np.testing.assert_array_equal(indices, indices2)
    
    def test_calculate_bootstrap_statistic(self, sample_data):
        """Test bootstrap statistic calculation"""
        n_bootstrap = 50
        indices = bootstrap_sample(sample_data, n_bootstrap, random_state=42)
        
        # Test with mean
        boot_means = calculate_bootstrap_statistic(sample_data, np.mean, indices)
        assert boot_means.shape == (n_bootstrap,)
        assert np.all(np.isfinite(boot_means))
        
        # Test with custom function
        def custom_stat(x):
            return np.array([np.mean(x), np.std(x)])
        
        boot_stats = calculate_bootstrap_statistic(sample_data, custom_stat, indices)
        assert boot_stats.shape == (n_bootstrap, 2)
    
    def test_percentile_method(self, sample_data):
        """Test percentile CI method"""
        n_bootstrap = 1000
        indices = bootstrap_sample(sample_data, n_bootstrap, random_state=42)
        boot_means = calculate_bootstrap_statistic(sample_data, np.mean, indices)
        
        lower, upper = percentile_method(boot_means, alpha=0.05)
        
        assert lower < upper
        assert lower < np.mean(sample_data) < upper
        
        # Test different alpha
        lower_90, upper_90 = percentile_method(boot_means, alpha=0.10)
        assert lower < lower_90 < upper_90 < upper  # 90% CI should be narrower
    
    def test_bca_method(self, sample_data):
        """Test BCa CI method"""
        n_bootstrap = 200  # Smaller for faster test
        indices = bootstrap_sample(sample_data, n_bootstrap, random_state=42)
        boot_means = calculate_bootstrap_statistic(sample_data, np.mean, indices)
        
        lower, upper = bca_method(
            sample_data, np.mean, boot_means, alpha=0.05
        )
        
        assert lower < upper
        assert np.isfinite(lower) and np.isfinite(upper)
        
        # BCa should give similar but potentially different bounds than percentile
        lower_p, upper_p = percentile_method(boot_means, alpha=0.05)
        assert abs(lower - lower_p) < 2  # Should be relatively close
        assert abs(upper - upper_p) < 2
    
    def test_basic_method(self, sample_data):
        """Test basic bootstrap CI method"""
        n_bootstrap = 1000
        indices = bootstrap_sample(sample_data, n_bootstrap, random_state=42)
        boot_means = calculate_bootstrap_statistic(sample_data, np.mean, indices)
        
        theta = np.mean(sample_data)
        lower, upper = basic_method(theta, boot_means, alpha=0.05)
        
        assert lower < upper
        assert lower < theta < upper
    
    def test_bootstrap_confidence_interval(self, sample_data):
        """Test main bootstrap CI function"""
        # Test percentile method
        value, (lower, upper) = bootstrap_confidence_interval(
            sample_data,
            np.mean,
            n_bootstrap=1000,
            alpha=0.05,
            method=BootstrapMethod.PERCENTILE,
            random_state=42
        )
        
        assert value == pytest.approx(np.mean(sample_data))
        assert lower < value < upper
        
        # Test BCa method
        value_bca, (lower_bca, upper_bca) = bootstrap_confidence_interval(
            sample_data,
            np.mean,
            n_bootstrap=200,
            alpha=0.05,
            method=BootstrapMethod.BCA,
            random_state=42
        )
        
        assert value_bca == pytest.approx(value)
        assert lower_bca < value_bca < upper_bca
        
        # Test with progress bar (just check it doesn't crash)
        _, _ = bootstrap_confidence_interval(
            sample_data,
            np.mean,
            n_bootstrap=10,
            show_progress=True
        )
    
    def test_bootstrap_vector_statistic(self, sample_data):
        """Test bootstrap with vector-valued statistics"""
        def quartiles(x):
            return np.percentile(x, [25, 50, 75])
        
        value, (lower, upper) = bootstrap_confidence_interval(
            sample_data,
            quartiles,
            n_bootstrap=500,
            random_state=42
        )
        
        assert value.shape == (3,)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert np.all(lower < value)
        assert np.all(value < upper)
    
    def test_bootstrap_matrix_ci(self):
        """Test bootstrap CI for matrix elements"""
        # Create a simple transition matrix
        matrix = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.3, 0.6]
        ])
        
        orig, lower, upper = bootstrap_matrix_ci(
            matrix,
            n_bootstrap=100,
            random_state=42
        )
        
        assert orig.shape == matrix.shape
        assert lower.shape == matrix.shape
        assert upper.shape == matrix.shape
        
        # Check bounds make sense (allowing for some tolerance due to randomness)
        # The CIs are approximate and based on synthetic data
        tolerance = 0.05
        assert np.all(lower <= orig + tolerance)
        assert np.all(orig - tolerance <= upper)
        
        # Check that CIs are reasonable width
        widths = upper - lower
        assert np.all(widths > 0)
        assert np.all(widths < 0.5)  # Not too wide
    
    def test_coverage_probability(self, known_distribution_data):
        """Test that 95% CI contains true mean ~95% of the time"""
        # This is a statistical test, so we allow some tolerance
        true_mean = 50.0
        n_trials = 100
        n_covered = 0
        
        for i in range(n_trials):
            # Sample from the data
            sample_indices = np.random.choice(
                len(known_distribution_data), size=50, replace=False
            )
            sample = known_distribution_data[sample_indices]
            
            # Calculate CI
            _, (lower, upper) = bootstrap_confidence_interval(
                sample,
                np.mean,
                n_bootstrap=200,
                alpha=0.05,
                random_state=i
            )
            
            # Check if true mean is covered
            if lower <= true_mean <= upper:
                n_covered += 1
        
        coverage = n_covered / n_trials
        # Should be close to 0.95 (allow 0.85-1.0 for small sample)
        assert 0.85 <= coverage <= 1.0
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Constant data
        const_data = np.ones(10)
        value, (lower, upper) = bootstrap_confidence_interval(
            const_data,
            np.mean,
            n_bootstrap=100
        )
        assert value == 1.0
        assert lower == 1.0
        assert upper == 1.0
        
        # Single value
        single_data = np.array([5.0])
        value, (lower, upper) = bootstrap_confidence_interval(
            single_data,
            np.mean,
            n_bootstrap=100
        )
        assert value == 5.0
        
        # Empty data should raise error
        with pytest.raises(ValueError, match="Cannot calculate bootstrap CI for empty data"):
            bootstrap_confidence_interval(
                np.array([]),
                np.mean,
                n_bootstrap=100
            )
    
    def test_bootstrap_methods_enum(self):
        """Test BootstrapMethod enum"""
        assert BootstrapMethod.PERCENTILE.value == "percentile"
        assert BootstrapMethod.BCA.value == "bca"
        assert BootstrapMethod.BASIC.value == "basic"
        assert BootstrapMethod.STUDENTIZED.value == "studentized"
    
    def test_custom_statistic(self, sample_data):
        """Test with custom statistic function"""
        def trimmed_mean(x, trim=0.1):
            return stats.trim_mean(x, trim)
        
        value, (lower, upper) = bootstrap_confidence_interval(
            sample_data,
            trimmed_mean,
            n_bootstrap=500,
            trim=0.1
        )
        
        assert lower < value < upper
        # Trimmed mean should be close to regular mean for normal data
        assert abs(value - np.mean(sample_data)) < 0.5