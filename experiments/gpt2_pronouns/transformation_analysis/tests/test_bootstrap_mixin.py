"""
Tests for BootstrapMixin class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..bootstrap_mixin import BootstrapMixin
from ..bootstrap_utils import BootstrapMethod
from ..output_schema import UnifiedAnalysisOutput, AnalysisMetadata, MetricWithCI
from datetime import datetime


class MockAnalysis(BootstrapMixin):
    """Mock analysis class for testing mixin"""
    def __init__(self, config=None):
        self.config = config or {}
        super().__init__()


class TestBootstrapMixin:
    """Test BootstrapMixin functionality"""
    
    @pytest.fixture
    def analysis(self):
        """Create analysis instance with mixin"""
        return MockAnalysis()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return np.random.normal(10, 2, 100)
    
    def test_initialization(self, analysis):
        """Test mixin initialization"""
        assert hasattr(analysis, 'bootstrap_config')
        assert analysis.bootstrap_config['n_bootstrap'] == 1000
        assert analysis.bootstrap_config['confidence_level'] == 0.95
        assert analysis.bootstrap_config['method'] == BootstrapMethod.PERCENTILE
        assert hasattr(analysis, '_bootstrap_cache')
    
    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        config = {
            'bootstrap': {
                'n_bootstrap': 500,
                'confidence_level': 0.90,
                'method': BootstrapMethod.BCA
            }
        }
        analysis = MockAnalysis(config)
        
        assert analysis.bootstrap_config['n_bootstrap'] == 500
        assert analysis.bootstrap_config['confidence_level'] == 0.90
        assert analysis.bootstrap_config['method'] == BootstrapMethod.BCA
    
    def test_bootstrap_metric(self, analysis, sample_data):
        """Test single metric bootstrap"""
        result = analysis.bootstrap_metric(
            sample_data,
            np.mean,
            metric_name="test_mean"
        )
        
        assert isinstance(result, MetricWithCI)
        assert result.value == pytest.approx(np.mean(sample_data))
        assert result.ci is not None
        assert result.ci.lower < result.value < result.ci.upper
        assert result.ci.confidence_level == 0.95
        assert result.ci.method == "percentile"
        assert result.ci.n_bootstrap == 1000
    
    def test_bootstrap_metric_caching(self, analysis, sample_data):
        """Test that metrics are cached"""
        # First call
        result1 = analysis.bootstrap_metric(
            sample_data,
            np.mean,
            metric_name="cached_mean"
        )
        
        # Second call with same data and name
        result2 = analysis.bootstrap_metric(
            sample_data,
            np.mean,
            metric_name="cached_mean"
        )
        
        # Should be the same object (cached)
        assert result1 is result2
        assert len(analysis._bootstrap_cache) == 1
    
    def test_bootstrap_multiple_metrics(self, analysis, sample_data):
        """Test multiple metrics calculation"""
        metrics = {
            'mean': np.mean,
            'std': np.std,
            'median': np.median
        }
        
        results = analysis.bootstrap_multiple_metrics(
            sample_data,
            metrics
        )
        
        assert len(results) == 3
        assert all(isinstance(r, MetricWithCI) for r in results.values())
        
        # Check values are reasonable
        assert results['mean'].value == pytest.approx(np.mean(sample_data))
        assert results['std'].value == pytest.approx(np.std(sample_data))
        assert results['median'].value == pytest.approx(np.median(sample_data))
        
        # Check CIs exist
        for metric_name, result in results.items():
            assert result.ci is not None
            assert result.ci.lower < result.value < result.ci.upper
    
    def test_bootstrap_matrix_metrics(self, analysis):
        """Test matrix metrics bootstrap"""
        matrix = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        
        metrics = {
            'trace': np.trace,
            'determinant': np.linalg.det,
            'frobenius_norm': np.linalg.norm
        }
        
        results = analysis.bootstrap_matrix_metrics(matrix, metrics)
        
        assert len(results) == 3
        assert results['trace'].value == pytest.approx(1.5)
        assert results['determinant'].value == pytest.approx(0.5)
        
        # Note: Matrix bootstrap returns approximate CIs
        for result in results.values():
            assert result.ci.method == "approximate"
    
    def test_add_confidence_intervals_to_output(self, analysis, sample_data):
        """Test adding CIs to output"""
        # Create mock output
        output = UnifiedAnalysisOutput(
            metadata=AnalysisMetadata(
                analysis_type="test",
                timestamp=datetime.now(),
                version="1.0",
                parameters={}
            ),
            data={},
            statistics={},
            summary={}
        )
        
        # Calculate some metrics
        analysis.bootstrap_metric(sample_data, np.mean, "test_metric")
        
        # Add to output
        analysis.add_confidence_intervals_to_output(output)
        
        assert hasattr(output, 'confidence_intervals')
        assert 'test_metric' in output.confidence_intervals
        assert 'value' in output.confidence_intervals['test_metric']
        assert 'confidence_interval' in output.confidence_intervals['test_metric']
    
    def test_clear_bootstrap_cache(self, analysis, sample_data):
        """Test cache clearing"""
        # Add some cached results
        analysis.bootstrap_metric(sample_data, np.mean, "metric1")
        analysis.bootstrap_metric(sample_data, np.std, "metric2")
        
        assert len(analysis._bootstrap_cache) == 2
        
        # Clear cache
        analysis.clear_bootstrap_cache()
        
        assert len(analysis._bootstrap_cache) == 0
    
    def test_set_bootstrap_config(self, analysis):
        """Test updating bootstrap configuration"""
        analysis.set_bootstrap_config(
            n_bootstrap=2000,
            confidence_level=0.99,
            show_progress=True
        )
        
        assert analysis.bootstrap_config['n_bootstrap'] == 2000
        assert analysis.bootstrap_config['confidence_level'] == 0.99
        assert analysis.bootstrap_config['show_progress'] == True
        
        # Test warning for unknown parameter
        with patch('logging.Logger.warning') as mock_warning:
            analysis.set_bootstrap_config(unknown_param=123)
            mock_warning.assert_called_once()
    
    def test_bootstrap_stratified_metric(self, analysis):
        """Test stratified bootstrap"""
        np.random.seed(42)
        data_dict = {
            'low': np.random.normal(5, 1, 50),
            'medium': np.random.normal(10, 1, 50),
            'high': np.random.normal(15, 1, 50)
        }
        
        results = analysis.bootstrap_stratified_metric(
            data_dict,
            np.mean,
            aggregate_func=np.mean
        )
        
        assert len(results) == 4  # 3 strata + aggregate
        assert 'low' in results
        assert 'medium' in results
        assert 'high' in results
        assert 'aggregate' in results
        
        # Check values are ordered correctly
        assert results['low'].value < results['medium'].value < results['high'].value
        
        # Check aggregate is reasonable
        assert 8 < results['aggregate'].value < 12
    
    def test_bootstrap_with_kwargs(self, analysis, sample_data):
        """Test bootstrap with additional kwargs for metric function"""
        def weighted_mean(x, weights):
            return np.average(x, weights=weights)
        
        weights = np.ones(len(sample_data))
        weights[:50] = 2  # Give more weight to first half
        
        result = analysis.bootstrap_metric(
            sample_data,
            weighted_mean,
            weights=weights
        )
        
        assert isinstance(result, MetricWithCI)
        assert result.ci is not None
    
    def test_metric_with_ci_serialization(self, analysis, sample_data):
        """Test MetricWithCI serialization"""
        result = analysis.bootstrap_metric(sample_data, np.mean)
        
        # Test to_dict
        result_dict = result.to_dict()
        
        assert 'value' in result_dict
        assert 'confidence_interval' in result_dict
        assert 'lower' in result_dict['confidence_interval']
        assert 'upper' in result_dict['confidence_interval']
        assert 'confidence_level' in result_dict['confidence_interval']
        assert 'method' in result_dict['confidence_interval']
        assert 'n_bootstrap' in result_dict['confidence_interval']
        
        # Check types for JSON serialization
        assert isinstance(result_dict['value'], float)
        assert isinstance(result_dict['confidence_interval']['lower'], float)
        assert isinstance(result_dict['confidence_interval']['upper'], float)
    
    def test_confidence_interval_methods(self, analysis, sample_data):
        """Test ConfidenceInterval methods"""
        result = analysis.bootstrap_metric(sample_data, np.mean)
        ci = result.ci
        
        # Test contains method
        assert ci.contains(result.value)
        assert not ci.contains(result.value - 100)
        assert not ci.contains(result.value + 100)
        
        # Test width method
        width = ci.width()
        assert width > 0
        assert width == pytest.approx(ci.upper - ci.lower)
    
    def test_different_bootstrap_methods(self, analysis, sample_data):
        """Test different bootstrap methods"""
        # Test BCa method
        analysis.set_bootstrap_config(
            method=BootstrapMethod.BCA,
            n_bootstrap=200  # Fewer for BCa (slower)
        )
        
        result_bca = analysis.bootstrap_metric(
            sample_data,
            np.mean,
            metric_name="bca_test"
        )
        
        assert result_bca.ci.method == "bca"
        
        # Test basic method
        analysis.set_bootstrap_config(method=BootstrapMethod.BASIC)
        analysis.clear_bootstrap_cache()
        
        result_basic = analysis.bootstrap_metric(
            sample_data,
            np.mean,
            metric_name="basic_test"
        )
        
        assert result_basic.ci.method == "basic"