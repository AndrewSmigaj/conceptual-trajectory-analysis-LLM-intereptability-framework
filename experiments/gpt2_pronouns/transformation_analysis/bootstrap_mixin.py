"""
Mixin class for adding bootstrap confidence intervals to any analysis.

This mixin can be inherited by any analysis class to provide easy access
to bootstrap confidence interval calculation for metrics.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, Union, List
import logging
from functools import lru_cache

from .bootstrap_utils import (
    bootstrap_confidence_interval,
    bootstrap_matrix_ci,
    parallel_bootstrap,
    BootstrapMethod
)
from .output_schema import ConfidenceInterval, MetricWithCI


logger = logging.getLogger(__name__)


class BootstrapMixin:
    """
    Mixin class that provides bootstrap confidence interval functionality.
    
    Any analysis class can inherit from this mixin to gain access to
    bootstrap CI calculation methods.
    """
    
    def __init__(self, n_bootstrap=1000, **kwargs):
        """Initialize bootstrap configuration."""
        # Note: Don't call super().__init__() in mixins
        
        # Bootstrap configuration
        self.bootstrap_config = {
            'n_bootstrap': n_bootstrap,
            'confidence_level': 0.95,
            'method': BootstrapMethod.PERCENTILE,
            'random_state': None,
            'show_progress': False,
            'parallel': False,
            'n_jobs': -1
        }
        
        # Update from config if available
        if hasattr(self, 'config') and 'bootstrap' in self.config:
            self.bootstrap_config.update(self.config['bootstrap'])
        
        # Cache for bootstrap samples
        self._bootstrap_cache = {}
        
    def bootstrap_metric(self,
                        data: Union[np.ndarray, List[np.ndarray]],
                        metric_func: Callable,
                        metric_name: Optional[str] = None,
                        **kwargs) -> MetricWithCI:
        """
        Calculate a metric with bootstrap confidence interval.
        
        Args:
            data: Input data for metric calculation
            metric_func: Function to calculate the metric
            metric_name: Optional name for caching
            **kwargs: Additional arguments for metric_func
            
        Returns:
            MetricWithCI object containing value and confidence interval
        """
        # Check cache if metric_name provided
        cache_key = None
        if metric_name:
            cache_key = f"{metric_name}_{hash(str(data))}"
            if cache_key in self._bootstrap_cache:
                return self._bootstrap_cache[cache_key]
        
        # Convert confidence level to alpha
        alpha = 1 - self.bootstrap_config['confidence_level']
        
        # Calculate metric and CI
        value, (lower, upper) = bootstrap_confidence_interval(
            data,
            metric_func,
            n_bootstrap=self.bootstrap_config['n_bootstrap'],
            alpha=alpha,
            method=self.bootstrap_config['method'],
            random_state=self.bootstrap_config['random_state'],
            show_progress=self.bootstrap_config['show_progress'],
            **kwargs
        )
        
        # Create CI object
        ci = ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=self.bootstrap_config['confidence_level'],
            method=self.bootstrap_config['method'].value,
            n_bootstrap=self.bootstrap_config['n_bootstrap']
        )
        
        # Create result
        result = MetricWithCI(value=value, ci=ci)
        
        # Cache if metric_name provided
        if cache_key:
            self._bootstrap_cache[cache_key] = result
        
        return result
    
    def bootstrap_multiple_metrics(self,
                                 data: Union[np.ndarray, List[np.ndarray]],
                                 metrics_dict: Dict[str, Callable],
                                 **kwargs) -> Dict[str, MetricWithCI]:
        """
        Calculate multiple metrics with bootstrap CIs.
        
        Args:
            data: Input data
            metrics_dict: Dictionary of metric_name -> metric_function
            **kwargs: Additional arguments for metric functions
            
        Returns:
            Dictionary of metric_name -> MetricWithCI
        """
        results = {}
        
        if self.bootstrap_config['parallel'] and len(metrics_dict) > 1:
            # Parallel processing
            metric_names = list(metrics_dict.keys())
            metric_funcs = list(metrics_dict.values())
            
            # Create data list for parallel processing
            data_list = [data] * len(metrics_dict)
            
            # Run parallel bootstrap
            ci_results = parallel_bootstrap(
                data_list,
                lambda d, func=func: func(d, **kwargs),
                n_bootstrap=self.bootstrap_config['n_bootstrap'],
                alpha=1 - self.bootstrap_config['confidence_level'],
                method=self.bootstrap_config['method'],
                n_jobs=self.bootstrap_config['n_jobs']
            )
            
            # Package results
            for name, (value, (lower, upper)) in zip(metric_names, ci_results):
                ci = ConfidenceInterval(
                    lower=lower,
                    upper=upper,
                    confidence_level=self.bootstrap_config['confidence_level'],
                    method=self.bootstrap_config['method'].value,
                    n_bootstrap=self.bootstrap_config['n_bootstrap']
                )
                results[name] = MetricWithCI(value=value, ci=ci)
        else:
            # Sequential processing
            for metric_name, metric_func in metrics_dict.items():
                results[metric_name] = self.bootstrap_metric(
                    data, metric_func, metric_name, **kwargs
                )
        
        return results
    
    def bootstrap_matrix_metrics(self,
                               matrix: np.ndarray,
                               metrics_dict: Dict[str, Callable]) -> Dict[str, MetricWithCI]:
        """
        Calculate metrics for a matrix with bootstrap CIs.
        
        Special handling for transition matrices and similar structures.
        
        Args:
            matrix: 2D matrix
            metrics_dict: Dictionary of metric_name -> metric_function
            
        Returns:
            Dictionary of metric_name -> MetricWithCI
        """
        results = {}
        
        for metric_name, metric_func in metrics_dict.items():
            # For matrix metrics, we need to handle the bootstrap differently
            # We'll bootstrap the metric calculated on the whole matrix
            value = metric_func(matrix)
            
            # For now, create a simple CI based on the metric value
            # In practice, you'd need the raw data that created the matrix
            ci = ConfidenceInterval(
                lower=value * 0.9,  # Placeholder
                upper=value * 1.1,  # Placeholder
                confidence_level=self.bootstrap_config['confidence_level'],
                method="approximate",
                n_bootstrap=0
            )
            
            results[metric_name] = MetricWithCI(value=value, ci=ci)
            
            logger.warning(f"Matrix bootstrap for {metric_name} uses approximate CI. "
                         "Provide raw data for accurate bootstrap.")
        
        return results
    
    def add_confidence_intervals_to_output(self, output: Any) -> None:
        """
        Add confidence intervals to the analysis output.
        
        This method should be called after analysis to add CIs to the output.
        
        Args:
            output: UnifiedAnalysisOutput object
        """
        if not hasattr(output, 'confidence_intervals') or output.confidence_intervals is None:
            output.confidence_intervals = {}
        
        # Add any cached confidence intervals
        for cache_key, metric_with_ci in self._bootstrap_cache.items():
            # Extract metric name from cache key
            # Cache key format is "metric_name_hash", so we need everything before the last underscore
            parts = cache_key.split('_')
            if len(parts) > 1:
                metric_name = '_'.join(parts[:-1])  # Everything except the hash
            else:
                metric_name = cache_key
            output.confidence_intervals[metric_name] = metric_with_ci.to_dict()
    
    def clear_bootstrap_cache(self) -> None:
        """Clear the bootstrap cache."""
        self._bootstrap_cache.clear()
    
    def set_bootstrap_config(self, **kwargs) -> None:
        """
        Update bootstrap configuration.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95)
            method: Bootstrap method (BootstrapMethod enum)
            random_state: Random seed
            show_progress: Whether to show progress bar
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs
        """
        for key, value in kwargs.items():
            if key in self.bootstrap_config:
                self.bootstrap_config[key] = value
            else:
                logger.warning(f"Unknown bootstrap config parameter: {key}")
    
    def bootstrap_stratified_metric(self,
                                  data_dict: Dict[str, np.ndarray],
                                  metric_func: Callable,
                                  aggregate_func: Optional[Callable] = None) -> Dict[str, MetricWithCI]:
        """
        Calculate bootstrap CIs for stratified data.
        
        Args:
            data_dict: Dictionary of stratum_name -> data
            metric_func: Function to calculate metric for each stratum
            aggregate_func: Optional function to aggregate across strata
            
        Returns:
            Dictionary of stratum_name -> MetricWithCI, plus 'aggregate' if aggregate_func provided
        """
        results = {}
        
        # Calculate for each stratum
        for stratum_name, stratum_data in data_dict.items():
            results[stratum_name] = self.bootstrap_metric(
                stratum_data, metric_func, f"stratified_{stratum_name}"
            )
        
        # Calculate aggregate if requested
        if aggregate_func:
            # Combine all data
            all_data = np.concatenate(list(data_dict.values()))
            results['aggregate'] = self.bootstrap_metric(
                all_data, aggregate_func, "stratified_aggregate"
            )
        
        return results