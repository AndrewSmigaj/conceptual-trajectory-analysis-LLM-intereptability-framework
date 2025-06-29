"""
Bootstrap utilities for calculating confidence intervals.

Provides functions for bootstrap resampling and confidence interval calculation
using various methods (percentile, BCa, basic, studentized).
"""

import numpy as np
from typing import Callable, Tuple, Optional, Union, List
from scipy import stats
from functools import partial
import warnings
from enum import Enum
from tqdm import tqdm


class BootstrapMethod(Enum):
    """Available bootstrap methods for confidence interval calculation."""
    PERCENTILE = "percentile"  # Simple percentile method
    BCA = "bca"  # Bias-corrected and accelerated
    BASIC = "basic"  # Basic bootstrap
    STUDENTIZED = "studentized"  # Bootstrap-t method


def bootstrap_sample(data: np.ndarray, 
                    n_bootstrap: int = 1000,
                    random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate bootstrap samples from data.
    
    Args:
        data: Original data array (n_samples, ...)
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Bootstrap indices array of shape (n_bootstrap, n_samples)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(data)
    
    # Pre-generate all bootstrap indices for efficiency
    bootstrap_indices = rng.randint(0, n_samples, size=(n_bootstrap, n_samples))
    
    return bootstrap_indices


def calculate_bootstrap_statistic(data: np.ndarray,
                                statistic_func: Callable,
                                bootstrap_indices: np.ndarray,
                                **kwargs) -> np.ndarray:
    """
    Calculate statistic for each bootstrap sample.
    
    Args:
        data: Original data
        statistic_func: Function to calculate statistic
        bootstrap_indices: Pre-generated bootstrap indices
        **kwargs: Additional arguments for statistic_func
        
    Returns:
        Array of bootstrap statistics
    """
    n_bootstrap = len(bootstrap_indices)
    
    # Calculate statistic on first sample to get shape
    first_sample = data[bootstrap_indices[0]]
    first_stat = statistic_func(first_sample, **kwargs)
    
    # Initialize array for all statistics
    if np.isscalar(first_stat):
        boot_stats = np.zeros(n_bootstrap)
    else:
        boot_stats = np.zeros((n_bootstrap,) + np.array(first_stat).shape)
    
    boot_stats[0] = first_stat
    
    # Calculate for remaining samples
    for i in range(1, n_bootstrap):
        sample = data[bootstrap_indices[i]]
        boot_stats[i] = statistic_func(sample, **kwargs)
    
    return boot_stats


def percentile_method(boot_stats: np.ndarray, 
                     alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals using percentile method.
    
    Args:
        boot_stats: Bootstrap statistics
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(boot_stats, lower_percentile, axis=0)
    upper = np.percentile(boot_stats, upper_percentile, axis=0)
    
    return lower, upper


def bca_method(data: np.ndarray,
               statistic_func: Callable,
               boot_stats: np.ndarray,
               alpha: float = 0.05,
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals using BCa (bias-corrected and accelerated) method.
    
    Args:
        data: Original data
        statistic_func: Function to calculate statistic
        boot_stats: Bootstrap statistics
        alpha: Significance level
        **kwargs: Additional arguments for statistic_func
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    # Calculate original statistic
    theta = statistic_func(data, **kwargs)
    
    # Calculate bias correction
    z0 = stats.norm.ppf(np.mean(boot_stats < theta, axis=0))
    
    # Calculate acceleration using jackknife
    n = len(data)
    jackknife_stats = np.zeros((n,) + np.array(theta).shape)
    
    for i in range(n):
        # Leave one out
        jack_data = np.delete(data, i, axis=0)
        jackknife_stats[i] = statistic_func(jack_data, **kwargs)
    
    # Calculate acceleration
    jack_mean = np.mean(jackknife_stats, axis=0)
    numerator = np.sum((jack_mean - jackknife_stats) ** 3, axis=0)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2, axis=0) ** 1.5)
    
    # Avoid division by zero
    acceleration = np.divide(numerator, denominator, 
                           out=np.zeros_like(numerator), 
                           where=denominator != 0)
    
    # Calculate adjusted percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    
    a1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)))
    a2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - acceleration * (z0 + z_1_alpha)))
    
    # Ensure percentiles are in valid range
    a1 = np.clip(a1, 0.001, 0.999)
    a2 = np.clip(a2, 0.001, 0.999)
    
    # Get BCa confidence intervals
    lower = np.percentile(boot_stats, a1 * 100, axis=0)
    upper = np.percentile(boot_stats, a2 * 100, axis=0)
    
    return lower, upper


def basic_method(theta: Union[float, np.ndarray],
                boot_stats: np.ndarray,
                alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals using basic bootstrap method.
    
    Args:
        theta: Original statistic value
        boot_stats: Bootstrap statistics
        alpha: Significance level
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Basic method: theta - (boot_upper - theta), theta - (boot_lower - theta)
    boot_lower = np.percentile(boot_stats, lower_percentile, axis=0)
    boot_upper = np.percentile(boot_stats, upper_percentile, axis=0)
    
    lower = 2 * theta - boot_upper
    upper = 2 * theta - boot_lower
    
    return lower, upper


def bootstrap_confidence_interval(data: np.ndarray,
                                statistic_func: Callable,
                                n_bootstrap: int = 1000,
                                alpha: float = 0.05,
                                method: BootstrapMethod = BootstrapMethod.PERCENTILE,
                                random_state: Optional[int] = None,
                                show_progress: bool = False,
                                **kwargs) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_func: Function that calculates the statistic
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level (default 0.05 for 95% CI)
        method: Bootstrap method to use
        random_state: Random seed for reproducibility
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments for statistic_func
        
    Returns:
        (statistic, (lower_ci, upper_ci)) tuple
    """
    # Check for empty data
    if len(data) == 0:
        raise ValueError("Cannot calculate bootstrap CI for empty data")
    
    # Calculate original statistic
    theta = statistic_func(data, **kwargs)
    
    # Generate bootstrap samples
    bootstrap_indices = bootstrap_sample(data, n_bootstrap, random_state)
    
    # Calculate bootstrap statistics
    if show_progress:
        boot_stats = []
        for indices in tqdm(bootstrap_indices, desc="Bootstrap"):
            sample = data[indices]
            boot_stats.append(statistic_func(sample, **kwargs))
        boot_stats = np.array(boot_stats)
    else:
        boot_stats = calculate_bootstrap_statistic(
            data, statistic_func, bootstrap_indices, **kwargs
        )
    
    # Calculate confidence intervals based on method
    if method == BootstrapMethod.PERCENTILE:
        lower, upper = percentile_method(boot_stats, alpha)
    elif method == BootstrapMethod.BCA:
        lower, upper = bca_method(data, statistic_func, boot_stats, alpha, **kwargs)
    elif method == BootstrapMethod.BASIC:
        lower, upper = basic_method(theta, boot_stats, alpha)
    elif method == BootstrapMethod.STUDENTIZED:
        warnings.warn("Studentized method not yet implemented, using percentile")
        lower, upper = percentile_method(boot_stats, alpha)
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")
    
    return theta, (lower, upper)


def bootstrap_matrix_ci(matrix: np.ndarray,
                       element_func: Optional[Callable] = None,
                       n_bootstrap: int = 1000,
                       alpha: float = 0.05,
                       method: BootstrapMethod = BootstrapMethod.PERCENTILE,
                       random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate bootstrap CIs for each element of a matrix.
    
    Special function for transition matrices and similar 2D structures.
    Note: This creates approximate CIs based on the matrix values.
    For accurate CIs, you need the raw data that created the matrix.
    
    Args:
        matrix: 2D matrix to bootstrap
        element_func: Optional function to apply to each element
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level
        method: Bootstrap method
        random_state: Random seed
        
    Returns:
        (original_matrix, lower_ci_matrix, upper_ci_matrix)
    """
    rows, cols = matrix.shape
    lower_ci = np.zeros_like(matrix)
    upper_ci = np.zeros_like(matrix)
    
    # If no element function provided, use identity
    if element_func is None:
        element_func = lambda x: x
    
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Bootstrap each element
    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            
            # For matrix elements, create approximate CI
            # This assumes some variability proportional to the value
            if value > 0:
                # Create synthetic data with reasonable variance
                std_dev = value * 0.1  # 10% coefficient of variation
                element_data = np.random.normal(value, std_dev, size=100)
                
                _, (lower, upper) = bootstrap_confidence_interval(
                    element_data,
                    np.mean,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                    method=method,
                    random_state=random_state if random_state is None else random_state + i*cols + j
                )
            else:
                # For zero values, CI is also zero
                lower = upper = 0.0
            
            lower_ci[i, j] = max(0, lower)  # Ensure non-negative for probabilities
            upper_ci[i, j] = min(1, upper) if value <= 1 else upper  # Cap at 1 for probabilities
    
    return matrix, lower_ci, upper_ci


def parallel_bootstrap(data_list: List[np.ndarray],
                      statistic_func: Callable,
                      n_bootstrap: int = 1000,
                      alpha: float = 0.05,
                      method: BootstrapMethod = BootstrapMethod.PERCENTILE,
                      n_jobs: int = -1,
                      **kwargs) -> List[Tuple[float, Tuple[float, float]]]:
    """
    Perform bootstrap in parallel for multiple datasets.
    
    Args:
        data_list: List of datasets to bootstrap
        statistic_func: Function to calculate statistic
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level
        method: Bootstrap method
        n_jobs: Number of parallel jobs (-1 for all cores)
        **kwargs: Additional arguments for statistic_func
        
    Returns:
        List of (statistic, (lower_ci, upper_ci)) tuples
    """
    try:
        from joblib import Parallel, delayed
        
        # Wrapper function for parallel execution
        def _bootstrap_single(data, idx):
            return bootstrap_confidence_interval(
                data, statistic_func, n_bootstrap, alpha, method,
                random_state=idx, **kwargs
            )
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_single)(data, i) 
            for i, data in enumerate(data_list)
        )
        
        return results
        
    except ImportError:
        warnings.warn("joblib not available, falling back to sequential processing")
        return [
            bootstrap_confidence_interval(
                data, statistic_func, n_bootstrap, alpha, method,
                random_state=i, **kwargs
            )
            for i, data in enumerate(data_list)
        ]