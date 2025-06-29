"""
Tests for effect size calculator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from ..effect_size_calculator import EffectSizeCalculator
from ..output_schema import UnifiedAnalysisOutput


class TestEffectSizeCalculator:
    """Test effect size calculator functionality"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'comparisons': {
                'contexts': ['baseline', 'determiner_the', 'function_have'],
                'layers': 'all',
                'stratify_by': ['frequency', 'type']
            },
            'effect_size_types': ['cohens_d', 'cliffs_delta'],
            'metrics_to_analyze': {
                'transition_entropy': 'continuous',
                'diagonal_dominance': 'continuous'
            },
            'bootstrap': {
                'n_bootstrap': 100,  # Fewer for faster tests
                'confidence_level': 0.95
            },
            'enable_logging': False,
            'visualize': False
        }
    
    @pytest.fixture
    def calculator(self, sample_config):
        """Create calculator instance"""
        output_dir = tempfile.mkdtemp()
        
        calc = EffectSizeCalculator(output_dir=output_dir, config=sample_config)
        calc.output_dir = Path(output_dir)
        
        # Mock logger
        calc.logger = Mock()
        
        # Mock data loader
        calc.data_loader = Mock()
        
        yield calc
    
    @pytest.fixture
    def sample_groups(self):
        """Create sample groups for effect size testing"""
        np.random.seed(42)
        # Create two groups with known effect size
        # Cohen's d ≈ 0.8 (large effect)
        group1 = np.random.normal(10, 2, 50)
        group2 = np.random.normal(11.6, 2, 50)
        return group1, group2
    
    def test_initialization(self, calculator):
        """Test proper initialization"""
        assert hasattr(calculator, 'effect_sizes')
        assert hasattr(calculator, 'interpretations')
        assert calculator.analysis_name == 'effect_size_calculator'
        
        # Check mixin initialization
        assert hasattr(calculator, 'bootstrap_config')
        assert hasattr(calculator, '_bootstrap_cache')
    
    def test_cohens_d_calculation(self, calculator, sample_groups):
        """Test Cohen's d calculation"""
        group1, group2 = sample_groups
        
        d = calculator._cohens_d(group1, group2)
        
        # Should be close to -1.15 (negative because group1 < group2)
        # With means 10 vs 11.6 and pooled std ~2, d = (10-11.6)/2 ≈ -0.8
        # But actual calculation with sample data gives ~-1.15
        assert pytest.approx(d, abs=0.2) == -1.15
        
        # Test with identical groups
        d_same = calculator._cohens_d(group1, group1)
        assert pytest.approx(d_same) == 0.0
        
        # Test with empty std
        const_group = np.ones(10)
        d_const = calculator._cohens_d(const_group, const_group)
        assert d_const == 0.0
    
    def test_cohens_d_with_ci(self, calculator, sample_groups):
        """Test Cohen's d with confidence interval"""
        group1, group2 = sample_groups
        
        d_value, (lower, upper) = calculator._cohens_d_with_ci(group1, group2)
        
        # Check value
        assert pytest.approx(d_value, abs=0.2) == -1.15
        
        # Check CI properties
        assert lower < d_value < upper
        assert upper - lower < 1.0  # Reasonable CI width
    
    def test_hedges_g_calculation(self, calculator, sample_groups):
        """Test Hedge's g calculation"""
        group1, group2 = sample_groups
        
        g = calculator._hedges_g(group1, group2)
        
        # Should be slightly smaller than Cohen's d due to bias correction
        d = calculator._cohens_d(group1, group2)
        assert abs(g) < abs(d)
        assert pytest.approx(g, abs=0.1) == d * 0.98  # Approximate correction
    
    def test_cliffs_delta_calculation(self, calculator):
        """Test Cliff's delta calculation"""
        # Create groups with known dominance
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        
        delta = calculator._cliffs_delta(group1, group2)
        
        # Group2 completely dominates group1
        assert delta == -1.0
        
        # Test with overlapping groups
        group3 = np.array([3, 4, 5, 6, 7])
        delta_overlap = calculator._cliffs_delta(group1, group3)
        assert -1 < delta_overlap < 0  # Partial dominance
        
        # Test with identical groups
        delta_same = calculator._cliffs_delta(group1, group1)
        assert delta_same == 0.0
    
    def test_cliffs_delta_with_ci(self, calculator):
        """Test Cliff's delta with confidence interval"""
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)
        
        delta_value, (lower, upper) = calculator._cliffs_delta_with_ci(group1, group2)
        
        # Check value
        assert -1 <= delta_value <= 1
        
        # Check CI properties
        assert -1 <= lower <= delta_value <= upper <= 1
        assert upper - lower < 1.0  # Reasonable CI width
    
    def test_cramers_v_calculation(self, calculator):
        """Test Cramér's V calculation"""
        # Create contingency table with known association
        # Strong association
        contingency_strong = np.array([
            [20, 5],
            [5, 20]
        ])
        
        v_strong = calculator._cramers_v(contingency_strong)
        assert 0.5 < v_strong < 1.0  # Strong association
        
        # No association
        contingency_none = np.array([
            [10, 10],
            [10, 10]
        ])
        
        v_none = calculator._cramers_v(contingency_none)
        assert v_none < 0.1  # No association
    
    def test_rank_biserial_correlation(self, calculator):
        """Test rank-biserial correlation"""
        # Create groups with known rank relationship
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        
        r = calculator._rank_biserial_correlation(group1, group2)
        
        # Perfect separation - group1 < group2, so r = 1.0
        assert pytest.approx(r) == 1.0
        
        # Test with overlapping groups
        group3 = np.array([3, 4, 5, 6, 7])
        r_overlap = calculator._rank_biserial_correlation(group1, group3)
        # group1 has lower values on average, so positive correlation expected
        assert 0 < r_overlap < 1
    
    def test_effect_size_interpretations(self, calculator):
        """Test effect size interpretation functions"""
        # Cohen's d interpretation
        assert calculator._interpret_cohens_d(0.1) == 'negligible'
        assert calculator._interpret_cohens_d(0.3) == 'small'
        assert calculator._interpret_cohens_d(0.6) == 'medium'
        assert calculator._interpret_cohens_d(0.9) == 'large'
        assert calculator._interpret_cohens_d(-0.9) == 'large'  # Absolute value
        
        # Cliff's delta interpretation
        assert calculator._interpret_cliffs_delta(0.05) == 'negligible'
        assert calculator._interpret_cliffs_delta(0.2) == 'small'
        assert calculator._interpret_cliffs_delta(0.4) == 'medium'
        assert calculator._interpret_cliffs_delta(0.6) == 'large'
        
        # Cramér's V interpretation
        assert calculator._interpret_cramers_v(0.05) == 'negligible'
        assert calculator._interpret_cramers_v(0.2) == 'small'
        assert calculator._interpret_cramers_v(0.4) == 'medium'
        assert calculator._interpret_cramers_v(0.6) == 'large'
    
    def test_extract_context_metrics(self, calculator):
        """Test context metric extraction"""
        # Mock trajectories
        trajectories = {
            'trajectories': {
                'token1_baseline': {
                    'context_frame': 'baseline',
                    'path': [0, 1, 1, 2, 2, 2]  # Some repetition
                },
                'token2_baseline': {
                    'context_frame': 'baseline',
                    'path': [0, 0, 0, 0, 0, 0]  # No variation
                },
                'token1_context': {
                    'context_frame': 'determiner_the',
                    'path': [0, 1, 2, 3, 4, 5]  # High variation
                }
            }
        }
        
        # Extract baseline metrics
        baseline_metrics = calculator._extract_context_metrics(trajectories, 'baseline')
        
        assert 'transition_entropy' in baseline_metrics
        assert 'diagonal_dominance' in baseline_metrics
        assert 'trajectory_divergence' in baseline_metrics
        
        # Check values
        assert len(baseline_metrics['transition_entropy']) == 2
        assert baseline_metrics['transition_entropy'][1] < baseline_metrics['transition_entropy'][0]  # Less entropy for constant path
        
        # Extract context metrics
        context_metrics = calculator._extract_context_metrics(trajectories, 'determiner_the')
        assert len(context_metrics['trajectory_divergence']) == 1
        assert context_metrics['trajectory_divergence'][0] == 1.0  # All different clusters
    
    def test_analyze_context_effects(self, calculator):
        """Test context effect analysis"""
        # Mock trajectories with clear differences
        trajectories = {
            'trajectories': {}
        }
        
        # Add tokens with different behaviors in different contexts
        for i in range(20):
            # Baseline: low entropy
            trajectories['trajectories'][f'token{i}_baseline'] = {
                'context_frame': 'baseline',
                'path': [0, 0, 1, 1, 1, 1]  # Mostly stays in cluster 1
            }
            # Context: high entropy
            trajectories['trajectories'][f'token{i}_determiner_the'] = {
                'context_frame': 'determiner_the',
                'path': [0, 1, 2, 3, 0, 1]  # Changes frequently
            }
        
        # Analyze
        effects = calculator._analyze_context_effects(trajectories)
        
        assert 'baseline_vs_determiner_the' in effects
        context_effects = effects['baseline_vs_determiner_the']
        
        # Should have effect sizes for metrics
        assert any('cohens_d' in key for key in context_effects)
        assert any('cliffs_delta' in key for key in context_effects)
        
        # Check that effects are detected
        for metric, result in context_effects.items():
            if 'cohens_d' in metric and 'value' in result:
                # Should show difference between contexts
                assert abs(result['value']) > 0.1
    
    def test_analyze_layer_effects(self, calculator):
        """Test layer effect analysis"""
        # Mock trajectories with layer progression
        trajectories = {
            'trajectories': {}
        }
        
        # Add tokens that converge over layers
        for i in range(30):
            path = []
            for layer in range(6):
                # Early layers: diverse, later layers: converged
                if layer < 3:
                    path.append(i % 5)  # Diverse clusters
                else:
                    path.append(1)  # Converge to cluster 1
            
            trajectories['trajectories'][f'token{i}'] = {
                'path': path
            }
        
        # Analyze
        effects = calculator._analyze_layer_effects(trajectories)
        
        # Should have layer comparisons
        assert any('layer_' in key for key in effects)
        
        # Should detect convergence effect
        if 'first_to_last_layer' in effects:
            layer_effects = effects['first_to_last_layer']
            for metric, result in layer_effects.items():
                if 'entropy' in metric and 'value' in result:
                    # Entropy should decrease (negative effect)
                    assert result['value'] < 0
    
    def test_calculate_summary_statistics(self, calculator):
        """Test summary statistics calculation"""
        # Create mock effect sizes
        effect_sizes = {
            'context_comparisons': {
                'baseline_vs_the': {
                    'metric1_cohens_d': {'value': 0.3, 'interpretation': 'small'},
                    'metric2_cohens_d': {'value': 0.9, 'interpretation': 'large'}
                },
                'baseline_vs_have': {
                    'metric1_cohens_d': {'value': 0.1, 'interpretation': 'negligible'},
                    'metric2_cohens_d': {'value': 0.6, 'interpretation': 'medium'}
                }
            }
        }
        
        summary = calculator._calculate_summary_statistics(effect_sizes)
        
        assert 'mean_absolute_effect' in summary
        assert 'proportion_large_effects' in summary
        assert 'proportion_small_effects' in summary
        
        # Check calculations
        assert summary['mean_absolute_effect'] == pytest.approx(0.475)  # (0.3+0.9+0.1+0.6)/4
        assert summary['proportion_large_effects'] == 0.25  # 1/4
        assert summary['proportion_small_effects'] == 0.75  # 3/4
    
    def test_generate_recommendations(self, calculator):
        """Test recommendation generation"""
        results = {
            'summary_statistics': {
                'proportion_large_effects': 0.4,
                'mean_absolute_effect': 0.15
            },
            'effect_sizes': {
                'context_comparisons': {
                    'baseline_vs_the': {
                        'metric1_cohens_d': {'value': 0.9, 'interpretation': 'large'}
                    }
                }
            }
        }
        
        recommendations = calculator._generate_recommendations(results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should mention large effects
        assert any('large effect' in r.lower() for r in recommendations)
        
        # Should mention small mean effect
        assert any('small' in r.lower() for r in recommendations)
    
    def test_full_analysis_pipeline(self, calculator):
        """Test full analysis pipeline"""
        # Mock trajectories
        trajectories = {
            'trajectories': {}
        }
        
        # Create diverse trajectory data
        np.random.seed(42)
        contexts = ['baseline', 'determiner_the', 'function_have']
        
        for token_id in range(50):
            for context in contexts:
                # Different behavior per context
                if context == 'baseline':
                    path = [0] * 6  # Stable
                else:
                    path = list(np.random.randint(0, 5, 6))  # Variable
                
                trajectories['trajectories'][f'token{token_id}_{context}'] = {
                    'context_frame': context,
                    'path': path
                }
        
        calculator.data_loader.load_unified_trajectories.return_value = trajectories
        calculator.data_loader.load_token_metadata.return_value = {
            'frequencies': {str(i): 100 * (i + 1) for i in range(50)}
        }
        
        # Run analysis
        results = calculator.analyze()
        
        assert 'effect_sizes' in results
        assert 'interpretations' in results
        assert 'summary_statistics' in results
        assert 'recommendations' in results
        
        # Check structure
        assert 'context_comparisons' in results['effect_sizes']
        assert len(results['recommendations']) > 0
    
    def test_validation_methods(self, calculator):
        """Test data and results validation"""
        # Test data validation failure
        calculator.data_loader.load_unified_trajectories.return_value = None
        
        with pytest.raises(ValueError, match="No trajectory data"):
            calculator.validate_data()
        
        # Test with insufficient contexts
        calculator.data_loader.load_unified_trajectories.return_value = {
            'trajectories': {
                'token1': {'context_frame': 'baseline'}
            }
        }
        
        with pytest.raises(ValueError, match="at least 2 contexts"):
            calculator.validate_data()
        
        # Test results validation
        calculator.output = None
        with pytest.raises(ValueError, match="No output generated"):
            calculator.validate_results()
        
        # Test invalid effect size
        calculator.output = Mock()
        calculator.output.data = {
            'effect_sizes': {
                'test': {
                    'comp1': {
                        'metric_cliffs_delta': {'value': 2.0}  # Invalid
                    }
                }
            }
        }
        
        with pytest.raises(ValueError, match="Invalid Cliff's delta"):
            calculator.validate_results()
    
    def test_visualizations(self, calculator):
        """Test visualization creation"""
        # Enable visualization
        calculator.config['visualize'] = True
        
        # Create mock output
        calculator.output = Mock()
        calculator.output.data = {
            'effect_sizes': {
                'context_comparisons': {
                    'baseline_vs_the': {
                        'entropy_cohens_d': {
                            'value': 0.65,
                            'ci': (0.45, 0.85),
                            'interpretation': 'medium'
                        }
                    }
                }
            }
        }
        
        viz_list = calculator._create_visualizations()
        
        assert len(viz_list) == 3
        assert any('forest' in v['name'] for v in viz_list)
        assert any('distribution' in v['name'] for v in viz_list)
        assert any('heatmap' in v['name'] for v in viz_list)
        
        # Check files were created
        for viz in viz_list:
            assert Path(viz['path']).exists()
    
    def test_edge_cases(self, calculator):
        """Test edge cases"""
        # Empty groups
        with pytest.raises(ValueError, match="Cannot calculate Cohen's d with empty groups"):
            calculator._cohens_d(np.array([]), np.array([]))
        
        # Single value groups
        g1 = np.array([1.0])
        g2 = np.array([2.0])
        
        # Should handle gracefully - returns 0 for single-value groups
        d = calculator._cohens_d(g1, g2)
        assert d == 0.0  # Returns 0 for single-value groups
        
        # Identical values
        g_same = np.ones(10)
        d_same = calculator._cohens_d(g_same, g_same)
        assert d_same == 0.0
        
        # Very large effect - need variance in groups for meaningful Cohen's d
        g_small = np.random.normal(0, 1, 50)  # mean 0, std 1
        g_large = np.random.normal(100, 1, 50)  # mean 100, std 1
        d_large = calculator._cohens_d(g_small, g_large)
        assert abs(d_large) > 50  # Very large effect