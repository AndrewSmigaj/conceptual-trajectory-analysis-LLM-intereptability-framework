"""
Tests for Predictive Transformation Model Analysis
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from ..predictive_transformation_model import PredictiveTransformationModel
from ..output_schema import UnifiedAnalysisOutput


class TestPredictiveTransformationModel:
    """Test predictive transformation model analysis"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'experiment_name': 'test_predictive',
            'output_dir': 'test_output',
            'k': 10,
            'layers': list(range(12)),
            'context_types': ['determiner_the', 'function_have'],
            'max_tokens': 100,
            'enable_logging': False
        }
    
    @pytest.fixture
    def analysis(self, sample_config):
        """Create analysis instance"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_config, f)
            config_path = f.name
        
        analysis = PredictiveTransformationModel(config_path)
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
        assert hasattr(analysis, 'models')
        assert hasattr(analysis, 'scalers')
        assert hasattr(analysis, 'vectorizers')
        assert hasattr(analysis, 'encoders')
        assert analysis.models == {}
        assert analysis.scalers == {}
    
    def test_extract_token_features(self, analysis):
        """Test token feature extraction"""
        token_data = {
            'frequencies': {'0': 1000, '1': 100},
            'types': {'0': 'function', '1': 'content'},
            'pos_tags': {'0': 'DT', '1': 'NN'},
            'semantic_categories': {'0': 'determiner', '1': 'noun'}
        }
        
        # Test regular token
        features = analysis._extract_token_features('the', 0, token_data)
        assert features is not None
        assert features['token_length'] == 3
        assert features['is_lowercase'] == True
        assert features['token_type'] == 'function'
        assert features['pos_tag'] == 'DT'
        assert features['log_frequency'] > 0
        
        # Test token with space
        features = analysis._extract_token_features(' cat', 1, token_data)
        assert features['starts_with_space'] == True
        assert features['token_type'] == 'content'
        
        # Test unknown token
        features = analysis._extract_token_features('xyz', 99, token_data)
        assert features['token_type'] == 'unknown'
        assert features['pos_tag'] == 'unknown'
    
    def test_prepare_ml_data(self, analysis):
        """Test ML data preparation"""
        # Mock data
        analysis.data_loader.load_unified_trajectories.return_value = {
            'baseline': {
                'the': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                'cat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
            },
            'determiner_the': {
                'the': [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                'cat': [6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]
            }
        }
        
        analysis.data_loader.load_token_metadata.return_value = {
            'frequencies': {'0': 1000, '1': 100},
            'types': {'0': 'function', '1': 'content'}
        }
        
        analysis.data_loader.get_all_tokens.return_value = ['the', 'cat']
        
        # Prepare data
        X, y, feature_names, token_indices = analysis._prepare_ml_data('determiner_the')
        
        assert X is not None
        assert len(X) == 2  # Two tokens
        assert len(y) == 2
        assert len(feature_names) > 0
        assert len(token_indices) == 2
        
        # Check transitions encoded
        assert len(np.unique(y)) <= 2  # At most 2 unique transitions
    
    def test_train_models(self, analysis):
        """Test model training"""
        # Create sample data
        np.random.seed(42)
        X_train = np.random.randn(80, 10)
        X_test = np.random.randn(20, 10)
        y_train = np.random.randint(0, 3, 80)
        y_test = np.random.randint(0, 3, 20)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Train models
        results = analysis._train_models(X_train, y_train, X_test, y_test, feature_names)
        
        assert 'logistic_regression' in results
        assert 'random_forest' in results
        assert 'neural_network' in results
        
        # Check metrics
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'confusion_matrix' in metrics
            assert 'cv_mean' in metrics
            assert 'cv_std' in metrics
            assert 0 <= metrics['accuracy'] <= 1
            assert metrics['n_features'] == 10
            assert metrics['n_classes'] == 3
    
    def test_analyze_feature_importance(self, analysis):
        """Test feature importance analysis"""
        # Create mock random forest
        rf_model = Mock()
        rf_model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01])
        analysis.models['random_forest'] = rf_model
        
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Analyze importance
        importance = analysis._analyze_feature_importance(feature_names)
        
        assert 'top_features' in importance
        assert len(importance['top_features']) <= 20
        assert importance['top_features'][0]['importance'] == 0.3
        assert importance['top_features'][0]['feature'] == 'feature_0'
        assert importance['cumulative_importance_top10'] == 1.0
    
    def test_analyze_context_predictions(self, analysis):
        """Test context prediction analysis"""
        # Mock data
        analysis.data_loader.load_token_metadata.return_value = {
            'types': {'0': 'function', '1': 'content', '2': 'function', '3': 'content'}
        }
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])  # Perfect predictions
        mock_model.score.return_value = 1.0
        analysis.models['random_forest'] = mock_model
        
        X_test = np.random.randn(4, 10)
        y_test = np.array([0, 1, 0, 1])
        test_indices = [0, 1, 2, 3]
        
        # Analyze
        results = analysis._analyze_context_predictions(X_test, y_test, test_indices)
        
        assert 'per_token_type' in results
        assert 'error_analysis' in results
        
        # Check perfect accuracy
        assert results['per_token_type']['function']['accuracy'] == 1.0
        assert results['per_token_type']['content']['accuracy'] == 1.0
        assert results['error_analysis']['n_errors'] == 0
    
    def test_analyze_overall_predictability(self, analysis):
        """Test overall predictability analysis"""
        results = {
            'model_performance': {
                'context1': {
                    'logistic_regression': {'accuracy': 0.7, 'n_classes': 3},
                    'random_forest': {'accuracy': 0.8, 'n_classes': 3}
                },
                'context2': {
                    'logistic_regression': {'accuracy': 0.75, 'n_classes': 3},
                    'random_forest': {'accuracy': 0.85, 'n_classes': 3}
                }
            }
        }
        
        # Analyze
        predictability = analysis._analyze_overall_predictability(results)
        
        assert 'average_accuracy' in predictability
        assert 'context_comparison' in predictability
        assert 'predictability_score' in predictability
        
        # Check calculations
        assert predictability['average_accuracy']['random_forest']['mean'] == 0.825
        assert predictability['context_comparison']['context1'] == 0.8
        assert predictability['context_comparison']['context2'] == 0.85
        
        # Predictability score should account for chance level
        assert 0 <= predictability['predictability_score'] <= 1
    
    def test_full_analysis_pipeline(self, analysis):
        """Test full analysis pipeline"""
        # Mock all data - need more tokens for stratified split
        n_tokens = 100  # Enough for 80/20 split with 10 classes
        analysis.data_loader.load_unified_trajectories.return_value = {
            'baseline': {f'token_{i}': [i % 10] * 12 for i in range(n_tokens)},
            'determiner_the': {f'token_{i}': [(i + 5) % 10] * 12 for i in range(n_tokens)}
        }
        
        analysis.data_loader.load_token_metadata.return_value = {
            'frequencies': {str(i): 100 * (i + 1) for i in range(n_tokens)},
            'types': {str(i): 'function' if i % 2 == 0 else 'content' for i in range(n_tokens)}
        }
        
        analysis.data_loader.get_all_tokens.return_value = [f'token_{i}' for i in range(n_tokens)]
        
        analysis.config['context_types'] = ['determiner_the']
        analysis.config['max_tokens'] = n_tokens
        
        # Run analysis
        results = analysis.analyze()
        
        assert 'model_performance' in results
        assert 'feature_importance' in results
        assert 'transformation_predictability' in results
        
        # Check some model was trained
        assert len(results['model_performance']) > 0
        if 'determiner_the' in results['model_performance']:
            assert len(results['model_performance']['determiner_the']) > 0
    
    def test_error_handling(self, analysis):
        """Test error handling"""
        # No data case
        analysis.data_loader.load_unified_trajectories.return_value = {}
        analysis.data_loader.get_all_tokens.return_value = []
        
        results = analysis.analyze()
        assert isinstance(results, dict)
        
        # Empty features case
        analysis.data_loader.load_unified_trajectories.return_value = {
            'baseline': {'token': [0] * 12},
            'context': {}
        }
        analysis.data_loader.get_all_tokens.return_value = ['token']
        
        X, y, _, _ = analysis._prepare_ml_data('context')
        assert X is None or len(X) == 0
    
    def test_generate_summary(self, analysis):
        """Test summary generation"""
        results = {
            'transformation_predictability': {
                'predictability_score': 0.75,
                'average_accuracy': {
                    'random_forest': {'mean': 0.85}
                }
            },
            'feature_importance': {
                'context1': {
                    'top_features': [
                        {'feature': 'log_frequency', 'importance': 0.3}
                    ]
                }
            },
            'per_context_analysis': {
                'context1': {
                    'per_token_type': {
                        'function': {'accuracy': 0.9},
                        'content': {'accuracy': 0.7}
                    }
                }
            }
        }
        
        summary = analysis._generate_summary(results)
        
        assert 'key_findings' in summary
        assert len(summary['key_findings']) > 0
        assert 'highly predictable' in summary['key_findings'][0]
        assert 'Random Forest' in summary['key_findings'][1]
        assert 'interpretation' in summary
        assert 'systematic rules' in summary['interpretation']