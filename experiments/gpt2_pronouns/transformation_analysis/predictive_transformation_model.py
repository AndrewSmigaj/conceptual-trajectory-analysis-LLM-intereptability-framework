"""
Predictive Transformation Model Analysis

Tests if context-induced transformations are learnable and generalizable by training
machine learning models to predict cluster transitions based on token properties.

This analysis:
1. Extracts features from tokens (frequency, type, semantic properties)
2. Trains multiple ML models (logistic regression, random forest, neural net)
3. Evaluates prediction accuracy on held-out tokens
4. Analyzes feature importance to understand what drives transformations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction import DictVectorizer
import warnings
warnings.filterwarnings('ignore')

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import UnifiedAnalysisOutput, PredictiveResults, AnalysisMetadata


class PredictiveTransformationModel(BaseTransformationAnalysis):
    """
    Trains machine learning models to predict cluster transitions from token properties.
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        self.encoders = {}
        self.feature_names = []
        
    def analyze(self) -> Dict[str, Any]:
        """Run predictive modeling analysis"""
        self.logger.info("Starting predictive transformation model analysis")
        
        results = {
            'model_performance': {},
            'feature_importance': {},
            'per_context_analysis': {},
            'transformation_predictability': {},
            'cross_validation_scores': {}
        }
        
        # Analyze each context type
        context_types = self.config.get('context_types', ['determiner_the', 'function_have'])
        
        for context_type in context_types:
            self.logger.info(f"Analyzing context: {context_type}")
            
            # Prepare features and targets
            X, y, feature_names, token_indices = self._prepare_ml_data(context_type)
            
            if X is None or len(X) == 0:
                self.logger.warning(f"No valid data for context {context_type}")
                continue
                
            # Split data
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, token_indices, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Train models
            models_performance = self._train_models(X_train, y_train, X_test, y_test, feature_names)
            results['model_performance'][context_type] = models_performance
            
            # Feature importance analysis
            if 'random_forest' in self.models:
                importance = self._analyze_feature_importance(feature_names)
                results['feature_importance'][context_type] = importance
            
            # Per-context detailed analysis
            context_analysis = self._analyze_context_predictions(X_test, y_test, test_idx)
            results['per_context_analysis'][context_type] = context_analysis
            
        # Overall transformation predictability
        results['transformation_predictability'] = self._analyze_overall_predictability(results)
        
        return results
    
    def _prepare_ml_data(self, context_type: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """Prepare features and targets for ML"""
        features = []
        targets = []
        token_indices = []
        
        # Get trajectories
        trajectories = self.data_loader.load_unified_trajectories(k=self.config['k'])
        
        # Get token metadata
        token_data = self.data_loader.load_token_metadata()
        
        # Process each token
        for token_idx, token in enumerate(self.data_loader.get_all_tokens()[:self.config.get('max_tokens', 1000)]):
            # Skip if no trajectory data
            if token not in trajectories['baseline'] or token not in trajectories.get(context_type, {}):
                continue
                
            # Extract features
            token_features = self._extract_token_features(token, token_idx, token_data)
            if token_features is None:
                continue
                
            # Get target (cluster transition at layer 0)
            baseline_cluster = trajectories['baseline'][token][0]
            context_cluster = trajectories[context_type][token][0]
            
            # Create transition label
            transition = f"{baseline_cluster}→{context_cluster}"
            
            features.append(token_features)
            targets.append(transition)
            token_indices.append(token_idx)
        
        if not features:
            return None, None, [], []
        
        # Vectorize features
        self.vectorizers[context_type] = DictVectorizer(sparse=False)
        X = self.vectorizers[context_type].fit_transform(features)
        
        # Encode targets
        self.encoders[context_type] = LabelEncoder()
        y = self.encoders[context_type].fit_transform(targets)
        
        # Get feature names
        feature_names = self.vectorizers[context_type].get_feature_names_out()
        
        # Scale features
        self.scalers[context_type] = StandardScaler()
        X = self.scalers[context_type].fit_transform(X)
        
        return X, y, list(feature_names), token_indices
    
    def _extract_token_features(self, token: str, token_idx: int, token_data: Dict) -> Optional[Dict[str, Any]]:
        """Extract features from a token"""
        features = {}
        
        # Basic token properties
        features['token_length'] = len(token)
        features['is_uppercase'] = token.isupper()
        features['is_lowercase'] = token.islower()
        features['is_capitalized'] = token[0].isupper() if token else False
        features['has_punctuation'] = any(c in token for c in '.,!?;:')
        features['is_numeric'] = any(c.isdigit() for c in token)
        features['starts_with_space'] = token.startswith(' ') if token else False
        
        # Frequency features
        if str(token_idx) in token_data.get('frequencies', {}):
            freq_data = token_data['frequencies'][str(token_idx)]
            features['log_frequency'] = np.log(freq_data + 1) if isinstance(freq_data, (int, float)) else 0
        else:
            features['log_frequency'] = 0
            
        # Token type
        if str(token_idx) in token_data.get('types', {}):
            features['token_type'] = token_data['types'][str(token_idx)]
        else:
            features['token_type'] = 'unknown'
            
        # POS tag if available
        if str(token_idx) in token_data.get('pos_tags', {}):
            features['pos_tag'] = token_data['pos_tags'][str(token_idx)]
        else:
            features['pos_tag'] = 'unknown'
            
        # Semantic category if available
        if str(token_idx) in token_data.get('semantic_categories', {}):
            features['semantic_category'] = token_data['semantic_categories'][str(token_idx)]
        else:
            features['semantic_category'] = 'unknown'
        
        return features
    
    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray, 
                     feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Train multiple ML models"""
        results = {}
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        for name, model in models.items():
            self.logger.info(f"Training {name}")
            
            # Train
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'accuracy': float(accuracy),
                'confusion_matrix': conf_matrix.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'n_features': len(feature_names),
                'n_classes': len(np.unique(y_train))
            }
            
            self.logger.info(f"{name} accuracy: {accuracy:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        
        return results
    
    def _analyze_feature_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance from random forest"""
        if 'random_forest' not in self.models:
            return {}
            
        rf_model = self.models['random_forest']
        importances = rf_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        top_features = []
        for i in range(min(20, len(feature_names))):  # Top 20 features
            idx = indices[i]
            top_features.append({
                'feature': feature_names[idx],
                'importance': float(importances[idx])
            })
        
        return {
            'top_features': top_features,
            'total_features': len(feature_names),
            'cumulative_importance_top10': float(sum(importances[indices[:10]]))
        }
    
    def _analyze_context_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                                   test_indices: List[int]) -> Dict[str, Any]:
        """Analyze predictions in detail"""
        results = {
            'per_token_type': {},
            'error_analysis': {},
            'transition_patterns': {}
        }
        
        # Get best model
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x].score(X_test, y_test))
        best_model = self.models[best_model_name]
        
        # Get predictions
        y_pred = best_model.predict(X_test)
        
        # Analyze by token type
        token_data = self.data_loader.load_token_metadata()
        token_types = {}
        
        for i, idx in enumerate(test_indices):
            token_type = token_data.get('types', {}).get(str(idx), 'unknown')
            if token_type not in token_types:
                token_types[token_type] = {'correct': 0, 'total': 0}
            
            token_types[token_type]['total'] += 1
            if y_pred[i] == y_test[i]:
                token_types[token_type]['correct'] += 1
        
        # Calculate accuracies by type
        for token_type, counts in token_types.items():
            acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            results['per_token_type'][token_type] = {
                'accuracy': float(acc),
                'n_samples': counts['total']
            }
        
        # Error analysis
        errors = []
        for i in range(len(y_test)):
            if y_pred[i] != y_test[i]:
                errors.append({
                    'predicted': int(y_pred[i]),
                    'actual': int(y_test[i]),
                    'token_idx': int(test_indices[i])
                })
        
        results['error_analysis'] = {
            'n_errors': len(errors),
            'error_rate': float(len(errors) / len(y_test)),
            'sample_errors': errors[:10]  # First 10 errors
        }
        
        return results
    
    def _analyze_overall_predictability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall transformation predictability"""
        analysis = {
            'average_accuracy': {},
            'context_comparison': {},
            'model_comparison': {},
            'predictability_score': 0.0
        }
        
        # Calculate average accuracy per model across contexts
        model_accuracies = {}
        for context, perf in results['model_performance'].items():
            for model, metrics in perf.items():
                if model not in model_accuracies:
                    model_accuracies[model] = []
                model_accuracies[model].append(metrics['accuracy'])
        
        for model, accuracies in model_accuracies.items():
            analysis['average_accuracy'][model] = {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies))
            }
        
        # Compare contexts
        context_scores = {}
        for context, perf in results['model_performance'].items():
            best_acc = max(metrics['accuracy'] for metrics in perf.values())
            context_scores[context] = best_acc
        
        analysis['context_comparison'] = context_scores
        
        # Overall predictability score (0-1)
        if model_accuracies:
            best_model_avg = max(np.mean(accs) for accs in model_accuracies.values())
            # Adjust for chance level (depends on number of classes)
            n_classes = np.mean([perf[list(perf.keys())[0]]['n_classes'] 
                               for perf in results['model_performance'].values()])
            chance_level = 1.0 / n_classes
            analysis['predictability_score'] = float((best_model_avg - chance_level) / (1.0 - chance_level))
        
        return analysis
    
    def validate_data(self) -> None:
        """Validate loaded data"""
        if not hasattr(self, 'data_loader') or self.data_loader is None:
            raise ValueError("Data loader not initialized")
            
        # Check we have trajectory data
        trajectories = self.data_loader.load_unified_trajectories(k=self.config['k'])
        if not trajectories:
            raise ValueError("No trajectory data found")
            
        # Check we have token metadata
        token_data = self.data_loader.load_token_metadata()
        if not token_data:
            raise ValueError("No token metadata found")
            
        self.logger.info("Data validation passed")
    
    def validate_results(self) -> None:
        """Validate analysis results"""
        if not hasattr(self, 'output') or self.output is None:
            raise ValueError("No output generated")
            
        # Check required fields
        if 'model_performance' not in self.output.data:
            raise ValueError("Missing model performance results")
            
        if 'transformation_predictability' not in self.output.data:
            raise ValueError("Missing predictability analysis")
            
        self.logger.info("Results validation passed")
    
    def _create_visualizations(self) -> List[Dict[str, Any]]:
        """Create visualizations for predictive modeling results"""
        viz_list = []
        
        # Feature importance plot
        viz_list.append({
            'name': 'feature_importance',
            'path': str(self.output_dir / 'feature_importance.png'),
            'type': 'bar_chart',
            'description': 'Top features driving cluster transitions'
        })
        
        # Model comparison plot
        viz_list.append({
            'name': 'model_comparison',
            'path': str(self.output_dir / 'model_comparison.png'),
            'type': 'grouped_bar',
            'description': 'Accuracy comparison across ML models'
        })
        
        # Confusion matrix heatmap
        viz_list.append({
            'name': 'confusion_matrix',
            'path': str(self.output_dir / 'confusion_matrix.png'),
            'type': 'heatmap',
            'description': 'Transition prediction confusion matrix'
        })
        
        return viz_list
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        predictability = results.get('transformation_predictability', {})
        
        # Key findings
        key_findings = []
        
        # Overall predictability
        pred_score = predictability.get('predictability_score', 0)
        if pred_score > 0.7:
            key_findings.append(f"Context transformations are highly predictable ({pred_score:.1%} above chance)")
        elif pred_score > 0.3:
            key_findings.append(f"Context transformations show moderate predictability ({pred_score:.1%} above chance)")
        else:
            key_findings.append("Context transformations show limited predictability from token features alone")
        
        # Best model
        avg_acc = predictability.get('average_accuracy', {})
        if avg_acc:
            best_model = max(avg_acc.keys(), key=lambda x: avg_acc[x]['mean'])
            best_acc = avg_acc[best_model]['mean']
            key_findings.append(f"{best_model.replace('_', ' ').title()} achieves {best_acc:.1%} average accuracy")
        
        # Feature importance
        if 'feature_importance' in results and results['feature_importance']:
            first_context = list(results['feature_importance'].keys())[0]
            top_feature = results['feature_importance'][first_context]['top_features'][0]
            key_findings.append(f"'{top_feature['feature']}' is the most predictive feature")
        
        # Token type differences
        type_accuracies = {}
        for context, analysis in results.get('per_context_analysis', {}).items():
            for token_type, metrics in analysis.get('per_token_type', {}).items():
                if token_type not in type_accuracies:
                    type_accuracies[token_type] = []
                type_accuracies[token_type].append(metrics['accuracy'])
        
        if type_accuracies:
            avg_by_type = {t: np.mean(accs) for t, accs in type_accuracies.items()}
            most_predictable = max(avg_by_type.keys(), key=lambda x: avg_by_type[x])
            least_predictable = min(avg_by_type.keys(), key=lambda x: avg_by_type[x])
            if avg_by_type[most_predictable] - avg_by_type[least_predictable] > 0.1:
                key_findings.append(f"{most_predictable} tokens are most predictable ({avg_by_type[most_predictable]:.1%})")
        
        return {
            'key_findings': key_findings,
            'interpretation': self._generate_interpretation(results),
            'next_steps': [
                "Investigate tokens with unpredictable transformations",
                "Test if adding activation features improves predictions",
                "Analyze if certain transformation patterns are context-specific"
            ]
        }
    
    def _generate_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate interpretation of results"""
        pred_score = results.get('transformation_predictability', {}).get('predictability_score', 0)
        
        if pred_score > 0.7:
            return ("The high predictability of cluster transitions suggests that context effects "
                   "follow systematic rules based on token properties. This supports the hypothesis "
                   "that transformers learn structured mappings for how context modifies representations.")
        elif pred_score > 0.3:
            return ("Moderate predictability indicates that while token properties influence how "
                   "context affects representations, additional factors (possibly learned patterns "
                   "or semantic relationships) also play a significant role.")
        else:
            return ("Low predictability from token features alone suggests that context transformations "
                   "are primarily driven by learned semantic patterns rather than surface properties. "
                   "This indicates sophisticated context-dependent processing in the model.")


if __name__ == "__main__":
    # Example usage
    analysis = PredictiveTransformationModel("config_unified.yaml")
    analysis.run()