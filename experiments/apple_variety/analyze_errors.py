"""Analyze misclassifications in apple quality routing predictions."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from concept_fragmentation.models.feedforward import FeedforwardNetwork

def load_model_and_data(config_path='experiments/apple_variety/config_test.yaml'):
    """Load the trained model and test data."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    df = pd.read_csv(config['dataset']['data_path'])
    
    # Filter to known routing classes
    known_routing = ['fresh_premium', 'fresh_standard', 'juice']
    df = df[df['routing'].isin(known_routing)].copy()
    
    # Filter varieties
    min_samples = config['dataset'].get('min_samples_per_variety', 1)
    if min_samples > 1:
        variety_counts = df['variety'].value_counts()
        valid_varieties = variety_counts[variety_counts >= min_samples].index
        df = df[df['variety'].isin(valid_varieties)].copy()
    
    # Get features
    feature_cols = config['dataset']['features']
    
    # Add engineered features
    df['sweetness_ratio'] = df['brix_numeric'] / (df['firmness_numeric'] + 1e-6)
    df['quality_index'] = (
        0.3 * df['brix_numeric'] / df['brix_numeric'].max() +
        0.3 * df['firmness_numeric'] / df['firmness_numeric'].max() +
        0.2 * df['size_numeric'] / df['size_numeric'].max() +
        0.2 * (1 - df['starch_numeric'] / df['starch_numeric'].max())
    )
    df['firmness_sugar_interaction'] = df['firmness_numeric'] * df['brix_numeric']
    df['maturity_size_ratio'] = df['starch_numeric'] / (df['size_numeric'] + 1e-6)
    df['seasonal_sugar_adjusted'] = df['brix_numeric'] * (1 + 0.1 * (df['season_numeric'] - 2))
    
    feature_cols.extend(['sweetness_ratio', 'quality_index', 
                       'firmness_sugar_interaction', 'maturity_size_ratio', 
                       'seasonal_sugar_adjusted'])
    
    # Prepare features and labels
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    routing_encoder = LabelEncoder()
    y_routing = routing_encoder.fit_transform(df['routing'])
    variety_encoder = LabelEncoder()
    y_variety = variety_encoder.fit_transform(df['variety'])
    
    # Get train/test split (recreate the same split used in training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, variety_train, variety_test = train_test_split(
        X, y_routing, y_variety, 
        test_size=config['dataset']['test_size'],
        random_state=config['experiment']['random_seed'],
        stratify=y_routing
    )
    
    # Further split for validation (to match training)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2,
        random_state=config['experiment']['random_seed'],
        stratify=y_train
    )
    
    # Load the trained model
    model_path = Path('experiments/apple_variety/results/apple_variety_test/model.pth')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Initialize model
    n_features = len(feature_cols)
    n_classes = len(routing_encoder.classes_)
    model_config = config['model']['architecture']
    
    model = FeedforwardNetwork(
        input_dim=n_features,
        output_dim=n_classes,
        hidden_layer_sizes=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        activation=model_config['activation']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Scale test data
    scaler = checkpoint['scaler']
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.numpy()
    
    return {
        'df': df.iloc[X_train.shape[0]:],  # Test portion of dataframe
        'X_test': X_test,
        'y_test': y_test,
        'predictions': predictions,
        'variety_test': variety_test,
        'routing_encoder': routing_encoder,
        'variety_encoder': variety_encoder,
        'feature_names': feature_cols
    }

def analyze_errors(data):
    """Perform comprehensive error analysis."""
    y_test = data['y_test']
    predictions = data['predictions']
    routing_classes = data['routing_encoder'].classes_
    variety_encoder = data['variety_encoder']
    variety_test = data['variety_test']
    
    # 1. Overall metrics
    print("="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, predictions, 
                              target_names=routing_classes))
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=routing_classes,
                yticklabels=routing_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('experiments/apple_variety/confusion_matrix.png')
    plt.close()
    
    # 3. Error analysis by variety
    print("\n" + "="*50)
    print("ERROR ANALYSIS BY VARIETY")
    print("="*50)
    
    variety_errors = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': []})
    
    for i in range(len(y_test)):
        variety_idx = variety_test[i]
        variety_name = variety_encoder.inverse_transform([variety_idx])[0]
        true_label = routing_classes[y_test[i]]
        pred_label = routing_classes[predictions[i]]
        
        variety_errors[variety_name]['total'] += 1
        if y_test[i] == predictions[i]:
            variety_errors[variety_name]['correct'] += 1
        else:
            variety_errors[variety_name]['errors'].append({
                'true': true_label,
                'pred': pred_label
            })
    
    # Sort by error rate
    error_rates = []
    for variety, stats in variety_errors.items():
        error_rate = 1 - (stats['correct'] / stats['total'])
        error_rates.append((variety, error_rate, stats))
    
    error_rates.sort(key=lambda x: x[1], reverse=True)
    
    print("\nVarieties with highest error rates:")
    for variety, error_rate, stats in error_rates[:10]:
        print(f"\n{variety}: {error_rate:.1%} error rate ({stats['total']} samples)")
        if stats['errors']:
            # Count error types
            error_types = defaultdict(int)
            for err in stats['errors']:
                error_types[f"{err['true']} -> {err['pred']}"] += 1
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count}")
    
    # 4. Feature analysis for misclassified samples
    print("\n" + "="*50)
    print("FEATURE ANALYSIS FOR MISCLASSIFIED SAMPLES")
    print("="*50)
    
    X_test = data['X_test']
    feature_names = data['feature_names']
    
    correct_mask = y_test == predictions
    incorrect_mask = ~correct_mask
    
    print(f"\nCorrect predictions: {np.sum(correct_mask)}")
    print(f"Incorrect predictions: {np.sum(incorrect_mask)}")
    
    # Compare feature distributions
    print("\nFeature differences (incorrect vs correct):")
    for i, feature in enumerate(feature_names[:5]):  # First 5 base features
        correct_mean = X_test[correct_mask, i].mean()
        incorrect_mean = X_test[incorrect_mask, i].mean()
        diff_pct = (incorrect_mean - correct_mean) / correct_mean * 100
        print(f"  {feature}: correct={correct_mean:.2f}, incorrect={incorrect_mean:.2f} ({diff_pct:+.1f}%)")
    
    # 5. Most confused class pairs
    print("\n" + "="*50)
    print("MOST CONFUSED CLASS PAIRS")
    print("="*50)
    
    class_pairs = defaultdict(int)
    for i in range(len(y_test)):
        if y_test[i] != predictions[i]:
            true_label = routing_classes[y_test[i]]
            pred_label = routing_classes[predictions[i]]
            class_pairs[f"{true_label} -> {pred_label}"] += 1
    
    sorted_pairs = sorted(class_pairs.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_pairs:
        print(f"  {pair}: {count} times")
    
    # Save detailed results
    results = {
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, predictions, 
                                                     target_names=routing_classes,
                                                     output_dict=True),
        'variety_error_rates': {v: {'error_rate': r, 'stats': s} 
                               for v, r, s in error_rates},
        'class_confusion_pairs': dict(sorted_pairs)
    }
    
    with open('experiments/apple_variety/error_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to error_analysis_results.json")

def main():
    """Run error analysis."""
    print("Loading model and data...")
    data = load_model_and_data()
    
    print("Analyzing errors...")
    analyze_errors(data)
    
    print("\nError analysis complete!")

if __name__ == '__main__':
    main()