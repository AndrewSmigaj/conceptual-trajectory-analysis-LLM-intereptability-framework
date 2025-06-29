"""K-fold cross-validation for apple quality routing model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import logging
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from concept_fragmentation.models.feedforward import FeedforwardNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess the apple dataset."""
    df = pd.read_csv(config['dataset']['data_path'])
    
    # Filter to known routing classes
    known_routing = ['fresh_premium', 'fresh_standard', 'juice']
    df = df[df['routing'].isin(known_routing)].copy()
    
    # Filter varieties with minimum samples
    min_samples = config['dataset'].get('min_samples_per_variety', 1)
    if min_samples > 1:
        variety_counts = df['variety'].value_counts()
        valid_varieties = variety_counts[variety_counts >= min_samples].index
        df = df[df['variety'].isin(valid_varieties)].copy()
    
    logger.info(f"Dataset size: {len(df)} samples")
    
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
    y = routing_encoder.fit_transform(df['routing'])
    
    return X, y, routing_encoder.classes_

def train_fold(X_train, y_train, X_val, y_val, config, device):
    """Train model on one fold."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model_config = config['model']['architecture']
    
    model = FeedforwardNetwork(
        input_dim=n_features,
        output_dim=n_classes,
        hidden_layer_sizes=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        activation=model_config['activation'],
        seed=config['experiment']['random_seed']
    ).to(device)
    
    # Setup training
    if config['training'].get('class_weights', False):
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop with early stopping
    n_epochs = config['training']['epochs']
    early_stopping_config = config['training']['early_stopping']
    patience = early_stopping_config['patience']
    min_delta = early_stopping_config['min_delta']
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    final_val_accuracy = 100 * val_correct / val_total
    
    return final_val_accuracy, len(train_losses)

def main():
    """Run k-fold cross-validation."""
    # Load configuration
    config_path = 'experiments/apple_variety/config_test.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    X, y, routing_classes = load_data(config)
    logger.info(f"Loaded {len(X)} samples with {len(routing_classes)} classes: {routing_classes}")
    
    # K-fold setup
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['experiment']['random_seed'])
    
    fold_accuracies = []
    fold_epochs = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\nFold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logger.info(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # Train on this fold
        val_accuracy, epochs = train_fold(X_train, y_train, X_val, y_val, config, device)
        
        fold_accuracies.append(val_accuracy)
        fold_epochs.append(epochs)
        
        logger.info(f"  Validation accuracy: {val_accuracy:.2f}%")
        logger.info(f"  Epochs trained: {epochs}")
    
    # Calculate statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_epochs = np.mean(fold_epochs)
    
    # Print results
    print("\n" + "="*50)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Number of folds: {n_folds}")
    print(f"Total samples: {len(X)}")
    print(f"\nFold accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    print(f"\nMean accuracy: {mean_accuracy:.2f}% (± {std_accuracy:.2f}%)")
    print(f"Mean epochs: {mean_epochs:.1f}")
    
    # Confidence interval (95%)
    confidence_interval = 1.96 * std_accuracy / np.sqrt(n_folds)
    print(f"95% Confidence interval: [{mean_accuracy - confidence_interval:.2f}%, {mean_accuracy + confidence_interval:.2f}%]")
    
    # Save results
    results = {
        'n_folds': n_folds,
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confidence_interval': confidence_interval,
        'fold_epochs': fold_epochs,
        'mean_epochs': mean_epochs,
        'config': config
    }
    
    output_file = 'cross_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()