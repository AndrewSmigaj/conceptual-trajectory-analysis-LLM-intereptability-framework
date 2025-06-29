"""Apple quality routing experiment with Concept Trajectory Analysis - Realistic version.

This experiment demonstrates how neural networks classify apple quality (routing)
using realistic variety-specific characteristics and tracks how different varieties 
flow through these quality predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from concept_fragmentation.experiments.base import BaseExperiment
from concept_fragmentation.experiments.config import ExperimentConfig
from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.activation.collector import ActivationCollector, CollectionConfig, ActivationFormat
from concept_fragmentation.clustering import select_optimal_k
from concept_fragmentation.analysis.cross_layer_metrics import (
    extract_paths, 
    compute_trajectory_fragmentation,
    analyze_cross_layer_metrics
)
from concept_fragmentation.visualization.sankey import SankeyGenerator
from concept_fragmentation.visualization.trajectory import TrajectoryVisualizer
from concept_fragmentation.visualization.configs import SankeyConfig, TrajectoryConfig
from concept_fragmentation.llm.analysis import ClusterAnalysis
from concept_fragmentation.llm.bias_audit import generate_bias_report

logger = logging.getLogger(__name__)


class AppleRealisticExperiment(BaseExperiment):
    """Experiment for apple quality routing with realistic variety-specific characteristics."""
    
    def __init__(self, config_path: str = "config_realistic.yaml"):
        """Initialize experiment with configuration."""
        
        # Load full config with new structure
        with open(config_path, 'r') as f:
            self.full_config = yaml.safe_load(f)
        
        # Set up base experiment config with correct name
        base_config = ExperimentConfig(
            name=self.full_config['experiment']['name'],
            description=self.full_config['experiment']['description'],
            output_dir=self.full_config['experiment']['output_dir'],
            random_seed=self.full_config['experiment']['random_seed']
        )
        
        super().__init__(base_config)
        self.config_path = config_path
        
        # Initialize encoders
        self.routing_encoder = LabelEncoder()
        self.variety_encoder = LabelEncoder()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_and_prepare_data(self) -> None:
        """Load and prepare the realistic synthetic apple dataset."""
        
        # Load dataset
        data_path = self.full_config['dataset']['data_path']
        df = pd.read_csv(data_path)
        
        # Filter out rows without routing
        known_routing = ['fresh_premium', 'fresh_standard', 'juice']
        df = df[df['routing'].isin(known_routing)].copy()
        
        logger.info(f"Dataset size: {len(df)} samples")
        logger.info(f"Routing distribution: {df['routing'].value_counts().to_dict()}")
        
        # Get numeric features
        numeric_features = self.full_config['dataset']['features']
        
        # Handle categorical features (variety)
        categorical_features = self.full_config['dataset'].get('categorical_features', [])
        
        # Prepare numeric features
        X_numeric = df[numeric_features].fillna(df[numeric_features].mean()).values
        
        # One-hot encode categorical features
        X_categorical = None
        if categorical_features:
            X_categorical_list = []
            for cat_feat in categorical_features:
                dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat)
                X_categorical_list.append(dummies.values)
            X_categorical = np.hstack(X_categorical_list)
            
            # Store feature names for later use
            cat_feature_names = []
            for cat_feat in categorical_features:
                cat_values = df[cat_feat].unique()
                cat_feature_names.extend([f"{cat_feat}_{val}" for val in sorted(cat_values)])
            
            # Combine numeric and categorical features
            X = np.hstack([X_numeric, X_categorical])
            self.feature_names = numeric_features + cat_feature_names
        else:
            X = X_numeric
            self.feature_names = numeric_features
        
        # Encode labels
        y_routing = self.routing_encoder.fit_transform(df['routing'])
        y_variety = self.variety_encoder.fit_transform(df['variety'])
        
        # Store class information
        self.routing_classes = self.routing_encoder.classes_
        self.variety_names = self.variety_encoder.classes_
        self.n_routing_classes = len(self.routing_classes)
        
        logger.info(f"Routing classes: {self.routing_classes}")
        logger.info(f"Number of varieties: {len(self.variety_names)}")
        logger.info(f"Total features: {X.shape[1]} ({len(numeric_features)} numeric + {X.shape[1] - len(numeric_features)} categorical)")
        
        # Train/test split
        X_train, X_test, y_train, y_test, variety_train, variety_test = train_test_split(
            X, y_routing, y_variety, 
            test_size=self.full_config['dataset']['test_size'],
            random_state=self.full_config['experiment']['random_seed'],
            stratify=y_routing
        )
        
        # Further split train into train/validation for early stopping
        X_train_split, X_val, y_train_split, y_val, variety_train_split, variety_val = train_test_split(
            X_train, y_train, variety_train,
            test_size=0.2,  # 20% of training data for validation
            random_state=self.full_config['experiment']['random_seed'],
            stratify=y_train
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for imbalanced dataset
        if self.full_config['training'].get('class_weights', False):
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_split),
                y=y_train_split
            )
            self.class_weights = torch.FloatTensor(class_weights).to(self.device)
            logger.info(f"Class weights: {dict(zip(self.routing_classes, class_weights))}")
        else:
            self.class_weights = None
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.LongTensor(y_train_split)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Store variety labels for analysis
        self.variety_train = variety_train_split
        self.variety_test = variety_test
        
        # Store original data for analysis
        self.X_train_original = X_train_split
        self.y_train_routing = y_train_split
        
        # Store economic information
        self.df_original = df
        self.economic_prices = self.full_config['analysis']['economic']['prices']
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        batch_size = self.full_config['training']['batch_size']
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model_config = self.full_config['model']['architecture']
        self.model = FeedforwardNetwork(
            input_dim=X.shape[1],  # Total features including one-hot encoded
            output_dim=self.n_routing_classes,
            hidden_layer_sizes=model_config['hidden_dims'],
            dropout_rate=model_config['dropout_rate'],
            activation=model_config['activation'],
            seed=self.full_config['experiment']['random_seed']
        )
        self.model.to(self.device)
        
        # Set up activation collector
        collection_config = CollectionConfig(
            device=str(self.device),
            format=ActivationFormat.NUMPY,
            include_metadata=True
        )
        self.activation_collector = ActivationCollector(collection_config)
        
        # Register the model for activation collection
        self.activation_collector.register_model(
            self.model,
            model_id='feedforward',
            include_patterns=['layer_*']  # Collect from all layers
        )
        
    def train_model(self) -> Dict[str, List[float]]:
        """Train the model with early stopping and class weights."""
        
        # Set up optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.full_config['training']['learning_rate'],
            weight_decay=self.full_config['training']['weight_decay']
        )
        
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Early stopping parameters
        early_stop_config = self.full_config['training']['early_stopping']
        patience = early_stop_config['patience']
        min_delta = early_stop_config['min_delta']
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        # Training loop
        epochs = self.full_config['training']['epochs']
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Store history
            history['train_losses'].append(avg_train_loss)
            history['train_accuracies'].append(train_accuracy)
            history['val_losses'].append(avg_val_loss)
            history['val_accuracies'].append(val_accuracy)
            
            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_accuracy:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_accuracy:.2f}%")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model from early stopping")
        
        # Final test evaluation
        self.model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        logger.info(f"Final Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        model_save_path = Path(self.config.output_dir) / "model.pth"
        torch.save(self.model.state_dict(), model_save_path)
        
        return history
    
    def collect_activations(self) -> Dict[int, np.ndarray]:
        """Collect activations for all training samples."""
        
        self.model.eval()
        
        # Collect activations for each layer
        all_activations = {0: [], 1: [], 2: [], 3: []}  # 4 layers total
        
        # Run forward pass and collect activations from model's internal storage
        with torch.no_grad():
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(self.device)
                _ = self.model(batch_x)
                
                # The model stores activations in self.activations
                all_activations[0].append(self.model.activations['input'].cpu().numpy())
                all_activations[1].append(self.model.activations['layer1'].cpu().numpy())
                all_activations[2].append(self.model.activations['layer2'].cpu().numpy())
                all_activations[3].append(self.model.activations['layer3'].cpu().numpy())
        
        # Concatenate activations from all batches
        final_activations = {}
        for layer_idx, acts_list in all_activations.items():
            final_activations[layer_idx] = np.vstack(acts_list)
        
        logger.info(f"Collected activations for {len(final_activations)} layers")
        for layer_idx, acts in final_activations.items():
            logger.info(f"  Layer {layer_idx}: shape {acts.shape}")
        
        return final_activations
    
    def cluster_activations(self, activations: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """Cluster activations at each layer with variety-aware clustering."""
        
        cluster_results = {}
        
        for layer_idx, layer_acts in activations.items():
            logger.info(f"Clustering layer {layer_idx} with shape {layer_acts.shape}")
            
            # Select optimal number of clusters
            optimal_k = select_optimal_k(
                layer_acts,
                min_k=self.full_config['clustering']['k_min'],
                max_k=self.full_config['clustering']['k_max'],
                method=self.full_config['clustering']['k_selection'],
                random_state=self.full_config['clustering']['random_state']
            )
            
            logger.info(f"Layer {layer_idx}: optimal k = {optimal_k}")
            
            # Perform clustering
            kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=self.full_config['clustering']['random_state'],
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(layer_acts)
            
            # Analyze cluster composition
            cluster_composition = self._analyze_cluster_composition(
                cluster_labels, 
                self.y_train_routing, 
                self.variety_train
            )
            
            cluster_results[layer_idx] = {
                'labels': cluster_labels,
                'centers': kmeans.cluster_centers_,
                'n_clusters': optimal_k,
                'composition': cluster_composition
            }
        
        return cluster_results
    
    def _analyze_cluster_composition(self, cluster_labels: np.ndarray, 
                                   routing_labels: np.ndarray,
                                   variety_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze what's in each cluster - both routing and variety."""
        
        composition = {'routing': {}, 'variety': {}}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            
            # Routing composition
            routing_in_cluster = routing_labels[mask]
            routing_counts = {}
            for routing_id in np.unique(routing_in_cluster):
                routing_name = self.routing_classes[routing_id]
                routing_counts[routing_name] = int(np.sum(routing_in_cluster == routing_id))
            composition['routing'][int(cluster_id)] = routing_counts
            
            # Variety composition
            variety_in_cluster = variety_labels[mask]
            variety_counts = {}
            for variety_id in np.unique(variety_in_cluster):
                variety_name = self.variety_names[variety_id]
                count = int(np.sum(variety_in_cluster == variety_id))
                if count > 2:  # Only show varieties with meaningful presence
                    variety_counts[variety_name] = count
            composition['variety'][int(cluster_id)] = variety_counts
        
        return composition
    
    def analyze_trajectories(self, cluster_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trajectories with variety-specific insights."""
        
        # Convert to format expected by extract_paths (just the labels)
        cluster_labels_str = {}
        for layer_idx, results in cluster_results.items():
            cluster_labels_str[f"layer_{layer_idx}"] = results['labels']
        
        # Extract paths
        paths_by_sample, layer_names = extract_paths(cluster_labels_str)
        
        # Compute fragmentation
        fragmentation = compute_trajectory_fragmentation(
            paths_by_sample,
            layer_names,
            labels=self.y_train_routing
        )
        
        # Skip cross-layer metrics for now (would need activations)
        cross_layer_metrics = {}
        
        # Variety-specific trajectory analysis
        variety_trajectories = self._analyze_variety_trajectories(
            paths_by_sample, 
            self.variety_train
        )
        
        # Economic impact analysis
        economic_impact = None
        if self.full_config['analysis']['economic']['compute_impact']:
            economic_impact = self._analyze_economic_impact(
                paths_by_sample,
                self.y_train_routing,
                self.variety_train
            )
        
        return {
            'paths': paths_by_sample,
            'fragmentation': fragmentation,
            'cross_layer_metrics': cross_layer_metrics,
            'variety_trajectories': variety_trajectories,
            'economic_impact': economic_impact
        }
    
    def _analyze_variety_trajectories(self, paths: List[List[int]], 
                                    variety_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze how different varieties flow through the network."""
        
        variety_flows = {}
        
        for variety_id in np.unique(variety_labels):
            variety_name = self.variety_names[variety_id]
            variety_mask = variety_labels == variety_id
            variety_paths = [paths[i] for i in range(len(paths)) if variety_mask[i]]
            
            if len(variety_paths) > 5:  # Only analyze varieties with enough samples
                # Find most common paths for this variety
                path_tuples = [tuple(path) for path in variety_paths]
                unique_paths, counts = np.unique(path_tuples, return_counts=True, axis=0)
                
                # Sort by frequency
                sorted_idx = np.argsort(counts)[::-1]
                top_paths = []
                for idx in sorted_idx[:3]:  # Top 3 paths
                    # Convert path tuple back to list
                    path = [int(x) for x in unique_paths[idx]]
                    top_paths.append({
                        'path': path,
                        'count': int(counts[idx]),
                        'percentage': float(counts[idx] / len(variety_paths) * 100)
                    })
                
                variety_flows[variety_name] = {
                    'total_samples': len(variety_paths),
                    'unique_paths': len(unique_paths),
                    'top_paths': top_paths
                }
        
        return variety_flows
    
    def _analyze_economic_impact(self, paths: List[List[int]], 
                               routing_labels: np.ndarray,
                               variety_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze economic impact of routing decisions."""
        
        prices = self.economic_prices
        
        # Calculate total value by routing
        routing_values = {}
        for routing_id, routing_name in enumerate(self.routing_classes):
            mask = routing_labels == routing_id
            count = np.sum(mask)
            value_per_lb = prices[routing_name]
            routing_values[routing_name] = {
                'count': int(count),
                'price_per_lb': value_per_lb,
                'total_value': float(count * value_per_lb)
            }
        
        # Calculate loss from juice routing by variety
        variety_losses = {}
        for variety_id in np.unique(variety_labels):
            variety_name = self.variety_names[variety_id]
            variety_mask = variety_labels == variety_id
            
            # Get base price for this variety
            variety_info = self.df_original[self.df_original['variety'] == variety_name].iloc[0]
            base_price = variety_info['variety_base_price']
            
            # Count juice routing for this variety
            juice_mask = (variety_mask) & (routing_labels == self.routing_encoder.transform(['juice'])[0])
            juice_count = np.sum(juice_mask)
            
            if juice_count > 0:
                loss_per_lb = base_price - prices['juice']
                total_loss = juice_count * loss_per_lb
                
                variety_losses[variety_name] = {
                    'juice_count': int(juice_count),
                    'total_count': int(np.sum(variety_mask)),
                    'juice_percentage': float(juice_count / np.sum(variety_mask) * 100),
                    'base_price': base_price,
                    'loss_per_lb': loss_per_lb,
                    'total_loss': float(total_loss)
                }
        
        # Sort by total loss
        sorted_losses = dict(sorted(variety_losses.items(), 
                                  key=lambda x: x[1]['total_loss'], 
                                  reverse=True))
        
        return {
            'routing_values': routing_values,
            'variety_losses': sorted_losses,
            'total_loss': sum(v['total_loss'] for v in variety_losses.values())
        }
    
    def generate_visualizations(self, cluster_results: Dict[int, Dict[str, Any]], 
                              trajectory_results: Dict[str, Any]) -> None:
        """Generate variety-aware visualizations."""
        
        output_dir = Path(self.config.output_dir)
        
        # Sankey diagrams for different layer windows
        windows = self.full_config['trajectory_analysis']['windows']
        
        for window_name, (start_layer, end_layer) in windows.items():
            logger.info(f"Generating Sankey diagram for {window_name} window")
            
            # Prepare data for visualization
            vis_data = self._prepare_visualization_data(
                cluster_results, 
                trajectory_results,
                start_layer, 
                end_layer
            )
            
            # Generate Sankey
            sankey_config = SankeyConfig(
                height=self.full_config['visualization']['sankey']['height'],
                width=self.full_config['visualization']['sankey']['width']
            )
            
            sankey_gen = SankeyGenerator(sankey_config)
            sankey_path = output_dir / f"sankey_{window_name}.html"
            
            # Create figure - pass the window name from the data
            window_key = list(vis_data['windowed_analysis'].keys())[0]
            fig = sankey_gen.create_figure(vis_data, window=window_key)
            
            # Save figure
            sankey_gen.save_figure(fig, sankey_path)
            
        # Generate trajectory visualization
        if trajectory_results:
            self._generate_trajectory_viz(cluster_results, trajectory_results)
    
    def _prepare_visualization_data(self, cluster_results: Dict[int, Dict[str, Any]],
                                  trajectory_results: Dict[str, Any],
                                  start_layer: int, end_layer: int) -> Dict[str, Any]:
        """Prepare data for variety-aware visualization in SankeyData format."""
        
        # Get paths for the window
        paths = trajectory_results['paths']
        window_layers = list(range(start_layer, end_layer + 1))
        
        # Count path frequencies
        path_counts = {}
        path_varieties = {}
        for i, path in enumerate(paths):
            window_path = tuple(path[start_layer:end_layer+1])
            path_counts[window_path] = path_counts.get(window_path, 0) + 1
            
            # Track varieties for each path
            if window_path not in path_varieties:
                path_varieties[window_path] = []
            variety_name = self.variety_names[self.variety_train[i]]
            path_varieties[window_path].append(variety_name)
        
        # Create archetypal paths (top paths)
        archetypal_paths = []
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        
        for path_tuple, count in sorted_paths[:self.full_config['visualization']['sankey']['top_n_paths']]:
            # Get most common varieties for this path
            variety_counter = Counter(path_varieties[path_tuple])
            top_varieties = [v[0] for v in variety_counter.most_common(3)]
            
            archetypal_paths.append({
                'path': list(path_tuple),
                'frequency': count,
                'representative_words': top_varieties,  # Using varieties as representatives
                'percentage': count / len(paths) * 100
            })
        
        # Create labels for clusters
        labels = {}
        for layer_idx, results in cluster_results.items():
            layer_key = f"L{layer_idx}"
            labels[layer_key] = {}
            
            for cluster_id in range(results['n_clusters']):
                cluster_key = f"L{layer_idx}_C{cluster_id}"
                
                # Get dominant routing for this cluster
                cluster_mask = results['labels'] == cluster_id
                cluster_routing = self.y_train_routing[cluster_mask]
                if len(cluster_routing) > 0:
                    routing_counts = Counter(cluster_routing)
                    dominant_routing = self.routing_classes[routing_counts.most_common(1)[0][0]]
                else:
                    dominant_routing = "unknown"
                
                labels[layer_key][cluster_key] = {
                    'label': f"C{cluster_id}:{dominant_routing[:4]}",
                    'count': int(np.sum(cluster_mask))
                }
        
        # Create windowed analysis structure
        window_name = f"{start_layer}_{end_layer}"
        windowed_analysis = {
            window_name: {
                'layers': window_layers,
                'total_paths': len(paths),
                'unique_paths': len(path_counts),
                'archetypal_paths': archetypal_paths
            }
        }
        
        return {
            'windowed_analysis': windowed_analysis,
            'labels': labels,
            'purity_data': None  # Skip purity for now
        }
    
    def _generate_trajectory_viz(self, cluster_results: Dict[int, Dict[str, Any]],
                               trajectory_results: Dict[str, Any]) -> None:
        """Generate 3D trajectory visualization."""
        
        output_dir = Path(self.config.output_dir)
        
        # Get all activations
        all_layer_acts = []
        for layer_idx in sorted(cluster_results.keys()):
            all_layer_acts.append(cluster_results[layer_idx]['centers'])
        
        # Configure visualization
        traj_config = TrajectoryConfig(
            reduction_method=self.full_config['visualization']['trajectory']['method'],
            dimensions=self.full_config['visualization']['trajectory']['n_components'],
            backend='plotly',
            color_by='class',
            show_arrows=True,
            show_layer_labels=True,
            show_legend=True
        )
        
        # Create visualizer
        visualizer = TrajectoryVisualizer(traj_config)
        
        # Skip trajectory visualization for now - would need to fix method name
        logger.info("Skipping 3D trajectory visualization for now")
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save all experiment results including variety-specific insights."""
        
        output_dir = Path(self.config.output_dir)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        # Save main results
        results_serializable = convert_to_serializable(results)
        results_path = output_dir / f"{self.config.name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save variety trajectory summary
        if 'trajectory_analysis' in results and 'variety_trajectories' in results['trajectory_analysis']:
            variety_summary_path = output_dir / "variety_trajectory_summary.txt"
            with open(variety_summary_path, 'w') as f:
                f.write("VARIETY TRAJECTORY ANALYSIS\n")
                f.write("="*60 + "\n\n")
                
                for variety, info in results['trajectory_analysis']['variety_trajectories'].items():
                    f.write(f"{variety}:\n")
                    f.write(f"  Total samples: {info['total_samples']}\n")
                    f.write(f"  Unique paths: {info['unique_paths']}\n")
                    f.write("  Top paths:\n")
                    for path_info in info['top_paths']:
                        f.write(f"    {path_info['path']} - {path_info['count']} samples ({path_info['percentage']:.1f}%)\n")
                    f.write("\n")
        
        # Save economic impact report
        if 'trajectory_analysis' in results and results['trajectory_analysis'].get('economic_impact'):
            economic_report_path = output_dir / "economic_impact_report.txt"
            with open(economic_report_path, 'w') as f:
                impact = results['trajectory_analysis']['economic_impact']
                
                f.write("ECONOMIC IMPACT ANALYSIS\n")
                f.write("="*60 + "\n\n")
                
                f.write("Routing Value Summary:\n")
                for routing, info in impact['routing_values'].items():
                    f.write(f"  {routing}: {info['count']} samples @ ${info['price_per_lb']:.2f}/lb = ${info['total_value']:.2f}\n")
                
                f.write(f"\nTotal Loss from Juice Routing: ${impact['total_loss']:.2f}\n\n")
                
                f.write("Variety Loss Analysis (sorted by total loss):\n")
                for variety, loss_info in impact['variety_losses'].items():
                    f.write(f"\n{variety}:\n")
                    f.write(f"  Juice routing: {loss_info['juice_count']}/{loss_info['total_count']} ({loss_info['juice_percentage']:.1f}%)\n")
                    f.write(f"  Loss per lb: ${loss_info['base_price']:.2f} - $0.06 = ${loss_info['loss_per_lb']:.2f}\n")
                    f.write(f"  Total loss: ${loss_info['total_loss']:.2f}\n")
        
        logger.info(f"Results saved to {output_dir}")
    
    def setup(self) -> None:
        """Set up the experiment (required by BaseExperiment)."""
        self.load_and_prepare_data()
        
    def execute(self) -> Dict[str, Any]:
        """Execute the experiment (required by BaseExperiment)."""
        # Train model
        logger.info("Training model...")
        training_history = self.train_model()
        
        # Collect activations
        logger.info("Collecting activations...")
        activations = self.collect_activations()
        
        # Store for later analysis
        self.activations = activations
        self.training_history = training_history
        
        return {
            'training_history': training_history,
            'activations': activations
        }
        
    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the results (required by BaseExperiment)."""
        # Cluster activations
        logger.info("Clustering activations...")
        cluster_results = self.cluster_activations(self.activations)
        
        # Analyze trajectories
        logger.info("Analyzing trajectories...")
        trajectory_results = self.analyze_trajectories(cluster_results)
        
        return {
            'cluster_results': cluster_results,
            'trajectory_analysis': trajectory_results
        }
        
    def visualize(self, results: Dict[str, Any]) -> None:
        """Generate visualizations (required by BaseExperiment)."""
        cluster_results = results['cluster_results']
        trajectory_results = results['trajectory_analysis']
        self.generate_visualizations(cluster_results, trajectory_results)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline."""
        
        logger.info(f"Starting experiment: {self.config.name}")
        
        # Setup
        logger.info("Setting up experiment...")
        self.setup()
        
        # Execute
        exec_results = self.execute()
        
        # Analyze
        analysis_results = self.analyze(exec_results)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self.visualize(analysis_results)
        
        # Compile results
        results = {
            'config': self.full_config,
            'training_history': self.training_history,
            'cluster_results': analysis_results['cluster_results'],
            'trajectory_analysis': analysis_results['trajectory_analysis'],
            'variety_names': self.variety_names.tolist(),
            'routing_classes': self.routing_classes.tolist()
        }
        
        # Save results
        self.save_results(results)
        
        logger.info("Experiment completed successfully!")
        
        return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiment
    experiment = AppleRealisticExperiment("experiments/apple_variety/config_realistic.yaml")
    results = experiment.run()
    
    # Print summary
    print("\nEXPERIMENT SUMMARY")
    print("="*60)
    print(f"Final test accuracy: {results['training_history']['val_accuracies'][-1]:.2f}%")
    print(f"Number of varieties: {len(results['variety_names'])}")
    print(f"Routing classes: {results['routing_classes']}")
    
    if 'economic_impact' in results['trajectory_analysis'] and results['trajectory_analysis']['economic_impact']:
        print(f"\nTotal economic loss from juice routing: ${results['trajectory_analysis']['economic_impact']['total_loss']:.2f}")