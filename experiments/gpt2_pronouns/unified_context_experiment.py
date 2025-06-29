"""
Unified Context Effects Experiment

Analyzes how context affects token trajectories using a unified clustering approach.
All activations are pooled and clustered together to create a common space.
"""

import json
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import logging
from tqdm import tqdm
from datetime import datetime
import pickle

# Import from existing infrastructure
import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from concept_fragmentation.experiments.base import BaseExperiment

# For GPT-2 activation extraction, we'll implement it here since the path doesn't exist
from transformers import GPT2Model, GPT2Tokenizer

class SimpleGPT2ActivationExtractor:
    """GPT-2 activation extractor."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        """Load GPT-2 model and tokenizer."""
        logger.info("Loading GPT-2 model...")
        self.model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def extract_activations(self, texts: List[str]) -> Dict:
        """Extract activations for a batch of texts."""
        # Tokenize texts
        encoded = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Extract hidden states for all layers
        hidden_states = outputs.hidden_states  # tuple of tensors, one per layer
        
        # Convert to format expected by experiment
        activations = {}
        for batch_idx in range(len(texts)):
            activations[batch_idx] = {}
            for position in range(input_ids.shape[1]):
                if attention_mask[batch_idx, position] == 1:  # Valid token
                    activations[batch_idx][position] = {}
                    for layer_idx, layer_hidden in enumerate(hidden_states):
                        # Extract activation for this token
                        activation = layer_hidden[batch_idx, position].cpu().numpy()
                        activations[batch_idx][position][layer_idx] = activation
        
        return {'activations': activations}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedContextExperiment(BaseExperiment):
    """Experiment to study context effects using unified clustering."""
    
    def __init__(self, config_path: str = "config_unified.yaml"):
        """Initialize the experiment."""
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config object with nested structure
        self.config = type('Config', (), config_dict['experiment'])()
        # Add output_dir from output config if not in experiment config
        if not hasattr(self.config, 'output_dir') and 'output' in config_dict and 'base_dir' in config_dict['output']:
            self.config.output_dir = config_dict['output']['base_dir']
        
        # Convert nested dicts to objects
        for key in ['model', 'analysis', 'data', 'output', 'clustering']:
            if key in config_dict:
                setattr(self.config, key, type(f'Config{key}', (), config_dict[key])())
        
        super().__init__(self.config)
        
        # Initialize components
        self.extractor = SimpleGPT2ActivationExtractor()
        self.test_cases = []
        self.activations = defaultdict(lambda: defaultdict(list))  # layer -> position -> activations
        self.trajectories = {}
        
        # Checkpointing
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_processed_idx = 0
        
    def setup(self) -> None:
        """Set up the experiment."""
        logger.info("Setting up unified context effects experiment")
        
        # Load test cases
        test_cases_path = Path(self.config.data.test_cases_path)
        with open(test_cases_path, 'r') as f:
            data = json.load(f)
            self.test_cases = data['test_cases']
            self.metadata = data['metadata']
            
        logger.info(f"Loaded {len(self.test_cases)} test cases")
        
        # Setup GPT-2 model
        self.extractor.setup_model()
        
        # Check for existing checkpoint
        self._load_checkpoint()
        
    def _load_checkpoint(self) -> None:
        """Load from checkpoint if available."""
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.last_processed_idx = checkpoint['last_processed_idx']
                logger.info(f"Resuming from checkpoint at index {self.last_processed_idx}")
                
            # Load partial activations
            activations_file = self.checkpoint_dir / "activations_partial.pkl"
            if activations_file.exists():
                with open(activations_file, 'rb') as f:
                    self.activations = pickle.load(f)
                logger.info(f"Loaded activations for {sum(len(acts) for layer_acts in self.activations.values() for acts in layer_acts.values())} examples")
                    
    def _save_checkpoint(self, idx: int) -> None:
        """Save checkpoint."""
        checkpoint = {
            'last_processed_idx': idx,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_dir / "latest_checkpoint.json", 'w') as f:
            json.dump(checkpoint, f)
            
        # Save activations
        with open(self.checkpoint_dir / "activations_partial.pkl", 'wb') as f:
            pickle.dump(dict(self.activations), f)
            
        logger.info(f"Checkpoint saved at index {idx}")
        
    def execute(self) -> Dict[str, Any]:
        """Execute the main experiment logic."""
        logger.info("Executing unified context effects experiment")
        
        # Phase 1: Extract activations
        self._extract_all_activations()
        
        # Save final activations
        self._save_activations()
        
        # Phase 2: Perform unified clustering
        cluster_models = self._perform_unified_clustering()
        
        # Phase 3: Build trajectories
        self._build_trajectories(cluster_models)
        
        # Save results
        self._save_results()
        
        return {
            'num_test_cases': len(self.test_cases),
            'num_trajectories': len(self.trajectories),
            'num_layers': getattr(self.config.model, 'num_layers', None) or getattr(self.config.model, 'n_layers', 12)
        }
        
    def _extract_all_activations(self) -> None:
        """Extract activations for all test cases."""
        batch_size = getattr(self.config.data, 'batch_size', 32)
        total_cases = len(self.test_cases)
        
        # Process in batches
        for batch_start in range(self.last_processed_idx, total_cases, batch_size):
            batch_end = min(batch_start + batch_size, total_cases)
            batch_cases = self.test_cases[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start}-{batch_end} of {total_cases}")
            
            # Extract texts and positions
            # Get texts - handle different field names
            texts = [case.get('full_text', case.get('text', '')) for case in batch_cases]
            # Calculate positions based on context
            positions = []
            for case in batch_cases:
                if case.get('context') == 'baseline':
                    positions.append(0)
                else:
                    # For context cases, target is after context
                    context_str = case.get('context_str', case.get('context', ''))
                    if context_str:
                        context_tokens = self.extractor.tokenizer.encode(context_str, add_special_tokens=False)
                        positions.append(len(context_tokens))
                    else:
                        positions.append(0)
            
            # Get activations from GPT-2
            batch_result = self.extractor.extract_activations(texts)
            batch_activations = batch_result['activations']
            
            # Store activations by layer and position
            for i, (case, position) in enumerate(zip(batch_cases, positions)):
                # Get activations for target token
                if i in batch_activations:
                    # Check if the target position exists in the activations
                    if position in batch_activations[i]:
                        # Get activations at target position for all layers
                        # Get number of layers from config
                        n_layers = getattr(self.config.model, 'num_layers', None) or getattr(self.config.model, 'n_layers', 12)
                        for layer in range(n_layers):
                            if layer in batch_activations[i][position]:
                                activation = batch_activations[i][position][layer]
                                
                                # Store with case index for reference
                                self.activations[layer][batch_start + i].append({
                                    'activation': activation,
                                    'case_idx': batch_start + i,
                                    'token_idx': case.get('token_id', case.get('token_idx')),
                                    'context': case.get('context', case.get('context_frame'))
                                })
                            
            # Save checkpoint periodically
            if batch_end % getattr(self.config.output, 'checkpoint_interval', 500) == 0:
                self._save_checkpoint(batch_end)
                
        # Final checkpoint
        self._save_checkpoint(total_cases)
        
    def _save_activations(self) -> None:
        """Save extracted activations."""
        output_file = self.output_dir / "unified_activations.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(dict(self.activations), f)
        logger.info(f"Saved activations to {output_file}")
        
    def _perform_unified_clustering(self) -> Dict[int, Any]:
        """Perform unified clustering on pooled activations."""
        from sklearn.cluster import KMeans
        import joblib
        
        cluster_models = {}
        cluster_dir = self.output_dir / "cluster_models"
        cluster_dir.mkdir(exist_ok=True)
        
        # Get k from clustering config or analysis config
        if hasattr(self.config, 'clustering') and hasattr(self.config.clustering, 'k_clusters'):
            k = self.config.clustering.k_clusters
        else:
            k = getattr(self.config.analysis, 'k_clusters', 20)
        
        # Get number of layers from config
        n_layers = getattr(self.config.model, 'num_layers', None) or getattr(self.config.model, 'n_layers', 12)
        for layer in range(n_layers):
            logger.info(f"Clustering layer {layer} with k={k}")
            
            # Collect all activations for this layer
            layer_activations = []
            activation_metadata = []
            
            for case_idx in sorted(self.activations[layer].keys()):
                for act_data in self.activations[layer][case_idx]:
                    layer_activations.append(act_data['activation'])
                    activation_metadata.append({
                        'case_idx': act_data['case_idx'],
                        'token_idx': act_data['token_idx'],
                        'context': act_data['context']
                    })
                    
            if not layer_activations:
                logger.warning(f"No activations for layer {layer}")
                continue
                
            # Convert to numpy array
            X = np.array(layer_activations)
            logger.info(f"  Layer {layer}: {X.shape[0]} activations, dim={X.shape[1]}")
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Save model
            model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
            joblib.dump(kmeans, model_path)
            
            # Save cluster assignments with metadata
            assignments_path = cluster_dir / f"assignments_layer_{layer}.json"
            assignments = []
            for i, (label, meta) in enumerate(zip(labels, activation_metadata)):
                assignments.append({
                    'cluster': int(label),
                    **meta
                })
            
            with open(assignments_path, 'w') as f:
                json.dump(assignments, f)
                
            cluster_models[layer] = kmeans
            
            # Print cluster statistics
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"  Cluster sizes: {dict(zip(unique, counts))}")
            
        return cluster_models
        
    def _build_trajectories(self, cluster_models: Dict[int, Any]) -> None:
        """Build trajectories using cluster assignments."""
        logger.info("Building trajectories from cluster assignments")
        
        # For each test case, get its trajectory
        for case_idx, case in enumerate(self.test_cases):
            trajectory = []
            
            # Get number of layers from config
            n_layers = getattr(self.config.model, 'num_layers', None) or getattr(self.config.model, 'n_layers', 12)
            for layer in range(n_layers):
                if layer in self.activations and case_idx in self.activations[layer]:
                    # Get the activation for this case
                    act_data = self.activations[layer][case_idx][0]  # Should only be one
                    activation = act_data['activation'].reshape(1, -1)
                    
                    # Predict cluster
                    if layer in cluster_models:
                        cluster = cluster_models[layer].predict(activation)[0]
                        trajectory.append(int(cluster))
                    else:
                        trajectory.append(-1)  # Missing
                else:
                    trajectory.append(-1)  # Missing
                    
            # Store trajectory
            key = f"{case['token_idx']}_{case['context_frame']}"
            self.trajectories[key] = {
                'token_idx': case['token_idx'],
                'token_str': case['token_str'],
                'context_frame': case['context_frame'],
                'path': trajectory,
                'case_idx': case_idx
            }
            
        logger.info(f"Built {len(self.trajectories)} trajectories")
        
    def _save_results(self) -> None:
        """Save trajectories and metadata."""
        results = {
            'metadata': {
                'experiment': 'unified_context_effects',
                'timestamp': datetime.now().isoformat(),
                'num_tokens': self.metadata['num_tokens'],
                'num_contexts': self.metadata['num_contexts'],
                'num_test_cases': len(self.test_cases),
                'k_clusters': getattr(self.config.analysis, 'k_clusters', 20)
            },
            'trajectories': self.trajectories
        }
        
        output_file = self.output_dir / "unified_trajectories.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved trajectories to {output_file}")
        
        # Also save visualization data
        vis_data = {
            'trajectories': self.trajectories,
            'context_frames': self.metadata['context_frames']
        }
        
        vis_file = self.output_dir / "visualization_data.json"
        with open(vis_file, 'w') as f:
            json.dump(vis_data, f, indent=2)
            
    def analyze(self) -> Dict[str, Any]:
        """Analyze the results (placeholder for now)."""
        # This will be implemented in separate analysis scripts
        return {
            'analysis': 'See analysis scripts for detailed results'
        }
        
    def visualize(self) -> Dict[str, str]:
        """Generate visualizations (placeholder for now)."""
        # This will be implemented using existing visualization tools
        logger.info("Visualizations will be generated using existing tools")
        return {}


def main():
    """Run the unified context experiment."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_unified.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = UnifiedContextExperiment(args.config)
    experiment.setup()
    results = experiment.execute()
    
    logger.info(f"Experiment complete: {results}")
    

if __name__ == "__main__":
    main()