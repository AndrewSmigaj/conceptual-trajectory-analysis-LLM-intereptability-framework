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

# Import GPT-2 tools
from transformers import GPT2Model, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGPT2ActivationExtractor:
    """Simple GPT-2 activation extractor."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        """Load GPT-2 model and tokenizer."""
        logger.info("Loading GPT-2 model...")
        self.model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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


class UnifiedContextExperiment(BaseExperiment):
    """Experiment to study context effects using unified clustering."""
    
    def __init__(self, config_path: str = "config_unified.yaml"):
        """Initialize the experiment."""
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create a proper config object that BaseExperiment expects
        class Config:
            def __init__(self, config_dict):
                self.name = config_dict['experiment']['name']
                self.output_dir = config_dict['output']['base_dir']
                
                # Add experiment fields
                for key, value in config_dict['experiment'].items():
                    setattr(self, key, value)
                
                # Add nested configs as objects
                for section in ['model', 'analysis', 'data', 'output', 'clustering']:
                    if section in config_dict:
                        section_obj = type(f'Config{section.title()}', (), {})()
                        for k, v in config_dict[section].items():
                            setattr(section_obj, k, v)
                        setattr(self, section, section_obj)
        
        self.config = Config(config_dict)
        self.config_dict = config_dict  # Keep original dict
        
        # Call parent constructor
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
        
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
            
        # Save partial activations
        activations_file = self.checkpoint_dir / "activations_partial.pkl"
        with open(activations_file, 'wb') as f:
            pickle.dump(dict(self.activations), f)
            
        logger.info(f"Saved checkpoint at index {idx}")
        
    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        logger.info("Running unified context effects experiment")
        
        # Phase 1: Extract activations
        self._extract_activations()
        
        # Save activations
        self._save_activations()
        
        # Phase 2: Perform unified clustering
        cluster_models = self._perform_unified_clustering()
        
        # Phase 3: Build trajectories
        self._build_trajectories(cluster_models)
        
        # Phase 4: Analyze results
        results = self._analyze_results()
        
        return results
    
    def _extract_activations(self) -> None:
        """Extract activations for all test cases."""
        logger.info("Extracting activations...")
        
        batch_size = self.config.data.batch_size
        total_cases = len(self.test_cases)
        
        # Process in batches starting from checkpoint
        for batch_start in range(self.last_processed_idx, total_cases, batch_size):
            batch_end = min(batch_start + batch_size, total_cases)
            batch_cases = self.test_cases[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start}-{batch_end} of {total_cases}")
            
            # Extract texts
            texts = [case['full_text'] for case in batch_cases]
            
            # Get activations from GPT-2
            batch_result = self.extractor.extract_activations(texts)
            batch_activations = batch_result['activations']
            
            # Store activations by layer and case
            for i, case in enumerate(batch_cases):
                case_idx = batch_start + i
                
                # Get the position of the target token
                # For baseline, it's position 0; for context, it's position 1
                if case['context'] == 'baseline':
                    target_position = 0
                else:
                    # Context + space + token
                    context_tokens = self.extractor.tokenizer.encode(case['context_str'], add_special_tokens=False)
                    target_position = len(context_tokens)  # Position after context
                
                # Extract activations for target token
                if i in batch_activations and target_position in batch_activations[i]:
                    for layer in range(self.config.model.n_layers):
                        if layer in batch_activations[i][target_position]:
                            activation = batch_activations[i][target_position][layer]
                            
                            # Store with metadata
                            self.activations[layer][case_idx].append({
                                'activation': activation,
                                'case_idx': case_idx,
                                'token_id': case['token_id'],
                                'token_str': case['token_str'],
                                'context': case['context']
                            })
                            
            # Save checkpoint periodically
            if batch_end % self.config.output.checkpoint_frequency == 0:
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
        
        k_clusters = self.config.clustering.k_clusters
        
        for layer in range(self.config.model.n_layers):
            logger.info(f"Clustering layer {layer} with k={k_clusters}")
            
            # Collect all activations for this layer
            layer_activations = []
            activation_metadata = []
            
            for case_idx in sorted(self.activations[layer].keys()):
                for act_data in self.activations[layer][case_idx]:
                    layer_activations.append(act_data['activation'])
                    activation_metadata.append({
                        'case_idx': act_data['case_idx'],
                        'token_id': act_data['token_id'],
                        'token_str': act_data['token_str'],
                        'context': act_data['context']
                    })
            
            if not layer_activations:
                logger.warning(f"No activations for layer {layer}")
                continue
            
            # Convert to numpy array
            X = np.array(layer_activations)
            logger.info(f"  Layer {layer}: {X.shape[0]} activations, dim={X.shape[1]}")
            
            # Normalize if specified
            if self.config.clustering.normalize:
                from sklearn.preprocessing import normalize
                X = normalize(X, norm='l2')
            
            # Perform k-means clustering
            kmeans = KMeans(
                n_clusters=k_clusters,
                random_state=self.config.clustering.random_state,
                n_init=self.config.clustering.n_init,
                max_iter=self.config.clustering.max_iter
            )
            labels = kmeans.fit_predict(X)
            
            # Save model
            model_path = cluster_dir / f"kmeans_layer_{layer}.pkl"
            joblib.dump(kmeans, model_path)
            logger.info(f"  Saved model to {model_path}")
            
            # Save cluster assignments
            assignments = []
            for i, (label, meta) in enumerate(zip(labels, activation_metadata)):
                assignments.append({
                    'cluster': int(label),
                    **meta
                })
            
            assignments_path = cluster_dir / f"assignments_layer_{layer}.json"
            with open(assignments_path, 'w') as f:
                json.dump(assignments, f, indent=2)
            
            cluster_models[layer] = kmeans
            
            # Print cluster statistics
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"  Cluster sizes: {dict(zip(unique, counts))}")
        
        return cluster_models
    
    def _build_trajectories(self, cluster_models: Dict[int, Any]) -> None:
        """Build trajectories from cluster assignments."""
        logger.info("Building trajectories...")
        
        trajectories = defaultdict(lambda: defaultdict(list))
        
        # Read cluster assignments for each layer
        cluster_dir = self.output_dir / "cluster_models"
        
        for layer in range(self.config.model.n_layers):
            assignments_path = cluster_dir / f"assignments_layer_{layer}.json"
            if not assignments_path.exists():
                continue
                
            with open(assignments_path, 'r') as f:
                assignments = json.load(f)
            
            # Group by case_idx
            for assignment in assignments:
                case_idx = assignment['case_idx']
                cluster = assignment['cluster']
                trajectories[case_idx][layer] = cluster
        
        # Convert to list format
        self.trajectories = {}
        for case_idx, layer_clusters in trajectories.items():
            # Ensure we have assignments for all layers
            trajectory = []
            for layer in range(self.config.model.n_layers):
                if layer in layer_clusters:
                    trajectory.append(layer_clusters[layer])
                else:
                    trajectory.append(-1)  # Missing assignment
            
            self.trajectories[case_idx] = {
                'trajectory': trajectory,
                'token_id': self.test_cases[case_idx]['token_id'],
                'token_str': self.test_cases[case_idx]['token_str'],
                'context': self.test_cases[case_idx]['context']
            }
        
        # Save trajectories
        output_file = self.output_dir / "unified_trajectories.json"
        with open(output_file, 'w') as f:
            json.dump(self.trajectories, f, indent=2)
        logger.info(f"Saved {len(self.trajectories)} trajectories to {output_file}")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze the results."""
        logger.info("Analyzing results...")
        
        # Calculate basic statistics
        n_tokens = len(set(t['token_id'] for t in self.trajectories.values()))
        n_contexts = len(set(t['context'] for t in self.trajectories.values()))
        
        results = {
            'metadata': {
                'n_tokens': n_tokens,
                'n_contexts': n_contexts,
                'n_trajectories': len(self.trajectories),
                'k_clusters': self.config.clustering.k_clusters,
                'n_layers': self.config.model.n_layers
            },
            'trajectories': self.trajectories
        }
        
        return results
    
    def save_results(self) -> None:
        """Save all results."""
        # Results are already saved during the run
        logger.info("All results have been saved to output directory")


def main():
    """Run the experiment with command line argument for config."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unified context effects experiment')
    parser.add_argument('config', nargs='?', default='config_unified.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = UnifiedContextExperiment(config_path=args.config)
    experiment.setup()
    results = experiment.run()
    experiment.save_results()
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()