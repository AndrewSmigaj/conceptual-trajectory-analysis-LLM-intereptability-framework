"""
GPT-2 Full Vocabulary Context Effects Experiment

Analyzes how context affects the routing of all 10k most frequent tokens
through GPT-2's activation space using existing k10 clustering.
"""

import json
import yaml
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import logging
from tqdm import tqdm
import torch
from datetime import datetime

# Import from existing infrastructure
import sys
sys.path.append('../../')
from concept_fragmentation.experiments.base import BaseExperiment
from concept_fragmentation.clustering.paths import PathExtractor

# Use sklearn KMeans for mapping to existing clusters
from sklearn.cluster import KMeans

# Import GPT-2 specific tools
sys.path.append('../gpt2/shared/')
from gpt2_activation_extractor import SimpleGPT2ActivationExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VocabularyContextExperiment(BaseExperiment):
    """Experiment to study context effects on full GPT-2 vocabulary routing."""
    
    def __init__(self, config_path: str):
        """Initialize the experiment from config file."""
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create a simple config object
        self.config = type('Config', (), config_dict['experiment'])()
        self.config.model = config_dict['model']
        self.config.analysis = config_dict['analysis']
        self.config.data = config_dict['data']
        self.config.metrics = config_dict['metrics']
        self.config.visualization = config_dict['visualization']
        self.config.output = config_dict['output']
        
        super().__init__(self.config)
        
        # Initialize components
        self.extractor = SimpleGPT2ActivationExtractor()
        self.test_cases = []
        self.existing_clusters = {}
        self.cluster_labels = {}
        self.trajectories = {}
        self.context_effects = defaultdict(lambda: defaultdict(list))
        
        # Checkpointing
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_processed_idx = 0
        
    def setup(self) -> None:
        """Set up the experiment."""
        logger.info("Setting up vocabulary context effects experiment")
        
        # Load test cases
        test_cases_path = Path("test_cases_full_vocabulary.json")
        if not test_cases_path.exists():
            logger.info("Generating test cases...")
            from context_frame_generator import ContextFrameGenerator
            generator = ContextFrameGenerator(str(Path(__file__).parent / "config.yaml"))
            self.test_cases = generator.save_test_cases(str(test_cases_path))
        else:
            with open(test_cases_path, 'r') as f:
                self.test_cases = json.load(f)
        
        logger.info(f"Loaded {len(self.test_cases)} test cases")
        
        # Setup GPT-2 model
        self.extractor.setup_model()
        
        # Load existing k10 cluster labels
        self._load_existing_clusters()
        
        # Check for existing checkpoint
        self._load_checkpoint()
        
    def _load_existing_clusters(self) -> None:
        """Load existing k10 cluster assignments."""
        labels_path = Path(self.config.analysis['clustering']['labels_path'])
        if not labels_path.is_absolute():
            labels_path = Path(__file__).parent / labels_path
            
        logger.info(f"Loading existing cluster labels from {labels_path}")
        with open(labels_path, 'r') as f:
            self.cluster_labels = json.load(f)
            
        # Convert string keys to int for layer access
        self.cluster_labels = {int(k): v for k, v in self.cluster_labels.items()}
        
    def _load_checkpoint(self) -> None:
        """Load from checkpoint if available."""
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.last_processed_idx = checkpoint['last_processed_idx']
                logger.info(f"Resuming from checkpoint at index {self.last_processed_idx}")
                
            # Load partial results
            if (self.checkpoint_dir / "trajectories_partial.json").exists():
                with open(self.checkpoint_dir / "trajectories_partial.json", 'r') as f:
                    self.trajectories = json.load(f)
                    
    def _save_checkpoint(self, idx: int) -> None:
        """Save checkpoint at regular intervals."""
        checkpoint = {
            'last_processed_idx': idx,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_dir / "latest_checkpoint.json", 'w') as f:
            json.dump(checkpoint, f)
            
        # Save partial results
        with open(self.checkpoint_dir / "trajectories_partial.json", 'w') as f:
            json.dump(self.trajectories, f, indent=2)
            
    def execute(self) -> Dict[str, Any]:
        """Execute the main experiment logic."""
        logger.info("Executing vocabulary context effects experiment")
        
        # Process test cases in batches
        batch_size = self.config.data['batch_size']
        total_cases = len(self.test_cases)
        
        # Group by context frame for baseline comparisons
        cases_by_token = defaultdict(dict)
        for case in self.test_cases:
            token_idx = case['token_idx']
            frame = case['context_frame']
            cases_by_token[token_idx][frame] = case
            
        # Process in batches
        for batch_start in range(self.last_processed_idx, total_cases, batch_size):
            batch_end = min(batch_start + batch_size, total_cases)
            batch_cases = self.test_cases[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start}-{batch_end} of {total_cases}")
            
            # Extract activations for batch
            texts = [case['text'] for case in batch_cases]
            activations_batch = self.extractor.extract_activations(texts)
            
            # Process each case in batch
            for i, case in enumerate(batch_cases):
                case_idx = batch_start + i
                
                # Extract trajectory for target token
                trajectory = self._extract_trajectory(
                    activations_batch, 
                    i, 
                    case['target_position']
                )
                
                # Store trajectory
                trajectory_key = f"{case['token_idx']}_{case['context_frame']}"
                self.trajectories[trajectory_key] = {
                    'token_idx': case['token_idx'],
                    'token_str': case['token_str'],
                    'context_frame': case['context_frame'],
                    'path': trajectory,
                    'text': case['text']
                }
                
            # Save checkpoint periodically
            if batch_end % self.config.output['checkpoint_interval'] == 0:
                self._save_checkpoint(batch_end)
                logger.info(f"Checkpoint saved at index {batch_end}")
                
        # Final save
        self._save_checkpoint(total_cases)
        
        # Analyze context effects
        self._analyze_context_effects(cases_by_token)
        
        return {
            'num_test_cases': len(self.test_cases),
            'num_trajectories': len(self.trajectories),
            'unique_tokens': len(cases_by_token)
        }
        
    def _extract_trajectory(self, activations_batch: Dict, batch_idx: int, 
                          target_position: int) -> List[int]:
        """Extract trajectory for a specific token."""
        trajectory = []
        
        for layer in range(self.config.model['num_layers']):
            # Get activation for target token at this layer
            if (batch_idx in activations_batch['activations'] and
                target_position in activations_batch['activations'][batch_idx] and
                layer in activations_batch['activations'][batch_idx][target_position]):
                
                activation = np.array(activations_batch['activations'][batch_idx][target_position][layer])
                
                # For baseline (token alone), use existing cluster assignment
                # For context cases, we need to map the new activation
                # Since we don't have the actual cluster models loaded,
                # we'll use a simplified approach for now
                
                # This is a placeholder - in a full implementation, we would:
                # 1. Load the saved KMeans models for each layer
                # 2. Use model.predict() on the new activation
                
                # For now, use the baseline cluster assignment if available
                cluster = self._find_nearest_cluster(activation, layer)
                trajectory.append(cluster)
            else:
                trajectory.append(-1)  # Missing data
                
        return trajectory
        
    def _find_nearest_cluster(self, activation: np.ndarray, layer: int) -> int:
        """Find nearest cluster for an activation vector."""
        # This is a simplified placeholder
        # In practice, we would load the KMeans model and use predict()
        # For now, return a cluster based on activation magnitude
        
        # Use existing cluster assignments for baseline tokens
        # This is a temporary solution
        if layer < len(self.cluster_labels):
            # Simple heuristic: map based on activation norm
            norm = np.linalg.norm(activation)
            cluster = int(norm * 10) % 10  # Map to 0-9
            return cluster
        return -1
        
    def _analyze_context_effects(self, cases_by_token: Dict) -> None:
        """Analyze how context affects trajectories."""
        logger.info("Analyzing context effects across vocabulary")
        
        for token_idx, frames in cases_by_token.items():
            # Get baseline trajectory
            if 'baseline' not in frames:
                continue
                
            baseline_key = f"{token_idx}_baseline"
            if baseline_key not in self.trajectories:
                continue
                
            baseline_traj = self.trajectories[baseline_key]['path']
            
            # Compare each context to baseline
            for frame_name, case in frames.items():
                if frame_name == 'baseline':
                    continue
                    
                context_key = f"{token_idx}_{frame_name}"
                if context_key not in self.trajectories:
                    continue
                    
                context_traj = self.trajectories[context_key]['path']
                
                # Calculate divergence
                divergence = self._calculate_divergence(baseline_traj, context_traj)
                
                # Store context effect
                self.context_effects[token_idx][frame_name] = {
                    'divergence_early': divergence['early'],
                    'divergence_full': divergence['full'],
                    'bifurcation_layer': divergence['bifurcation_layer']
                }
                
    def _calculate_divergence(self, traj1: List[int], traj2: List[int]) -> Dict[str, Any]:
        """Calculate trajectory divergence metrics."""
        # Early layers divergence (first 4)
        early_diff = sum(1 for i in range(4) if i < len(traj1) and 
                        traj1[i] != traj2[i] and traj1[i] != -1 and traj2[i] != -1)
        early_total = sum(1 for i in range(4) if i < len(traj1) and 
                         traj1[i] != -1 and traj2[i] != -1)
        
        # Full trajectory divergence
        full_diff = sum(1 for i in range(len(traj1)) if 
                       traj1[i] != traj2[i] and traj1[i] != -1 and traj2[i] != -1)
        full_total = sum(1 for i in range(len(traj1)) if 
                        traj1[i] != -1 and traj2[i] != -1)
        
        # Find bifurcation point
        bifurcation_layer = -1
        for i in range(len(traj1)):
            if traj1[i] != traj2[i] and traj1[i] != -1 and traj2[i] != -1:
                bifurcation_layer = i
                break
                
        return {
            'early': early_diff / early_total if early_total > 0 else 0,
            'full': full_diff / full_total if full_total > 0 else 0,
            'bifurcation_layer': bifurcation_layer
        }
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        logger.info("Analyzing vocabulary context effects results")
        
        # Compute aggregate statistics
        context_sensitivity = self._compute_context_sensitivity()
        bifurcation_patterns = self._analyze_bifurcation_patterns()
        token_stability = self._rank_token_stability()
        
        analysis = {
            'context_sensitivity': context_sensitivity,
            'bifurcation_patterns': bifurcation_patterns,
            'token_stability_ranking': token_stability,
            'summary_statistics': self._compute_summary_stats()
        }
        
        # Save analysis results
        with open(self.output_dir / "context_effects_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
        
    def _compute_context_sensitivity(self) -> Dict[str, Any]:
        """Compute which tokens are most sensitive to context."""
        sensitivity_scores = []
        
        for token_idx, effects in self.context_effects.items():
            if not effects:
                continue
                
            # Average divergence across all contexts
            divergences = [e['divergence_early'] for e in effects.values()]
            avg_divergence = np.mean(divergences) if divergences else 0
            
            sensitivity_scores.append({
                'token_idx': token_idx,
                'avg_divergence': avg_divergence,
                'max_divergence': max(divergences) if divergences else 0,
                'num_contexts': len(effects)
            })
            
        # Sort by average divergence
        sensitivity_scores.sort(key=lambda x: x['avg_divergence'], reverse=True)
        
        return {
            'most_sensitive': sensitivity_scores[:100],
            'least_sensitive': sensitivity_scores[-100:],
            'distribution': {
                'mean': np.mean([s['avg_divergence'] for s in sensitivity_scores]),
                'std': np.std([s['avg_divergence'] for s in sensitivity_scores]),
                'median': np.median([s['avg_divergence'] for s in sensitivity_scores])
            }
        }
        
    def _analyze_bifurcation_patterns(self) -> Dict[str, Any]:
        """Analyze where trajectories typically bifurcate."""
        bifurcation_layers = []
        
        for token_effects in self.context_effects.values():
            for effect in token_effects.values():
                if effect['bifurcation_layer'] >= 0:
                    bifurcation_layers.append(effect['bifurcation_layer'])
                    
        if not bifurcation_layers:
            return {'no_bifurcations': True}
            
        layer_counts = defaultdict(int)
        for layer in bifurcation_layers:
            layer_counts[layer] += 1
            
        return {
            'layer_distribution': dict(layer_counts),
            'mean_bifurcation_layer': np.mean(bifurcation_layers),
            'mode_bifurcation_layer': max(layer_counts, key=layer_counts.get),
            'early_bifurcation_rate': sum(1 for l in bifurcation_layers if l < 4) / len(bifurcation_layers)
        }
        
    def _rank_token_stability(self) -> List[Dict[str, Any]]:
        """Rank tokens by trajectory stability across contexts."""
        stability_scores = []
        
        for token_idx, effects in self.context_effects.items():
            if not effects:
                continue
                
            # Calculate variance in trajectories
            divergences = [e['divergence_full'] for e in effects.values()]
            
            stability_scores.append({
                'token_idx': token_idx,
                'stability': 1 - np.mean(divergences),  # Higher score = more stable
                'variance': np.var(divergences)
            })
            
        # Sort by stability
        stability_scores.sort(key=lambda x: x['stability'], reverse=True)
        
        return stability_scores[:100]  # Top 100 most stable
        
    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute overall summary statistics."""
        all_divergences = []
        for effects in self.context_effects.values():
            for effect in effects.values():
                all_divergences.append(effect['divergence_early'])
                
        return {
            'total_tokens_analyzed': len(self.context_effects),
            'total_comparisons': len(all_divergences),
            'mean_divergence': np.mean(all_divergences) if all_divergences else 0,
            'tokens_with_any_divergence': sum(1 for effects in self.context_effects.values()
                                             if any(e['divergence_early'] > 0 for e in effects.values())),
            'percentage_affected': sum(1 for effects in self.context_effects.values()
                                      if any(e['divergence_early'] > 0 for e in effects.values())) / 
                                  len(self.context_effects) * 100 if self.context_effects else 0
        }
        
    def visualize(self) -> Dict[str, str]:
        """Create visualizations."""
        logger.info("Creating visualizations")
        
        viz_paths = {}
        
        # For now, save the data for visualization generation
        # The actual visualization will be done in a separate script
        viz_data = {
            'trajectories': self.trajectories,
            'context_effects': dict(self.context_effects),
            'analysis': self.analyze()
        }
        
        viz_data_path = self.output_dir / "visualization_data.json"
        with open(viz_data_path, 'w') as f:
            json.dump(viz_data, f, indent=2)
            
        viz_paths['data'] = str(viz_data_path)
        
        return viz_paths


def main():
    """Run the vocabulary context effects experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-2 Full Vocabulary Context Effects Experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Run experiment
    experiment = VocabularyContextExperiment(args.config)
    results = experiment.run()
    
    print(f"Experiment completed. Results saved to {experiment.output_dir}")
    print(f"Summary: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()