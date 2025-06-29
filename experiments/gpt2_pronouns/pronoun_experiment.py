"""
GPT-2 Pronoun Context Steering Experiment

This experiment investigates how context tokens influence the trajectory bifurcation
of pronouns in GPT-2, focusing on the first 4 layers where the split occurs.
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging

# Import from existing infrastructure
import sys
sys.path.append('../../')
from concept_fragmentation.experiments.base import BaseExperiment
from concept_fragmentation.clustering.paths import PathExtractor

# For now, comment out visualizations that aren't used in basic execution
# from concept_fragmentation.visualization.sankey import SankeyGenerator
# from concept_fragmentation.visualization.d3_sankey import D3SankeyGenerator

# Use sklearn KMeans directly
from sklearn.cluster import KMeans

# Import GPT-2 specific tools
sys.path.append('../gpt2/shared/')
from gpt2_activation_extractor import SimpleGPT2ActivationExtractor

logger = logging.getLogger(__name__)


class PronounContextExperiment(BaseExperiment):
    """Experiment to study context-dependent pronoun trajectory bifurcation."""
    
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
        self.sentences = []
        self.labels = []
        self.activations = None
        self.trajectories = {}
        self.tds_scores = {}
        
    def setup(self) -> None:
        """Set up the experiment."""
        logger.info("Setting up pronoun context steering experiment")
        
        # Generate two-token probing sentences
        self._generate_sentences()
        
        # Setup GPT-2 model
        self.extractor.setup_model()
        
    def _generate_sentences(self) -> None:
        """Generate all two-token probing sentences."""
        self.sentences = []
        self.labels = []
        
        # Generate sentences for each pronoun-context combination
        for pronoun in self.config.data['pronouns']:
            # Baseline (pronoun alone)
            self.sentences.append(pronoun)
            self.labels.append({
                'pronoun': pronoun,
                'context': '',
                'context_type': 'neutral'
            })
            
            # Function word contexts
            for context in self.config.data['contexts']['function_words']:
                self.sentences.append(f"{context} {pronoun}")
                self.labels.append({
                    'pronoun': pronoun,
                    'context': context,
                    'context_type': 'function'
                })
            
            # Content word contexts
            for context in self.config.data['contexts']['content_words']:
                self.sentences.append(f"{context} {pronoun}")
                self.labels.append({
                    'pronoun': pronoun,
                    'context': context,
                    'context_type': 'content'
                })
        
        logger.info(f"Generated {len(self.sentences)} probing sentences")
        
    def execute(self) -> Dict[str, Any]:
        """Execute the main experiment logic."""
        logger.info("Executing pronoun context steering experiment")
        
        # Extract activations
        logger.info("Extracting GPT-2 activations...")
        self.activations = self.extractor.extract_activations(self.sentences)
        
        # Load existing clustering models from GPT-2 10k study
        # For now, we'll create new clusterers (in real implementation, load existing)
        clusterers = self._setup_clustering()
        
        # Extract trajectories
        logger.info("Extracting trajectories...")
        self.trajectories = self._extract_trajectories(clusterers)
        
        # Calculate TDS scores
        logger.info("Calculating Trajectory Divergence Scores...")
        self.tds_scores = self._calculate_tds_scores()
        
        return {
            'num_sentences': len(self.sentences),
            'num_trajectories': len(self.trajectories),
            'tds_summary': self._summarize_tds_scores()
        }
        
    def _setup_clustering(self) -> Dict[int, KMeans]:
        """Setup clustering for each layer."""
        clusterers = {}
        k = self.config.analysis['clustering']['k_per_layer']
        
        for layer in range(self.config.model['num_layers']):
            clusterers[layer] = KMeans(
                n_clusters=k,
                random_state=self.config.analysis['clustering']['random_state']
            )
            
            # In real implementation, load existing cluster models
            # For now, fit on current data
            layer_activations = []
            for sent_idx in self.activations['activations']:
                for token_idx in self.activations['activations'][sent_idx]:
                    if layer in self.activations['activations'][sent_idx][token_idx]:
                        act = self.activations['activations'][sent_idx][token_idx][layer]
                        layer_activations.append(act)
            
            if layer_activations:
                # Ensure correct dtype for sklearn
                clusterers[layer].fit(np.array(layer_activations, dtype=np.float64))
                
        return clusterers
        
    def _extract_trajectories(self, clusterers: Dict[int, KMeans]) -> Dict[str, List[int]]:
        """Extract trajectories for all sentences."""
        trajectories = {}
        
        for sent_idx, label in enumerate(self.labels):
            trajectory = []
            
            # Get pronoun token index (last token in sentence)
            tokens = self.activations['tokens'][sent_idx]
            pronoun_idx = len(tokens) - 1
            
            # Extract trajectory across layers
            for layer in range(self.config.model['num_layers']):
                if (sent_idx in self.activations['activations'] and 
                    pronoun_idx in self.activations['activations'][sent_idx] and
                    layer in self.activations['activations'][sent_idx][pronoun_idx]):
                    
                    activation = self.activations['activations'][sent_idx][pronoun_idx][layer]
                    # Ensure correct dtype for sklearn
                    activation = np.array(activation, dtype=np.float64)
                    cluster = clusterers[layer].predict(activation.reshape(1, -1))[0]
                    trajectory.append(cluster)
                else:
                    trajectory.append(-1)  # Missing data
            
            # Create unique key for this sentence
            key = f"{label['pronoun']}|{label['context']}|{label['context_type']}"
            trajectories[key] = trajectory
            
        return trajectories
        
    def _calculate_tds_scores(self) -> Dict[str, Dict[str, float]]:
        """Calculate Trajectory Divergence Scores."""
        tds_scores = defaultdict(dict)
        
        for pronoun in self.config.data['pronouns']:
            # Get baseline trajectory (pronoun alone)
            baseline_key = f"{pronoun}||neutral"
            if baseline_key not in self.trajectories:
                continue
            baseline_trajectory = self.trajectories[baseline_key]
            
            # Compare with each context
            for context_type in ['function', 'content']:
                context_list = self.config.data['contexts'][f'{context_type}_words']
                
                for context in context_list:
                    context_key = f"{pronoun}|{context}|{context_type}"
                    if context_key not in self.trajectories:
                        continue
                    
                    context_trajectory = self.trajectories[context_key]
                    
                    # Calculate TDS for early layers (first 4)
                    early_diff = sum(1 for i in range(4) 
                                   if baseline_trajectory[i] != context_trajectory[i])
                    tds_early = early_diff / 4.0
                    
                    # Calculate TDS for all layers
                    full_diff = sum(1 for i in range(12) 
                                  if baseline_trajectory[i] != context_trajectory[i])
                    tds_full = full_diff / 12.0
                    
                    # Store scores
                    tds_key = f"{pronoun}|{context}"
                    tds_scores[tds_key] = {
                        'pronoun': pronoun,
                        'context': context,
                        'context_type': context_type,
                        'tds_early': tds_early,
                        'tds_full': tds_full,
                        'bifurcation_layer': self._find_bifurcation_layer(
                            baseline_trajectory, context_trajectory
                        )
                    }
        
        return dict(tds_scores)
    
    def _find_bifurcation_layer(self, traj1: List[int], traj2: List[int]) -> int:
        """Find the layer where trajectories first diverge."""
        for i in range(len(traj1)):
            if traj1[i] != traj2[i]:
                return i
        return -1  # No divergence
        
    def _summarize_tds_scores(self) -> Dict[str, Any]:
        """Summarize TDS scores by context type."""
        summary = {
            'function': {'early': [], 'full': []},
            'content': {'early': [], 'full': []}
        }
        
        for scores in self.tds_scores.values():
            context_type = scores['context_type']
            summary[context_type]['early'].append(scores['tds_early'])
            summary[context_type]['full'].append(scores['tds_full'])
        
        # Calculate means
        for context_type in ['function', 'content']:
            if summary[context_type]['early']:
                summary[context_type]['early_mean'] = np.mean(summary[context_type]['early'])
                summary[context_type]['full_mean'] = np.mean(summary[context_type]['full'])
            else:
                summary[context_type]['early_mean'] = 0
                summary[context_type]['full_mean'] = 0
                
        return summary
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        logger.info("Analyzing pronoun context steering results")
        
        analysis = {
            'tds_analysis': self._analyze_tds_patterns(),
            'trajectory_analysis': self._analyze_trajectory_patterns(),
            'statistical_tests': self._run_statistical_tests()
        }
        
        return analysis
        
    def _analyze_tds_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in TDS scores."""
        # Group by context type
        function_tds_early = []
        content_tds_early = []
        
        for scores in self.tds_scores.values():
            if scores['context_type'] == 'function':
                function_tds_early.append(scores['tds_early'])
            else:
                content_tds_early.append(scores['tds_early'])
        
        return {
            'function_tds_early_mean': np.mean(function_tds_early) if function_tds_early else 0,
            'content_tds_early_mean': np.mean(content_tds_early) if content_tds_early else 0,
            'function_tds_early_std': np.std(function_tds_early) if function_tds_early else 0,
            'content_tds_early_std': np.std(content_tds_early) if content_tds_early else 0,
            'effect_size': (np.mean(function_tds_early) - np.mean(content_tds_early)) 
                          if function_tds_early and content_tds_early else 0
        }
        
    def _analyze_trajectory_patterns(self) -> Dict[str, Any]:
        """Analyze trajectory patterns."""
        # Find common bifurcation points
        bifurcation_layers = []
        for scores in self.tds_scores.values():
            if scores['bifurcation_layer'] >= 0:
                bifurcation_layers.append(scores['bifurcation_layer'])
        
        return {
            'mean_bifurcation_layer': np.mean(bifurcation_layers) if bifurcation_layers else -1,
            'mode_bifurcation_layer': max(set(bifurcation_layers), key=bifurcation_layers.count) 
                                     if bifurcation_layers else -1,
            'bifurcation_distribution': dict(zip(*np.unique(bifurcation_layers, return_counts=True)))
                                       if bifurcation_layers else {}
        }
        
    def _run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical tests on results."""
        from scipy import stats
        
        # Group TDS scores by context type
        function_tds = [s['tds_early'] for s in self.tds_scores.values() 
                       if s['context_type'] == 'function']
        content_tds = [s['tds_early'] for s in self.tds_scores.values() 
                      if s['context_type'] == 'content']
        
        # Run t-test
        if function_tds and content_tds:
            t_stat, p_value = stats.ttest_ind(function_tds, content_tds)
        else:
            t_stat, p_value = 0, 1
            
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
    def visualize(self) -> Dict[str, str]:
        """Create visualizations."""
        logger.info("Creating visualizations")
        
        viz_paths = {}
        
        # Create sankey diagrams
        if self.config.visualization['sankey']['show_early_only']:
            early_path = self._create_early_sankey()
            viz_paths['sankey_early'] = str(early_path)
            
        if self.config.visualization['sankey']['show_full_network']:
            full_path = self._create_full_sankey()
            viz_paths['sankey_full'] = str(full_path)
            
        return viz_paths
        
    def _create_early_sankey(self) -> Path:
        """Create sankey diagram for first 4 layers."""
        # Prepare data for sankey
        paths_data = []
        for key, trajectory in self.trajectories.items():
            pronoun, context, context_type = key.split('|')
            path = trajectory[:4]  # First 4 layers only
            paths_data.append({
                'path': [int(x) for x in path],  # Convert numpy types to Python int
                'label': f"{pronoun}_{context}",
                'context_type': context_type
            })
        
        # Save trajectory data for later visualization
        import json
        trajectory_file = self.output_dir / "early_trajectories.json"
        with open(trajectory_file, 'w') as f:
            json.dump(paths_data, f, indent=2)
        
        # Generate placeholder sankey
        output_path = self.output_dir / "sankey_early_layers.html"
        output_path.write_text("<html><body><h1>Early Layers Sankey</h1><p>Trajectories saved to early_trajectories.json</p></body></html>")
        
        return output_path
        
    def _create_full_sankey(self) -> Path:
        """Create sankey diagram for full network."""
        # Prepare data for sankey
        paths_data = []
        for key, trajectory in self.trajectories.items():
            pronoun, context, context_type = key.split('|')
            paths_data.append({
                'path': [int(x) for x in trajectory],  # Convert numpy types to Python int
                'label': f"{pronoun}_{context}",
                'context_type': context_type
            })
        
        # Save trajectory data for later visualization
        import json
        trajectory_file = self.output_dir / "full_trajectories.json"
        with open(trajectory_file, 'w') as f:
            json.dump(paths_data, f, indent=2)
        
        # Generate placeholder sankey
        output_path = self.output_dir / "sankey_full_network.html"
        output_path.write_text("<html><body><h1>Full Network Sankey</h1><p>Trajectories saved to full_trajectories.json</p></body></html>")
        
        return output_path


def main():
    """Run the pronoun context steering experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-2 Pronoun Context Steering Experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Run experiment
    experiment = PronounContextExperiment(args.config)
    results = experiment.run()
    
    print(f"Experiment completed. Results saved to {experiment.output_dir}")
    print(f"Summary: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()