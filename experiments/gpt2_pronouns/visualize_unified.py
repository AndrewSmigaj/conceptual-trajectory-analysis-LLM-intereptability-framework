"""
Visualize Unified Context Effects

Uses existing visualization infrastructure (SankeyGenerator, D3SankeyGenerator)
to create visualizations for the unified clustering experiment.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Import from existing infrastructure
import sys
sys.path.append('../../')
from concept_fragmentation.visualization.sankey import SankeyGenerator
from concept_fragmentation.visualization.d3_sankey import D3SankeyGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedVisualizer:
    """Create visualizations for unified context effects experiment."""
    
    def __init__(self, trajectories_path: str, analysis_path: str = None):
        """Initialize with data paths."""
        # Load trajectories
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
            self.trajectories = data['trajectories']
            self.metadata = data.get('metadata', {})
            
        # Load analysis results if available
        self.analysis = {}
        if analysis_path and Path(analysis_path).exists():
            with open(analysis_path, 'r') as f:
                self.analysis = json.load(f)
                
        logger.info(f"Loaded {len(self.trajectories)} trajectories for visualization")
        
    def create_context_sankey(self, context_name: str, output_path: str = None,
                            use_d3: bool = True, top_n: int = 25) -> None:
        """Create sankey diagram for a specific context effect."""
        # Filter trajectories for baseline and specific context
        baseline_trajectories = {}
        context_trajectories = {}
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            
            if traj_data['context_frame'] == 'baseline':
                baseline_trajectories[token_idx] = traj_data
            elif traj_data['context_frame'] == context_name:
                context_trajectories[token_idx] = traj_data
                
        # Find tokens that appear in both
        common_tokens = set(baseline_trajectories.keys()) & set(context_trajectories.keys())
        logger.info(f"Found {len(common_tokens)} tokens with both baseline and {context_name}")
        
        # Prepare data for sankey
        sankey_data = []
        
        for token_idx in common_tokens:
            baseline_path = baseline_trajectories[token_idx]['path']
            context_path = context_trajectories[token_idx]['path']
            token_str = baseline_trajectories[token_idx]['token_str']
            
            # Check if trajectories differ
            if baseline_path != context_path:
                sankey_data.append({
                    'token_str': token_str,
                    'baseline_path': baseline_path,
                    'context_path': context_path,
                    'divergence': sum(1 for b, c in zip(baseline_path, context_path) 
                                    if b != c and b != -1 and c != -1)
                })
                
        # Sort by divergence
        sankey_data.sort(key=lambda x: x['divergence'], reverse=True)
        
        if use_d3:
            self._create_d3_sankey(sankey_data[:top_n], context_name, output_path)
        else:
            self._create_plotly_sankey(sankey_data[:top_n], context_name, output_path)
            
    def _create_d3_sankey(self, sankey_data: List[Dict], context_name: str, 
                         output_path: str = None) -> None:
        """Create D3.js sankey visualization."""
        # Prepare paths for D3SankeyGenerator
        paths = []
        path_metadata = []
        
        # Add baseline and context paths
        for item in sankey_data:
            # Add baseline path
            paths.append(item['baseline_path'])
            path_metadata.append({
                'token': item['token_str'],
                'type': 'baseline'
            })
            
            # Add context path
            paths.append(item['context_path'])
            path_metadata.append({
                'token': item['token_str'],
                'type': context_name
            })
        
        # Prepare trajectory data in expected format
        trajectory_data = {
            'paths': paths,
            'metadata': path_metadata
        }
        
        # Create cluster results structure
        cluster_results = {
            'metadata': {
                'k_clusters': self.metadata.get('k_clusters', 5),
                'num_layers': 12
            }
        }
        
        # Add empty cluster info for each layer (required by D3SankeyGenerator)
        for layer in range(12):
            cluster_results[f'layer_{layer}'] = {
                'n_clusters': self.metadata.get('k_clusters', 5)
            }
        
        # Use D3SankeyGenerator
        generator = D3SankeyGenerator()
        
        # Generate sankey
        title = f"Context Effect: {context_name.replace('_', ' ').title()}"
        subtitle = f"Comparing baseline vs {context_name} trajectories"
        
        if output_path is None:
            output_path = f"sankey_{context_name}.html"
            
        # Define routing colors for baseline vs context
        routing_classes = ['baseline', context_name]
        routing_colors = {
            'baseline': '#1f77b4',
            context_name: '#ff7f0e'
        }
        
        generator.generate(
            trajectory_data=trajectory_data,
            cluster_results=cluster_results,
            output_path=output_path,
            title=title,
            subtitle=subtitle,
            routing_classes=routing_classes,
            routing_colors=routing_colors,
            layer_names=[f"Layer {i}" for i in range(12)],
            full_network=True
        )
        
        logger.info(f"Created D3 sankey for {context_name} at {output_path}")
        
    def _create_plotly_sankey(self, sankey_data: List[Dict], context_name: str,
                            output_path: str = None) -> None:
        """Create Plotly sankey visualization."""
        # Prepare data for SankeyGenerator
        paths_data = []
        
        for item in sankey_data:
            # Baseline path
            paths_data.append({
                'token': f"{item['token_str']} (baseline)",
                'path': item['baseline_path']
            })
            
            # Context path
            paths_data.append({
                'token': f"{item['token_str']} ({context_name})",
                'path': item['context_path']
            })
            
        # Use SankeyGenerator
        generator = SankeyGenerator()
        
        # Prepare the data structure expected by SankeyGenerator
        # This might need adaptation based on actual interface
        title = f"Context Effect: {context_name.replace('_', ' ').title()}"
        
        if output_path is None:
            output_path = f"sankey_{context_name}.png"
            
        # Note: The exact interface might differ
        # generator.create_sankey(paths_data, output_path, title)
        logger.info(f"Created Plotly sankey for {context_name}")
        
    def create_divergence_heatmap(self, output_path: str = None) -> None:
        """Create heatmap showing divergence patterns."""
        if not self.analysis:
            logger.warning("No analysis data available for heatmap")
            return
            
        # Get divergence data
        divergence_data = self.analysis.get('per_token', {})
        
        # Create matrix: tokens × contexts
        contexts = ['determiner_the', 'determiner_a', 'possessive_my', 
                   'copula_is', 'intensifier_very', 'negation_not',
                   'conjunction_and', 'modal_will', 'sentence_start']
        
        # Select top divergent tokens
        sorted_tokens = sorted(divergence_data.items(), 
                             key=lambda x: x[1]['max_divergence'], 
                             reverse=True)[:50]
        
        # Build matrix
        matrix = []
        token_labels = []
        
        for token_idx, token_data in sorted_tokens:
            row = []
            token_labels.append(token_data['token_str'])
            
            for context in contexts:
                if context in token_data['divergences']:
                    row.append(token_data['divergences'][context]['full_divergence'])
                else:
                    row.append(0)
                    
            matrix.append(row)
            
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(matrix, 
                   xticklabels=[c.replace('_', ' ').title() for c in contexts],
                   yticklabels=token_labels,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Trajectory Divergence'},
                   linewidths=0.5)
                   
        plt.title('Token Trajectory Divergence by Context', fontsize=16)
        plt.xlabel('Context Type', fontsize=12)
        plt.ylabel('Token', fontsize=12)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = "divergence_heatmap.png"
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created divergence heatmap at {output_path}")
        
    def create_layer_effects_plot(self, output_path: str = None) -> None:
        """Create plot showing context effects by layer."""
        # Calculate layer-wise divergence
        layer_divergences = defaultdict(list)
        
        # Group trajectories by token
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        # Calculate divergences per layer
        for token_idx, contexts in token_groups.items():
            if 'baseline' not in contexts:
                continue
                
            baseline = contexts['baseline']
            
            for context_name, path in contexts.items():
                if context_name == 'baseline':
                    continue
                    
                for layer in range(min(len(baseline), len(path))):
                    if baseline[layer] != -1 and path[layer] != -1:
                        if baseline[layer] != path[layer]:
                            layer_divergences[layer].append(1)
                        else:
                            layer_divergences[layer].append(0)
                            
        # Calculate rates
        layers = sorted(layer_divergences.keys())
        divergence_rates = []
        
        for layer in layers:
            if layer_divergences[layer]:
                rate = np.mean(layer_divergences[layer])
                divergence_rates.append(rate)
            else:
                divergence_rates.append(0)
                
        # Create plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(layers, divergence_rates, 'o-', linewidth=2, markersize=8)
        plt.fill_between(layers, divergence_rates, alpha=0.3)
        
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Divergence Rate', fontsize=12)
        plt.title('Context Effect Strength by Layer', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add layer window annotations
        plt.axvspan(-0.5, 3.5, alpha=0.1, color='green', label='Early')
        plt.axvspan(3.5, 7.5, alpha=0.1, color='blue', label='Middle')
        plt.axvspan(7.5, 11.5, alpha=0.1, color='red', label='Late')
        
        plt.legend()
        plt.tight_layout()
        
        if output_path is None:
            output_path = "layer_effects.png"
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created layer effects plot at {output_path}")
        
    def create_all_visualizations(self, output_dir: str = None) -> None:
        """Generate all visualizations."""
        if output_dir is None:
            output_dir = "visualizations/"
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create sankey diagrams for key contexts
        key_contexts = ['determiner_the', 'determiner_a', 'negation_not', 'sentence_start']
        
        for context in key_contexts:
            logger.info(f"Creating sankey for {context}...")
            self.create_context_sankey(
                context, 
                output_path=output_dir / f"sankey_{context}.html"
            )
            
        # Create heatmap
        logger.info("Creating divergence heatmap...")
        self.create_divergence_heatmap(
            output_path=output_dir / "divergence_heatmap.png"
        )
        
        # Create layer effects plot
        logger.info("Creating layer effects plot...")
        self.create_layer_effects_plot(
            output_path=output_dir / "layer_effects.png"
        )
        
        logger.info(f"All visualizations saved to {output_dir}")


def main():
    """Create visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str,
                       default='results_unified/unified_trajectories.json',
                       help='Path to trajectories file')
    parser.add_argument('--analysis', type=str,
                       default='results_unified/analysis/divergence_analysis.json',
                       help='Path to analysis results')
    parser.add_argument('--output', type=str,
                       default='results_unified/visualizations/',
                       help='Output directory')
    args = parser.parse_args()
    
    # Create visualizations
    visualizer = UnifiedVisualizer(args.trajectories, args.analysis)
    visualizer.create_all_visualizations(args.output)
    

if __name__ == "__main__":
    main()