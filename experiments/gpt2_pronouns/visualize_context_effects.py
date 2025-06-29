"""
Comprehensive Visualizations for Context Effects

Creates trajectory heatmaps, context effect distributions, bifurcation analysis plots,
and token stability rankings as specified in config.yaml.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class ContextEffectVisualizer:
    """Generate comprehensive visualizations for context effects analysis."""
    
    def __init__(self, results_dir: str = "results/"):
        """Initialize with paths to analysis results."""
        self.results_dir = Path(results_dir)
        
        # Load all analysis results
        self.trajectories = self._load_json("visualization_data.json")
        self.statistics = self._load_json("statistical_report.json")
        self.clustering_analysis = self._load_json("clustering_analysis/analysis_summary.json")
        self.pattern_discovery = self._load_json("pattern_discovery/pattern_discovery_summary.json")
        
        # Load token information
        token_path = Path("../gpt2/all_tokens/top_10k_tokens_full.json")
        if token_path.exists():
            with open(token_path, 'r') as f:
                tokens = json.load(f)
                self.token_info = {i: t for i, t in enumerate(tokens)}
        else:
            self.token_info = {}
            
        logger.info(f"Loaded data from {self.results_dir}")
        
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from results directory."""
        path = self.results_dir / filename
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"File not found: {path}")
            return {}
            
    def create_trajectory_heatmap(self, num_tokens: int = 200, save_path: str = None):
        """Create heatmap showing trajectory changes across contexts."""
        # Group trajectories by token
        token_trajectories = defaultdict(dict)
        
        for key, traj_data in self.trajectories.get('trajectories', {}).items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data['path'][:4]  # First 4 layers
            
        # Create matrix for heatmap
        context_types = ['baseline', 'determiner_the', 'determiner_a', 'pronoun_i', 
                        'pronoun_they', 'preposition_with', 'preposition_of',
                        'sentence_start_is', 'sentence_start_are']
        
        # Select most affected tokens
        affected_tokens = self._get_most_affected_tokens(num_tokens)
        
        # Build heatmap data
        heatmap_data = []
        token_labels = []
        
        for token_idx in affected_tokens:
            if token_idx not in token_trajectories:
                continue
                
            token_str = self.token_info.get(token_idx, {}).get('token_str', f'token_{token_idx}')
            token_labels.append(token_str)
            
            row = []
            baseline = token_trajectories[token_idx].get('baseline', [-1]*4)
            
            for context in context_types:
                if context in token_trajectories[token_idx]:
                    trajectory = token_trajectories[token_idx][context]
                    # Calculate divergence from baseline
                    divergence = sum(1 for b, t in zip(baseline, trajectory) 
                                   if b != -1 and t != -1 and b != t) / 4
                    row.append(divergence)
                else:
                    row.append(0)
                    
            heatmap_data.append(row)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, min(20, len(token_labels) * 0.3)))
        
        sns.heatmap(heatmap_data, 
                   xticklabels=[c.replace('_', ' ').title() for c in context_types],
                   yticklabels=token_labels,
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': 'Trajectory Divergence'},
                   linewidths=0.5,
                   ax=ax)
                   
        ax.set_title('Token Trajectory Changes by Context', fontsize=16, pad=20)
        ax.set_xlabel('Context Type', fontsize=12)
        ax.set_ylabel('Token', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory heatmap to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def _get_most_affected_tokens(self, n: int) -> List[int]:
        """Get tokens most affected by context."""
        # Try to load from analysis results
        sensitive_tokens_path = self.results_dir / "clustering_analysis/context_sensitive_tokens.json"
        
        if sensitive_tokens_path.exists():
            with open(sensitive_tokens_path, 'r') as f:
                sensitive = json.load(f)
                return [t['token_idx'] for t in sensitive[:n]]
        else:
            # Fall back to computing from trajectories
            token_effects = defaultdict(float)
            token_groups = defaultdict(dict)
            
            for key, traj_data in self.trajectories.get('trajectories', {}).items():
                token_idx = traj_data['token_idx']
                context = traj_data['context_frame']
                token_groups[token_idx][context] = traj_data['path']
                
            for token_idx, contexts in token_groups.items():
                if 'baseline' in contexts:
                    baseline = contexts['baseline']
                    for ctx, traj in contexts.items():
                        if ctx != 'baseline':
                            div = sum(1 for b, t in zip(baseline[:4], traj[:4])
                                    if b != -1 and t != -1 and b != t) / 4
                            token_effects[token_idx] = max(token_effects[token_idx], div)
                            
            # Return top n
            sorted_tokens = sorted(token_effects.items(), key=lambda x: x[1], reverse=True)
            return [t[0] for t in sorted_tokens[:n]]
            
    def create_context_effect_distributions(self, save_path: str = None):
        """Create distribution plots for context effects."""
        # Load statistical report
        if not self.statistics:
            logger.warning("No statistical report found")
            return
            
        effect_sizes = self.statistics.get('effect_sizes', {})
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Context Effect Distributions', fontsize=16)
        
        # 1. Cohen's d by context type
        ax = axes[0, 0]
        contexts = list(effect_sizes.keys())
        cohens_d = [effect_sizes[c]['cohens_d'] for c in contexts]
        
        bars = ax.bar(range(len(contexts)), cohens_d)
        ax.set_xticks(range(len(contexts)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in contexts], rotation=45, ha='right')
        ax.set_ylabel("Cohen's d")
        ax.set_title("Effect Size by Context Type")
        ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Small effect')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Medium effect')
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.9, label='Large effect')
        ax.legend()
        
        # Color bars by effect size
        for i, bar in enumerate(bars):
            if abs(cohens_d[i]) < 0.2:
                bar.set_color('lightgray')
            elif abs(cohens_d[i]) < 0.5:
                bar.set_color('lightblue')
            elif abs(cohens_d[i]) < 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
                
        # 2. Distribution of trajectory divergences
        ax = axes[0, 1]
        all_divergences = []
        
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.get('trajectories', {}).items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        for token_idx, contexts in token_groups.items():
            if 'baseline' in contexts:
                baseline = contexts['baseline']
                for ctx, traj in contexts.items():
                    if ctx != 'baseline':
                        div = sum(1 for b, t in zip(baseline[:4], traj[:4])
                                if b != -1 and t != -1 and b != t) / 4
                        all_divergences.append(div)
                        
        ax.hist(all_divergences, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Trajectory Divergence')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Trajectory Divergences')
        ax.axvline(x=np.mean(all_divergences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_divergences):.3f}')
        ax.legend()
        
        # 3. Token type analysis
        ax = axes[1, 0]
        token_type_stats = self.statistics.get('token_type_analysis', {})
        
        if token_type_stats:
            types = [t for t in token_type_stats.keys() if t != 'anova']
            mean_effects = [token_type_stats[t]['mean_effect'] for t in types]
            
            ax.bar(range(len(types)), mean_effects)
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=45, ha='right')
            ax.set_ylabel('Mean Effect')
            ax.set_title('Context Sensitivity by Token Type')
            
        # 4. Layer-wise divergence rates
        ax = axes[1, 1]
        layer_stats = self.statistics.get('layer_statistics', {})
        
        if layer_stats:
            layers = sorted([int(k.split('_')[1]) for k in layer_stats.keys()])
            divergence_rates = [layer_stats[f'layer_{l}']['divergence_rate'] for l in layers]
            
            ax.plot(layers, divergence_rates, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Divergence Rate')
            ax.set_title('Context Effect Strength by Layer')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved context effect distributions to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def create_bifurcation_analysis(self, save_path: str = None):
        """Create bifurcation analysis plots."""
        # Load trajectory transitions
        transitions_path = self.results_dir / "pattern_discovery/trajectory_transitions.json"
        
        if not transitions_path.exists():
            logger.warning("No trajectory transitions data found")
            return
            
        with open(transitions_path, 'r') as f:
            transitions = json.load(f)
            
        # Create interactive Sankey diagram for top transitions
        fig = self._create_transition_sankey(transitions)
        
        if save_path:
            # Save as HTML for interactivity
            html_path = save_path.replace('.png', '.html').replace('.pdf', '.html')
            fig.write_html(html_path)
            logger.info(f"Saved bifurcation analysis to {html_path}")
            
            # Also save static version
            fig.write_image(save_path)
            logger.info(f"Saved static bifurcation analysis to {save_path}")
        else:
            fig.show()
            
    def _create_transition_sankey(self, transitions: Dict) -> go.Figure:
        """Create Sankey diagram showing trajectory transitions."""
        common_transitions = transitions.get('common_transitions', [])[:20]  # Top 20
        
        if not common_transitions:
            return go.Figure()
            
        # Build node and link data
        nodes = []
        node_map = {}
        links = []
        
        for trans in common_transitions:
            # Create nodes for from and to paths
            from_key = f"Baseline: {trans['from_path']}"
            to_key = f"{trans['context']}: {trans['to_path']}"
            
            if from_key not in node_map:
                node_map[from_key] = len(nodes)
                nodes.append(from_key)
                
            if to_key not in node_map:
                node_map[to_key] = len(nodes)
                nodes.append(to_key)
                
            # Add link
            links.append({
                'source': node_map[from_key],
                'target': node_map[to_key],
                'value': trans['count'],
                'label': f"{trans['context']} ({trans['count']} tokens)"
            })
            
        # Create Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="blue"
            ),
            link=dict(
                source=[l['source'] for l in links],
                target=[l['target'] for l in links],
                value=[l['value'] for l in links],
                label=[l['label'] for l in links]
            )
        )])
        
        fig.update_layout(
            title="Top Trajectory Transitions by Context",
            font_size=10,
            height=800
        )
        
        return fig
        
    def create_token_stability_rankings(self, save_path: str = None):
        """Create token stability ranking visualization."""
        # Calculate stability scores
        stability_scores = self._calculate_stability_scores()
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Most Stable Tokens', 'Most Unstable Tokens'),
            row_heights=[0.5, 0.5]
        )
        
        # Top 20 most stable
        stable = sorted(stability_scores.items(), key=lambda x: x[1])[:20]
        stable_tokens = [self.token_info.get(t[0], {}).get('token_str', f'tok_{t[0]}') for t in stable]
        stable_scores = [t[1] for t in stable]
        
        fig.add_trace(
            go.Bar(x=stable_tokens, y=stable_scores, name='Stable', marker_color='green'),
            row=1, col=1
        )
        
        # Top 20 most unstable
        unstable = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        unstable_tokens = [self.token_info.get(t[0], {}).get('token_str', f'tok_{t[0]}') for t in unstable]
        unstable_scores = [t[1] for t in unstable]
        
        fig.add_trace(
            go.Bar(x=unstable_tokens, y=unstable_scores, name='Unstable', marker_color='red'),
            row=2, col=1
        )
        
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text='Instability Score', row=1, col=1)
        fig.update_yaxes(title_text='Instability Score', row=2, col=1)
        
        fig.update_layout(
            title='Token Stability Rankings',
            showlegend=False,
            height=800
        )
        
        if save_path:
            # Save as HTML
            html_path = save_path.replace('.png', '.html').replace('.pdf', '.html')
            fig.write_html(html_path)
            logger.info(f"Saved token stability rankings to {html_path}")
            
            # Also save static version
            if save_path.endswith('.png') or save_path.endswith('.pdf'):
                fig.write_image(save_path)
                logger.info(f"Saved static token stability rankings to {save_path}")
        else:
            fig.show()
            
    def _calculate_stability_scores(self) -> Dict[int, float]:
        """Calculate stability score for each token."""
        stability_scores = {}
        
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.get('trajectories', {}).items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        for token_idx, contexts in token_groups.items():
            if 'baseline' not in contexts:
                continue
                
            baseline = contexts['baseline']
            divergences = []
            
            for ctx, traj in contexts.items():
                if ctx != 'baseline':
                    div = sum(1 for b, t in zip(baseline[:4], traj[:4])
                            if b != -1 and t != -1 and b != t) / 4
                    divergences.append(div)
                    
            if divergences:
                # Stability score = mean divergence * variance
                # Higher score = more unstable
                stability_scores[token_idx] = np.mean(divergences) * np.var(divergences)
                
        return stability_scores
        
    def create_trajectory_embeddings_plot(self, method: str = 'pca', save_path: str = None):
        """Create 2D embedding of trajectories colored by context effects."""
        # Collect trajectory data
        trajectories = []
        labels = []
        context_labels = []
        
        for key, traj_data in self.trajectories.get('trajectories', {}).items():
            if len(traj_data['path']) >= 4 and -1 not in traj_data['path'][:4]:
                trajectories.append(traj_data['path'][:4])
                labels.append(traj_data['token_idx'])
                context_labels.append(traj_data['context_frame'])
                
        if not trajectories:
            logger.warning("No valid trajectories for embedding")
            return
            
        trajectories = np.array(trajectories)
        
        # Compute embeddings
        if method == 'pca':
            reducer = PCA(n_components=2)
            embeddings = reducer.fit_transform(trajectories)
            title = 'PCA Embedding of Token Trajectories'
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(trajectories)
            title = 't-SNE Embedding of Token Trajectories'
            
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        
        # Color by context
        unique_contexts = list(set(context_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_contexts)))
        
        for i, context in enumerate(unique_contexts):
            mask = [c == context for c in context_labels]
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c=[colors[i]], label=context.replace('_', ' ').title(),
                       alpha=0.6, s=50)
                       
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory embeddings to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def generate_all_visualizations(self, output_dir: str = None):
        """Generate all visualization types."""
        if output_dir is None:
            output_dir = self.results_dir / "figures"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Generating comprehensive visualizations...")
        
        # 1. Trajectory heatmap
        logger.info("Creating trajectory heatmap...")
        self.create_trajectory_heatmap(
            save_path=output_dir / "trajectory_heatmap.png"
        )
        
        # 2. Context effect distributions
        logger.info("Creating context effect distributions...")
        self.create_context_effect_distributions(
            save_path=output_dir / "context_effect_distributions.png"
        )
        
        # 3. Bifurcation analysis
        logger.info("Creating bifurcation analysis...")
        self.create_bifurcation_analysis(
            save_path=output_dir / "bifurcation_analysis.html"
        )
        
        # 4. Token stability rankings
        logger.info("Creating token stability rankings...")
        self.create_token_stability_rankings(
            save_path=output_dir / "token_stability_rankings.html"
        )
        
        # 5. Trajectory embeddings (both PCA and t-SNE)
        logger.info("Creating trajectory embeddings...")
        self.create_trajectory_embeddings_plot(
            method='pca',
            save_path=output_dir / "trajectory_embeddings_pca.png"
        )
        
        # Don't do t-SNE for large datasets as it's slow
        if len(self.trajectories.get('trajectories', {})) < 10000:
            self.create_trajectory_embeddings_plot(
                method='tsne',
                save_path=output_dir / "trajectory_embeddings_tsne.png"
            )
            
        logger.info(f"All visualizations saved to {output_dir}")
        
        # Create summary README
        readme_content = """# Context Effects Visualization Results

## Generated Visualizations

1. **trajectory_heatmap.png** - Shows how each token's trajectory changes across different contexts
2. **context_effect_distributions.png** - Statistical distributions of context effects
3. **bifurcation_analysis.html** - Interactive Sankey diagram of trajectory transitions
4. **token_stability_rankings.html** - Rankings of most and least stable tokens
5. **trajectory_embeddings_pca.png** - 2D PCA projection of trajectory space

## Key Findings

See the analysis summary files for detailed statistics and findings.
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(readme_content)
            

def main():
    """Run visualization generation."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/',
                       help='Directory containing analysis results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for figures')
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ContextEffectVisualizer(args.results)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(args.output)
    

if __name__ == "__main__":
    main()