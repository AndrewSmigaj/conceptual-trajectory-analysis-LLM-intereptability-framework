"""
Generate Paper-Ready Figures and Tables

Creates publication-quality figures and LaTeX tables for the paper.
Focuses on the most impactful visualizations and findings.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

# Configure matplotlib for LaTeX
plt.rcParams.update({
    'text.usetex': False,  # Set to True if LaTeX is available
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperFigureGenerator:
    """Generate publication-ready figures and tables."""
    
    def __init__(self, results_dir: str = "results/"):
        """Initialize with analysis results."""
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "paper_figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load all necessary data."""
        # Load trajectories
        traj_path = self.results_dir / "visualization_data.json"
        if traj_path.exists():
            with open(traj_path, 'r') as f:
                data = json.load(f)
                self.trajectories = data.get('trajectories', {})
        else:
            self.trajectories = {}
            
        # Load statistics
        stats_path = self.results_dir / "statistical_report.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.statistics = json.load(f)
        else:
            self.statistics = {}
            
        # Load pattern discovery
        patterns_path = self.results_dir / "pattern_discovery/archetypal_paths.json"
        if patterns_path.exists():
            with open(patterns_path, 'r') as f:
                self.patterns = json.load(f)
        else:
            self.patterns = {}
            
        # Load token info
        token_path = Path("../gpt2/all_tokens/top_10k_tokens_full.json")
        if token_path.exists():
            with open(token_path, 'r') as f:
                tokens = json.load(f)
                self.token_info = {i: t for i, t in enumerate(tokens)}
        else:
            self.token_info = {}
            
    def create_main_effect_figure(self):
        """Create main figure showing context effects overview."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 1. Effect sizes by context
        ax = axes[0, 0]
        effect_sizes = self.statistics.get('effect_sizes', {})
        
        contexts = list(effect_sizes.keys())
        cohens_d = [abs(effect_sizes[c]['cohens_d']) for c in contexts]
        
        # Sort by effect size
        sorted_indices = np.argsort(cohens_d)[::-1]
        contexts = [contexts[i] for i in sorted_indices]
        cohens_d = [cohens_d[i] for i in sorted_indices]
        
        bars = ax.barh(range(len(contexts)), cohens_d)
        ax.set_yticks(range(len(contexts)))
        ax.set_yticklabels([self._format_context_name(c) for c in contexts])
        ax.set_xlabel("Cohen's d (absolute value)")
        ax.set_title("(a) Effect Size by Context Type")
        
        # Add significance thresholds
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.9)
        
        # 2. Layer-wise effects
        ax = axes[0, 1]
        layer_stats = self.statistics.get('layer_statistics', {})
        
        layers = []
        divergence_rates = []
        
        for layer_key, stats in sorted(layer_stats.items()):
            layer_num = int(layer_key.split('_')[1])
            if layer_num < 4:  # Focus on early layers
                layers.append(layer_num)
                divergence_rates.append(stats['divergence_rate'])
                
        ax.plot(layers, divergence_rates, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Divergence Rate')
        ax.set_title('(b) Context Effects by Layer')
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        
        # 3. Token type sensitivity
        ax = axes[1, 0]
        token_types = self.statistics.get('token_type_analysis', {})
        
        # Filter out non-type entries
        type_data = {k: v for k, v in token_types.items() if k != 'anova'}
        
        if type_data:
            types = list(type_data.keys())
            mean_effects = [type_data[t]['mean_effect'] for t in types]
            
            # Sort by effect
            sorted_indices = np.argsort(mean_effects)[::-1]
            types = [types[i] for i in sorted_indices]
            mean_effects = [mean_effects[i] for i in sorted_indices]
            
            bars = ax.bar(range(len(types)), mean_effects)
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=45, ha='right')
            ax.set_ylabel('Mean Effect')
            ax.set_title('(c) Sensitivity by Token Type')
            
        # 4. Distribution of effects
        ax = axes[1, 1]
        
        # Calculate all effects
        all_effects = []
        token_groups = defaultdict(dict)
        
        for key, traj_data in self.trajectories.items():
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
                        all_effects.append(div)
                        
        ax.hist(all_effects, bins=30, alpha=0.7, edgecolor='black', density=True)
        ax.set_xlabel('Trajectory Divergence')
        ax.set_ylabel('Density')
        ax.set_title('(d) Distribution of Context Effects')
        ax.axvline(x=np.mean(all_effects), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_effects):.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in ['pdf', 'png']:
            save_path = self.figures_dir / f"main_context_effects.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved main effect figure to {save_path}")
            
        plt.close()
        
    def _format_context_name(self, context: str) -> str:
        """Format context name for display."""
        replacements = {
            'determiner_the': 'the [TOKEN]',
            'determiner_a': 'a [TOKEN]',
            'pronoun_i': 'I [TOKEN]',
            'pronoun_they': 'they [TOKEN]',
            'preposition_with': 'with [TOKEN]',
            'preposition_of': 'of [TOKEN]',
            'sentence_start_is': '[TOKEN] is',
            'sentence_start_are': '[TOKEN] are',
            'baseline': 'Baseline'
        }
        return replacements.get(context, context.replace('_', ' ').title())
        
    def create_trajectory_examples_figure(self):
        """Create figure showing example trajectory changes."""
        # Find interesting examples
        examples = self._find_interesting_examples()
        
        if not examples:
            logger.warning("No interesting examples found")
            return
            
        # Create figure
        n_examples = min(4, len(examples))
        fig, axes = plt.subplots(n_examples, 1, figsize=(10, n_examples * 2.5))
        
        if n_examples == 1:
            axes = [axes]
            
        for idx, (token_idx, data) in enumerate(examples[:n_examples]):
            ax = axes[idx]
            
            token_str = self.token_info.get(token_idx, {}).get('token_str', f'token_{token_idx}')
            
            # Plot trajectories
            contexts_to_plot = ['baseline', 'determiner_the', 'pronoun_i', 'sentence_start_is']
            colors = ['black', 'blue', 'red', 'green']
            
            for ctx, color in zip(contexts_to_plot, colors):
                if ctx in data:
                    trajectory = data[ctx]['path'][:4]
                    layers = list(range(len(trajectory)))
                    
                    # Add small jitter for visibility
                    jittered_traj = [t + np.random.normal(0, 0.05) for t in trajectory]
                    
                    ax.plot(layers, jittered_traj, 'o-', color=color, 
                           label=self._format_context_name(ctx), 
                           linewidth=2, markersize=8, alpha=0.7)
                           
            ax.set_xlabel('Layer')
            ax.set_ylabel('Cluster ID')
            ax.set_title(f'Token: "{token_str}"')
            ax.legend()
            ax.set_xticks(range(4))
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save
        for fmt in ['pdf', 'png']:
            save_path = self.figures_dir / f"trajectory_examples.{fmt}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory examples to {save_path}")
            
        plt.close()
        
    def _find_interesting_examples(self) -> List[Tuple[int, Dict]]:
        """Find tokens with interesting trajectory patterns."""
        interesting = []
        
        # Group trajectories by token
        token_trajectories = defaultdict(dict)
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data
            
        # Find tokens with high divergence
        for token_idx, contexts in token_trajectories.items():
            if 'baseline' not in contexts:
                continue
                
            baseline = contexts['baseline']['path']
            max_div = 0
            
            for ctx, traj_data in contexts.items():
                if ctx != 'baseline':
                    div = sum(1 for b, t in zip(baseline[:4], traj_data['path'][:4])
                            if b != -1 and t != -1 and b != t) / 4
                    max_div = max(max_div, div)
                    
            if max_div > 0.5:  # High divergence
                interesting.append((token_idx, contexts))
                
        # Sort by divergence
        interesting.sort(key=lambda x: max([
            sum(1 for b, t in zip(x[1]['baseline']['path'][:4], x[1][c]['path'][:4])
                if b != -1 and t != -1 and b != t) / 4
            for c in x[1] if c != 'baseline'
        ]), reverse=True)
        
        return interesting
        
    def create_latex_tables(self):
        """Generate LaTeX tables for the paper."""
        # Table 1: Summary statistics
        self._create_summary_table()
        
        # Table 2: Effect sizes by context
        self._create_effect_sizes_table()
        
        # Table 3: Top sensitive tokens
        self._create_sensitive_tokens_table()
        
    def _create_summary_table(self):
        """Create summary statistics table."""
        summary = self.statistics.get('summary', {})
        
        latex = r"""\begin{table}[ht]
\centering
\caption{Summary Statistics of Context Effects}
\label{tab:summary}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Token-Context Pairs & %d \\
Mean Trajectory Divergence & %.3f \\
Median Trajectory Divergence & %.3f \\
Std. Dev. of Divergence & %.3f \\
Tokens with Any Effect & %d (%.1f\%%) \\
Tokens with Large Effect ($>0.5$) & %d (%.1f\%%) \\
\bottomrule
\end{tabular}
\end{table}""" % (
            summary.get('total_comparisons', 0),
            summary.get('mean_effect', 0),
            summary.get('median_effect', 0),
            summary.get('std_effect', 0),
            summary.get('tokens_with_any_effect', 0),
            summary.get('tokens_with_any_effect', 0) / max(summary.get('total_comparisons', 1), 1) * 100,
            summary.get('tokens_with_large_effect', 0),
            summary.get('tokens_with_large_effect', 0) / max(summary.get('total_comparisons', 1), 1) * 100
        )
        
        with open(self.figures_dir / "summary_table.tex", 'w') as f:
            f.write(latex)
            
        logger.info("Created summary statistics table")
        
    def _create_effect_sizes_table(self):
        """Create effect sizes table."""
        effect_sizes = self.statistics.get('effect_sizes', {})
        
        latex = r"""\begin{table}[ht]
\centering
\caption{Effect Sizes by Context Type}
\label{tab:effect_sizes}
\begin{tabular}{lrrrl}
\toprule
\textbf{Context} & \textbf{Cohen's d} & \textbf{Mean Effect} & \textbf{N} & \textbf{Interpretation} \\
\midrule
"""
        
        # Sort by Cohen's d
        sorted_contexts = sorted(effect_sizes.items(), 
                               key=lambda x: abs(x[1]['cohens_d']), 
                               reverse=True)
        
        for context, data in sorted_contexts:
            latex += "%s & %.3f & %.3f & %d & %s \\\\\n" % (
                self._format_context_name(context).replace('[TOKEN]', r'\texttt{[TOKEN]}'),
                data['cohens_d'],
                data['mean_effect'],
                data['n_tokens'],
                data['interpretation'].capitalize()
            )
            
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        with open(self.figures_dir / "effect_sizes_table.tex", 'w') as f:
            f.write(latex)
            
        logger.info("Created effect sizes table")
        
    def _create_sensitive_tokens_table(self):
        """Create table of most context-sensitive tokens."""
        # Load sensitive tokens
        sensitive_path = self.results_dir / "clustering_analysis/context_sensitive_tokens.json"
        
        if not sensitive_path.exists():
            logger.warning("No sensitive tokens data found")
            return
            
        with open(sensitive_path, 'r') as f:
            sensitive_tokens = json.load(f)[:10]  # Top 10
            
        latex = r"""\begin{table}[ht]
\centering
\caption{Top 10 Most Context-Sensitive Tokens}
\label{tab:sensitive_tokens}
\begin{tabular}{llrr}
\toprule
\textbf{Token} & \textbf{Type} & \textbf{Max Divergence} & \textbf{Most Affecting Context} \\
\midrule
"""
        
        for token_data in sensitive_tokens:
            token_idx = token_data['token_idx']
            token_info = self.token_info.get(token_idx, {})
            token_str = token_info.get('token_str', f'token_{token_idx}')
            token_type = token_info.get('token_type', 'unknown')
            
            # Get most affecting context
            if token_data['most_affecting_contexts']:
                top_context = token_data['most_affecting_contexts'][0][0]
                top_context_formatted = self._format_context_name(top_context)
            else:
                top_context_formatted = 'N/A'
                
            latex += r"\texttt{%s} & %s & %.3f & %s \\" % (
                token_str.replace('_', r'\_').replace('#', r'\#').replace('$', r'\$'),
                token_type.capitalize(),
                token_data['max_divergence'],
                top_context_formatted.replace('[TOKEN]', '')
            )
            latex += "\n"
            
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        with open(self.figures_dir / "sensitive_tokens_table.tex", 'w') as f:
            f.write(latex)
            
        logger.info("Created sensitive tokens table")
        
    def generate_all_paper_materials(self):
        """Generate all paper figures and tables."""
        logger.info("Generating paper materials...")
        
        # Create figures
        self.create_main_effect_figure()
        self.create_trajectory_examples_figure()
        
        # Create tables
        self.create_latex_tables()
        
        # Create figure list for paper
        figure_list = """# Paper Figures and Tables

## Figures

1. **main_context_effects.pdf** - Overview of context effects (4 panels)
   - (a) Effect sizes by context type
   - (b) Layer-wise divergence rates
   - (c) Token type sensitivity
   - (d) Distribution of effects

2. **trajectory_examples.pdf** - Example trajectory changes for selected tokens

## Tables

1. **summary_table.tex** - Summary statistics of context effects
2. **effect_sizes_table.tex** - Cohen's d effect sizes by context type
3. **sensitive_tokens_table.tex** - Top 10 most context-sensitive tokens

## Usage

Include in LaTeX document:
```latex
\\input{figures/summary_table}
\\input{figures/effect_sizes_table}
\\input{figures/sensitive_tokens_table}

\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{figures/main_context_effects}
    \\caption{Overview of context effects on token trajectories.}
    \\label{fig:main_effects}
\\end{figure}
```
"""
        
        with open(self.figures_dir / "README.md", 'w') as f:
            f.write(figure_list)
            
        logger.info(f"All paper materials saved to {self.figures_dir}")
        

def main():
    """Generate paper figures and tables."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/',
                       help='Directory containing analysis results')
    args = parser.parse_args()
    
    # Create generator
    generator = PaperFigureGenerator(args.results)
    
    # Generate all materials
    generator.generate_all_paper_materials()
    

if __name__ == "__main__":
    main()