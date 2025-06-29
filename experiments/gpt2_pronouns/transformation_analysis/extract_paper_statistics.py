#!/usr/bin/env python
"""
Extract key statistics from analysis results for the paper.
"""

import json
from pathlib import Path
import numpy as np


def extract_statistics():
    """Extract key statistics from available results."""
    
    results_dir = Path("results_paper")
    statistics = {}
    
    # 1. Stratified Transition Analysis
    stratified_path = results_dir / "stratified_transition" / "stratified_transition_analysis_results.json"
    if stratified_path.exists():
        with open(stratified_path, 'r') as f:
            data = json.load(f)
            
        statistics['stratified_transition'] = {
            'mean_entropy': data['statistics'].get('mean_entropy'),
            'mean_sparsity': data['statistics'].get('mean_sparsity'),
            'mean_mi': data['statistics'].get('mean_mi'),
            'mean_diagonal_dominance': data['statistics'].get('mean_diagonal_dominance'),
            'sparsity_vs_random': data['statistics'].get('sparsity_improvement_vs_random'),
            'contexts_analyzed': len(data['data']['transition_matrices'])
        }
        
        # Extract stratified results
        if 'stratified_results' in data['data']:
            strat = data['data']['stratified_results']
            
            # By frequency
            if 'by_frequency' in strat:
                freq_stats = {}
                for level, metrics in strat['by_frequency'].items():
                    freq_stats[level] = {
                        'n_tokens': metrics['n_tokens'],
                        'avg_entropy': metrics['avg_entropy'],
                        'avg_sparsity': metrics['avg_sparsity']
                    }
                statistics['stratified_by_frequency'] = freq_stats
            
            # By type  
            if 'by_type' in strat:
                type_stats = {}
                for token_type, metrics in strat['by_type'].items():
                    type_stats[token_type] = {
                        'n_tokens': metrics['n_tokens'],
                        'avg_entropy': metrics['avg_entropy'],
                        'avg_sparsity': metrics['avg_sparsity']
                    }
                statistics['stratified_by_type'] = type_stats
    
    # 2. Effect Sizes (if available)
    effect_path = results_dir / "effect_sizes" / "effect_size_calculator_results.json"
    if effect_path.exists():
        with open(effect_path, 'r') as f:
            data = json.load(f)
            
        statistics['effect_sizes'] = {
            'mean_cohens_d': data['statistics'].get('mean_cohens_d'),
            'mean_hedges_g': data['statistics'].get('mean_hedges_g'),
            'mean_cliffs_delta': data['statistics'].get('mean_cliffs_delta'),
            'large_effects_percentage': data['statistics'].get('large_effects_percentage')
        }
    
    # 3. Permutation Test (if available)
    perm_path = results_dir / "permutation_significance" / "permutation_significance_test_results.json"
    if perm_path.exists() and Path(perm_path).stat().st_size > 0:
        with open(perm_path, 'r') as f:
            data = json.load(f)
            
        statistics['permutation_test'] = {
            'n_significant_layers': data['statistics'].get('n_significant_layers'),
            'avg_p_value': data['statistics'].get('avg_p_value'),
            'min_p_value': data['statistics'].get('min_p_value')
        }
    
    # 4. Information Theory (if available)
    info_path = results_dir / "information_theory" / "information_theory_metrics_results.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            data = json.load(f)
            
        statistics['information_theory'] = {
            'avg_mutual_information': data['statistics'].get('avg_mutual_information'),
            'avg_kl_divergence': data['statistics'].get('avg_kl_divergence'),
            'avg_entropy': data['statistics'].get('avg_entropy'),
            'avg_js_divergence': data['statistics'].get('avg_js_divergence')
        }
    
    return statistics


def format_for_paper(statistics):
    """Format statistics for inclusion in paper."""
    
    print("="*60)
    print("KEY STATISTICS FOR PAPER")
    print("="*60)
    
    # Overall transformation metrics
    if 'stratified_transition' in statistics:
        st = statistics['stratified_transition']
        print("\n## Transformation Characteristics")
        if st['mean_entropy'] is not None:
            print(f"- Mean transition entropy: {st['mean_entropy']:.3f} bits")
        if st['mean_sparsity'] is not None:
            print(f"- Mean sparsity: {st['mean_sparsity']:.3f}")
        if st['sparsity_vs_random'] is not None:
            print(f"- Sparsity vs random baseline: {st['sparsity_vs_random']:.1%} sparser")
        if st['mean_mi'] is not None:
            print(f"- Mean mutual information: {st['mean_mi']:.3f}")
        if st['mean_diagonal_dominance'] is not None:
            print(f"- Mean diagonal dominance: {st['mean_diagonal_dominance']:.3f}")
        print(f"- Contexts analyzed: {st['contexts_analyzed']}")
    
    # Stratified results
    if 'stratified_by_frequency' in statistics:
        print("\n## Stratification by Frequency")
        for level, metrics in statistics['stratified_by_frequency'].items():
            print(f"\n{level.capitalize()} frequency tokens (n={metrics['n_tokens']}):")
            print(f"  - Entropy: {metrics['avg_entropy']:.3f}")
            print(f"  - Sparsity: {metrics['avg_sparsity']:.3f}")
    
    if 'stratified_by_type' in statistics:
        print("\n## Stratification by Token Type")
        for token_type, metrics in statistics['stratified_by_type'].items():
            print(f"\n{token_type.capitalize()} tokens (n={metrics['n_tokens']}):")
            print(f"  - Entropy: {metrics['avg_entropy']:.3f}")
            print(f"  - Sparsity: {metrics['avg_sparsity']:.3f}")
    
    # Effect sizes
    if 'effect_sizes' in statistics:
        es = statistics['effect_sizes']
        print("\n## Effect Sizes")
        print(f"- Mean Cohen's d: {es['mean_cohens_d']:.3f}")
        print(f"- Mean Hedge's g: {es['mean_hedges_g']:.3f}")
        print(f"- Mean Cliff's delta: {es['mean_cliffs_delta']:.3f}")
        print(f"- Large effects: {es['large_effects_percentage']:.1%}")
    
    # Statistical significance
    if 'permutation_test' in statistics:
        pt = statistics['permutation_test']
        print("\n## Statistical Significance")
        print(f"- Significant layers: {pt['n_significant_layers']}/12")
        print(f"- Average p-value: {pt['avg_p_value']:.4f}")
        print(f"- Minimum p-value: {pt['min_p_value']:.4f}")
    
    # Information theory
    if 'information_theory' in statistics:
        it = statistics['information_theory']
        print("\n## Information Theory Metrics")
        print(f"- Average mutual information: {it['avg_mutual_information']:.3f} bits")
        print(f"- Average KL divergence: {it['avg_kl_divergence']:.3f}")
        print(f"- Average entropy: {it['avg_entropy']:.3f} bits")
        print(f"- Average JS divergence: {it['avg_js_divergence']:.3f}")
    
    # Save to file
    output_path = Path("results_paper/paper_statistics.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n\nStatistics saved to: {output_path}")
    
    # Create LaTeX table snippets
    print("\n" + "="*60)
    print("LATEX TABLE SNIPPETS")
    print("="*60)
    
    # Table 1: Overall metrics
    print("\n% Table 1: Overall Transformation Metrics")
    if 'stratified_transition' in statistics:
        st = statistics['stratified_transition']
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\caption{Overall transformation characteristics across contexts}")
        print(r"\begin{tabular}{lc}")
        print(r"\hline")
        print(r"Metric & Value \\")
        print(r"\hline")
        print(f"Mean transition entropy & {st['mean_entropy']:.3f} bits \\\\")
        print(f"Mean sparsity & {st['mean_sparsity']:.3f} \\\\")
        print(f"Sparsity vs random & {st['sparsity_vs_random']:.1%} sparser \\\\")
        print(f"Mutual information & {st['mean_mi']:.3f} \\\\")
        print(f"Diagonal dominance & {st['mean_diagonal_dominance']:.3f} \\\\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


if __name__ == "__main__":
    stats = extract_statistics()
    format_for_paper(stats)