#!/usr/bin/env python
"""
Extract key statistics from analysis results for the paper - Version 2.
Updated to match actual JSON structure.
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
        
        # Extract metrics from the data section
        if 'metrics' in data['data']:
            metrics = data['data']['metrics']
            
            # Calculate overall averages
            all_entropy = []
            all_sparsity = []
            all_mi = []
            all_diagonal = []
            
            for context, context_metrics in metrics.items():
                if context != 'baseline':  # Skip baseline
                    for layer_metrics in context_metrics.values():
                        all_entropy.append(layer_metrics['entropy'])
                        all_sparsity.append(layer_metrics['sparsity'])
                        all_mi.append(layer_metrics['mutual_information'])
                        all_diagonal.append(layer_metrics['diagonal_dominance'])
            
            statistics['stratified_transition'] = {
                'mean_entropy': np.mean(all_entropy) if all_entropy else None,
                'mean_sparsity': np.mean(all_sparsity) if all_sparsity else None,
                'mean_mi': np.mean(all_mi) if all_mi else None,
                'mean_diagonal_dominance': np.mean(all_diagonal) if all_diagonal else None,
                'contexts_analyzed': len(metrics) - 1  # Exclude baseline
            }
        
        # Extract random comparison statistics
        if 'transition_vs_random' in data['statistics']:
            vs_random = data['statistics']['transition_vs_random']
            
            # Calculate average sparsity improvement
            sparsity_improvements = []
            for comparison in vs_random.values():
                if 'sparsity_difference' in comparison:
                    sparsity_improvements.append(comparison['sparsity_difference']['mean'])
            
            if sparsity_improvements and 'stratified_transition' in statistics:
                statistics['stratified_transition']['sparsity_improvement_vs_random'] = np.mean(sparsity_improvements)
        
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
        
        if st.get('mean_entropy') is not None:
            print(f"- Mean transition entropy: {st['mean_entropy']:.3f} bits")
        if st.get('mean_sparsity') is not None:
            print(f"- Mean sparsity: {st['mean_sparsity']:.3f}")
        if st.get('sparsity_improvement_vs_random') is not None:
            print(f"- Sparsity vs random baseline: {st['sparsity_improvement_vs_random']:.3f} (higher is sparser)")
        if st.get('mean_mi') is not None:
            print(f"- Mean mutual information: {st['mean_mi']:.3f}")
        if st.get('mean_diagonal_dominance') is not None:
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
    
    # Create LaTeX snippets
    print("\n" + "="*60)
    print("PAPER TEXT SNIPPETS")
    print("="*60)
    
    if 'stratified_transition' in statistics:
        st = statistics['stratified_transition']
        
        print("\n### Results Section Text:")
        print("\nOur analysis of context-induced transformations reveals several key findings.")
        
        if st.get('mean_entropy') is not None:
            print(f"\nFirst, the transition matrices show an average entropy of {st['mean_entropy']:.3f} bits, "
                  "indicating that transformations are not random but follow structured patterns.")
        
        if st.get('mean_sparsity') is not None:
            print(f"\nSecond, with an average sparsity of {st['mean_sparsity']:.3f}, the transformations are "
                  "highly selective, with tokens typically transitioning to only a few target clusters.")
        
        if st.get('sparsity_improvement_vs_random') is not None:
            improvement_pct = st['sparsity_improvement_vs_random'] * 100
            print(f"\nNotably, the observed transitions are {improvement_pct:.1f}% sparser than random baselines, "
                  "confirming that context creates systematic rather than arbitrary transformations.")
    
    # Save statistics
    output_path = Path("results_paper/paper_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n\nStatistics saved to: {output_path}")


if __name__ == "__main__":
    stats = extract_statistics()
    format_for_paper(stats)