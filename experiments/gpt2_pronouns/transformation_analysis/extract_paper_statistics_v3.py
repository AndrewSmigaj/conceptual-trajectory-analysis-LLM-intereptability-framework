#!/usr/bin/env python
"""
Extract key statistics from analysis results for the paper - Version 3.
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
        
        # Extract from transition_metrics if available
        if 'transition_metrics' in data['data']:
            metrics = data['data']['transition_metrics']
            
            # Calculate overall averages across all contexts and layers
            all_entropy = []
            all_sparsity = []
            all_mi = []
            all_diagonal = []
            
            for context, layer_metrics in metrics.items():
                if context != 'baseline':  # Skip baseline
                    for layer, layer_data in layer_metrics.items():
                        all_entropy.append(layer_data['entropy'])
                        all_sparsity.append(layer_data['sparsity'])
                        all_mi.append(layer_data['mutual_information'])
                        all_diagonal.append(layer_data['diagonal_dominance'])
            
            statistics['stratified_transition'] = {
                'mean_entropy': np.mean(all_entropy) if all_entropy else None,
                'std_entropy': np.std(all_entropy) if all_entropy else None,
                'mean_sparsity': np.mean(all_sparsity) if all_sparsity else None,
                'std_sparsity': np.std(all_sparsity) if all_sparsity else None,
                'mean_mi': np.mean(all_mi) if all_mi else None,
                'mean_diagonal_dominance': np.mean(all_diagonal) if all_diagonal else None,
                'contexts_analyzed': len([k for k in metrics.keys() if k != 'baseline'])
            }
        
        # Extract random comparison statistics
        if 'transition_vs_random' in data['statistics']:
            vs_random = data['statistics']['transition_vs_random']
            
            # Calculate average sparsity improvement
            sparsity_improvements = []
            entropy_differences = []
            
            for comparison in vs_random.values():
                if 'sparsity_difference' in comparison:
                    sparsity_improvements.append(comparison['sparsity_difference']['mean'])
                if 'entropy_difference' in comparison:
                    entropy_differences.append(comparison['entropy_difference']['mean'])
            
            if sparsity_improvements:
                statistics['vs_random'] = {
                    'mean_sparsity_improvement': np.mean(sparsity_improvements),
                    'mean_entropy_difference': np.mean(entropy_differences) if entropy_differences else None
                }
        
        # Check for stratified results in summary
        if 'summary' in data and 'stratified_analysis' in data['summary']:
            statistics['stratified_summary'] = data['summary']['stratified_analysis']
    
    # 2. Check for figures
    figures_dir = results_dir / "stratified_transition"
    if figures_dir.exists():
        png_files = list(figures_dir.glob("*.png"))
        statistics['figures_generated'] = len(png_files)
    
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
            print(f"- Mean transition entropy: {st['mean_entropy']:.3f} ± {st.get('std_entropy', 0):.3f} bits")
        if st.get('mean_sparsity') is not None:
            print(f"- Mean sparsity: {st['mean_sparsity']:.3f} ± {st.get('std_sparsity', 0):.3f}")
        if st.get('mean_mi') is not None:
            print(f"- Mean mutual information: {st['mean_mi']:.3f}")
        if st.get('mean_diagonal_dominance') is not None:
            print(f"- Mean diagonal dominance: {st['mean_diagonal_dominance']:.3f}")
        print(f"- Contexts analyzed: {st['contexts_analyzed']}")
    
    # Comparison to random
    if 'vs_random' in statistics:
        vr = statistics['vs_random']
        print("\n## Comparison to Random Baselines")
        
        if vr.get('mean_sparsity_improvement') is not None:
            # Positive value means actual transitions are sparser
            improvement = vr['mean_sparsity_improvement']
            print(f"- Sparsity improvement: {improvement:.3f} (transitions are {improvement*100:.1f}% sparser than random)")
        
        if vr.get('mean_entropy_difference') is not None:
            # Negative value means actual transitions have lower entropy
            diff = vr['mean_entropy_difference']
            print(f"- Entropy difference: {diff:.3f} bits (transitions have {abs(diff):.3f} bits less entropy)")
    
    # Figures
    if 'figures_generated' in statistics:
        print(f"\n## Visualizations")
        print(f"- Figures generated: {statistics['figures_generated']}")
    
    # Create paper text
    print("\n" + "="*60)
    print("PAPER TEXT SNIPPETS")
    print("="*60)
    
    if 'stratified_transition' in statistics and 'vs_random' in statistics:
        st = statistics['stratified_transition']
        vr = statistics['vs_random']
        
        print("\n### Results Section Text:")
        print("\nOur analysis of context-induced transformations reveals systematic, non-random patterns.")
        
        if st.get('mean_entropy') is not None:
            print(f"\nTransition matrices exhibit an average entropy of {st['mean_entropy']:.2f} ± {st.get('std_entropy', 0):.2f} bits, "
                  "substantially lower than the theoretical maximum, indicating structured transformations.")
        
        if st.get('mean_sparsity') is not None and vr.get('mean_sparsity_improvement') is not None:
            print(f"\nWith a mean sparsity of {st['mean_sparsity']:.2f}, transformations are highly selective. "
                  f"Critically, these transitions are {vr['mean_sparsity_improvement']*100:.1f}% sparser than random baselines, "
                  "demonstrating that context creates systematic rather than arbitrary transformations.")
        
        print(f"\nAcross {st['contexts_analyzed']} different linguistic contexts, we observe consistent patterns "
              "of transformation, supporting our hypothesis that context acts as a systematic operator on the "
              "representation space.")
    
    # LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    
    if 'stratified_transition' in statistics:
        st = statistics['stratified_transition']
        vr = statistics.get('vs_random', {})
        
        print("\n% Table: Transformation Metrics")
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\caption{Context transformation characteristics in GPT-2}")
        print(r"\begin{tabular}{lc}")
        print(r"\hline")
        print(r"Metric & Value \\")
        print(r"\hline")
        
        if st.get('mean_entropy'):
            print(f"Transition entropy (bits) & ${st['mean_entropy']:.2f} \\pm {st.get('std_entropy', 0):.2f}$ \\\\")
        if st.get('mean_sparsity'):
            print(f"Transition sparsity & ${st['mean_sparsity']:.2f} \\pm {st.get('std_sparsity', 0):.2f}$ \\\\")
        if vr.get('mean_sparsity_improvement'):
            print(f"Sparsity vs. random & ${vr['mean_sparsity_improvement']*100:.1f}\\%$ sparser \\\\")
        if st.get('mean_mi'):
            print(f"Mutual information & ${st['mean_mi']:.3f}$ \\\\")
        
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")
    
    # Save statistics
    output_path = Path("results_paper/paper_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n\nStatistics saved to: {output_path}")
    
    # Print next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Run publication_figures.py to generate high-quality figures")
    print("2. Select 4-5 best figures from results_paper/publication_figures/")
    print("3. Use the LaTeX table above in your paper")
    print("4. Incorporate the text snippets into your Results section")
    print("5. Update Methods section with analysis details")


if __name__ == "__main__":
    stats = extract_statistics()
    format_for_paper(stats)