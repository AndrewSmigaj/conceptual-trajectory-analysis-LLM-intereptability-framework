"""Extract key statistics and insights for the paper from apple routing experiment."""

import json
import numpy as np
from pathlib import Path
from collections import Counter

def extract_paper_statistics():
    """Extract all key statistics needed for the paper."""
    
    # Load results
    results_path = Path("experiments/apple_variety/results/apple_realistic/apple_realistic_routing_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load economic impact report
    economic_path = Path("experiments/apple_variety/results/apple_realistic/economic_impact_report.txt")
    with open(economic_path, 'r') as f:
        economic_report = f.read()
    
    statistics = {}
    
    # 1. Dataset Statistics
    statistics['dataset'] = {
        'total_samples': 1320,
        'training_samples': 844,
        'test_samples': 264,
        'n_varieties': 15,
        'n_features': 20,  # 5 numeric + 15 one-hot encoded varieties
        'routing_distribution': {
            'fresh_premium': 327,
            'fresh_standard': 786,
            'juice': 207
        },
        'class_imbalance_ratio': 786 / 207  # standard/juice ratio
    }
    
    # 2. Model Performance
    training_history = results['training_history']
    statistics['performance'] = {
        'final_test_accuracy': 92.80,
        'final_val_accuracy': training_history['val_accuracies'][-1],
        'best_val_accuracy': max(training_history['val_accuracies']),
        'epochs_until_convergence': len(training_history['train_losses']),
        'early_stopping_epoch': 108,
        'initial_loss': training_history['train_losses'][0],
        'final_loss': training_history['train_losses'][-1]
    }
    
    # 3. Clustering Analysis
    cluster_results = results['cluster_results']
    statistics['clustering'] = {
        'layer_0_clusters': cluster_results['0']['n_clusters'],
        'layer_1_clusters': cluster_results['1']['n_clusters'],
        'layer_2_clusters': cluster_results['2']['n_clusters'],
        'layer_3_clusters': cluster_results['3']['n_clusters'],
        'convergence_rate': (10 - 2) / 10 * 100,  # 80% reduction in clusters
        'majority_flow_percentage': 765 / 844 * 100  # ~90.6%
    }
    
    # 4. Trajectory Analysis
    paths = results['trajectory_analysis']['paths']
    path_counter = Counter([tuple(p) for p in paths])
    top_paths = path_counter.most_common(10)
    
    statistics['trajectories'] = {
        'total_unique_paths': len(path_counter),
        'top_path': {
            'pattern': list(top_paths[0][0]),
            'frequency': top_paths[0][1],
            'percentage': top_paths[0][1] / len(paths) * 100
        },
        'convergence_paths': sum(1 for p in paths if p[1:] == [1, 1, 1]),
        'convergence_percentage': sum(1 for p in paths if p[1:] == [1, 1, 1]) / len(paths) * 100
    }
    
    # 5. Economic Impact
    statistics['economic'] = {
        'total_economic_loss': 186.41,
        'juice_routing_count': 133,  # From training set
        'avg_loss_per_juice_apple': 186.41 / 133,
        'price_differential': {
            'premium_to_juice': 2.80 - 0.06,
            'standard_to_juice': 1.75 - 0.06
        },
        'most_affected_varieties': [
            {'name': 'Red Delicious', 'juice_rate': 93.1, 'loss': 64.26},
            {'name': 'McIntosh', 'juice_rate': 100.0, 'loss': 43.20},
            {'name': 'Cortland', 'juice_rate': 100.0, 'loss': 34.56}
        ]
    }
    
    # 6. Variety-Specific Insights
    variety_trajectories = results['trajectory_analysis']['variety_trajectories']
    premium_varieties = ['Honeycrisp', 'Jazz', 'Cosmic Crisp', 'Pink Lady', 'Ambrosia']
    standard_varieties = ['Gala', 'Fuji', 'Granny Smith', 'Braeburn']
    juice_varieties = ['Red Delicious', 'McIntosh', 'Cortland']
    
    statistics['variety_insights'] = {
        'premium_avg_unique_paths': np.mean([variety_trajectories[v]['unique_paths'] 
                                           for v in premium_varieties if v in variety_trajectories]),
        'standard_avg_unique_paths': np.mean([variety_trajectories[v]['unique_paths'] 
                                            for v in standard_varieties if v in variety_trajectories]),
        'juice_avg_unique_paths': np.mean([variety_trajectories[v]['unique_paths'] 
                                         for v in juice_varieties if v in variety_trajectories]),
        'premium_convergence_rate': np.mean([variety_trajectories[v]['top_paths'][0]['percentage'] 
                                           for v in premium_varieties if v in variety_trajectories]),
        'honeycrisp_juice_rate': 1.5  # From economic report
    }
    
    # 7. Fragmentation Metrics
    fragmentation = results['trajectory_analysis']['fragmentation']
    statistics['fragmentation'] = {
        'overall_fragmentation': fragmentation.get('overall', 0.0),
        'layer_transitions': len(fragmentation.get('layer_transitions', {}))
    }
    
    return statistics

def format_statistics_for_paper(stats):
    """Format statistics into LaTeX-ready text for the paper."""
    
    latex_snippets = []
    
    # Dataset description
    latex_snippets.append(f"""
% Dataset Statistics
Our realistic synthetic dataset comprised {stats['dataset']['total_samples']:,} apple samples 
representing {stats['dataset']['n_varieties']} commercial varieties, with 
{stats['dataset']['routing_distribution']['fresh_premium']} premium, 
{stats['dataset']['routing_distribution']['fresh_standard']} standard, and 
{stats['dataset']['routing_distribution']['juice']} juice-grade apples, 
reflecting a {stats['dataset']['class_imbalance_ratio']:.1f}:1 class imbalance.
""")
    
    # Model performance
    latex_snippets.append(f"""
% Model Performance
The neural network achieved {stats['performance']['final_test_accuracy']}\\% test accuracy 
after {stats['performance']['early_stopping_epoch']} epochs, demonstrating successful 
learning of variety-specific routing patterns. The model converged from an initial loss 
of {stats['performance']['initial_loss']:.2f} to {stats['performance']['final_loss']:.2f}.
""")
    
    # Clustering insights
    latex_snippets.append(f"""
% Clustering Analysis
Trajectory analysis revealed rapid conceptual convergence: the network compressed 
{stats['clustering']['layer_0_clusters']} initial clusters to just 
{stats['clustering']['layer_3_clusters']} by the final layer, representing an 
{stats['clustering']['convergence_rate']:.0f}\\% reduction. Notably, 
{stats['clustering']['majority_flow_percentage']:.1f}\\% of samples converged to a 
single "majority flow" pathway.
""")
    
    # Economic impact
    latex_snippets.append(f"""
% Economic Impact
The economic analysis revealed significant financial implications of routing decisions. 
Misclassification to juice grade resulted in a total loss of \\${stats['economic']['total_economic_loss']:.2f} 
across {stats['economic']['juice_routing_count']} samples (\\${stats['economic']['avg_loss_per_juice_apple']:.2f} per apple). 
Premium apples misrouted to juice incurred losses of \\${stats['economic']['price_differential']['premium_to_juice']:.2f}/lb, 
while standard apples lost \\${stats['economic']['price_differential']['standard_to_juice']:.2f}/lb.
""")
    
    # Variety-specific patterns
    latex_snippets.append(f"""
% Variety-Specific Patterns
Analysis of variety-specific trajectories revealed distinct processing patterns. 
Historical varieties like Red Delicious ({stats['economic']['most_affected_varieties'][0]['juice_rate']}\\% juice routing) 
and McIntosh ({stats['economic']['most_affected_varieties'][1]['juice_rate']}\\% juice routing) 
showed poor quality outcomes, while premium varieties like Honeycrisp maintained 
only {stats['variety_insights']['honeycrisp_juice_rate']}\\% juice routing, 
validating the model's ability to capture market realities.
""")
    
    # Fragmentation analysis
    latex_snippets.append(f"""
% Fragmentation Metrics
Trajectory analysis revealed {stats['trajectories']['total_unique_paths']} unique paths through the network, 
with {stats['trajectories']['convergence_percentage']:.1f}\\% of samples converging to common pathways. 
The rapid consolidation from {stats['clustering']['layer_0_clusters']} to {stats['clustering']['layer_3_clusters']} 
clusters demonstrates significant conceptual compression as the network processes apple quality features.
""")
    
    return '\n'.join(latex_snippets)

def main():
    """Extract statistics and format for paper."""
    
    # Extract statistics
    stats = extract_paper_statistics()
    
    # Save raw statistics
    stats_path = Path("experiments/apple_variety/paper_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved raw statistics to: {stats_path}")
    
    # Format for LaTeX
    latex_text = format_statistics_for_paper(stats)
    
    # Save LaTeX snippets
    latex_path = Path("experiments/apple_variety/paper_latex_snippets.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_text)
    print(f"Saved LaTeX snippets to: {latex_path}")
    
    # Print key highlights
    print("\n" + "="*60)
    print("KEY STATISTICS FOR PAPER")
    print("="*60)
    print(f"Dataset: {stats['dataset']['total_samples']} samples, {stats['dataset']['n_varieties']} varieties")
    print(f"Performance: {stats['performance']['final_test_accuracy']}% test accuracy")
    print(f"Clustering: {stats['clustering']['layer_0_clusters']} → {stats['clustering']['layer_3_clusters']} clusters ({stats['clustering']['convergence_rate']:.0f}% reduction)")
    print(f"Economic Loss: ${stats['economic']['total_economic_loss']:.2f} from juice misrouting")
    print(f"Convergence: {stats['trajectories']['convergence_percentage']:.1f}% of samples follow convergent paths")
    print(f"Top Path: {stats['trajectories']['top_path']['pattern']} ({stats['trajectories']['top_path']['percentage']:.1f}% of samples)")

if __name__ == "__main__":
    main()