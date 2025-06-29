import json
import numpy as np

# Load the results
with open('experiments/apple_variety/results/apple_variety_test/apple_quality_routing_test_results.json', 'r') as f:
    data = json.load(f)

results = data.get('results', {})
analysis = results.get('analysis', {})

print("=== APPLE QUALITY ROUTING EXPERIMENT RESULTS ===\n")

# Basic metrics
print(f"Test Accuracy: {results.get('test_accuracy', 'N/A')}%")
print(f"Training Samples: {results.get('n_train', 'N/A')}")
print(f"Test Samples: {results.get('n_test', 'N/A')}")
print()

# Routing information
routing = analysis.get('variety_routing', {})
if routing:
    print("Routing Classes:", routing.get('routing_classes', []))
    print()
    
    # By variety analysis
    by_variety = routing.get('by_variety', {})
    if by_variety:
        print("=== VARIETY-SPECIFIC ROUTING RESULTS ===")
        
        # Collect routing accuracy data
        total_correct = 0
        total_samples = 0
        routing_confusion = {'fresh_premium': {}, 'fresh_standard': {}, 'juice': {}}
        
        for variety, info in by_variety.items():
            if isinstance(info, dict) and 'samples' in info:
                print(f"\n{variety}:")
                print(f"  Total Samples: {info.get('count', 0)}")
                
                # Count routing distribution
                routing_dist = {}
                for sample in info['samples']:
                    if 'predicted_route' in sample:
                        route = sample['predicted_route']
                        routing_dist[route] = routing_dist.get(route, 0) + 1
                
                if routing_dist:
                    print(f"  Routing Distribution: {routing_dist}")
                    
                # Economic impact if available
                if 'economic_impact' in info:
                    print(f"  Economic Impact: ${info['economic_impact']:.2f}")

# Look for confusion matrix or accuracy metrics
train_metrics = analysis.get('train_metrics', {})
test_metrics = analysis.get('test_metrics', {})

if 'confusion_matrix' in test_metrics:
    print("\n=== CONFUSION MATRIX ===")
    cm = test_metrics['confusion_matrix']
    print("(Note: This may be variety classification, not routing)")
    
# LLM Analysis
llm_analysis = analysis.get('llm_analysis', {})
if llm_analysis:
    print("\n=== LLM ANALYSIS SUMMARY ===")
    if 'cluster_interpretations' in llm_analysis:
        print("\nCluster Interpretations Available")
    if 'routing_patterns' in llm_analysis:
        print("Routing Patterns Analysis Available")

# Problematic varieties
problematic = routing.get('problematic_varieties', [])
if problematic:
    print("\n=== PROBLEMATIC VARIETIES ===")
    for i, var in enumerate(problematic[:10]):
        print(f"{i+1}. {var['variety']}: {var['issue']} (score: {var['score']:.2f})")