"""
Test Unified Context Experiment with Small Dataset

Quick test to verify the pipeline works before running the full experiment.
"""

import json
from unified_context_experiment import UnifiedContextExperiment

def create_small_test_cases():
    """Create a small subset of test cases for testing."""
    # Load full test cases
    with open('test_cases_unified.json', 'r') as f:
        data = json.load(f)
    
    # Take first 100 test cases (10 tokens x 10 contexts)
    small_data = {
        'metadata': data['metadata'],
        'test_cases': data['test_cases'][:100]
    }
    
    # Save small test cases
    with open('test_cases_unified_small.json', 'w') as f:
        json.dump(small_data, f, indent=2)
    
    print(f"Created small test set with {len(small_data['test_cases'])} test cases")

def run_small_test():
    """Run experiment on small dataset."""
    # Create small config
    small_config = """
experiment:
  name: "gpt2_unified_context_effects_small"
  description: "Small test of unified clustering"
  output_dir: "./results_unified_small/"

model:
  name: "gpt2"
  type: "transformer"
  num_layers: 12
  hidden_size: 768
  
analysis:
  k_clusters: 5  # Fewer clusters for small test
  early_layers: [0, 1, 2, 3]
  all_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  
  windows:
    early: [0, 1, 2, 3]
    middle: [4, 5, 6, 7]
    late: [8, 9, 10, 11]

data:
  test_cases_path: "test_cases_unified_small.json"
  tokens_path: "selected_tokens_unified.json"
  batch_size: 10

output:
  checkpoint_interval: 50
  save_activations: true
  save_trajectories: true
  save_cluster_models: true
  generate_sankey: true
  generate_heatmaps: true
  generate_statistics: true
"""
    
    # Save config
    with open('config_unified_small.yaml', 'w') as f:
        f.write(small_config)
    
    # Run experiment
    print("Running small unified context experiment...")
    experiment = UnifiedContextExperiment('config_unified_small.yaml')
    experiment.setup()
    results = experiment.execute()
    
    print(f"\nSmall test complete: {results}")
    return results

if __name__ == "__main__":
    # Create small dataset
    create_small_test_cases()
    
    # Run test
    results = run_small_test()