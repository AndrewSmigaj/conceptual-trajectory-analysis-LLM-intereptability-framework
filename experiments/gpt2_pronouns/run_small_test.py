"""
Run a small test of the vocabulary context experiment
"""

import json
import torch
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np

# Load small test batch
with open('test_cases_small.json', 'r') as f:
    test_cases = json.load(f)

print(f"Testing with {len(test_cases)} cases")

# Load model and tokenizer
print("Loading GPT-2 model and tokenizer...")
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

# Load cluster labels
cluster_labels_path = Path("../gpt2/all_tokens/clustering_results_k10/all_labels_k10.json")
print(f"Loading cluster labels from {cluster_labels_path}")

if cluster_labels_path.exists():
    with open(cluster_labels_path, 'r') as f:
        cluster_labels = json.load(f)
    print(f"Loaded cluster labels for {len(cluster_labels)} layers")
else:
    print("ERROR: Cluster labels file not found!")
    print(f"Expected path: {cluster_labels_path.absolute()}")
    exit(1)

# Process a few examples
trajectories = {}
print("\nProcessing test cases...")

for i, test_case in enumerate(test_cases[:5]):  # Just process 5 for quick test
    text = test_case['text']
    token_idx = test_case['token_idx']
    context_frame = test_case['context_frame']
    
    print(f"\n{i+1}. Token idx: {token_idx}, Context: {context_frame}")
    print(f"   Text: '{text}'")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    # Get activations
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of tensors, one per layer
    
    # Extract trajectory for target token
    target_position = test_case['target_position']
    trajectory = []
    
    for layer_idx, layer_activations in enumerate(hidden_states):
        if layer_idx < len(hidden_states) - 1:  # Skip embedding layer
            activation = layer_activations[0, target_position].numpy()
            
            # Map to cluster - simplified version
            # In real implementation, we'd use the actual cluster centroids
            layer_clusters = cluster_labels.get(str(layer_idx), [])
            if token_idx < len(layer_clusters):
                cluster_id = layer_clusters[token_idx]
            else:
                cluster_id = 0  # Default
                
            trajectory.append(cluster_id)
    
    print(f"   Trajectory: {trajectory[:4]} (first 4 layers)")
    
    # Store trajectory
    key = f"{token_idx}_{context_frame}"
    trajectories[key] = {
        'token_idx': token_idx,
        'context_frame': context_frame,
        'path': trajectory
    }

print(f"\nSuccessfully processed {len(trajectories)} trajectories")
print("Pipeline test complete!")

# Save test results
test_output = {
    'trajectories': trajectories,
    'metadata': {
        'n_test_cases': len(test_cases),
        'n_processed': len(trajectories)
    }
}

with open('test_results.json', 'w') as f:
    json.dump(test_output, f, indent=2)
    
print("\nTest results saved to test_results.json")