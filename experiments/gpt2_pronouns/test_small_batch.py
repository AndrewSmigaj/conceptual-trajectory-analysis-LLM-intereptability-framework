"""
Test script to verify the pipeline works with a small batch
"""

import json
from vocabulary_context_experiment import VocabularyContextExperiment

# Load test cases
with open('test_cases_full_vocabulary.json', 'r') as f:
    test_cases = json.load(f)

# Take only first 100 test cases for quick testing
small_test_cases = test_cases[:100]

# Save small batch
with open('test_cases_small.json', 'w') as f:
    json.dump(small_test_cases, f)

print(f"Created small test batch with {len(small_test_cases)} cases")

# Modify config for small test
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create test config
test_config = config.copy()
test_config['experiment']['name'] = 'gpt2_small_test'
test_config['experiment']['output_dir'] = './test_results/'
test_config['data']['batch_size'] = 50

with open('test_config.yaml', 'w') as f:
    yaml.dump(test_config, f)

print("Created test config")
print("Run with: python vocabulary_context_experiment.py --config test_config.yaml")