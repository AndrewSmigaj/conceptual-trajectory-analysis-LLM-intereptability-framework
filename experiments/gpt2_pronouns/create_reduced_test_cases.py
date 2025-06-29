#!/usr/bin/env python3
"""
Create reduced test cases with 5,000 tokens × 20 key contexts for paper.
"""

import json
import random
from pathlib import Path

def create_reduced_test_cases(output_file="reduced_test_cases_20contexts.json"):
    """Create test cases with all tokens but only key contexts."""
    
    print("Creating reduced test cases with 20 key contexts...")
    
    # Load the full test cases
    with open("expanded_test_cases.json", 'r') as f:
        full_data = json.load(f)
    
    all_test_cases = full_data['test_cases']
    metadata = full_data['metadata']
    
    # Define the 20 key contexts for the paper
    key_contexts = [
        "baseline",      # No context
        "the",          # Definite article
        "a",            # Indefinite article
        "he",           # Personal pronoun (masculine)
        "she",          # Personal pronoun (feminine)
        "it",           # Personal pronoun (neuter)
        "is",           # Copula (present)
        "was",          # Copula (past)
        "in",           # Preposition (location)
        "on",           # Preposition (surface)
        "with",         # Preposition (accompaniment)
        "and",          # Conjunction (coordinating)
        "but",          # Conjunction (contrasting)
        "not",          # Negation
        "said",         # Past tense verb
        "will",         # Modal auxiliary
        "good",         # Adjective
        "time",         # Common noun
        ".",            # Punctuation
        "sentence_start" # Special position marker
    ]
    
    print(f"Using {len(key_contexts)} key contexts")
    print(f"Contexts: {key_contexts}")
    
    # Filter test cases to only include key contexts
    reduced_test_cases = []
    token_ids_seen = set()
    
    for case in all_test_cases:
        if case['context'] in key_contexts:
            reduced_test_cases.append(case)
            token_ids_seen.add(case['token_id'])
    
    print(f"Selected {len(reduced_test_cases)} test cases")
    print(f"Covering {len(token_ids_seen)} unique tokens")
    
    # Update metadata
    new_metadata = {
        "n_tokens": len(token_ids_seen),
        "n_contexts": len(key_contexts),
        "total_test_cases": len(reduced_test_cases),
        "contexts": key_contexts,
        "token_ids": sorted(list(token_ids_seen)),
        "description": "Reduced dataset with 5,000 tokens × 20 key contexts for comprehensive paper analysis"
    }
    
    # Save to file
    output_data = {
        "metadata": new_metadata,
        "test_cases": reduced_test_cases
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCreated {len(reduced_test_cases)} test cases")
    print(f"Expected: ~{len(token_ids_seen)} tokens × {len(key_contexts)} contexts = ~{len(token_ids_seen) * len(key_contexts)} cases")
    print(f"Actual: {len(reduced_test_cases)} cases")
    print(f"Saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary by context:")
    context_counts = {}
    for case in reduced_test_cases:
        ctx = case['context']
        context_counts[ctx] = context_counts.get(ctx, 0) + 1
    
    for ctx in key_contexts:
        count = context_counts.get(ctx, 0)
        print(f"  {ctx:15} : {count:,} test cases")
    
    return output_data

if __name__ == "__main__":
    create_reduced_test_cases()