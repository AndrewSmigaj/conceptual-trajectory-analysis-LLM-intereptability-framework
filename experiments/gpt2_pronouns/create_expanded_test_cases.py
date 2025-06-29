#!/usr/bin/env python3
"""
Create expanded test cases with more tokens for comprehensive analysis.
"""

import json
import random
from pathlib import Path
from transformers import GPT2Tokenizer
from comprehensive_contexts import get_all_contexts, get_context_info

def create_expanded_test_cases(n_tokens=10000, output_file="expanded_test_cases.json"):
    """Create test cases with specified number of tokens."""
    
    print(f"Creating test cases with {n_tokens} tokens...")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Get all single token IDs (excluding special tokens)
    single_token_ids = []
    vocab_size = len(tokenizer)
    
    print("Identifying single-token words...")
    for i in range(vocab_size):
        if i % 5000 == 0:
            print(f"  Progress: {i}/{vocab_size} tokens checked...")
            
        # Skip special tokens
        if i in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
            continue
            
        # Decode token
        token_str = tokenizer.decode([i])
        
        # Re-encode to check if it's truly single token
        reencoded = tokenizer.encode(token_str, add_special_tokens=False)
        
        if len(reencoded) == 1 and reencoded[0] == i:
            single_token_ids.append(i)
    
    print(f"Found {len(single_token_ids)} single-token words in vocabulary")
    
    # Sample tokens with stratification
    if n_tokens >= len(single_token_ids):
        # Use all tokens
        selected_tokens = single_token_ids
        print(f"Using all {len(selected_tokens)} single tokens")
    else:
        # Sample with preference for common tokens
        # Sort by token string length as proxy for frequency (shorter = more common)
        token_lengths = [(i, len(tokenizer.decode([i]).strip())) for i in single_token_ids]
        token_lengths.sort(key=lambda x: x[1])
        
        # Take more from common (short) tokens
        n_common = int(n_tokens * 0.6)
        n_random = n_tokens - n_common
        
        common_tokens = [t[0] for t in token_lengths[:len(token_lengths)//3]]
        selected_tokens = random.sample(common_tokens, min(n_common, len(common_tokens)))
        
        # Add random tokens from rest
        remaining = [t for t in single_token_ids if t not in selected_tokens]
        selected_tokens.extend(random.sample(remaining, min(n_random, len(remaining))))
        
        print(f"Selected {len(selected_tokens)} tokens ({n_common} common, {n_random} random)")
    
    # Get all contexts
    contexts = get_all_contexts()
    print(f"Using {len(contexts)} contexts (including baseline)")
    
    # Create test cases
    test_cases = []
    for token_id in selected_tokens:
        token_str = tokenizer.decode([token_id])
        
        for context in contexts:
            if context == "baseline":
                # No context case
                test_case = {
                    "id": f"t{token_id}_baseline",
                    "token_id": token_id,
                    "token_str": token_str,
                    "context": "baseline",
                    "context_str": "",
                    "full_text": token_str
                }
            elif context == "sentence_start":
                # Special position marker
                test_case = {
                    "id": f"t{token_id}_sentence_start",
                    "token_id": token_id,
                    "token_str": token_str,
                    "context": "sentence_start",
                    "context_str": "",
                    "full_text": token_str,
                    "is_sentence_start": True
                }
            else:
                # Regular context
                test_case = {
                    "id": f"t{token_id}_{context}",
                    "token_id": token_id,
                    "token_str": token_str,
                    "context": context,
                    "context_str": context,
                    "full_text": f"{context} {token_str}"
                }
            
            # Add context info
            context_info = get_context_info(context)
            test_case.update(context_info)
            
            test_cases.append(test_case)
    
    # Create metadata
    metadata = {
        "n_tokens": len(selected_tokens),
        "n_contexts": len(contexts),
        "total_test_cases": len(test_cases),
        "contexts": contexts,
        "token_ids": selected_tokens
    }
    
    # Save to file
    output_data = {
        "metadata": metadata,
        "test_cases": test_cases
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCreated {len(test_cases)} test cases")
    print(f"Saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Tokens: {len(selected_tokens)}")
    print(f"  Contexts: {len(contexts)}")
    print(f"  Total combinations: {len(test_cases)}")
    print(f"  Expected trajectories: {len(test_cases):,}")
    
    return output_data

if __name__ == "__main__":
    # Create test cases with 10,000 tokens
    create_expanded_test_cases(n_tokens=10000)