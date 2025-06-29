#!/usr/bin/env python3
"""
Fast version of test case creation using pre-filtering.
"""

import json
import random
from pathlib import Path
from transformers import GPT2Tokenizer
from comprehensive_contexts import get_all_contexts, get_context_info
import time

def create_test_cases_fast(n_tokens=10000, output_file="expanded_test_cases.json"):
    """Create test cases more efficiently."""
    
    start_time = time.time()
    print(f"Creating test cases with {n_tokens} tokens (fast version)...")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Use a pre-filtered list of common single tokens
    # These are known single-token words in GPT-2
    common_single_tokens = []
    
    # Add common words that are known to be single tokens
    test_words = [
        # Common function words
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "may", "might", "shall", "should", "to",
        "of", "in", "on", "at", "by", "for", "with", "from",
        "and", "but", "or", "nor", "yet", "so", "he", "she", "it",
        "they", "we", "I", "you", "my", "your", "his", "her",
        "not", "no", "yes", "this", "that", "these", "those",
        
        # Common content words
        "time", "person", "year", "way", "day", "man", "thing", "world",
        "life", "hand", "part", "child", "eye", "woman", "place", "work",
        "week", "case", "point", "government", "company", "number", "group",
        "said", "make", "go", "take", "come", "see", "know", "get", "give",
        "find", "think", "tell", "ask", "work", "seem", "feel", "try",
        "good", "new", "first", "last", "long", "great", "little", "own",
        "other", "old", "right", "big", "high", "different", "small",
        "very", "just", "now", "then", "also", "well", "only", "even",
        
        # Numbers and punctuation  
        "one", "two", "three", "four", "five", "ten", "hundred",
        ".", ",", "?", "!", ":", ";", "'", '"',
    ]
    
    print("Checking common words...")
    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            common_single_tokens.append(tokens[0])
    
    print(f"Found {len(common_single_tokens)} common single tokens")
    
    # Now scan vocabulary more efficiently
    print("Scanning vocabulary for more single tokens...")
    vocab_size = len(tokenizer)
    
    # Sample random tokens to check
    random_indices = random.sample(range(vocab_size), min(20000, vocab_size))
    
    checked = 0
    for i in random_indices:
        if checked % 1000 == 0 and checked > 0:
            print(f"  Checked {checked} random tokens, found {len(common_single_tokens)} single tokens so far...")
        
        checked += 1
        
        # Skip if already in list
        if i in common_single_tokens:
            continue
            
        # Quick check - decode and look at length
        token_str = tokenizer.decode([i])
        
        # Skip if too long (likely not single token on re-encode)
        if len(token_str) > 20:
            continue
            
        # Check if it re-encodes to single token
        try:
            reencoded = tokenizer.encode(token_str, add_special_tokens=False)
            if len(reencoded) == 1 and reencoded[0] == i:
                common_single_tokens.append(i)
        except:
            continue
    
    print(f"Total single tokens found: {len(common_single_tokens)}")
    
    # Remove duplicates
    single_token_ids = list(set(common_single_tokens))
    print(f"Unique single tokens: {len(single_token_ids)}")
    
    # Sample requested number
    if n_tokens >= len(single_token_ids):
        selected_tokens = single_token_ids
    else:
        selected_tokens = random.sample(single_token_ids, n_tokens)
    
    print(f"Selected {len(selected_tokens)} tokens")
    
    # Get all contexts
    contexts = get_all_contexts()
    print(f"Using {len(contexts)} contexts")
    
    # Create test cases
    print("Creating test cases...")
    test_cases = []
    total_cases = len(selected_tokens) * len(contexts)
    created = 0
    
    for token_id in selected_tokens:
        token_str = tokenizer.decode([token_id])
        
        for context in contexts:
            if created % 10000 == 0 and created > 0:
                print(f"  Created {created}/{total_cases} test cases...")
            
            created += 1
            
            if context == "baseline":
                test_case = {
                    "id": f"t{token_id}_baseline",
                    "token_id": token_id,
                    "token_str": token_str,
                    "context": "baseline",
                    "context_str": "",
                    "full_text": token_str
                }
            elif context == "sentence_start":
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
    
    print(f"Saving {len(test_cases)} test cases to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Created {len(test_cases)} test cases")
    print(f"Saved to {output_file}")
    
    return output_data

if __name__ == "__main__":
    # Create with 5000 tokens as a reasonable compromise
    create_test_cases_fast(n_tokens=5000)