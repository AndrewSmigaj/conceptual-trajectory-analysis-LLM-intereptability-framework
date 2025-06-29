"""
Select Diverse Tokens for Unified Context Effects Experiment

Selects 1,000 diverse tokens from the 10k vocabulary ensuring balanced
representation across different token types and linguistic categories.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_token_vocabulary(vocab_path: str) -> List[Dict[str, Any]]:
    """Load the 10k token vocabulary."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    logger.info(f"Loaded {len(tokens)} tokens from vocabulary")
    return tokens


def categorize_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize tokens by type and linguistic properties."""
    categories = defaultdict(list)
    
    for i, token_info in enumerate(tokens):
        token_info['index'] = i  # Add index for reference
        token_type = token_info.get('token_type', 'unknown')
        
        # Basic categorization by token type
        categories[f'type_{token_type}'].append(token_info)
        
        # Additional linguistic categorization
        token_str = token_info.get('token_str', '').strip()
        
        # Frequency-based categories
        if i < 200:  # Top 200 most frequent
            categories['high_frequency'].append(token_info)
        elif i < 1000:
            categories['medium_frequency'].append(token_info)
        else:
            categories['low_frequency'].append(token_info)
            
        # Length-based categories
        if len(token_str) <= 3:
            categories['short_tokens'].append(token_info)
        elif len(token_str) >= 8:
            categories['long_tokens'].append(token_info)
            
        # Special categories
        if token_info.get('is_punctuation', False):
            categories['punctuation'].append(token_info)
            
        # Linguistic function categories (heuristic)
        if token_str.lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
            categories['determiners'].append(token_info)
        elif token_str.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her']:
            categories['pronouns'].append(token_info)
        elif token_str.lower() in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'am']:
            categories['copula'].append(token_info)
        elif token_str.lower() in ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of']:
            categories['prepositions'].append(token_info)
        elif token_str.lower() in ['and', 'or', 'but', 'nor', 'yet', 'so']:
            categories['conjunctions'].append(token_info)
        elif token_str.lower() in ['not', 'no', 'never', 'neither', 'none']:
            categories['negations'].append(token_info)
            
    return dict(categories)


def select_diverse_subset(categories: Dict[str, List[Dict[str, Any]]], 
                         target_count: int = 1000) -> List[Dict[str, Any]]:
    """Select diverse subset ensuring balanced representation."""
    selected = []
    selected_indices = set()
    
    # Selection strategy:
    # 1. High-frequency words (200) - these are important for language structure
    # 2. Content words by type (600) - balanced across types
    # 3. Function words (100) - pronouns, determiners, prepositions
    # 4. Subword tokens (100) - to study composition effects
    
    # 1. High-frequency words (take top 200)
    high_freq = categories.get('high_frequency', [])[:200]
    for token in high_freq:
        if token['index'] not in selected_indices:
            selected.append(token)
            selected_indices.add(token['index'])
    logger.info(f"Selected {len(selected)} high-frequency tokens")
    
    # 2. Content words by type (aim for 600 total)
    content_types = ['type_complete_word', 'type_subword']
    tokens_per_type = 300
    
    for cat in content_types:
        available = [t for t in categories.get(cat, []) 
                    if t['index'] not in selected_indices]
        
        # Skip very high frequency (already included) and very low frequency
        filtered = [t for t in available if 200 <= t['index'] < 5000]
        
        # Random sample
        sample_size = min(tokens_per_type, len(filtered))
        if sample_size > 0:
            sampled = random.sample(filtered, sample_size)
            for token in sampled:
                selected.append(token)
                selected_indices.add(token['index'])
                
    logger.info(f"Total after content words: {len(selected)}")
    
    # 3. Function words (100)
    function_categories = ['pronouns', 'determiners', 'prepositions', 
                          'conjunctions', 'copula', 'negations']
    
    function_words = []
    for cat in function_categories:
        available = [t for t in categories.get(cat, []) 
                    if t['index'] not in selected_indices]
        function_words.extend(available)
    
    # Deduplicate and sample
    unique_function = list({t['index']: t for t in function_words}.values())
    sample_size = min(100, len(unique_function))
    if sample_size > 0:
        sampled = random.sample(unique_function, sample_size)
        for token in sampled:
            selected.append(token)
            selected_indices.add(token['index'])
            
    logger.info(f"Total after function words: {len(selected)}")
    
    # 4. Fill remaining slots with diverse tokens
    remaining_needed = target_count - len(selected)
    if remaining_needed > 0:
        all_available = []
        for cat, tokens in categories.items():
            if cat.startswith('type_'):
                available = [t for t in tokens if t['index'] not in selected_indices]
                all_available.extend(available)
        
        # Deduplicate
        unique_available = list({t['index']: t for t in all_available}.values())
        
        # Prioritize mid-frequency tokens
        unique_available.sort(key=lambda t: abs(t['index'] - 2500))
        
        for token in unique_available[:remaining_needed]:
            selected.append(token)
            selected_indices.add(token['index'])
            
    # Sort by index to maintain some frequency ordering
    selected.sort(key=lambda t: t['index'])
    
    logger.info(f"Final selection: {len(selected)} tokens")
    
    # Print distribution summary
    type_dist = defaultdict(int)
    for token in selected:
        type_dist[token.get('token_type', 'unknown')] += 1
    
    logger.info("Token type distribution:")
    for token_type, count in sorted(type_dist.items()):
        logger.info(f"  {token_type}: {count}")
        
    return selected


def save_selected_tokens(tokens: List[Dict[str, Any]], output_path: str) -> None:
    """Save selected tokens with metadata."""
    output_data = {
        'total_tokens': len(tokens),
        'selection_criteria': {
            'high_frequency': 200,
            'content_words': 600,
            'function_words': 100,
            'diverse_fill': 100
        },
        'tokens': tokens
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved {len(tokens)} selected tokens to {output_path}")
    
    # Also save a simple list for easy loading
    simple_list = [t['index'] for t in tokens]
    simple_path = Path(output_path).with_name('selected_token_indices.json')
    with open(simple_path, 'w') as f:
        json.dump(simple_list, f)
    logger.info(f"Saved token indices to {simple_path}")


def main():
    """Main function to select diverse tokens."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load vocabulary
    vocab_path = "../gpt2/all_tokens/top_10k_tokens_full.json"
    tokens = load_token_vocabulary(vocab_path)
    
    # Categorize tokens
    categories = categorize_tokens(tokens)
    
    # Print category statistics
    logger.info("\nToken categories found:")
    for cat, tokens_list in sorted(categories.items()):
        logger.info(f"  {cat}: {len(tokens_list)} tokens")
    
    # Select diverse subset
    selected = select_diverse_subset(categories, target_count=1000)
    
    # Save results
    output_path = "selected_tokens_unified.json"
    save_selected_tokens(selected, output_path)
    
    # Print some examples
    logger.info("\nExample selected tokens:")
    for i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        if i < len(selected):
            token = selected[i]
            logger.info(f"  {i}: '{token['token_str']}' (type: {token['token_type']}, "
                       f"idx: {token['index']})")


if __name__ == "__main__":
    main()