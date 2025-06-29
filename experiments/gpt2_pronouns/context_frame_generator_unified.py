"""
Context Frame Generator for Unified Clustering Experiment

Generates test cases with 10 different context frames for the selected 1,000 tokens.
Handles special cases and filtering for valid combinations.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedContextFrameGenerator:
    """Generate context frames for unified clustering analysis."""
    
    def __init__(self, selected_tokens_path: str = "selected_tokens_unified.json"):
        """Initialize with selected tokens."""
        # Load selected tokens
        with open(selected_tokens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.tokens = data['tokens']
            
        logger.info(f"Loaded {len(self.tokens)} selected tokens")
        
        # Define context frames
        self.context_frames = {
            'baseline': '{TOKEN}',           # Token alone
            'determiner_the': 'the {TOKEN}',  # Definite article
            'determiner_a': 'a {TOKEN}',      # Indefinite article  
            'possessive_my': 'my {TOKEN}',    # Possessive
            'copula_is': 'is {TOKEN}',        # Copula preceding
            'intensifier_very': 'very {TOKEN}', # Intensifier
            'negation_not': 'not {TOKEN}',    # Negation
            'conjunction_and': 'and {TOKEN}',  # Conjunction
            'modal_will': 'will {TOKEN}',      # Modal verb
            'sentence_start': '<|endoftext|> {TOKEN}'  # Start token
        }
        
        logger.info(f"Using {len(self.context_frames)} context frames")
        
    def should_skip_combination(self, token_info: Dict[str, Any], 
                               context_name: str) -> bool:
        """Determine if a token-context combination should be skipped."""
        token_str = token_info['token_str'].strip()
        token_type = token_info.get('token_type', '')
        
        # Skip punctuation with determiners and certain other contexts
        if token_info.get('is_punctuation', False) or token_type == 'punctuation':
            if context_name in ['determiner_the', 'determiner_a', 'possessive_my', 
                               'intensifier_very']:
                return True
                
        # Skip numbers with determiners and intensifiers
        if 'number' in token_type:
            if context_name in ['determiner_the', 'determiner_a', 'intensifier_very']:
                return True
                
        # Skip if token already starts with space and context adds another
        # (This helps avoid double spaces, though GPT-2 can handle them)
        if token_str.startswith(' ') and context_name == 'baseline':
            # Baseline is fine - just the token as-is
            return False
            
        # Skip subword tokens (starting with ##) with determiners
        if token_str.startswith('##'):
            if context_name in ['determiner_the', 'determiner_a', 'possessive_my']:
                return True
                
        return False
        
    def get_target_position(self, context_frame: str, token_str: str) -> int:
        """Determine the position of the target token in the tokenized sequence."""
        # For most frames, target is at position 1
        # For baseline, it's at position 0
        # For sentence_start with <|endoftext|>, it's at position 1
        
        if context_frame == '{TOKEN}':
            return 0
        else:
            # Count spaces before {TOKEN} to estimate position
            # This is approximate - actual position depends on tokenization
            prefix = context_frame.split('{TOKEN}')[0]
            
            # Special case for sentence start token
            if '<|endoftext|>' in prefix:
                return 1  # <|endoftext|> is a single token
            
            # Count words in prefix (simple approximation)
            words = prefix.strip().split()
            return len(words)
            
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate all test cases."""
        test_cases = []
        skipped = 0
        
        for token_info in self.tokens:
            token_str = token_info['token_str']
            token_idx = token_info['index']
            
            for context_name, context_frame in self.context_frames.items():
                # Check if should skip
                if self.should_skip_combination(token_info, context_name):
                    skipped += 1
                    continue
                    
                # Generate test case
                text = context_frame.format(TOKEN=token_str)
                
                # Clean up any double spaces (though GPT-2 handles them fine)
                text = ' '.join(text.split())
                
                test_case = {
                    'text': text,
                    'token_idx': token_idx,
                    'token_str': token_str,
                    'context_frame': context_name,
                    'target_position': self.get_target_position(context_frame, token_str),
                    'token_type': token_info.get('token_type', 'unknown')
                }
                
                test_cases.append(test_case)
                
        logger.info(f"Generated {len(test_cases)} test cases ({skipped} skipped)")
        return test_cases
        
    def save_test_cases(self, output_path: str = "test_cases_unified.json") -> None:
        """Generate and save test cases."""
        test_cases = self.generate_test_cases()
        
        # Add metadata
        output_data = {
            'metadata': {
                'total_cases': len(test_cases),
                'num_tokens': len(self.tokens),
                'num_contexts': len(self.context_frames),
                'context_frames': self.context_frames
            },
            'test_cases': test_cases
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(test_cases)} test cases to {output_path}")
        
        # Print examples
        logger.info("\nExample test cases:")
        for i in [0, 1, 2, 100, 500, 1000, 5000, -1]:
            if 0 <= i < len(test_cases):
                case = test_cases[i]
                logger.info(f"  {i}: '{case['text']}' "
                           f"(token: '{case['token_str']}', "
                           f"context: {case['context_frame']}, "
                           f"pos: {case['target_position']})")
                           
        # Print distribution
        context_counts = {}
        for case in test_cases:
            ctx = case['context_frame']
            context_counts[ctx] = context_counts.get(ctx, 0) + 1
            
        logger.info("\nTest cases per context:")
        for ctx, count in sorted(context_counts.items()):
            logger.info(f"  {ctx}: {count}")


def main():
    """Generate test cases for unified experiment."""
    generator = UnifiedContextFrameGenerator()
    generator.save_test_cases()
    

if __name__ == "__main__":
    main()