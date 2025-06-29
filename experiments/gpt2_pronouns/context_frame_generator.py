"""
Context Frame Generator for Full Vocabulary Analysis

Generates all test cases by applying context frames to the full 10k vocabulary.
Handles special cases like punctuation and subword tokens appropriately.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextFrameGenerator:
    """Generate context frames for all tokens in vocabulary."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load token list
        token_path = Path(self.config['data']['token_list_path'])
        if not token_path.is_absolute():
            token_path = Path(config_path).parent / token_path
            
        with open(token_path, 'r') as f:
            self.tokens = json.load(f)
            
        self.context_frames = self.config['data']['context_frames']
        logger.info(f"Loaded {len(self.tokens)} tokens and {len(self.context_frames)} context frames")
        
    def should_skip_token(self, token_info: Dict[str, Any]) -> bool:
        """Determine if a token should be skipped for certain contexts."""
        token_str = token_info['token_str']
        
        # Skip pure punctuation for some contexts
        if token_info['is_punctuation']:
            # Punctuation doesn't make sense with determiners
            return True
            
        # Skip tokens that already start with space for sentence-initial contexts
        if token_info['has_leading_space'] and 'sentence_start' in self.current_frame:
            return True
            
        return False
        
    def apply_frame(self, frame: str, token_str: str) -> str:
        """Apply a context frame to a token."""
        return frame.replace("[TOKEN]", token_str)
        
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate all test cases with context frames."""
        test_cases = []
        total_skipped = 0
        
        for token_idx, token_info in enumerate(self.tokens):
            token_str = token_info['token_str']
            token_id = token_info['token_id']
            
            for frame_name, frame_template in self.context_frames.items():
                self.current_frame = frame_name
                
                # Check if this combination should be skipped
                if self.should_skip_token(token_info) and frame_name != 'baseline':
                    total_skipped += 1
                    continue
                    
                # Generate the test case
                test_text = self.apply_frame(frame_template, token_str)
                
                test_case = {
                    'test_id': f"{token_idx}_{frame_name}",
                    'token_idx': token_idx,
                    'token_id': token_id,
                    'token_str': token_str,
                    'context_frame': frame_name,
                    'text': test_text,
                    'target_position': self._get_target_position(frame_template, test_text),
                    'token_info': {
                        'has_leading_space': token_info['has_leading_space'],
                        'is_subword': token_info['is_subword'],
                        'is_punctuation': token_info['is_punctuation'],
                        'token_type': token_info['token_type']
                    }
                }
                
                test_cases.append(test_case)
                
        logger.info(f"Generated {len(test_cases)} test cases ({total_skipped} skipped)")
        return test_cases
        
    def _get_target_position(self, frame_template: str, test_text: str) -> int:
        """Determine the position of the target token in the generated text."""
        # Count tokens before [TOKEN] in template
        before_token = frame_template.split("[TOKEN]")[0].strip()
        if not before_token:
            return 0
        else:
            # Simple tokenization - this is approximate
            return len(before_token.split()) 
            
    def save_test_cases(self, output_path: str = "test_cases.json"):
        """Generate and save all test cases."""
        test_cases = self.generate_test_cases()
        
        # Save full dataset
        with open(output_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
            
        # Save summary statistics
        stats = self._compute_statistics(test_cases)
        stats_path = output_path.replace('.json', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Saved {len(test_cases)} test cases to {output_path}")
        return test_cases
        
    def _compute_statistics(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics about the generated test cases."""
        frame_counts = {}
        token_type_counts = {}
        
        for case in test_cases:
            frame = case['context_frame']
            token_type = case['token_info']['token_type']
            
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
            token_type_counts[token_type] = token_type_counts.get(token_type, 0) + 1
            
        return {
            'total_cases': len(test_cases),
            'unique_tokens': len(self.tokens),
            'context_frames': len(self.context_frames),
            'cases_per_frame': frame_counts,
            'token_types': token_type_counts,
            'average_cases_per_token': len(test_cases) / len(self.tokens)
        }


def main():
    """Generate test cases for the full vocabulary analysis."""
    generator = ContextFrameGenerator()
    
    # Generate and save test cases
    output_path = "test_cases_full_vocabulary.json"
    test_cases = generator.save_test_cases(output_path)
    
    # Show sample of generated cases
    print("\nSample test cases:")
    for i, case in enumerate(test_cases[:5]):
        print(f"\n{i+1}. Token: '{case['token_str']}' | Frame: {case['context_frame']}")
        print(f"   Text: '{case['text']}'")
        print(f"   Target position: {case['target_position']}")


if __name__ == "__main__":
    main()