"""
Data generation for pronoun context steering experiment.

Generates two-token probing sentences for testing how context influences
pronoun trajectory bifurcation in GPT-2.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_probing_sentences(config: Dict) -> Tuple[List[str], List[Dict]]:
    """
    Generate two-token probing sentences and their labels.
    
    Returns:
        sentences: List of sentences
        labels: List of label dictionaries with pronoun, context, and context_type
    """
    sentences = []
    labels = []
    
    pronouns = config['data']['pronouns']
    contexts = config['data']['contexts']
    
    # Generate all combinations
    for pronoun in pronouns:
        # Baseline: pronoun alone
        sentences.append(pronoun)
        labels.append({
            'pronoun': pronoun,
            'context': '',
            'context_type': 'neutral',
            'sentence': pronoun
        })
        
        # Function word contexts
        for context in contexts['function_words']:
            sentence = f"{context} {pronoun}"
            sentences.append(sentence)
            labels.append({
                'pronoun': pronoun,
                'context': context,
                'context_type': 'function',
                'sentence': sentence
            })
        
        # Content word contexts  
        for context in contexts['content_words']:
            sentence = f"{context} {pronoun}"
            sentences.append(sentence)
            labels.append({
                'pronoun': pronoun,
                'context': context,
                'context_type': 'content',
                'sentence': sentence
            })
    
    return sentences, labels


def save_probing_data(sentences: List[str], labels: List[Dict], output_dir: str = "."):
    """Save probing sentences and labels to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sentences (one per line)
    sentences_file = output_path / "probing_sentences.txt"
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    # Save labels as JSON
    labels_file = output_path / "probing_labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    # Save summary statistics
    stats = {
        'total_sentences': len(sentences),
        'num_pronouns': len(set(label['pronoun'] for label in labels)),
        'num_function_contexts': len(set(label['context'] for label in labels 
                                       if label['context_type'] == 'function')),
        'num_content_contexts': len(set(label['context'] for label in labels 
                                      if label['context_type'] == 'content')),
        'sentences_per_pronoun': len(sentences) // len(set(label['pronoun'] for label in labels))
    }
    
    stats_file = output_path / "probing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generated {len(sentences)} probing sentences")
    print(f"Saved to: {output_path}")
    print(f"Statistics: {stats}")


def generate_analysis_groups(labels: List[Dict]) -> Dict[str, List[int]]:
    """
    Generate groupings for analysis.
    
    Returns dictionary mapping group names to sentence indices.
    """
    groups = {
        'all': list(range(len(labels))),
        'baseline': [],
        'function_context': [],
        'content_context': []
    }
    
    # Group by pronoun
    for pronoun in set(label['pronoun'] for label in labels):
        groups[f'pronoun_{pronoun}'] = []
    
    # Group by context word
    for context in set(label['context'] for label in labels if label['context']):
        groups[f'context_{context}'] = []
    
    # Populate groups
    for idx, label in enumerate(labels):
        if label['context_type'] == 'neutral':
            groups['baseline'].append(idx)
        elif label['context_type'] == 'function':
            groups['function_context'].append(idx)
        elif label['context_type'] == 'content':
            groups['content_context'].append(idx)
        
        groups[f'pronoun_{label["pronoun"]}'].append(idx)
        
        if label['context']:
            groups[f'context_{label["context"]}'].append(idx)
    
    return groups


def main():
    """Generate probing data for the experiment."""
    # Load config
    config = load_config()
    
    # Generate sentences
    sentences, labels = generate_probing_sentences(config)
    
    # Save data
    save_probing_data(sentences, labels, "data")
    
    # Generate analysis groups
    groups = generate_analysis_groups(labels)
    
    # Save groups
    groups_file = Path("data") / "analysis_groups.json"
    with open(groups_file, 'w') as f:
        json.dump(groups, f, indent=2)
    
    print(f"\nAnalysis groups created: {list(groups.keys())}")
    
    # Print example sentences
    print("\nExample sentences:")
    for i in range(min(10, len(sentences))):
        print(f"  {sentences[i]:20} [{labels[i]['context_type']}]")


if __name__ == "__main__":
    main()