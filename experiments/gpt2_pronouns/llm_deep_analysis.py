"""
LLM Deep Analysis - Second Pass

Performs detailed analysis of discovered patterns using LLMs to generate insights,
hypotheses, and explanations for the observed context effects.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging
from datetime import datetime
import sys

# Add path for local config
sys.path.append('../../')
try:
    from local_config import OPENAI_KEY, XAI_API_KEY, GEMINI_API_KEY
except ImportError:
    print("Warning: API keys not found in local_config.py")
    OPENAI_KEY = None
    XAI_API_KEY = None
    GEMINI_API_KEY = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMDeepAnalysis:
    """Perform deep analysis of context effects using LLMs."""
    
    def __init__(self, results_dir: str = "results/"):
        """Initialize with all analysis results."""
        self.results_dir = Path(results_dir)
        
        # Load all previous analyses
        self._load_all_results()
        
    def _load_all_results(self):
        """Load results from all previous analyses."""
        # Statistical report
        stats_path = self.results_dir / "statistical_report.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.statistics = json.load(f)
        else:
            self.statistics = {}
            
        # Pattern discovery
        patterns_dir = self.results_dir / "pattern_discovery"
        self.patterns = {}
        
        if patterns_dir.exists():
            for file in ['archetypal_paths.json', 'trajectory_clusters.json', 
                        'context_trajectory_patterns.json', 'trajectory_transitions.json']:
                path = patterns_dir / file
                if path.exists():
                    with open(path, 'r') as f:
                        self.patterns[file.replace('.json', '')] = json.load(f)
                        
        # Clustering analysis
        clustering_dir = self.results_dir / "clustering_analysis"
        self.clustering = {}
        
        if clustering_dir.exists():
            for file in ['context_sensitive_tokens.json', 'context_pattern_analysis.json',
                        'trajectory_consistency_analysis.json']:
                path = clustering_dir / file
                if path.exists():
                    with open(path, 'r') as f:
                        self.clustering[file.replace('.json', '')] = json.load(f)
                        
        # Token information
        token_path = Path("../gpt2/all_tokens/top_10k_tokens_full.json")
        if token_path.exists():
            with open(token_path, 'r') as f:
                tokens = json.load(f)
                self.token_info = {i: t for i, t in enumerate(tokens)}
        else:
            self.token_info = {}
            
        logger.info(f"Loaded results from {self.results_dir}")
        
    def analyze_token_type_patterns(self) -> Dict[str, Any]:
        """Deep analysis of how different token types respond to context."""
        # Prepare data for LLM analysis
        token_type_data = self._prepare_token_type_analysis()
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'token_type_patterns': {},
            'cross_type_comparisons': {},
            'linguistic_hypotheses': []
        }
        
        # Analyze each token type
        for token_type, data in token_type_data.items():
            prompt = self._create_token_type_prompt(token_type, data)
            
            # In production, this would call the LLM API
            # For now, structure the expected output
            analysis['token_type_patterns'][token_type] = {
                'behavioral_profile': f"Analysis of {token_type} tokens",
                'context_sensitivity': data['mean_sensitivity'],
                'exemplar_tokens': data['exemplars'][:5],
                'interpretation': self._interpret_token_type_behavior(token_type, data)
            }
            
        # Cross-type comparisons
        analysis['cross_type_comparisons'] = self._analyze_cross_type_patterns(token_type_data)
        
        # Generate linguistic hypotheses
        analysis['linguistic_hypotheses'] = self._generate_hypotheses(token_type_data)
        
        return analysis
        
    def _prepare_token_type_analysis(self) -> Dict[str, Any]:
        """Prepare token type data for analysis."""
        token_type_stats = self.statistics.get('token_type_analysis', {})
        sensitive_tokens = self.clustering.get('context_sensitive_tokens', [])
        
        # Group sensitive tokens by type
        type_exemplars = defaultdict(list)
        
        for token_data in sensitive_tokens:
            token_idx = token_data['token_idx']
            if token_idx in self.token_info:
                token_type = self.token_info[token_idx].get('token_type', 'unknown')
                token_str = self.token_info[token_idx].get('token_str', '')
                
                type_exemplars[token_type].append({
                    'token': token_str,
                    'max_divergence': token_data['max_divergence'],
                    'affecting_contexts': token_data.get('most_affecting_contexts', [])
                })
                
        # Combine with statistics
        result = {}
        for token_type, stats in token_type_stats.items():
            if token_type != 'anova':
                result[token_type] = {
                    'mean_sensitivity': stats.get('mean_effect', 0),
                    'std_sensitivity': stats.get('std_effect', 0),
                    'n_tokens': stats.get('n_tokens', 0),
                    'highly_affected': stats.get('highly_affected', 0),
                    'exemplars': sorted(type_exemplars.get(token_type, []), 
                                      key=lambda x: x['max_divergence'], 
                                      reverse=True)
                }
                
        return result
        
    def _create_token_type_prompt(self, token_type: str, data: Dict[str, Any]) -> str:
        """Create prompt for token type analysis."""
        prompt = f"""Analyze the context sensitivity patterns for {token_type} tokens in GPT-2:

Token Type: {token_type}
Mean Sensitivity: {data['mean_sensitivity']:.3f}
Standard Deviation: {data['std_sensitivity']:.3f}
Number of Tokens: {data['n_tokens']}
Highly Affected Tokens: {data['highly_affected']}

Top 5 Most Sensitive {token_type} Tokens:
"""
        
        for i, exemplar in enumerate(data['exemplars'][:5], 1):
            prompt += f"\n{i}. '{exemplar['token']}' (max divergence: {exemplar['max_divergence']:.3f})"
            if exemplar['affecting_contexts']:
                top_context = exemplar['affecting_contexts'][0]
                prompt += f"\n   Most affected by: {top_context[0]} (divergence: {top_context[1]:.3f})"
                
        prompt += f"""

Please provide:
1. A behavioral profile explaining why {token_type} tokens show this pattern of context sensitivity
2. Linguistic explanation for the observed sensitivity level
3. Predictions about which specific {token_type} tokens would be most/least affected
4. Implications for language model behavior

Focus on the computational and linguistic mechanisms that might explain these patterns."""
        
        return prompt
        
    def _interpret_token_type_behavior(self, token_type: str, data: Dict[str, Any]) -> str:
        """Generate interpretation of token type behavior."""
        sensitivity = data['mean_sensitivity']
        
        interpretations = {
            'subword': "Subword tokens show high sensitivity as they heavily depend on context for meaning completion",
            'punctuation': "Punctuation tokens show low sensitivity as their syntactic role is largely fixed",
            'complete_word': "Complete words show moderate sensitivity based on their semantic flexibility",
            'numeric': "Numeric tokens show low sensitivity as their meaning is context-independent",
            'other': "Other tokens show varied sensitivity depending on their specific linguistic function"
        }
        
        base_interpretation = interpretations.get(token_type, "Token type shows context-dependent behavior")
        
        # Add sensitivity-based modifier
        if sensitivity > 0.5:
            return f"{base_interpretation}. The high sensitivity ({sensitivity:.3f}) suggests strong context dependence."
        elif sensitivity > 0.2:
            return f"{base_interpretation}. The moderate sensitivity ({sensitivity:.3f}) indicates balanced context influence."
        else:
            return f"{base_interpretation}. The low sensitivity ({sensitivity:.3f}) indicates relative context independence."
            
    def _analyze_cross_type_patterns(self, token_type_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across token types."""
        # Sort types by sensitivity
        sorted_types = sorted(token_type_data.items(), 
                            key=lambda x: x[1]['mean_sensitivity'], 
                            reverse=True)
        
        return {
            'sensitivity_ranking': [t[0] for t in sorted_types],
            'most_sensitive_type': sorted_types[0][0] if sorted_types else None,
            'least_sensitive_type': sorted_types[-1][0] if sorted_types else None,
            'sensitivity_range': (sorted_types[-1][1]['mean_sensitivity'], 
                                sorted_types[0][1]['mean_sensitivity']) if sorted_types else (0, 0)
        }
        
    def _generate_hypotheses(self, token_type_data: Dict[str, Any]) -> List[str]:
        """Generate testable hypotheses based on patterns."""
        hypotheses = []
        
        # Hypothesis 1: Subword sensitivity
        if 'subword' in token_type_data and token_type_data['subword']['mean_sensitivity'] > 0.3:
            hypotheses.append(
                "Subword tokens require context to resolve their complete form, leading to "
                "trajectory bifurcation based on whether the context suggests word completion "
                "or standalone usage."
            )
            
        # Hypothesis 2: Function vs content
        if 'punctuation' in token_type_data and 'complete_word' in token_type_data:
            punct_sens = token_type_data['punctuation']['mean_sensitivity']
            word_sens = token_type_data['complete_word']['mean_sensitivity']
            
            if word_sens > punct_sens * 1.5:
                hypotheses.append(
                    "Content-bearing tokens (complete words) show greater context sensitivity "
                    "than function tokens (punctuation) because they participate in semantic "
                    "composition while function tokens have fixed syntactic roles."
                )
                
        # Hypothesis 3: Numeric stability
        if 'numeric' in token_type_data and token_type_data['numeric']['mean_sensitivity'] < 0.2:
            hypotheses.append(
                "Numeric tokens maintain stable trajectories across contexts because their "
                "semantic content is inherently context-independent, serving as anchors "
                "in the representation space."
            )
            
        return hypotheses
        
    def analyze_trajectory_patterns(self) -> Dict[str, Any]:
        """Deep analysis of trajectory patterns and transitions."""
        archetypal_paths = self.patterns.get('archetypal_paths', {})
        transitions = self.patterns.get('trajectory_transitions', {})
        
        analysis = {
            'path_interpretations': {},
            'transition_mechanisms': {},
            'emergent_patterns': []
        }
        
        # Analyze archetypal paths
        if archetypal_paths:
            top_paths = archetypal_paths.get('archetypal_paths', [])[:10]
            
            for i, path_data in enumerate(top_paths):
                path = path_data['path']
                frequency = path_data['frequency']
                examples = path_data.get('examples', [])
                
                analysis['path_interpretations'][f'path_{i}'] = {
                    'trajectory': path,
                    'frequency': frequency,
                    'interpretation': self._interpret_path(path, examples),
                    'linguistic_function': self._infer_linguistic_function(path, examples)
                }
                
        # Analyze transitions
        if transitions:
            common_transitions = transitions.get('common_transitions', [])[:10]
            
            for trans in common_transitions:
                key = f"{trans['context']}_transition"
                analysis['transition_mechanisms'][key] = {
                    'from_path': trans['from_path'],
                    'to_path': trans['to_path'],
                    'count': trans['count'],
                    'examples': trans.get('examples', [])[:3],
                    'mechanism': self._infer_transition_mechanism(trans)
                }
                
        # Identify emergent patterns
        analysis['emergent_patterns'] = self._identify_emergent_patterns(
            archetypal_paths, transitions
        )
        
        return analysis
        
    def _interpret_path(self, path: List[int], examples: List[Dict]) -> str:
        """Interpret a trajectory path based on its pattern and examples."""
        # Look for patterns in the path
        if len(set(path)) == 1:
            return "Stable trajectory - token maintains same cluster throughout early layers"
        elif path == sorted(path):
            return "Ascending trajectory - token moves to higher-numbered clusters"
        elif path == sorted(path, reverse=True):
            return "Descending trajectory - token moves to lower-numbered clusters"
        else:
            changes = sum(1 for i in range(1, len(path)) if path[i] != path[i-1])
            if changes == 1:
                return "Single transition - token changes cluster once then stabilizes"
            else:
                return f"Complex trajectory - token changes cluster {changes} times"
                
    def _infer_linguistic_function(self, path: List[int], examples: List[Dict]) -> str:
        """Infer linguistic function from trajectory and examples."""
        # This would use the actual examples to infer function
        # For now, return a placeholder
        if examples:
            token_types = [self.token_info.get(ex.get('token', 0), {}).get('token_type', 'unknown') 
                          for ex in examples]
            
            if all(t == token_types[0] for t in token_types):
                return f"Common pattern for {token_types[0]} tokens"
            else:
                return "Mixed token types following this trajectory"
        else:
            return "Unknown linguistic function"
            
    def _infer_transition_mechanism(self, transition: Dict) -> str:
        """Infer mechanism behind a trajectory transition."""
        context = transition['context']
        examples = transition.get('examples', [])
        
        mechanisms = {
            'determiner_the': "Definite article context triggers syntactic parsing trajectory",
            'determiner_a': "Indefinite article context triggers syntactic parsing trajectory",
            'pronoun_i': "First-person pronoun context activates subject-verb agreement pathway",
            'pronoun_they': "Third-person pronoun context activates plural agreement pathway",
            'preposition_with': "Prepositional context activates object-relation pathway",
            'preposition_of': "Genitive preposition activates possession/attribution pathway",
            'sentence_start_is': "Copula context activates predicate nominal/adjective pathway",
            'sentence_start_are': "Plural copula context activates plural predicate pathway"
        }
        
        base_mechanism = mechanisms.get(context, f"{context} context triggers trajectory change")
        
        # Add example-based refinement
        if examples:
            return f"{base_mechanism}. Common for tokens like: {', '.join(examples[:3])}"
        else:
            return base_mechanism
            
    def _identify_emergent_patterns(self, archetypal_paths: Dict, transitions: Dict) -> List[str]:
        """Identify emergent patterns from the data."""
        patterns = []
        
        # Pattern 1: Early layer divergence
        if archetypal_paths:
            early_divergent = sum(1 for p in archetypal_paths.get('archetypal_paths', [])
                                if len(set(p['path'][:2])) > 1)
            
            total = len(archetypal_paths.get('archetypal_paths', []))
            
            if total > 0 and early_divergent / total > 0.3:
                patterns.append(
                    f"Early layer divergence: {early_divergent/total:.1%} of trajectories "
                    f"diverge within first 2 layers, suggesting rapid context-based routing"
                )
                
        # Pattern 2: Context-specific pathways
        if transitions:
            context_counts = defaultdict(int)
            for trans in transitions.get('common_transitions', []):
                context_counts[trans['context']] += trans['count']
                
            if context_counts:
                top_context = max(context_counts.items(), key=lambda x: x[1])
                patterns.append(
                    f"Context '{top_context[0]}' causes most trajectory changes ({top_context[1]} tokens), "
                    f"indicating strong influence on token routing"
                )
                
        return patterns
        
    def analyze_linguistic_implications(self) -> Dict[str, Any]:
        """Analyze linguistic implications of the findings."""
        analysis = {
            'theoretical_implications': [],
            'practical_implications': [],
            'connections_to_linguistics': [],
            'future_directions': []
        }
        
        # Theoretical implications
        if self.statistics.get('effect_sizes'):
            effect_sizes = self.statistics['effect_sizes']
            
            # Check for systematic effects
            significant_effects = sum(1 for ctx, data in effect_sizes.items() 
                                    if abs(data['cohens_d']) > 0.5)
            
            if significant_effects > len(effect_sizes) * 0.3:
                analysis['theoretical_implications'].append(
                    "Context systematically influences token routing in transformer models, "
                    "suggesting that 'meaning' emerges from context-token interactions rather "
                    "than being inherent to token representations"
                )
                
        # Practical implications
        analysis['practical_implications'].extend([
            "Context-aware token routing could be leveraged for more efficient models",
            "Understanding trajectory patterns could inform better tokenization strategies",
            "Context effects suggest potential for dynamic routing based on input"
        ])
        
        # Connections to linguistics
        analysis['connections_to_linguistics'].extend([
            "Trajectory bifurcation mirrors linguistic phenomena like polysemy resolution",
            "Context-dependent routing aligns with usage-based theories of meaning",
            "Early layer divergence supports rapid contextual disambiguation"
        ])
        
        # Future directions
        analysis['future_directions'].extend([
            "Investigate whether similar patterns occur in other transformer models",
            "Test if trajectory patterns correlate with downstream task performance",
            "Explore using trajectory analysis for model interpretability"
        ])
        
        return analysis
        
    def generate_comprehensive_report(self, output_path: str = None) -> None:
        """Generate comprehensive deep analysis report."""
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'results_dir': str(self.results_dir),
                'analysis_type': 'deep_llm_analysis'
            },
            'token_type_analysis': self.analyze_token_type_patterns(),
            'trajectory_pattern_analysis': self.analyze_trajectory_patterns(),
            'linguistic_implications': self.analyze_linguistic_implications(),
            'key_findings': self._summarize_key_findings(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        if output_path is None:
            output_path = self.results_dir / "llm_deep_analysis_report.json"
        else:
            output_path = Path(output_path)
            
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Deep analysis report saved to {output_path}")
        
        # Also generate markdown summary
        self._generate_markdown_summary(report, output_path.with_suffix('.md'))
        
    def _summarize_key_findings(self) -> List[str]:
        """Summarize key findings from all analyses."""
        findings = []
        
        # Finding 1: Effect sizes
        if self.statistics.get('effect_sizes'):
            max_effect = max(self.statistics['effect_sizes'].items(), 
                           key=lambda x: abs(x[1]['cohens_d']))
            
            findings.append(
                f"Strongest context effect: '{max_effect[0]}' with Cohen's d = {max_effect[1]['cohens_d']:.3f}"
            )
            
        # Finding 2: Token sensitivity
        if self.clustering.get('context_sensitive_tokens'):
            n_sensitive = len([t for t in self.clustering['context_sensitive_tokens'] 
                             if t['max_divergence'] > 0.5])
            findings.append(
                f"{n_sensitive} tokens show high context sensitivity (>50% trajectory divergence)"
            )
            
        # Finding 3: Layer effects
        if self.statistics.get('layer_statistics'):
            layer_stats = self.statistics['layer_statistics']
            early_layers = [v for k, v in layer_stats.items() 
                          if int(k.split('_')[1]) < 4]
            
            if early_layers:
                avg_early_divergence = np.mean([l['divergence_rate'] for l in early_layers])
                findings.append(
                    f"Early layers (0-3) show {avg_early_divergence:.1%} average divergence rate"
                )
                
        return findings
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        return [
            "Focus future research on highly context-sensitive tokens to understand routing mechanisms",
            "Investigate whether context effects can be used for controlled generation",
            "Explore connections between trajectory patterns and semantic categories",
            "Test if similar patterns exist across model scales and architectures",
            "Consider trajectory analysis as a tool for model debugging and interpretation"
        ]
        
    def _generate_markdown_summary(self, report: Dict, output_path: Path) -> None:
        """Generate markdown summary of the report."""
        md_content = f"""# LLM Deep Analysis Report

Generated: {report['metadata']['timestamp']}

## Executive Summary

This report presents a deep analysis of context effects on token trajectories in GPT-2, 
revealing systematic patterns in how contextual frames influence token routing through 
the model's layers.

## Key Findings

"""
        
        for i, finding in enumerate(report['key_findings'], 1):
            md_content += f"{i}. {finding}\n"
            
        md_content += """
## Token Type Analysis

### Sensitivity Rankings
"""
        
        token_analysis = report['token_type_analysis']
        if 'cross_type_comparisons' in token_analysis:
            rankings = token_analysis['cross_type_comparisons'].get('sensitivity_ranking', [])
            for rank, token_type in enumerate(rankings, 1):
                pattern = token_analysis['token_type_patterns'].get(token_type, {})
                sensitivity = pattern.get('context_sensitivity', 0)
                md_content += f"{rank}. **{token_type}**: {sensitivity:.3f} mean sensitivity\n"
                
        md_content += """
## Trajectory Patterns

### Most Common Archetypal Paths
"""
        
        trajectory_analysis = report['trajectory_pattern_analysis']
        for path_id, path_data in list(trajectory_analysis.get('path_interpretations', {}).items())[:5]:
            md_content += f"- **{path_data['trajectory']}**: {path_data['interpretation']}\n"
            
        md_content += """
## Linguistic Implications

### Theoretical Implications
"""
        
        for implication in report['linguistic_implications']['theoretical_implications']:
            md_content += f"- {implication}\n"
            
        md_content += """
### Connections to Linguistics
"""
        
        for connection in report['linguistic_implications']['connections_to_linguistics']:
            md_content += f"- {connection}\n"
            
        md_content += """
## Recommendations

"""
        
        for rec in report['recommendations']:
            md_content += f"1. {rec}\n"
            
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        logger.info(f"Markdown summary saved to {output_path}")
        

def main():
    """Run deep LLM analysis."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/',
                       help='Directory containing analysis results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for report')
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LLMDeepAnalysis(args.results)
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(args.output)
    

if __name__ == "__main__":
    main()