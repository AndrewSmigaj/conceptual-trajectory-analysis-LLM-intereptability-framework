"""
LLM Full Vocabulary Analysis

Uses large language models to analyze trajectory patterns across the full 10k vocabulary.
Identifies linguistic patterns, generates hypotheses, and discovers unexpected groupings.
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


class LLMFullVocabularyAnalysis:
    """Analyze trajectory patterns using LLMs for pattern discovery."""
    
    def __init__(self, trajectories_path: str, statistics_path: str = None, 
                 token_info_path: str = None):
        """Initialize with trajectory and analysis data."""
        # Load trajectories
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
            
        if 'trajectories' in data:
            self.trajectories = data['trajectories']
        else:
            self.trajectories = data
            
        # Load statistical analysis if available
        self.statistics = {}
        if statistics_path and Path(statistics_path).exists():
            with open(statistics_path, 'r') as f:
                self.statistics = json.load(f)
                
        # Load token information
        self.token_info = {}
        if token_info_path:
            token_path = Path(token_info_path)
            if not token_path.is_absolute():
                token_path = Path(trajectories_path).parent.parent / "gpt2/all_tokens/top_10k_tokens_full.json"
                
            if token_path.exists():
                with open(token_path, 'r') as f:
                    tokens = json.load(f)
                    self.token_info = {i: t for i, t in enumerate(tokens)}
                    
        logger.info(f"Loaded {len(self.trajectories)} trajectories for analysis")
        
    def prepare_analysis_batches(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Prepare trajectory data in batches for LLM analysis."""
        # Group trajectories by token
        token_trajectories = defaultdict(dict)
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data
            
        # Create batches of tokens with their trajectory data
        batches = []
        token_items = list(token_trajectories.items())
        
        for i in range(0, len(token_items), batch_size):
            batch_tokens = token_items[i:i+batch_size]
            
            batch_data = []
            for token_idx, contexts in batch_tokens:
                token_str = self.token_info.get(token_idx, {}).get('token_str', f'token_{token_idx}')
                token_type = self.token_info.get(token_idx, {}).get('token_type', 'unknown')
                
                # Get trajectory changes
                baseline = contexts.get('baseline', {}).get('path', [])
                context_effects = {}
                
                for context_name, traj_data in contexts.items():
                    if context_name != 'baseline':
                        trajectory = traj_data.get('path', [])
                        divergence = self._calculate_divergence(baseline, trajectory)
                        context_effects[context_name] = {
                            'divergence': divergence,
                            'bifurcation_layer': self._find_bifurcation(baseline, trajectory)
                        }
                        
                batch_data.append({
                    'token': token_str,
                    'token_type': token_type,
                    'baseline_trajectory': baseline[:4],  # First 4 layers
                    'context_effects': context_effects
                })
                
            batches.append({
                'batch_id': i // batch_size,
                'tokens': batch_data
            })
            
        return batches
        
    def _calculate_divergence(self, traj1: List[int], traj2: List[int]) -> float:
        """Calculate trajectory divergence."""
        if not traj1 or not traj2:
            return 0
            
        divergence = 0
        valid = 0
        
        for i in range(min(4, len(traj1), len(traj2))):  # Focus on first 4 layers
            if traj1[i] != -1 and traj2[i] != -1:
                if traj1[i] != traj2[i]:
                    divergence += 1
                valid += 1
                
        return divergence / valid if valid > 0 else 0
        
    def _find_bifurcation(self, traj1: List[int], traj2: List[int]) -> int:
        """Find first layer where trajectories diverge."""
        for i in range(min(len(traj1), len(traj2))):
            if traj1[i] != -1 and traj2[i] != -1 and traj1[i] != traj2[i]:
                return i
        return -1
        
    def analyze_with_llm(self, batch: Dict[str, Any], model: str = "gpt-4") -> Dict[str, Any]:
        """Analyze a batch of tokens with LLM."""
        # Prepare prompt
        prompt = self._create_analysis_prompt(batch)
        
        # Call LLM (placeholder - would use actual API)
        # In production, this would use OpenAI API or similar
        logger.info(f"Analyzing batch {batch['batch_id']} with {len(batch['tokens'])} tokens")
        
        # For now, return structured placeholder
        return {
            'batch_id': batch['batch_id'],
            'patterns': [],
            'hypotheses': [],
            'unexpected_findings': []
        }
        
    def _create_analysis_prompt(self, batch: Dict[str, Any]) -> str:
        """Create structured prompt for LLM analysis."""
        prompt = """Analyze the following token trajectory data from GPT-2:

Each token has a baseline trajectory (when appearing alone) and shows different trajectories when preceded by various context words. The trajectory is the sequence of cluster assignments through the first 4 layers of the network.

Token Data:
"""
        
        for token_data in batch['tokens'][:20]:  # Limit to prevent prompt overflow
            prompt += f"\nToken: '{token_data['token']}' (type: {token_data['token_type']})\n"
            prompt += f"  Baseline trajectory: {token_data['baseline_trajectory']}\n"
            
            if token_data['context_effects']:
                prompt += "  Context effects:\n"
                for context, effect in token_data['context_effects'].items():
                    prompt += f"    - {context}: divergence={effect['divergence']:.2f}, "
                    prompt += f"bifurcates at layer {effect['bifurcation_layer']}\n"
            else:
                prompt += "  No context effects observed\n"
                
        prompt += """
Please analyze this data and provide:

1. PATTERNS: What patterns do you observe in how different token types respond to context?
2. LINGUISTIC INSIGHTS: What linguistic principles might explain the observed trajectories?
3. UNEXPECTED FINDINGS: Any surprising or counterintuitive observations?
4. HYPOTHESES: What testable hypotheses emerge from this data?

Focus on:
- Which tokens are most/least affected by context
- Whether certain contexts have systematic effects
- Any relationships between token type and context sensitivity
- Potential mechanisms for context influence on routing

Provide your analysis in a structured format.
"""
        
        return prompt
        
    def synthesize_findings(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize findings across all batches."""
        synthesis = {
            'major_patterns': [],
            'linguistic_insights': [],
            'unexpected_findings': [],
            'testable_hypotheses': [],
            'summary': ""
        }
        
        # Aggregate patterns across batches
        pattern_counts = defaultdict(int)
        
        for analysis in all_analyses:
            for pattern in analysis.get('patterns', []):
                pattern_counts[pattern] += 1
                
        # Include patterns that appear in multiple batches
        synthesis['major_patterns'] = [
            pattern for pattern, count in pattern_counts.items() 
            if count >= len(all_analyses) * 0.2  # Appears in 20% of batches
        ]
        
        return synthesis
        
    def generate_focused_analysis(self, focus: str = "high_sensitivity") -> Dict[str, Any]:
        """Generate analysis focused on specific phenomena."""
        focused_data = []
        
        # Group trajectories by token
        token_trajectories = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_trajectories[token_idx][context] = traj_data
            
        if focus == "high_sensitivity":
            # Find tokens with high context sensitivity
            for token_idx, contexts in token_trajectories.items():
                if 'baseline' not in contexts:
                    continue
                    
                baseline = contexts['baseline']['path']
                max_divergence = 0
                
                for context_name, traj_data in contexts.items():
                    if context_name != 'baseline':
                        divergence = self._calculate_divergence(baseline, traj_data['path'])
                        max_divergence = max(max_divergence, divergence)
                        
                if max_divergence > 0.5:  # High sensitivity threshold
                    token_str = self.token_info.get(token_idx, {}).get('token_str', f'token_{token_idx}')
                    focused_data.append({
                        'token': token_str,
                        'max_divergence': max_divergence,
                        'trajectories': contexts
                    })
                    
        elif focus == "token_type_patterns":
            # Analyze patterns by token type
            type_patterns = defaultdict(list)
            
            for token_idx, contexts in token_trajectories.items():
                if token_idx in self.token_info:
                    token_type = self.token_info[token_idx].get('token_type', 'unknown')
                    
                    # Calculate average context effect
                    if 'baseline' in contexts:
                        baseline = contexts['baseline']['path']
                        divergences = []
                        
                        for context_name, traj_data in contexts.items():
                            if context_name != 'baseline':
                                div = self._calculate_divergence(baseline, traj_data['path'])
                                divergences.append(div)
                                
                        if divergences:
                            avg_divergence = np.mean(divergences)
                            type_patterns[token_type].append(avg_divergence)
                            
            focused_data = {
                token_type: {
                    'mean_sensitivity': np.mean(effects),
                    'std_sensitivity': np.std(effects),
                    'n_tokens': len(effects)
                }
                for token_type, effects in type_patterns.items()
            }
            
        return {
            'focus': focus,
            'data': focused_data,
            'timestamp': datetime.now().isoformat()
        }
        
    def save_analysis(self, output_dir: Path) -> None:
        """Run complete analysis and save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Prepare batches
        logger.info("Preparing analysis batches...")
        batches = self.prepare_analysis_batches(batch_size=100)
        
        # Analyze each batch (placeholder for actual LLM calls)
        logger.info(f"Analyzing {len(batches)} batches...")
        batch_analyses = []
        
        for batch in batches[:5]:  # Limit for testing
            analysis = self.analyze_with_llm(batch)
            batch_analyses.append(analysis)
            
        # Synthesize findings
        logger.info("Synthesizing findings...")
        synthesis = self.synthesize_findings(batch_analyses)
        
        with open(output_dir / "llm_synthesis.json", 'w') as f:
            json.dump(synthesis, f, indent=2)
            
        # Generate focused analyses
        logger.info("Generating focused analyses...")
        
        high_sensitivity = self.generate_focused_analysis("high_sensitivity")
        with open(output_dir / "high_sensitivity_tokens.json", 'w') as f:
            json.dump(high_sensitivity, f, indent=2)
            
        type_patterns = self.generate_focused_analysis("token_type_patterns")
        with open(output_dir / "token_type_patterns.json", 'w') as f:
            json.dump(type_patterns, f, indent=2)
            
        # Create summary report
        summary = {
            'total_tokens_analyzed': len(set(t['token_idx'] for t in self.trajectories.values())),
            'total_trajectories': len(self.trajectories),
            'num_batches': len(batches),
            'analysis_timestamp': datetime.now().isoformat(),
            'high_sensitivity_tokens': len(high_sensitivity.get('data', [])),
            'token_types_analyzed': len(type_patterns.get('data', {}))
        }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"LLM analysis complete. Results saved to {output_dir}")
        

def main():
    """Run LLM analysis on vocabulary trajectories."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str,
                       default='results/visualization_data.json',
                       help='Path to trajectory data')
    parser.add_argument('--statistics', type=str,
                       default='results/statistical_report.json',
                       help='Path to statistical analysis')
    parser.add_argument('--output', type=str,
                       default='results/llm_analysis/',
                       help='Output directory')
    args = parser.parse_args()
    
    # Run analysis
    analyzer = LLMFullVocabularyAnalysis(
        args.trajectories,
        args.statistics
    )
    analyzer.save_analysis(Path(args.output))
    

if __name__ == "__main__":
    main()