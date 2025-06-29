"""
Analyze Unified Trajectories

Uses existing PathExtractor to analyze trajectory patterns and calculate
divergence metrics for the unified context effects experiment.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging

# Import from existing infrastructure
import sys
sys.path.append('../../')
from concept_fragmentation.clustering.paths import PathExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedTrajectoryAnalyzer:
    """Analyze trajectories from unified clustering experiment."""
    
    def __init__(self, trajectories_path: str):
        """Initialize with trajectory data."""
        # Load trajectories
        with open(trajectories_path, 'r') as f:
            data = json.load(f)
            self.trajectories = data['trajectories']
            self.metadata = data.get('metadata', {})
            
        logger.info(f"Loaded {len(self.trajectories)} trajectories")
        
        # Initialize PathExtractor
        self.path_extractor = PathExtractor()
        
        # Group trajectories by token
        self.token_groups = self._group_by_token()
        
    def _group_by_token(self) -> Dict[int, Dict[str, Any]]:
        """Group trajectories by token index."""
        groups = defaultdict(dict)
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            groups[token_idx][context] = traj_data
            
        return dict(groups)
        
    def calculate_divergence_scores(self) -> Dict[str, Any]:
        """Calculate trajectory divergence scores for all tokens."""
        divergence_results = {
            'per_token': {},
            'by_context': defaultdict(list),
            'summary': {}
        }
        
        for token_idx, contexts in self.token_groups.items():
            if 'baseline' not in contexts:
                logger.warning(f"No baseline for token {token_idx}")
                continue
                
            baseline_path = contexts['baseline']['path']
            token_str = contexts['baseline']['token_str']
            
            token_divergences = {}
            
            # Compare each context to baseline
            for context_name, traj_data in contexts.items():
                if context_name == 'baseline':
                    continue
                    
                context_path = traj_data['path']
                
                # Calculate divergence metrics
                divergence = self._calculate_divergence(baseline_path, context_path)
                token_divergences[context_name] = divergence
                
                # Store for by-context analysis
                divergence_results['by_context'][context_name].append({
                    'token_idx': token_idx,
                    'token_str': token_str,
                    **divergence
                })
                
            # Store per-token results
            divergence_results['per_token'][token_idx] = {
                'token_str': token_str,
                'divergences': token_divergences,
                'max_divergence': max(d['full_divergence'] for d in token_divergences.values()) if token_divergences else 0,
                'mean_divergence': np.mean([d['full_divergence'] for d in token_divergences.values()]) if token_divergences else 0
            }
            
        # Calculate summary statistics
        all_divergences = []
        for token_data in divergence_results['per_token'].values():
            for div in token_data['divergences'].values():
                all_divergences.append(div['full_divergence'])
                
        divergence_results['summary'] = {
            'mean_divergence': np.mean(all_divergences) if all_divergences else 0,
            'std_divergence': np.std(all_divergences) if all_divergences else 0,
            'max_divergence': max(all_divergences) if all_divergences else 0,
            'tokens_affected': sum(1 for t in divergence_results['per_token'].values() 
                                 if t['max_divergence'] > 0),
            'total_comparisons': len(all_divergences)
        }
        
        return divergence_results
        
    def _calculate_divergence(self, path1: List[int], path2: List[int]) -> Dict[str, Any]:
        """Calculate detailed divergence metrics between two paths."""
        # Early layers (0-3)
        early_diff = 0
        early_total = 0
        for i in range(min(4, len(path1), len(path2))):
            if path1[i] != -1 and path2[i] != -1:
                if path1[i] != path2[i]:
                    early_diff += 1
                early_total += 1
                
        # Middle layers (4-7)
        middle_diff = 0
        middle_total = 0
        for i in range(4, min(8, len(path1), len(path2))):
            if path1[i] != -1 and path2[i] != -1:
                if path1[i] != path2[i]:
                    middle_diff += 1
                middle_total += 1
                
        # Late layers (8-11)
        late_diff = 0
        late_total = 0
        for i in range(8, min(len(path1), len(path2))):
            if path1[i] != -1 and path2[i] != -1:
                if path1[i] != path2[i]:
                    late_diff += 1
                late_total += 1
                
        # Full trajectory
        full_diff = early_diff + middle_diff + late_diff
        full_total = early_total + middle_total + late_total
        
        # Find bifurcation point
        bifurcation_layer = -1
        for i in range(min(len(path1), len(path2))):
            if path1[i] != -1 and path2[i] != -1 and path1[i] != path2[i]:
                bifurcation_layer = i
                break
                
        return {
            'early_divergence': early_diff / early_total if early_total > 0 else 0,
            'middle_divergence': middle_diff / middle_total if middle_total > 0 else 0,
            'late_divergence': late_diff / late_total if late_total > 0 else 0,
            'full_divergence': full_diff / full_total if full_total > 0 else 0,
            'bifurcation_layer': bifurcation_layer,
            'divergent_layers': full_diff
        }
        
    def find_archetypal_patterns(self) -> Dict[str, Any]:
        """Use PathExtractor to find archetypal trajectory patterns."""
        # Convert trajectories to format expected by PathExtractor
        paths_by_layer = defaultdict(lambda: defaultdict(list))
        
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            path = traj_data['path']
            
            # Add to each layer
            for layer, cluster in enumerate(path):
                if cluster != -1:  # Skip missing data
                    paths_by_layer[layer][cluster].append(token_idx)
                    
        # Find archetypal paths
        # Note: PathExtractor expects a specific format, we may need to adapt
        logger.info("Finding archetypal trajectory patterns...")
        
        # For now, find common full trajectories
        trajectory_counts = defaultdict(int)
        trajectory_examples = defaultdict(list)
        
        for key, traj_data in self.trajectories.items():
            path_tuple = tuple(traj_data['path'])
            trajectory_counts[path_tuple] += 1
            
            if len(trajectory_examples[path_tuple]) < 5:  # Keep up to 5 examples
                trajectory_examples[path_tuple].append({
                    'token_str': traj_data['token_str'],
                    'context': traj_data['context_frame']
                })
                
        # Sort by frequency
        common_trajectories = sorted(trajectory_counts.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:50]  # Top 50
        
        archetypal_patterns = []
        for path, count in common_trajectories:
            archetypal_patterns.append({
                'path': list(path),
                'count': count,
                'frequency': count / len(self.trajectories),
                'examples': trajectory_examples[path]
            })
            
        return {
            'num_unique_trajectories': len(trajectory_counts),
            'archetypal_patterns': archetypal_patterns
        }
        
    def analyze_context_effects(self) -> Dict[str, Any]:
        """Analyze effects of each context type."""
        context_analysis = {}
        
        divergence_results = self.calculate_divergence_scores()
        
        for context_name, token_divergences in divergence_results['by_context'].items():
            if not token_divergences:
                continue
                
            # Calculate statistics for this context
            full_divs = [d['full_divergence'] for d in token_divergences]
            early_divs = [d['early_divergence'] for d in token_divergences]
            bifurcation_layers = [d['bifurcation_layer'] for d in token_divergences 
                                 if d['bifurcation_layer'] >= 0]
            
            context_analysis[context_name] = {
                'mean_divergence': np.mean(full_divs),
                'std_divergence': np.std(full_divs),
                'max_divergence': max(full_divs),
                'mean_early_divergence': np.mean(early_divs),
                'tokens_affected': sum(1 for d in full_divs if d > 0),
                'strong_effects': sum(1 for d in full_divs if d > 0.5),
                'mean_bifurcation_layer': np.mean(bifurcation_layers) if bifurcation_layers else -1
            }
            
        # Rank contexts by effect strength
        ranked_contexts = sorted(context_analysis.items(), 
                               key=lambda x: x[1]['mean_divergence'], 
                               reverse=True)
        
        return {
            'context_effects': dict(context_analysis),
            'ranked_contexts': ranked_contexts
        }
        
    def identify_stable_tokens(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Identify tokens that are stable across contexts."""
        divergence_results = self.calculate_divergence_scores()
        
        stable_tokens = []
        
        for token_idx, token_data in divergence_results['per_token'].items():
            if token_data['max_divergence'] < threshold:
                stable_tokens.append({
                    'token_idx': token_idx,
                    'token_str': token_data['token_str'],
                    'max_divergence': token_data['max_divergence'],
                    'mean_divergence': token_data['mean_divergence']
                })
                
        # Sort by stability (lowest divergence first)
        stable_tokens.sort(key=lambda x: x['max_divergence'])
        
        return stable_tokens
        
    def save_analysis(self, output_dir: Path) -> None:
        """Save all analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Calculate all analyses
        logger.info("Calculating divergence scores...")
        divergence_results = self.calculate_divergence_scores()
        
        logger.info("Finding archetypal patterns...")
        archetypal_patterns = self.find_archetypal_patterns()
        
        logger.info("Analyzing context effects...")
        context_effects = self.analyze_context_effects()
        
        logger.info("Identifying stable tokens...")
        stable_tokens = self.identify_stable_tokens()
        
        # Save results
        with open(output_dir / "divergence_analysis.json", 'w') as f:
            json.dump(divergence_results, f, indent=2)
            
        with open(output_dir / "archetypal_patterns.json", 'w') as f:
            json.dump(archetypal_patterns, f, indent=2)
            
        with open(output_dir / "context_effects_analysis.json", 'w') as f:
            json.dump(context_effects, f, indent=2)
            
        with open(output_dir / "stable_tokens.json", 'w') as f:
            json.dump(stable_tokens, f, indent=2)
            
        # Create summary report
        summary = {
            'experiment': 'unified_context_effects',
            'total_trajectories': len(self.trajectories),
            'unique_tokens': len(self.token_groups),
            'divergence_summary': divergence_results['summary'],
            'top_5_context_effects': context_effects['ranked_contexts'][:5],
            'num_stable_tokens': len(stable_tokens),
            'num_archetypal_patterns': len(archetypal_patterns['archetypal_patterns'])
        }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_dir}")


def main():
    """Run trajectory analysis."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str, 
                       default='results_unified/unified_trajectories.json',
                       help='Path to trajectories file')
    parser.add_argument('--output', type=str,
                       default='results_unified/analysis/',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    # Run analysis
    analyzer = UnifiedTrajectoryAnalyzer(args.trajectories)
    analyzer.save_analysis(Path(args.output))
    

if __name__ == "__main__":
    main()