"""
Validation Analysis

Performs robustness checks and validation of findings to ensure reliability.
Includes permutation tests, bootstrap analysis, and cross-validation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging
from tqdm import tqdm
import random
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAnalysis:
    """Validate findings through robustness checks."""
    
    def __init__(self, results_dir: str = "results/", seed: int = 42):
        """Initialize with analysis results."""
        self.results_dir = Path(results_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load all necessary data for validation."""
        # Load trajectories
        traj_path = self.results_dir / "visualization_data.json"
        if traj_path.exists():
            with open(traj_path, 'r') as f:
                data = json.load(f)
                self.trajectories = data.get('trajectories', {})
        else:
            self.trajectories = {}
            logger.warning("No trajectory data found")
            
        # Load statistics
        stats_path = self.results_dir / "statistical_report.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.statistics = json.load(f)
        else:
            self.statistics = {}
            
        # Load token info
        token_path = Path("../gpt2/all_tokens/top_10k_tokens_full.json")
        if token_path.exists():
            with open(token_path, 'r') as f:
                tokens = json.load(f)
                self.token_info = {i: t for i, t in enumerate(tokens)}
        else:
            self.token_info = {}
            
        logger.info(f"Loaded {len(self.trajectories)} trajectories for validation")
        
    def permutation_test(self, n_permutations: int = 1000) -> Dict[str, Any]:
        """Perform permutation test to validate context effects."""
        logger.info(f"Running permutation test with {n_permutations} permutations...")
        
        # Calculate observed statistics
        observed_effects = self._calculate_context_effects()
        
        # Permutation test
        permuted_effects = []
        
        for i in tqdm(range(n_permutations), desc="Permutations"):
            # Shuffle context labels
            shuffled_trajectories = self._shuffle_context_labels()
            
            # Calculate effects with shuffled labels
            permuted_effect = self._calculate_context_effects(shuffled_trajectories)
            permuted_effects.append(permuted_effect['mean_effect'])
            
        # Calculate p-values
        observed_mean = observed_effects['mean_effect']
        permuted_effects = np.array(permuted_effects)
        
        p_value = np.mean(permuted_effects >= observed_mean)
        
        results = {
            'observed_mean_effect': observed_mean,
            'permuted_mean': np.mean(permuted_effects),
            'permuted_std': np.std(permuted_effects),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_percentile': stats.percentileofscore(permuted_effects, observed_mean)
        }
        
        # Create visualization
        self._plot_permutation_results(observed_mean, permuted_effects)
        
        return results
        
    def _calculate_context_effects(self, trajectories: Dict = None) -> Dict[str, float]:
        """Calculate context effect statistics."""
        if trajectories is None:
            trajectories = self.trajectories
            
        # Group by token
        token_groups = defaultdict(dict)
        for key, traj_data in trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data['path']
            
        # Calculate effects
        effects = []
        for token_idx, contexts in token_groups.items():
            if 'baseline' in contexts:
                baseline = contexts['baseline']
                for ctx, traj in contexts.items():
                    if ctx != 'baseline':
                        divergence = sum(1 for b, t in zip(baseline[:4], traj[:4])
                                       if b != -1 and t != -1 and b != t) / 4
                        effects.append(divergence)
                        
        return {
            'mean_effect': np.mean(effects) if effects else 0,
            'std_effect': np.std(effects) if effects else 0,
            'n_effects': len(effects)
        }
        
    def _shuffle_context_labels(self) -> Dict:
        """Shuffle context labels while preserving structure."""
        shuffled = {}
        
        # Get all context frames
        contexts = list(set(t['context_frame'] for t in self.trajectories.values()))
        
        # Create shuffled mapping
        shuffled_contexts = contexts.copy()
        random.shuffle(shuffled_contexts)
        context_map = dict(zip(contexts, shuffled_contexts))
        
        # Apply shuffling
        for key, traj_data in self.trajectories.items():
            shuffled_data = traj_data.copy()
            shuffled_data['context_frame'] = context_map[traj_data['context_frame']]
            shuffled[key] = shuffled_data
            
        return shuffled
        
    def _plot_permutation_results(self, observed: float, permuted: np.ndarray):
        """Plot permutation test results."""
        plt.figure(figsize=(10, 6))
        
        plt.hist(permuted, bins=50, alpha=0.7, density=True, 
                label=f'Permuted (n={len(permuted)})')
        plt.axvline(observed, color='red', linestyle='--', linewidth=2,
                   label=f'Observed ({observed:.3f})')
        plt.axvline(np.mean(permuted), color='blue', linestyle='--', linewidth=2,
                   label=f'Permuted mean ({np.mean(permuted):.3f})')
        
        plt.xlabel('Mean Context Effect')
        plt.ylabel('Density')
        plt.title('Permutation Test Results')
        plt.legend()
        
        save_path = self.results_dir / "validation" / "permutation_test.png"
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved permutation test plot to {save_path}")
        
    def bootstrap_confidence_intervals(self, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals for key statistics."""
        logger.info(f"Running bootstrap analysis with {n_bootstrap} samples...")
        
        # Group trajectories by token
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data
            
        # Calculate bootstrap samples
        bootstrap_results = defaultdict(list)
        
        token_indices = list(token_groups.keys())
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap samples"):
            # Resample tokens with replacement
            sampled_tokens = np.random.choice(token_indices, 
                                            size=len(token_indices), 
                                            replace=True)
            
            # Calculate statistics for this sample
            sample_effects = []
            context_effects = defaultdict(list)
            
            for token_idx in sampled_tokens:
                contexts = token_groups[token_idx]
                
                if 'baseline' in contexts:
                    baseline = contexts['baseline']['path']
                    
                    for ctx, traj_data in contexts.items():
                        if ctx != 'baseline':
                            divergence = sum(1 for b, t in zip(baseline[:4], traj_data['path'][:4])
                                           if b != -1 and t != -1 and b != t) / 4
                            sample_effects.append(divergence)
                            context_effects[ctx].append(divergence)
                            
            # Store results
            bootstrap_results['mean_effect'].append(np.mean(sample_effects))
            bootstrap_results['std_effect'].append(np.std(sample_effects))
            
            for ctx, effects in context_effects.items():
                bootstrap_results[f'mean_{ctx}'].append(np.mean(effects))
                
        # Calculate confidence intervals
        confidence_intervals = {}
        
        for metric, values in bootstrap_results.items():
            values = np.array(values)
            confidence_intervals[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_95': (np.percentile(values, 2.5), np.percentile(values, 97.5)),
                'ci_99': (np.percentile(values, 0.5), np.percentile(values, 99.5))
            }
            
        return confidence_intervals
        
    def cross_validation_stability(self, n_folds: int = 5) -> Dict[str, Any]:
        """Test stability of findings using cross-validation."""
        logger.info(f"Running {n_folds}-fold cross-validation...")
        
        # Group trajectories by token
        token_groups = defaultdict(dict)
        for key, traj_data in self.trajectories.items():
            token_idx = traj_data['token_idx']
            context = traj_data['context_frame']
            token_groups[token_idx][context] = traj_data
            
        # Prepare data for cross-validation
        token_indices = np.array(list(token_groups.keys()))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        
        cv_results = defaultdict(list)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(token_indices)):
            train_tokens = token_indices[train_idx]
            test_tokens = token_indices[test_idx]
            
            # Calculate statistics on training set
            train_effects = []
            for token_idx in train_tokens:
                contexts = token_groups[token_idx]
                if 'baseline' in contexts:
                    baseline = contexts['baseline']['path']
                    for ctx, traj_data in contexts.items():
                        if ctx != 'baseline':
                            divergence = sum(1 for b, t in zip(baseline[:4], traj_data['path'][:4])
                                           if b != -1 and t != -1 and b != t) / 4
                            train_effects.append(divergence)
                            
            # Calculate statistics on test set
            test_effects = []
            for token_idx in test_tokens:
                contexts = token_groups[token_idx]
                if 'baseline' in contexts:
                    baseline = contexts['baseline']['path']
                    for ctx, traj_data in contexts.items():
                        if ctx != 'baseline':
                            divergence = sum(1 for b, t in zip(baseline[:4], traj_data['path'][:4])
                                           if b != -1 and t != -1 and b != t) / 4
                            test_effects.append(divergence)
                            
            cv_results['train_mean'].append(np.mean(train_effects))
            cv_results['test_mean'].append(np.mean(test_effects))
            cv_results['train_std'].append(np.std(train_effects))
            cv_results['test_std'].append(np.std(test_effects))
            
        # Calculate stability metrics
        stability_results = {
            'train_mean_avg': np.mean(cv_results['train_mean']),
            'test_mean_avg': np.mean(cv_results['test_mean']),
            'train_mean_std': np.std(cv_results['train_mean']),
            'test_mean_std': np.std(cv_results['test_mean']),
            'generalization_gap': np.mean(cv_results['train_mean']) - np.mean(cv_results['test_mean']),
            'cv_coefficient': np.std(cv_results['test_mean']) / np.mean(cv_results['test_mean'])
        }
        
        # Plot CV results
        self._plot_cv_results(cv_results)
        
        return stability_results
        
    def _plot_cv_results(self, cv_results: Dict):
        """Plot cross-validation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot means
        folds = range(1, len(cv_results['train_mean']) + 1)
        ax1.plot(folds, cv_results['train_mean'], 'o-', label='Train', markersize=8)
        ax1.plot(folds, cv_results['test_mean'], 's-', label='Test', markersize=8)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Mean Effect')
        ax1.set_title('Cross-Validation: Mean Effects')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot standard deviations
        ax2.plot(folds, cv_results['train_std'], 'o-', label='Train', markersize=8)
        ax2.plot(folds, cv_results['test_std'], 's-', label='Test', markersize=8)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Std. Dev.')
        ax2.set_title('Cross-Validation: Standard Deviations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.results_dir / "validation" / "cross_validation.png"
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cross-validation plot to {save_path}")
        
    def sensitivity_analysis(self) -> Dict[str, Any]:
        """Analyze sensitivity to analysis parameters."""
        logger.info("Running sensitivity analysis...")
        
        sensitivity_results = {
            'layer_depth_sensitivity': self._test_layer_depth_sensitivity(),
            'context_subset_sensitivity': self._test_context_subset_sensitivity(),
            'token_subset_sensitivity': self._test_token_subset_sensitivity()
        }
        
        return sensitivity_results
        
    def _test_layer_depth_sensitivity(self) -> Dict[str, Any]:
        """Test sensitivity to number of layers analyzed."""
        results = {}
        
        for n_layers in [1, 2, 3, 4, 6, 8]:
            effects = []
            
            token_groups = defaultdict(dict)
            for key, traj_data in self.trajectories.items():
                token_idx = traj_data['token_idx']
                context = traj_data['context_frame']
                token_groups[token_idx][context] = traj_data
                
            for token_idx, contexts in token_groups.items():
                if 'baseline' in contexts:
                    baseline = contexts['baseline']['path']
                    for ctx, traj_data in contexts.items():
                        if ctx != 'baseline':
                            divergence = sum(1 for b, t in zip(baseline[:n_layers], 
                                                              traj_data['path'][:n_layers])
                                           if b != -1 and t != -1 and b != t) / n_layers
                            effects.append(divergence)
                            
            results[f'{n_layers}_layers'] = {
                'mean_effect': np.mean(effects),
                'std_effect': np.std(effects)
            }
            
        return results
        
    def _test_context_subset_sensitivity(self) -> Dict[str, Any]:
        """Test sensitivity to which contexts are included."""
        context_types = ['determiner_the', 'determiner_a', 'pronoun_i', 'pronoun_they',
                        'preposition_with', 'preposition_of', 'sentence_start_is', 'sentence_start_are']
        
        results = {}
        
        # Test each context individually
        for context in context_types:
            effects = []
            
            token_groups = defaultdict(dict)
            for key, traj_data in self.trajectories.items():
                token_idx = traj_data['token_idx']
                ctx = traj_data['context_frame']
                token_groups[token_idx][ctx] = traj_data
                
            for token_idx, contexts in token_groups.items():
                if 'baseline' in contexts and context in contexts:
                    baseline = contexts['baseline']['path']
                    traj = contexts[context]['path']
                    
                    divergence = sum(1 for b, t in zip(baseline[:4], traj[:4])
                                   if b != -1 and t != -1 and b != t) / 4
                    effects.append(divergence)
                    
            if effects:
                results[context] = {
                    'mean_effect': np.mean(effects),
                    'n_tokens': len(effects)
                }
                
        return results
        
    def _test_token_subset_sensitivity(self) -> Dict[str, Any]:
        """Test sensitivity to token subset selection."""
        results = {}
        
        # Test different token type subsets
        token_types = defaultdict(list)
        
        for token_idx, info in self.token_info.items():
            token_types[info.get('token_type', 'unknown')].append(token_idx)
            
        for token_type, token_indices in token_types.items():
            if len(token_indices) < 10:  # Skip rare types
                continue
                
            effects = []
            
            token_groups = defaultdict(dict)
            for key, traj_data in self.trajectories.items():
                if traj_data['token_idx'] in token_indices:
                    token_idx = traj_data['token_idx']
                    context = traj_data['context_frame']
                    token_groups[token_idx][context] = traj_data
                    
            for token_idx, contexts in token_groups.items():
                if 'baseline' in contexts:
                    baseline = contexts['baseline']['path']
                    for ctx, traj_data in contexts.items():
                        if ctx != 'baseline':
                            divergence = sum(1 for b, t in zip(baseline[:4], traj_data['path'][:4])
                                           if b != -1 and t != -1 and b != t) / 4
                            effects.append(divergence)
                            
            if effects:
                results[token_type] = {
                    'mean_effect': np.mean(effects),
                    'n_effects': len(effects)
                }
                
        return results
        
    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        validation_dir = self.results_dir / "validation"
        validation_dir.mkdir(exist_ok=True)
        
        # Run all validation tests
        report = {
            'metadata': {
                'seed': self.seed,
                'n_trajectories': len(self.trajectories)
            },
            'permutation_test': self.permutation_test(),
            'bootstrap_intervals': self.bootstrap_confidence_intervals(),
            'cross_validation': self.cross_validation_stability(),
            'sensitivity_analysis': self.sensitivity_analysis()
        }
        
        # Save report
        with open(validation_dir / "validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate summary
        self._generate_validation_summary(report, validation_dir)
        
        logger.info(f"Validation report saved to {validation_dir}")
        
    def _generate_validation_summary(self, report: Dict, output_dir: Path):
        """Generate human-readable validation summary."""
        summary = f"""# Validation Analysis Summary

## Permutation Test
- **Observed mean effect**: {report['permutation_test']['observed_mean_effect']:.4f}
- **Permuted mean**: {report['permutation_test']['permuted_mean']:.4f}
- **P-value**: {report['permutation_test']['p_value']:.4f}
- **Significant**: {'Yes' if report['permutation_test']['significant'] else 'No'}
- **Effect percentile**: {report['permutation_test']['effect_size_percentile']:.1f}%

## Bootstrap Confidence Intervals
"""
        
        for metric, data in report['bootstrap_intervals'].items():
            if metric == 'mean_effect':
                summary += f"\n### Overall Mean Effect\n"
                summary += f"- Mean: {data['mean']:.4f}\n"
                summary += f"- 95% CI: [{data['ci_95'][0]:.4f}, {data['ci_95'][1]:.4f}]\n"
                summary += f"- 99% CI: [{data['ci_99'][0]:.4f}, {data['ci_99'][1]:.4f}]\n"
                
        summary += f"""
## Cross-Validation Stability
- **Train mean**: {report['cross_validation']['train_mean_avg']:.4f} ± {report['cross_validation']['train_mean_std']:.4f}
- **Test mean**: {report['cross_validation']['test_mean_avg']:.4f} ± {report['cross_validation']['test_mean_std']:.4f}
- **Generalization gap**: {report['cross_validation']['generalization_gap']:.4f}
- **CV coefficient**: {report['cross_validation']['cv_coefficient']:.4f}

## Sensitivity Analysis

### Layer Depth Sensitivity
"""
        
        layer_sens = report['sensitivity_analysis']['layer_depth_sensitivity']
        for layers, data in sorted(layer_sens.items()):
            summary += f"- {layers}: {data['mean_effect']:.4f}\n"
            
        summary += """
## Conclusions

"""
        
        # Add automated conclusions
        if report['permutation_test']['significant']:
            summary += "✓ Context effects are statistically significant (permutation test)\n"
        else:
            summary += "✗ Context effects are NOT statistically significant\n"
            
        if abs(report['cross_validation']['generalization_gap']) < 0.01:
            summary += "✓ Findings are stable across data subsets (cross-validation)\n"
        else:
            summary += "⚠ Some instability detected across data subsets\n"
            
        if report['cross_validation']['cv_coefficient'] < 0.1:
            summary += "✓ Low variability in cross-validation results\n"
        else:
            summary += "⚠ High variability in cross-validation results\n"
            
        with open(output_dir / "validation_summary.md", 'w') as f:
            f.write(summary)
            

def main():
    """Run validation analysis."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/',
                       help='Directory containing analysis results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Create validator
    validator = ValidationAnalysis(args.results, args.seed)
    
    # Generate validation report
    validator.generate_validation_report()
    

if __name__ == "__main__":
    main()