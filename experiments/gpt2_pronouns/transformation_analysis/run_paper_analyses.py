#!/usr/bin/env python
"""
Run essential analyses for the paper with optimized configurations.
This script runs only the most important analyses needed for the paper.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the specific analyses we need
from transformation_analysis.stratified_transition_analysis import StratifiedTransitionAnalysis
from transformation_analysis.permutation_significance_test import PermutationSignificanceTest
from transformation_analysis.effect_size_calculator import EffectSizeCalculator
from transformation_analysis.information_theory_metrics import InformationTheoryMetrics
from transformation_analysis.publication_figures import PublicationFigures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_essential_analyses():
    """Run only the essential analyses needed for the paper."""
    
    results_summary = {}
    
    # Common configuration
    base_config = {
        'k_clusters': 10,
        'contexts_to_analyze': ['baseline', 'determiner_the', 'determiner_a', 
                               'copula_is', 'modal_will', 'sentence_start']
    }
    
    logger.info("="*60)
    logger.info("Running Essential Analyses for Paper")
    logger.info("="*60)
    
    # 1. Stratified Transition Analysis (most important)
    logger.info("\n1. Running Stratified Transition Analysis...")
    try:
        analysis = StratifiedTransitionAnalysis(
            output_dir="results_paper/stratified_transition",
            config={**base_config, 'n_random_baselines': 100}
        )
        output = analysis.run()
        results_summary['stratified_transition'] = {
            'status': 'success',
            'mean_entropy': output.statistics.get('mean_entropy'),
            'mean_sparsity': output.statistics.get('mean_sparsity')
        }
        logger.info("✓ Stratified transition analysis completed")
    except Exception as e:
        logger.error(f"✗ Stratified transition failed: {e}")
        results_summary['stratified_transition'] = {'status': 'failed', 'error': str(e)}
    
    # 2. Permutation Significance Test
    logger.info("\n2. Running Permutation Significance Test...")
    try:
        analysis = PermutationSignificanceTest(
            output_dir="results_paper/permutation_significance",
            config={**base_config, 'n_permutations': 100}  # Reduced for speed
        )
        output = analysis.run()
        results_summary['permutation_significance'] = {
            'status': 'success',
            'significant_layers': output.statistics.get('n_significant_layers'),
            'avg_p_value': output.statistics.get('avg_p_value')
        }
        logger.info("✓ Permutation significance test completed")
    except Exception as e:
        logger.error(f"✗ Permutation significance failed: {e}")
        results_summary['permutation_significance'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Effect Size Calculation
    logger.info("\n3. Running Effect Size Calculation...")
    try:
        analysis = EffectSizeCalculator(
            output_dir="results_paper/effect_sizes",
            config=base_config
        )
        output = analysis.run()
        results_summary['effect_sizes'] = {
            'status': 'success',
            'mean_cohens_d': output.statistics.get('mean_cohens_d'),
            'mean_hedges_g': output.statistics.get('mean_hedges_g')
        }
        logger.info("✓ Effect size calculation completed")
    except Exception as e:
        logger.error(f"✗ Effect size calculation failed: {e}")
        results_summary['effect_sizes'] = {'status': 'failed', 'error': str(e)}
    
    # 4. Information Theory Metrics (limited bootstrap for speed)
    logger.info("\n4. Running Information Theory Metrics...")
    try:
        analysis = InformationTheoryMetrics(
            output_dir="results_paper/information_theory",
            config={**base_config, 'n_bootstrap': 50}  # Very limited for speed
        )
        output = analysis.run()
        results_summary['information_theory'] = {
            'status': 'success',
            'avg_mutual_information': output.statistics.get('avg_mutual_information'),
            'avg_kl_divergence': output.statistics.get('avg_kl_divergence')
        }
        logger.info("✓ Information theory metrics completed")
    except Exception as e:
        logger.error(f"✗ Information theory metrics failed: {e}")
        results_summary['information_theory'] = {'status': 'failed', 'error': str(e)}
    
    # 5. Publication Figures
    logger.info("\n5. Generating Publication Figures...")
    try:
        analysis = PublicationFigures(
            output_dir="results_paper/publication_figures",
            config={
                **base_config,
                'dpi': 300,
                'formats': ['png', 'pdf'],
                'example_token': 'light'
            }
        )
        output = analysis.run()
        results_summary['publication_figures'] = {
            'status': 'success',
            'n_figures': output.statistics.get('n_figures')
        }
        logger.info("✓ Publication figures generated")
    except Exception as e:
        logger.error(f"✗ Publication figures failed: {e}")
        results_summary['publication_figures'] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)
    
    successful = sum(1 for r in results_summary.values() if r['status'] == 'success')
    total = len(results_summary)
    
    logger.info(f"Completed {successful}/{total} analyses successfully")
    
    for analysis_name, result in results_summary.items():
        status_icon = "✓" if result['status'] == 'success' else "✗"
        logger.info(f"{status_icon} {analysis_name}: {result['status']}")
        if result['status'] == 'success':
            # Log key metrics
            for key, value in result.items():
                if key not in ['status', 'error'] and value is not None:
                    logger.info(f"  - {key}: {value}")
    
    logger.info("\nResults saved to: results_paper/")
    logger.info("\nNext steps:")
    logger.info("1. Review results in results_paper/*/[analysis]_results.json")
    logger.info("2. Select figures from results_paper/publication_figures/")
    logger.info("3. Extract key statistics for the paper")
    
    return results_summary


if __name__ == "__main__":
    start_time = datetime.now()
    results = run_essential_analyses()
    duration = datetime.now() - start_time
    logger.info(f"\nTotal runtime: {duration}")