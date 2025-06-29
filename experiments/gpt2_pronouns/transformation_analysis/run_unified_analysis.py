#!/usr/bin/env python
"""
Run transformation analyses on unified clustering data.

This script runs all completed analyses on the unified context experiment data
to test the hypothesis that context creates systematic transformations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all analysis classes
from transformation_analysis.stratified_transition_analysis import StratifiedTransitionAnalysis
from transformation_analysis.clustering_stability_test import ClusteringStabilityTest
from transformation_analysis.permutation_significance_test import PermutationSignificanceTest
from transformation_analysis.predictive_transformation_model import PredictiveTransformationModel
from transformation_analysis.procrustes_cv_analysis import ProcrustesAnalysis
from transformation_analysis.subspace_alignment_analysis import SubspaceAlignmentAnalysis
from transformation_analysis.linguistic_grouping_analysis import LinguisticGroupingAnalysis
from transformation_analysis.bootstrap_confidence_intervals import BootstrapConfidenceIntervals
from transformation_analysis.effect_size_calculator import EffectSizeCalculator
from transformation_analysis.comprehensive_validation_suite import ComprehensiveValidationSuite

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'unified_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_analysis(analysis_class, name, config=None):
    """Run a single analysis and handle errors."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {name}")
    logger.info(f"{'='*60}")
    
    try:
        # Create output directory for this analysis
        output_dir = f"results_transformation/{name}"
        
        # Initialize and run
        analysis = analysis_class(output_dir=output_dir, config=config)
        output = analysis.run()
        
        logger.info(f"✓ {name} completed successfully")
        logger.info(f"  Output saved to: {output_dir}")
        
        # Log key statistics if available
        if hasattr(output, 'statistics') and output.statistics:
            logger.info(f"  Key statistics:")
            for key, value in output.statistics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    - {key}: {value:.4f}")
                    
        return True, output
        
    except Exception as e:
        logger.error(f"✗ {name} failed with error: {str(e)}")
        logger.exception(e)
        return False, None


def main():
    """Run all analyses on unified data."""
    logger.info("Starting unified transformation analysis pipeline")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Check that unified data exists
    unified_data_path = Path("../results_unified/unified_trajectories_k10.json")
    if not unified_data_path.exists():
        logger.error(f"Unified data not found at {unified_data_path}")
        return
        
    logger.info(f"Using unified data from: {unified_data_path}")
    
    # Load to check basic stats
    with open(unified_data_path, 'r') as f:
        data = json.load(f)
        n_trajectories = len(data.get('trajectories', {}))
        logger.info(f"Loaded {n_trajectories} trajectories")
    
    # Define analyses to run in order
    analyses = [
        # High priority - core analyses
        (StratifiedTransitionAnalysis, "stratified_transition", {
            'k_clusters': 10,
            'contexts_to_analyze': ['baseline', 'determiner_the', 'determiner_a', 
                                  'possessive_my', 'copula_is', 'modal_will',
                                  'function_have', 'function_with'],
            'visualize': True
        }),
        
        (ClusteringStabilityTest, "clustering_stability", {
            'k_clusters': 10,
            'n_iterations': 10,
            'contexts_to_test': ['baseline', 'determiner_the']
        }),
        
        (PermutationSignificanceTest, "permutation_significance", {
            'n_permutations': 1000,
            'contexts_to_test': ['baseline', 'determiner_the', 'copula_is']
        }),
        
        (PredictiveTransformationModel, "predictive_model", {
            'test_size': 0.2,
            'models': ['logistic', 'random_forest'],
            'contexts_to_predict': ['determiner_the', 'copula_is']
        }),
        
        (ProcrustesAnalysis, "procrustes_analysis", {
            'n_folds': 5,
            'contexts_to_analyze': ['determiner_the', 'determiner_a', 'copula_is']
        }),
        
        # Medium priority - deeper analyses
        (SubspaceAlignmentAnalysis, "subspace_alignment", {
            'n_components': 50,
            'contexts_to_analyze': ['determiner_the', 'copula_is', 'modal_will']
        }),
        
        (LinguisticGroupingAnalysis, "linguistic_grouping", {
            'grouping_properties': ['pos_tag', 'token_type', 'frequency_bin'],
            'min_group_size': 10
        }),
        
        (EffectSizeCalculator, "effect_sizes", {
            'effect_size_types': ['cohens_d', 'cliffs_delta'],
            'comparisons': {
                'contexts': ['baseline', 'determiner_the', 'copula_is'],
                'stratify_by': ['frequency', 'type']
            }
        }),
        
        (ComprehensiveValidationSuite, "validation_suite", {
            'k_values': [5, 10, 15, 20],
            'clustering_methods': ['kmeans', 'hierarchical', 'dbscan'],
            'normalization_methods': ['none', 'standard', 'minmax'],
            'n_samples': 1000  # Subsample for efficiency
        })
    ]
    
    # Run analyses
    results = {}
    successful = 0
    failed = 0
    
    for analysis_class, name, config in analyses:
        success, output = run_analysis(analysis_class, name, config)
        if success:
            successful += 1
            results[name] = output
        else:
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total analyses: {len(analyses)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(analyses)*100:.1f}%")
    
    # Save combined results summary
    summary_path = f"results_transformation/unified_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_analyses': len(analyses),
        'successful': successful,
        'failed': failed,
        'analyses_run': [name for _, name, _ in analyses],
        'data_source': str(unified_data_path)
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("Analysis pipeline complete!")


if __name__ == "__main__":
    main()