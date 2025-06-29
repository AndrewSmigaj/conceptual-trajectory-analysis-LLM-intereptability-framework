#!/usr/bin/env python
"""
Run information theory metrics and publication figure generation.

This script runs the two analyses needed for publication:
1. Information theory metrics for theoretical grounding
2. Publication-quality figures for the paper
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation_analysis.information_theory_metrics import InformationTheoryMetrics
from transformation_analysis.publication_figures import PublicationFigures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run publication analyses."""
    logger.info("Starting publication analyses")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Configuration
    config = {
        'k_clusters': 10,
        'contexts_to_analyze': ['baseline', 'determiner_the', 'determiner_a', 
                               'copula_is', 'modal_will', 'sentence_start']
    }
    
    # 1. Run information theory metrics
    logger.info("\n" + "="*60)
    logger.info("Running Information Theory Metrics")
    logger.info("="*60)
    
    try:
        info_analysis = InformationTheoryMetrics(
            output_dir="results_transformation/information_theory",
            config={**config, 'n_bootstrap': 1000}
        )
        
        info_output = info_analysis.run()
        
        logger.info("✓ Information theory analysis completed successfully")
        
        # Log key metrics
        if hasattr(info_output, 'statistics'):
            for key, value in info_output.statistics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  - {key}: {value:.4f}")
                    
    except Exception as e:
        logger.error(f"✗ Information theory analysis failed: {str(e)}")
        logger.exception(e)
        
    # 2. Run publication figure generation
    logger.info("\n" + "="*60)
    logger.info("Running Publication Figure Generation")
    logger.info("="*60)
    
    try:
        fig_analysis = PublicationFigures(
            output_dir="results_transformation/publication_figures",
            config={
                **config,
                'style': 'nature',
                'dpi': 300,
                'formats': ['png', 'pdf', 'svg'],
                'example_token': 'light'
            }
        )
        
        fig_output = fig_analysis.run()
        
        logger.info("✓ Publication figures generated successfully")
        
        # Log generated figures
        if hasattr(fig_output, 'data'):
            figures = fig_output.data
            logger.info(f"  Generated {len(figures)} figures:")
            for fig_name in figures:
                logger.info(f"    - {fig_name}")
                
    except Exception as e:
        logger.error(f"✗ Publication figure generation failed: {str(e)}")
        logger.exception(e)
        
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info("Both analyses completed. Check results_transformation/ for outputs:")
    logger.info("  - information_theory/: Information-theoretic metrics and analysis")
    logger.info("  - publication_figures/: High-quality figures for paper")
    logger.info("\nNext steps:")
    logger.info("  1. Review information theory results for theoretical insights")
    logger.info("  2. Select figures for paper inclusion")
    logger.info("  3. Write figure captions using the manifest")
    logger.info("  4. Integrate findings into paper narrative")


if __name__ == "__main__":
    main()