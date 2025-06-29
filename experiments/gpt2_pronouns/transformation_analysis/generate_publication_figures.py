#!/usr/bin/env python
"""
Generate publication-quality figures for the paper.
Standalone script with optimized settings.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation_analysis.publication_figures import PublicationFigures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate publication figures."""
    
    logger.info("="*60)
    logger.info("Generating Publication-Quality Figures")
    logger.info("="*60)
    
    # Configuration for high-quality figures
    config = {
        'k_clusters': 10,
        'dpi': 300,  # High resolution for publication
        'formats': ['png', 'pdf'],  # Both formats for flexibility
        'example_token': 'light',  # Polysemous word for showcase
        'style': 'nature',  # Nature/Science formatting
        'contexts_to_analyze': [
            'baseline', 
            'determiner_the', 
            'determiner_a',
            'copula_is', 
            'modal_will', 
            'sentence_start'
        ]
    }
    
    try:
        # Create analysis instance
        analysis = PublicationFigures(
            output_dir="results_paper/publication_figures",
            config=config
        )
        
        # Run figure generation
        logger.info("Starting figure generation...")
        output = analysis.run()
        
        logger.info("✓ Publication figures generated successfully")
        
        # Report what was created
        if hasattr(output, 'data') and output.data:
            logger.info(f"\nGenerated {len(output.data)} figures:")
            for fig_name, files in output.data.items():
                logger.info(f"\n{fig_name}:")
                for fmt, path in files.items():
                    logger.info(f"  - {fmt}: {path}")
        
        # Report manifest information
        if hasattr(output, 'summary') and 'manifest' in output.summary:
            manifest = output.summary['manifest']
            logger.info("\n" + "="*60)
            logger.info("FIGURE DESCRIPTIONS")
            logger.info("="*60)
            
            for fig_name, info in manifest.items():
                logger.info(f"\n{fig_name}:")
                logger.info(f"  Description: {info['description']}")
                logger.info(f"  Caption: {info['caption']}")
        
    except Exception as e:
        logger.error(f"✗ Figure generation failed: {str(e)}")
        logger.exception(e)
        return 1
    
    # Print next steps
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("1. Review figures in: results_paper/publication_figures/")
    logger.info("2. Select 4-5 best figures for the paper:")
    logger.info("   - trajectory_fan_plot: Shows divergence patterns")
    logger.info("   - token_type_metrics: Compares token types")
    logger.info("   - context_similarity_dendrogram: Context relationships")
    logger.info("   - single_token_showcase: Deep dive example")
    logger.info("   - layer_evolution: Metric changes across layers")
    logger.info("3. Use the provided captions from the manifest")
    logger.info("4. Ensure figures are referenced in the text")
    
    return 0


if __name__ == "__main__":
    start_time = datetime.now()
    exit_code = main()
    duration = datetime.now() - start_time
    logger.info(f"\nTotal runtime: {duration}")
    sys.exit(exit_code)