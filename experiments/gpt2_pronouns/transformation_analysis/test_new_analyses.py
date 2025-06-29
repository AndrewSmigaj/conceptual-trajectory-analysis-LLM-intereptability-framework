#!/usr/bin/env python
"""
Quick test of the new analyses with reduced parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation_analysis.information_theory_metrics import InformationTheoryMetrics
from transformation_analysis.publication_figures import PublicationFigures


def main():
    """Test both new analyses with minimal configuration."""
    
    print("Testing Information Theory Metrics...")
    try:
        info_analysis = InformationTheoryMetrics(
            output_dir="test_output/information_theory",
            config={
                'k_clusters': 10,
                'n_bootstrap': 10,  # Very small for testing
                'contexts_to_analyze': ['baseline', 'determiner_the'],
                'enable_logging': False
            }
        )
        info_output = info_analysis.run()
        print("✓ Information theory analysis completed")
        print(f"  - Statistics: {info_output.statistics}")
    except Exception as e:
        print(f"✗ Information theory analysis failed: {e}")
        
    print("\nTesting Publication Figures...")
    try:
        fig_analysis = PublicationFigures(
            output_dir="test_output/publication_figures",
            config={
                'k_clusters': 10,
                'dpi': 100,  # Lower for testing
                'formats': ['png'],
                'example_token': 'light',
                'enable_logging': False
            }
        )
        fig_output = fig_analysis.run()
        print("✓ Publication figures completed")
        print(f"  - Generated {len(fig_output.data)} figures")
    except Exception as e:
        print(f"✗ Publication figures failed: {e}")


if __name__ == "__main__":
    main()