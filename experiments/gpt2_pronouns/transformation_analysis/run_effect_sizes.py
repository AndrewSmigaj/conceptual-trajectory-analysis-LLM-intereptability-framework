#!/usr/bin/env python
"""Run just the effect size analysis."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformation_analysis.effect_size_calculator import EffectSizeCalculator

# Run effect size analysis
print("Running effect size calculator...")
analysis = EffectSizeCalculator(
    output_dir="results_transformation/effect_sizes",
    config={
        'k_clusters': 10,
        'effect_size_types': ['cohens_d', 'cliffs_delta'],
        'comparisons': {
            'contexts': ['baseline', 'determiner_the', 'copula_is'],
            'stratify_by': ['frequency', 'type']
        }
    }
)

output = analysis.run()
print("Effect size analysis complete!")
print(f"Results saved to: results_transformation/effect_sizes/")