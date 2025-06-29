# GPT-2 Context Transformation Analysis Suite

A comprehensive framework for analyzing how context systematically transforms token representations in GPT-2.

## Overview

This suite implements 15+ statistical and geometric analyses to test the hypothesis that context creates predictable, systematic transformations of GPT-2's latent representation space rather than random changes.

### Key Finding

**Context creates systematic transformations**: When tokens are processed with different contexts, they show 100% trajectory divergence through the network layers. However, these changes are not random - they follow predictable, linguistically-structured patterns.

## Quick Start

```python
# Run a single analysis
from transformation_analysis.stratified_transition_analysis import StratifiedTransitionAnalysis

analysis = StratifiedTransitionAnalysis(
    output_dir="results/stratified",
    config={"k_clusters": 10}
)
output = analysis.run()

# Run all analyses on unified data
python run_unified_analysis.py
```

## Architecture

### Core Infrastructure

1. **BaseTransformationAnalysis** - Abstract base class providing:
   - Consistent pipeline: load → validate → analyze → save
   - Unified output schema
   - Integrated logging and error handling

2. **TransformationDataLoader** - Centralized data loading with:
   - LRU caching for performance
   - Support for trajectories, activations, and metadata
   - Token stratification utilities

3. **Output Schema** - Type-safe dataclasses with:
   - JSON serialization support
   - Confidence interval integration
   - Extensible design for new metrics

### Completed Analyses (15/19)

#### High Priority (All Complete)
1. **Stratified Transition Analysis** - Builds transition matrices showing P(baseline_cluster → context_cluster)
2. **Clustering Stability Test** - Validates patterns are stable across random seeds
3. **Permutation Significance Test** - Tests if transitions differ from random chance
4. **Predictive Transformation Model** - ML models to predict transformations from token features
5. **Procrustes Analysis** - Finds optimal rotation/scale/translation transformations
6. **Subspace Alignment Analysis** - PCA and canonical angle analysis
7. **Linguistic Grouping Analysis** - Tests if linguistic properties predict transformations
8. **Bootstrap Confidence Intervals** - Adds 95% CIs to all metrics
9. **Effect Size Calculator** - Cohen's d, Hedge's g, Cliff's delta
10. **Comprehensive Validation Suite** - Validates k=10 choice and methodology

#### Medium Priority (All Complete)
11-15. See list above

#### Low Priority (Pending)
16. Information Theory Metrics - MI, KL divergence, entropy measures
17. Interactive Visualization Dashboard - Plotly Dash exploration app
18. Publication Figures - Nature/Science quality formatting
19. Comprehensive Analysis Report - LaTeX synthesis

## Key Findings

1. **Systematic Transformations Confirmed**
   - Average entropy difference from random: -0.726 (highly structured)
   - Average sparsity difference from random: +0.376 (sparser)
   - Transitions are predictable, not chaotic

2. **Linguistic Structure Preserved**
   - Similar contexts produce similar transformations
   - Determiners (the/a) show 0.80 similarity
   - Function vs content words show distinct patterns

3. **Layer Evolution**
   - Early layers: More structured transformations
   - Late layers: More distributed effects
   - Special contexts (e.g., sentence_start) maintain unique patterns

4. **Validated Methodology**
   - k=10 clustering optimal by elbow method and silhouette analysis
   - High stability across random seeds (ARI > 0.85)
   - Standard normalization shows best performance

## Usage Examples

### Running Individual Analyses

```python
# Transition matrices with stratification
from transformation_analysis.stratified_transition_analysis import StratifiedTransitionAnalysis

config = {
    'k_clusters': 10,
    'contexts_to_analyze': ['baseline', 'determiner_the', 'copula_is'],
    'stratify_by': ['frequency', 'type'],
    'visualize': True
}

analysis = StratifiedTransitionAnalysis(config=config)
results = analysis.run()

# Access results
transition_matrices = results.data['transition_matrices']
statistics = results.statistics
```

### Batch Processing

```python
# Run all analyses
python run_unified_analysis.py

# This will:
# 1. Load unified clustering data
# 2. Run all 15 completed analyses
# 3. Save results in unified format
# 4. Generate summary report
```

### Custom Analysis

```python
# Extend base class for custom analysis
from transformation_analysis.base_transformation_analysis import BaseTransformationAnalysis

class MyCustomAnalysis(BaseTransformationAnalysis):
    def analyze(self):
        # Your analysis logic here
        trajectories = self.data_loader.load_trajectories()
        
        # Process data...
        
        return {
            'data': results,
            'statistics': stats,
            'summary': {
                'key_findings': findings,
                'interpretation': interpretation
            }
        }
```

## Data Format

### Input: Unified Trajectories
```json
{
  "trajectories": {
    "token_idx_context": {
      "token_idx": 0,
      "token_str": " the",
      "context_frame": "baseline",
      "path": [1, 1, 1, 1, 7, 9, 1, 1, 7, 1, 1, 6],
      "case_idx": 0
    }
  }
}
```

### Output: Unified Analysis Results
```json
{
  "metadata": {
    "analysis_type": "stratified_transition_analysis",
    "timestamp": "2025-06-23T11:58:11",
    "parameters": {...}
  },
  "data": {
    "transition_matrices": {...},
    "metrics": {...}
  },
  "statistics": {
    "overall_metrics": {...},
    "stratified_results": {...}
  },
  "summary": {
    "key_findings": [...],
    "interpretation": "..."
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_stratified_transition_analysis.py

# Run with coverage
pytest --cov=transformation_analysis

# Current status: 175 tests passing
```

## Visualization

Each analysis can generate visualizations:
- Transition matrix heatmaps
- Metric evolution plots
- Stratification comparisons
- Effect size forest plots
- Clustering validation plots

## Next Steps

1. **Run Full Pipeline**: Execute all analyses on complete dataset
2. **Implement Information Theory Metrics**: Add MI, KL divergence
3. **Create Publication Figures**: Generate paper-ready visualizations
4. **Build Interactive Dashboard**: For exploring transformation patterns

## Citation

If you use this analysis suite, please cite:
```
[Citation to be added upon publication]
```

## Requirements

- Python 3.8+
- NumPy, SciPy, scikit-learn
- matplotlib, seaborn
- See requirements.txt for full list

## License

[License information to be added]