# Key Findings: Apple Quality Routing with CTA

## Executive Summary

The apple quality routing experiment demonstrates CTA's effectiveness in revealing how neural networks process hierarchical decisions with economic consequences. Our analysis of a 92.8% accurate routing model uncovered rapid conceptual convergence (10→2 clusters), variety-specific processing patterns, and quantifiable economic impacts ($186.41 loss from misrouting).

## Main Results

### 1. Conceptual Convergence Pattern
- **Layer 0**: 10 semantically meaningful clusters ("Premium-Leaning", "Standard Dominant", etc.)
- **Layers 1-3**: Rapid convergence to 2 clusters (90.6% majority flow, 9.4% minority paths)
- **Interpretation**: Network learns variety-specific calibrations early, then applies general routing rules

### 2. Economic Impact Quantification
- **Total Loss**: $186.41 from 133 juice-routed samples in training set
- **Premium Loss**: $2.74/lb when misrouted to juice
- **Standard Loss**: $1.69/lb when misrouted to juice
- **Variety Patterns**: 
  - Red Delicious: 93.1% juice routing ($64.26 loss)
  - Honeycrisp: 1.5% juice routing ($3.18 loss)

### 3. Trajectory Insights
- **13 unique paths** through the network
- **Dominant pattern**: [*, 1, 1, 1] - 90.6% of samples
- **Minority paths**: Represent specialized processing for edge cases
- **Variety preservation**: Early clusters maintain variety identity before convergence

### 4. Semantic Cluster Labels
Successfully generated interpretable labels:
- Layer 0: "Premium-Leaning (68 samples)", "Standard Dominant (220 samples)", etc.
- Layers 1-3: "Majority Flow (765 samples)" vs "Minority Patterns (79 samples)"

## Implications for the Paper

### Methodological Validation
- Demonstrates CTA works beyond language models
- Shows economic impact quantification capability
- Validates semantic labeling approach with LLMs

### Key Contributions
1. **Domain Generalization**: CTA reveals meaningful patterns in agricultural AI
2. **Economic Integration**: Method can incorporate real-world costs/benefits
3. **Interpretability**: Semantic labels make neural processing understandable to domain experts
4. **Practical Value**: Identifies which varieties/features lead to economic losses

### Figures for Paper
1. **D3 Sankey Diagram**: Shows 10→2 cluster convergence with semantic labels
2. **Economic Impact Chart**: Variety-specific losses from juice misrouting
3. **Trajectory Distribution**: Histogram of path frequencies

## Statistical Highlights for Abstract/Introduction

- "Applied to agricultural AI, CTA revealed how a neural network routing apples to quality grades compresses 10 initial clusters to 2, with 90.6% of samples converging to a common pathway"
- "Economic analysis showed $186.41 in losses from misrouting, with historical varieties like Red Delicious (93.1% juice routing) validating market realities"
- "The method generated interpretable cluster labels ('Premium-Leaning', 'Standard Dominant') that domain experts could validate"

## Code and Reproducibility

All code, data, and visualizations are available at:
- Main experiment: `experiments/apple_variety/run_realistic_experiment.py`
- D3 visualizations: `experiments/apple_variety/create_d3_sankey.py`
- Statistics extraction: `experiments/apple_variety/paper_statistics.py`

The realistic synthetic dataset (1,320 samples) captures actual variety characteristics and market dynamics while being fully reproducible.