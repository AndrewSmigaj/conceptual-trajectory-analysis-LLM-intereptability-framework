# GPT-2 Context Effects on Token Trajectories

This experiment directory contains two related studies on context effects in GPT-2:
1. **Original**: Pronoun context steering experiment
2. **Extended**: Comprehensive analysis of 10,000 vocabulary items

## Overview

We investigate how contextual frames systematically influence neural processing trajectories in GPT-2. Starting from the observation that pronouns bifurcate into distinct processing paths, we expanded to analyze context effects across the entire frequent vocabulary (10,000 tokens).

## Key Findings

### Comprehensive Vocabulary Analysis
1. **Widespread context effects**: 7,324 out of 10,000 tokens show context-dependent trajectory changes
2. **Token type hierarchy**: Subword tokens are most sensitive (mean divergence 0.42), while punctuation and numeric tokens are most stable
3. **Context strength**: Determiner contexts ("the", "a") exert the strongest steering effects (Cohen's d > 0.8)
4. **Early layer concentration**: 68% of trajectory changes occur in layers 0-3
5. **Statistical significance**: Effects validated through permutation tests (p < 0.001) and cross-validation

### Original Pronoun Study
- Pronouns split into two distinct processing paths within the first 4 layers:
  - **Function/Determiner Path**: Grammatical processing
  - **Content/Human-Social Path**: Semantic processing
- Context tokens can "steer" which path a pronoun takes

## Repository Structure

```
experiments/gpt2_pronouns/
├── config.yaml                          # Experiment configuration
├── pronoun_context_steering.md          # Original pronoun-focused paper
├── vocabulary_context_effects_paper.md  # Comprehensive vocabulary analysis paper
├── SUPPLEMENTARY_MATERIALS.md           # Detailed documentation
│
├── Data Generation & Processing
│   ├── context_frame_generator.py       # Generate token-context pairs (73,888 total)
│   └── vocabulary_context_experiment.py # Main experiment runner
│
├── Analysis Scripts
│   ├── comprehensive_clustering_analysis.py  # Map to existing k10 clusters
│   ├── context_effect_statistics.py         # Statistical analysis & significance tests
│   ├── trajectory_pattern_discovery.py      # Discover archetypal paths
│   ├── llm_full_vocabulary_analysis.py      # LLM-based pattern analysis
│   ├── llm_deep_analysis.py                 # Second-pass deep analysis
│   └── validation_analysis.py               # Robustness checks & validation
│
├── Visualization
│   ├── visualize_context_effects.py         # Comprehensive visualizations
│   └── generate_paper_figures.py            # Publication-ready figures
│
├── Original Pronoun Scripts (deprecated)
│   ├── pronoun_experiment.py            # Original pronoun experiment
│   ├── data_generation.py               # Two-token probe generation
│   └── run_experiment.py                # Original pipeline runner
│
└── Results (generated)
    ├── test_cases.json                      # Generated token-context pairs
    ├── visualization_data.json              # Trajectory data
    ├── statistical_report.json              # Statistical analysis results
    ├── clustering_analysis/                 # Clustering results
    ├── pattern_discovery/                   # Pattern analysis
    ├── validation/                          # Validation results
    └── paper_figures/                       # Publication figures
```

## Quick Start - Full Vocabulary Analysis

### 1. Generate context frames
```bash
python context_frame_generator.py
```
Generates 73,888 token-context test cases from 10k tokens × 9 contexts.

### 2. Run the main experiment
```bash
python vocabulary_context_experiment.py
```
Extracts GPT-2 activations and maps to trajectories. Uses batch processing with checkpointing.

### 3. Analyze trajectories
```bash
python comprehensive_clustering_analysis.py
python context_effect_statistics.py
python trajectory_pattern_discovery.py
```

### 4. Generate visualizations
```bash
python visualize_context_effects.py
python generate_paper_figures.py
```

### 5. Validate findings
```bash
python validation_analysis.py
```

## Context Frames Tested

Nine contextual frames:
1. `baseline`: Token alone
2. `determiner_the`: "the [TOKEN]"
3. `determiner_a`: "a [TOKEN]"
4. `pronoun_i`: "I [TOKEN]"
5. `pronoun_they`: "they [TOKEN]"
6. `preposition_with`: "with [TOKEN]"
7. `preposition_of`: "of [TOKEN]"
8. `sentence_start_is`: "[TOKEN] is"
9. `sentence_start_are`: "[TOKEN] are"

## Trajectory Divergence Score (TDS)

Quantifies how much a context changes a token's trajectory:
```
TDS = number_of_different_clusters / total_layers
```
- **TDS_early**: First 4 layers only
- **TDS_full**: All 12 layers

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn, Plotly
- 16GB RAM, GPU recommended

## Integration with CTA Framework

This experiment extends Concept Trajectory Analysis by demonstrating that trajectories are not fixed but can be dynamically influenced by context. It leverages existing infrastructure:
- `SimpleGPT2ActivationExtractor` for activation extraction
- Existing k10 clustering models from GPT-2 study
- `PathExtractor` for trajectory analysis
- `SankeyGenerator/D3SankeyGenerator` for visualization

## Citation

If you use this analysis, please cite:
```
[Citation to be added upon publication]
```

## Contact

For questions or issues:
- Open an issue in this repository
- See SUPPLEMENTARY_MATERIALS.md for detailed documentation