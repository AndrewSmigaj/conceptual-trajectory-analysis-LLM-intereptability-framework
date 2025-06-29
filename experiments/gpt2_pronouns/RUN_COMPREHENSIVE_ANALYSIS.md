# Comprehensive Context Analysis Instructions

## Overview
This comprehensive analysis expands the context transformation study to:
- **10,000+ tokens** (vs current 1,000)
- **117 single-token contexts** (vs current 9)
- **Activation-based distances** for better dendrograms
- **k=20 clusters** for finer granularity

## Steps to Run the Analysis

### 1. Create Expanded Test Cases (10k tokens)
```bash
python create_expanded_test_cases.py
```
This creates `expanded_test_cases.json` with ~1.17 million test cases.

### 2. Run the Experiment (Overnight)
```bash
python run_comprehensive_experiment.py
```
This will:
- Extract activations for all token-context combinations
- Perform unified clustering with k=20
- Save checkpoints every 1000 tokens (can resume if interrupted)
- Output to `results_comprehensive/`

Expected runtime: 8-12 hours on GPU, 24-48 hours on CPU

### 3. Analyze Results
```bash
python analyze_comprehensive_results.py
```
This creates improved dendrograms using:
- Trajectory-based distances
- Activation-based distances (if saved)

### 4. Extract Paper Subset
```bash
python prepare_paper_subset.py
```
Filters results to the 24 contexts used in the paper.

### 5. Generate Paper Figures
```bash
cd transformation_analysis
python fix_publication_figures.py
```

## Benefits of Comprehensive Analysis

1. **Statistical Power**: 1.17M trajectories vs 10k current
2. **Context Diversity**: 117 contexts across all linguistic categories
3. **Better Dendrograms**: Activation-based distances show real structure
4. **Robust Evidence**: Full vocabulary coverage

## Paper Contexts (24 selected)
- **Baseline**: no context
- **Articles**: the, a
- **Pronouns**: he, she, it
- **Prepositions**: in, on, with
- **Conjunctions**: and, but
- **Auxiliaries**: is, was, will, can
- **Content words**: time, person, said, make, good, new
- **Special**: not, ., sentence_start

## Expected Outputs
- `results_comprehensive/unified_trajectories_k20.json`: All trajectories
- `results_comprehensive/dendrogram_trajectory_based.pdf`: Improved dendrogram
- `results_comprehensive/dendrogram_activation_based.pdf`: Best dendrogram
- `results_paper_subset/paper_trajectories_k20.json`: Filtered for paper