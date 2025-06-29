# Context Transformation Analysis - GPT-2 Unified Clustering Experiment

## Current Status (2025-06-23)

### What We've Done
1. **Unified Clustering Experiment Completed**
   - Selected 1,000 diverse tokens from GPT-2 vocabulary
   - Generated 9,944 test cases (1,000 tokens × ~10 context frames)
   - Extracted activations for all test cases
   - Performed k=10 clustering on pooled activations (all contexts together)
   - Built trajectories and analyzed divergence

2. **Key Files Created**
   - `results_unified/unified_activations.pkl` - All 119,328 activations (12 layers × 9,944 cases)
   - `results_unified/cluster_models_k10/` - KMeans models for each layer
   - `results_unified/unified_trajectories_k10.json` - All trajectories with k=10
   - `results_unified/analysis_k10/divergence_analysis_k10.json` - Divergence analysis

### Initial Finding: 100% Divergence
- **Every single token** shows complete trajectory divergence when context is added
- Divergence occurs immediately at layer 0 (embedding layer)
- No token maintains its trajectory across any context
- Sample trajectories for " the":
  ```
  baseline:        [1, 1, 1, 1, 7, 9, 1, 1, 7, 1, 1, 6]
  determiner_the:  [9, 9, 4, 7, 6, 0, 0, 2, 5, 5, 0, 3]
  determiner_a:    [9, 9, 4, 7, 6, 0, 8, 4, 4, 5, 3, 3]
  possessive_my:   [9, 9, 4, 7, 6, 0, 0, 2, 5, 5, 0, 0]
  ```

### The Realization
This 100% divergence is **expected behavior** for transformers! Context *should* change representations - that's how attention works. The real insight is that these changes might be **systematic transformations** rather than random reorganization.

## Transformation Analysis Plan

### Core Hypothesis
Context creates a systematic transformation of the representation space. The latent space structure is preserved but transformed (shifted/rotated/scaled) in predictable ways based on the grammatical/semantic nature of the context.

### Phase 1: Cluster Transition Matrices
**Script**: `analyze_cluster_transitions.py`

Build 10×10 transition matrices for each context type showing P(baseline cluster i → context cluster j).

Expected patterns:
- Determiners might create similar transformations
- Grammatical contexts (is, will) might show different patterns
- Transitions should be sparse and systematic, not uniform

### Phase 2: Transformation Consistency Analysis
**Script**: `analyze_transformation_consistency.py`

Test if transformations are token-independent:
- Group tokens by baseline cluster
- Check if tokens from same baseline cluster consistently map to same context clusters
- Calculate entropy of transitions to measure consistency
- Compare transformation patterns across token types

### Phase 3: Find Linear Transformations in Activation Space
**Script**: `find_activation_transformations.py`

Work with actual activation vectors, not just cluster assignments:
- Extract activation pairs (baseline, context) for same token
- Test for linear transformation using Procrustes analysis
- Find optimal transformation matrix T such that: context_activation ≈ T × baseline_activation
- Check if T is consistent across tokens

### Phase 4: Layer-wise Transformation Analysis
**Script**: `layer_wise_transformation_analysis.py`

Analyze how transformations evolve across layers:
- Early layers (0-3): Expect grammatical/syntactic transformations
- Middle layers (4-7): Mixed syntactic/semantic
- Late layers (8-11): Semantic/task-specific transformations

Compare with known transformer behavior where early layers handle syntax and late layers handle semantics.

## Technical Implementation Details

### Data Structure
```python
# Trajectories are stored as:
{
  "token_idx_contextname": {
    "token_idx": int,
    "token_str": str,
    "context_frame": str,
    "path": [cluster_0, cluster_1, ..., cluster_11],
    "case_idx": int
  }
}
```

### Key Functions Needed
1. `build_transition_matrix(trajectories, baseline_context, target_context, layer)`
2. `find_procrustes_transform(activations_baseline, activations_context)`
3. `calculate_transformation_entropy(transition_matrix)`
4. `visualize_transition_patterns(transition_matrices)`

### Expected Findings
1. **Systematic Patterns**: Transition matrices should show clear structure
2. **Context Grouping**: Similar contexts (determiners) should produce similar transformations
3. **Layer Evolution**: Transformations should become more complex/semantic in later layers
4. **Predictability**: Given a token's baseline cluster, we should be able to predict its context cluster

## Quick Start for Next Session

```bash
# Navigate to experiment directory
cd /mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization/experiments/gpt2_pronouns

# Check current state
ls -la results_unified/
ls -la results_unified/cluster_models_k10/

# The trajectories are ready at:
# results_unified/unified_trajectories_k10.json

# The activations are at:
# results_unified/unified_activations.pkl

# Ready to implement transformation analysis!
```

## Why This Matters
If context effects are systematic transformations rather than chaotic reorganization, it suggests:
1. Transformers learn structured rules for how context modifies meaning
2. These transformations might be interpretable and predictable
3. We could potentially manipulate representations by applying learned transformations
4. This aligns with linguistic theory about how context modifies word meaning

## Next Steps
1. Implement Phase 1: Build and visualize transition matrices
2. Look for patterns in how different context types transform the space
3. Test the linear transformation hypothesis
4. Connect findings to transformer architecture and attention mechanisms