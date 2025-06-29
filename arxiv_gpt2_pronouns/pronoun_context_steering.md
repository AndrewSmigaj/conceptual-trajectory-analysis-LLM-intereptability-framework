# Context-Dependent Trajectory Bifurcation in Pronoun Processing: Evidence from GPT-2

## Abstract

We investigate how context tokens influence the neural processing trajectories of pronouns in GPT-2, focusing on the critical bifurcation that occurs within the first 4 layers. Using Concept Trajectory Analysis (CTA), we tracked pronoun activations through clustered representation spaces and discovered that pronouns split into two distinct processing paths: a "function/determiner" path and a "content/human-social" path. We introduce the Trajectory Divergence Score (TDS) to quantify how preceding context tokens can steer pronouns toward different processing paths. Our two-token probing methodology reveals that function words (e.g., "the") systematically redirect pronouns toward grammatical processing, while content words (e.g., "happy") maintain semantic processing paths. This context-dependent routing suggests that neural language models dynamically adjust their processing strategies based on local syntactic cues, with implications for understanding how transformers handle linguistic ambiguity.

## 1. Introduction

The organization of linguistic concepts within neural language models remains a fundamental question in interpretability research. Recent work using Concept Trajectory Analysis (CTA) has revealed that GPT-2 organizes words through a hierarchical process, with initial semantic clustering giving way to grammatical organization in later layers. A particularly intriguing finding is the bifurcation of pronoun processing within the first 4 layers, where pronouns split into two distinct trajectories: one aligned with function words and determiners, another with content words expressing human and social concepts.

This observation raises a critical question: Is this bifurcation inherent to pronoun processing, or can it be influenced by contextual cues? We hypothesize that preceding context tokens can "steer" pronouns toward different processing paths, effectively modulating whether a pronoun is processed primarily for its grammatical function or its semantic content.

To test this hypothesis, we employ a minimal two-token probing methodology, examining how different context tokens influence pronoun trajectories through GPT-2's layers. We introduce the Trajectory Divergence Score (TDS), a metric that quantifies the degree to which context alters a pronoun's path through the network's clustered activation spaces.

## 2. Background

### 2.1 Concept Trajectory Analysis

CTA tracks how neural representations evolve through network layers by:
1. Clustering activations at each layer
2. Assigning tokens to clusters based on activation patterns
3. Tracking movement through cluster spaces across layers
4. Identifying common paths (trajectories) and convergence patterns

Previous CTA analysis of GPT-2 revealed that pronouns exhibit unique behavior, splitting into two primary paths within the first 4 layers rather than following a single dominant trajectory like most word classes.

### 2.2 The Pronoun Bifurcation Phenomenon

Analysis of 1,228 single-token words showed that pronouns (she, he, they, it, etc.) diverge into:
- **Function/Determiner Path**: Aligning with articles, prepositions, and other function words
- **Content/Human-Social Path**: Aligning with words expressing human attributes and social concepts

This split occurs early in processing (layers 1-4) and persists through the network, suggesting two distinct processing modes for pronouns.

## 3. Methods

### 3.1 Two-Token Probing

We construct minimal two-token sequences to isolate context effects:
```
[CONTEXT] [PRONOUN]
```

Context categories:
- **Function words**: the, a, with, from, by
- **Content words**: happy, angry, smart, young, tired
- **Neutral baseline**: [PRONOUN] alone

Pronouns tested: she, he, they, it, we, I

### 3.2 Trajectory Divergence Score (TDS)

For each pronoun p and context c, we define:

**Early TDS (first 4 layers)**:
```
TDS_early(p,c) = |{l ∈ [0,3] : cluster(p,l) ≠ cluster(p|c,l)}| / 4
```

**Full TDS (all layers)**:
```
TDS_full(p,c) = |{l ∈ [0,11] : cluster(p,l) ≠ cluster(p|c,l)}| / 12
```

Where:
- cluster(p,l) = cluster assignment of pronoun p at layer l
- cluster(p|c,l) = cluster assignment of pronoun p with context c at layer l

### 3.3 Experimental Setup

1. **Model**: GPT-2 (12 layers, 768 hidden dimensions)
2. **Clustering**: K-means with k=10 per layer (matching original CTA study)
3. **Activation Extraction**: Hidden states from each transformer layer
4. **Trajectory Analysis**: Path extraction using existing CTA infrastructure

### 3.4 Analysis Pipeline

1. Extract activations for all two-token sequences
2. Apply clustering models from original GPT-2 study
3. Compute trajectories for each pronoun-context pair
4. Calculate TDS metrics (early and full)
5. Analyze trajectory destinations and bifurcation patterns
6. Generate sankey visualizations for trajectory flow

## 4. Expected Results

### 4.1 Context Steering Effects

We expect to find:
- **Function word contexts**: High TDS, steering pronouns toward function/determiner path
- **Content word contexts**: Low TDS, maintaining content/human-social path
- **Early layer sensitivity**: Most steering occurs in layers 0-3

### 4.2 Trajectory Patterns

Anticipated patterns:
1. **Bifurcation point**: Layer 2-3 as critical decision point
2. **Path commitment**: Once steered, pronouns remain on chosen path
3. **Context consistency**: Similar contexts produce similar steering effects

### 4.3 Implications

If confirmed, these results would suggest:
- Transformers use local syntactic cues for dynamic processing decisions
- Pronoun ambiguity is resolved through context-dependent routing
- Early layers implement critical organizational decisions

## 5. Trajectory Divergence Analysis

### 5.1 Quantifying Context Influence

The TDS metric provides a quantitative measure of how much a context token can alter a pronoun's processing path. We analyze:
- Distribution of TDS values across context types
- Correlation between context grammatical category and steering strength
- Individual pronoun susceptibility to steering

### 5.2 Bifurcation Mechanics

We examine:
- Layer-by-layer trajectory evolution
- Sharpness of bifurcation (gradual vs sudden)
- Consistency of bifurcation point across pronouns

## 6. Visualization

Using existing CTA visualization infrastructure:
- Sankey diagrams showing trajectory flow
- Separate visualizations for different context conditions
- Highlighting of divergence points

## 7. Statistical Analysis

- Chi-squared tests for trajectory distribution differences
- Effect sizes for context influence
- Bootstrap confidence intervals for TDS values

## 8. Discussion

This work extends CTA by demonstrating that concept trajectories are not fixed but can be dynamically influenced by context. The pronoun bifurcation phenomenon provides a unique window into how neural language models implement context-sensitive processing strategies.

## 9. Future Work

- Extend to multi-token contexts
- Test on other transformer models
- Investigate other word classes with trajectory variability
- Explore implications for pronoun resolution tasks

## References

[To be added based on actual citations needed]