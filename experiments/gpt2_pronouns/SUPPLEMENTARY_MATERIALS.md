# Supplementary Materials: Context-Induced Trajectory Modulation in GPT-2

## Table of Contents

1. [Data Description](#1-data-description)
2. [Analysis Pipeline](#2-analysis-pipeline)
3. [Reproduction Instructions](#3-reproduction-instructions)
4. [Data Formats](#4-data-formats)
5. [Statistical Methods](#5-statistical-methods)
6. [Additional Results](#6-additional-results)
7. [Code Documentation](#7-code-documentation)
8. [Computational Requirements](#8-computational-requirements)

## 1. Data Description

### 1.1 Token Dataset
- **Source**: GPT-2 tokenizer vocabulary
- **Size**: 10,000 most frequent tokens
- **File**: `../gpt2/all_tokens/top_10k_tokens_full.json`
- **Format**: JSON array with token information
  ```json
  {
    "token_str": "example",
    "token_id": 1234,
    "token_type": "complete_word",
    "frequency": 0.0023,
    "is_punctuation": false,
    "is_subword": false
  }
  ```

### 1.2 Context Frames
Nine context frames tested:
1. `baseline`: Token alone
2. `determiner_the`: "the [TOKEN]"
3. `determiner_a`: "a [TOKEN]"
4. `pronoun_i`: "I [TOKEN]"
5. `pronoun_they`: "they [TOKEN]"
6. `preposition_with`: "with [TOKEN]"
7. `preposition_of`: "of [TOKEN]"
8. `sentence_start_is`: "[TOKEN] is"
9. `sentence_start_are`: "[TOKEN] are"

### 1.3 Trajectory Data
- **Total pairs**: 73,888 (after filtering)
- **Format**: Token-context pairs with cluster assignments
- **Layers**: 12 (0-11)
- **Clusters per layer**: 10 (k=10)

## 2. Analysis Pipeline

### 2.1 Overview
```
1. Context Frame Generation
   ├── Load 10k tokens
   ├── Generate context pairs
   └── Filter invalid combinations
   
2. Activation Extraction
   ├── Process in batches (1000)
   ├── Extract GPT-2 hidden states
   └── Save checkpoints
   
3. Trajectory Mapping
   ├── Load pre-trained k10 clusters
   ├── Map activations to clusters
   └── Build trajectory sequences
   
4. Statistical Analysis
   ├── Calculate divergence scores
   ├── Compute effect sizes
   ├── Test significance
   └── Identify patterns
   
5. Visualization
   ├── Generate heatmaps
   ├── Create distribution plots
   ├── Build sankey diagrams
   └── Export paper figures
```

### 2.2 Key Scripts

1. **context_frame_generator.py**: Generates token-context pairs
2. **vocabulary_context_experiment.py**: Runs main experiment
3. **comprehensive_clustering_analysis.py**: Maps to clusters
4. **context_effect_statistics.py**: Statistical analysis
5. **trajectory_pattern_discovery.py**: Pattern identification
6. **llm_full_vocabulary_analysis.py**: LLM-based analysis
7. **visualize_context_effects.py**: Creates visualizations
8. **generate_paper_figures.py**: Publication figures
9. **llm_deep_analysis.py**: Deep pattern analysis
10. **validation_analysis.py**: Robustness checks

## 3. Reproduction Instructions

### 3.1 Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Required Dependencies
```
numpy>=1.21.0
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=0.24.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
tqdm>=4.62.0
pyyaml>=5.4.0
```

### 3.3 Running the Analysis
```bash
# 1. Generate context frames
python context_frame_generator.py

# 2. Run main experiment (this takes several hours)
python vocabulary_context_experiment.py

# 3. Perform clustering analysis
python comprehensive_clustering_analysis.py

# 4. Generate statistics
python context_effect_statistics.py

# 5. Discover patterns
python trajectory_pattern_discovery.py

# 6. Create visualizations
python visualize_context_effects.py

# 7. Generate paper figures
python generate_paper_figures.py

# 8. Run validation
python validation_analysis.py
```

### 3.4 Using Pre-computed Results
If you want to skip the computation-heavy steps, pre-computed results are available:
- Trajectories: `results/visualization_data.json`
- Statistics: `results/statistical_report.json`
- Patterns: `results/pattern_discovery/`
- Clustering: `results/clustering_analysis/`

## 4. Data Formats

### 4.1 Trajectory Format
```json
{
  "trajectories": {
    "token_idx_context": {
      "token_idx": 123,
      "token_str": "example",
      "context_frame": "determiner_the",
      "path": [3, 7, 2, 5, 8, 1, 4, 6, 9, 0, 2, 3],
      "activations_norm": [0.823, 0.756, ...],
      "timestamp": "2024-01-15T10:30:00"
    }
  }
}
```

### 4.2 Statistical Report Format
```json
{
  "effect_sizes": {
    "context_name": {
      "cohens_d": 0.84,
      "mean_effect": 0.31,
      "std_effect": 0.18,
      "n_tokens": 8924,
      "interpretation": "large"
    }
  },
  "independence_tests": {...},
  "token_type_analysis": {...},
  "layer_statistics": {...},
  "outliers": [...]
}
```

### 4.3 Pattern Discovery Format
```json
{
  "archetypal_paths": [
    {
      "path": [0, 1, 3, 3],
      "frequency": 0.23,
      "count": 5421,
      "examples": [
        {"token": "the", "context": "baseline"},
        {"token": "and", "context": "pronoun_i"}
      ]
    }
  ]
}
```

## 5. Statistical Methods

### 5.1 Trajectory Divergence Score (TDS)
```python
def calculate_tds(baseline_path, context_path, layers):
    divergent = sum(1 for i in layers 
                   if baseline_path[i] != context_path[i] 
                   and baseline_path[i] != -1 
                   and context_path[i] != -1)
    valid = sum(1 for i in layers 
               if baseline_path[i] != -1 
               and context_path[i] != -1)
    return divergent / valid if valid > 0 else 0
```

### 5.2 Cohen's d Effect Size
```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
    return (np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var)
```

### 5.3 Permutation Test
- Null hypothesis: Context labels are exchangeable
- Test statistic: Mean trajectory divergence
- Permutations: 1000
- Significance level: α = 0.05

### 5.4 Bootstrap Confidence Intervals
- Method: Percentile bootstrap
- Samples: 1000
- Confidence levels: 95% and 99%

## 6. Additional Results

### 6.1 Extended Token Type Analysis

| Token Type | Subtype | Mean Sensitivity | N Tokens |
|------------|---------|------------------|----------|
| Subword | Prefix | 0.38 ± 0.19 | 892 |
| Subword | Suffix | 0.44 ± 0.22 | 1,449 |
| Complete | Noun | 0.31 ± 0.17 | 1,843 |
| Complete | Verb | 0.29 ± 0.16 | 1,205 |
| Complete | Adjective | 0.27 ± 0.15 | 987 |
| Complete | Function | 0.23 ± 0.14 | 857 |

### 6.2 Layer-Specific Effects by Context

| Context | L0 | L1 | L2 | L3 | L4-11 avg |
|---------|----|----|----|----|-----------|
| the [TOKEN] | 0.22 | 0.34 | 0.41 | 0.35 | 0.18 |
| a [TOKEN] | 0.20 | 0.31 | 0.38 | 0.33 | 0.16 |
| they [TOKEN] | 0.15 | 0.24 | 0.29 | 0.26 | 0.14 |
| I [TOKEN] | 0.14 | 0.22 | 0.27 | 0.24 | 0.13 |

### 6.3 Trajectory Stability Across Contexts

Tokens showing complete stability (0% divergence) across all contexts:
- Punctuation: 68.4% stable
- Numeric: 71.2% stable
- Function words: 42.3% stable
- Content words: 18.7% stable
- Subwords: 8.2% stable

## 7. Code Documentation

### 7.1 Key Functions

#### context_frame_generator.py
```python
class ContextFrameGenerator:
    def generate_test_cases(self, tokens, context_frames):
        """Generate all valid token-context combinations."""
        
    def should_skip_token(self, token_info):
        """Determine if token should be skipped for certain contexts."""
        
    def save_test_cases(self, output_path):
        """Save generated test cases to JSON."""
```

#### vocabulary_context_experiment.py
```python
class VocabularyContextExperiment(BaseExperiment):
    def extract_batch_activations(self, batch):
        """Extract activations for a batch of inputs."""
        
    def map_to_clusters(self, activations, layer):
        """Map activations to existing k10 clusters."""
        
    def save_checkpoint(self, batch_idx):
        """Save progress checkpoint for recovery."""
```

### 7.2 Configuration Options

config.yaml parameters:
```yaml
experiment:
  batch_size: 1000  # Tokens per batch
  checkpoint_interval: 5000  # Save frequency
  
analysis:
  early_layers: [0, 1, 2, 3]  # Focus layers
  clustering:
    method: "existing_k10"  # Use pre-trained clusters
    
visualization:
  top_n_paths: 50  # Paths to show in sankey
  heatmap_tokens: 200  # Tokens in heatmap
```

## 8. Computational Requirements

### 8.1 Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB for full dataset and results

### 8.2 Runtime
- Context generation: ~5 minutes
- Activation extraction: ~4-6 hours (GPU)
- Clustering analysis: ~30 minutes
- Statistical analysis: ~20 minutes
- Visualization: ~15 minutes
- Total: ~6-8 hours

### 8.3 Memory Usage
- Peak GPU memory: ~6GB
- Peak RAM: ~12GB
- Disk space for results: ~2GB

## 9. Contact and Support

For questions about the analysis or issues with reproduction:
- GitHub repository: [URL]
- Issues: [URL]/issues
- Email: [contact email]

## 10. Version Information

- Analysis version: 1.0.0
- GPT-2 model: `gpt2` (124M parameters)
- Transformers version: 4.20.0
- Python version: 3.8+

Last updated: January 2024