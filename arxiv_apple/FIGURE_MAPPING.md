# Figure Mapping for Apple CTA Paper

This document maps the figures referenced in the LaTeX paper to their actual locations.

## Required Figures

### 1. Main Sankey Diagram
- **LaTeX Reference**: `figures/apple_sankey_full_network.png`
- **Source**: `experiments/apple_variety/results/apple_realistic/d3_sankey_full_network.html`
- **Action**: Export to PNG using manual instructions or scripts
- **Status**: Export instructions provided

### 2. Economic Impact Chart (if needed)
- **LaTeX Reference**: `figures/apple_economic_impact.png`
- **Source**: `experiments/apple_variety/results/apple_realistic/economic_impact.png` or generate from data
- **Action**: Create bar chart from economic_impact_report.txt data
- **Status**: May need to generate

### 3. Trajectory Distribution (if needed)
- **LaTeX Reference**: `figures/apple_trajectory_distribution.png`
- **Source**: Generate from trajectory analysis data
- **Action**: Create from paper_statistics.json
- **Status**: May need to generate

### 4. Variety Convergence (if needed)
- **LaTeX Reference**: `figures/apple_variety_convergence.png`
- **Source**: Generate from variety_trajectory_summary.txt
- **Action**: Create convergence rate visualization
- **Status**: May need to generate

## Current Status

The main paper currently only references the full network sankey diagram. Additional figures from figure_captions.tex can be added if needed, but are not currently referenced in the main text.

## To Complete Figures:

1. Export the D3 sankey to PNG:
   - Follow instructions in `experiments/apple_variety/MANUAL_EXPORT_INSTRUCTIONS.md`
   - Save as `arxiv_apple/figures/apple_sankey_full_network.png`

2. Create figures directory:
   ```bash
   mkdir -p arxiv_apple/figures
   ```

3. Once sankey is exported, the paper will be ready to compile.