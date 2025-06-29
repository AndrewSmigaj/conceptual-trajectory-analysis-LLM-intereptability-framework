# GPT-2 Pronoun Context Steering Paper - arXiv Submission

## Paper Title
Context-Dependent Trajectory Bifurcation in Pronoun Processing: Evidence from GPT-2

## Abstract
We investigate how context tokens influence the neural processing trajectories of pronouns in GPT-2, focusing on the critical bifurcation that occurs within the first 4 layers...

## Submission Checklist

### Prerequisites
- [ ] Run the experiment: `cd experiments/gpt2_pronouns && ./run_experiment.sh`
- [ ] Verify results generated in `experiments/gpt2_pronouns/results/`

### Paper Components
- [ ] `pronoun_cta_paper.tex` - Main LaTeX file
- [ ] `figures/` - Contains sankey diagrams and other visualizations
- [ ] `references.bib` - Bibliography file

### Figures to Export
1. **sankey_early_layers.html** → `figures/sankey_early_layers.pdf`
   - Shows pronoun trajectories in first 4 layers
   - Highlights bifurcation patterns

2. **sankey_full_network.html** → `figures/sankey_full_network.pdf`
   - Shows complete 12-layer trajectories
   - Demonstrates path persistence

### Compilation
```bash
# Compile the paper
pdflatex pronoun_cta_paper.tex
bibtex pronoun_cta_paper
pdflatex pronoun_cta_paper.tex
pdflatex pronoun_cta_paper.tex

# Or use the compile script
./compile_paper.sh
```

### Final Check
- [ ] PDF compiles without errors
- [ ] All figures are included and properly referenced
- [ ] Results section contains actual experimental data
- [ ] Statistical tests are reported with p-values
- [ ] TDS scores show expected pattern (function > content)

## Key Results to Highlight
1. Function words show significantly higher TDS than content words
2. Most trajectory divergence occurs in layers 2-3
3. Statistical validation confirms context steering effect
4. Implications for understanding transformer processing