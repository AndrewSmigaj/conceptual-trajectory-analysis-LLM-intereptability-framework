# Apple CTA Paper - Submission Package

## Paper Title
"From Honeycrisp to Juice: Using Concept Trajectory Analysis to Understand Quality Routing Decisions in Apple Processing"

## Files in Submission Package

### Main Files
1. **apple_cta_paper.tex** - Main LaTeX paper (complete)
2. **figures/** - Directory for paper figures
   - `apple_sankey_full_network.png` - Main trajectory visualization (needs export)

### Compilation Instructions

1. **Export the Sankey Diagram**:
   ```bash
   # Follow instructions in experiments/apple_variety/MANUAL_EXPORT_INSTRUCTIONS.md
   # Save PNG to arxiv_apple/figures/apple_sankey_full_network.png
   ```

2. **Compile the Paper**:
   ```bash
   cd arxiv_apple
   pdflatex apple_cta_paper.tex
   bibtex apple_cta_paper
   pdflatex apple_cta_paper.tex
   pdflatex apple_cta_paper.tex
   ```

3. **Check Output**:
   - Verify `apple_cta_paper.pdf` is generated
   - Check that figure appears correctly
   - Ensure all references are resolved

## Submission Checklist

- [ ] Export D3 sankey to PNG format
- [ ] Place in figures/apple_sankey_full_network.png
- [ ] Compile LaTeX to PDF
- [ ] Review PDF for formatting issues
- [ ] Check figure quality and readability
- [ ] Verify all sections are complete
- [ ] Update author information in LaTeX

## Paper Statistics

- **Length**: ~12 pages (including references and appendices)
- **Sections**: 9 main sections + 3 appendices
- **Figures**: 1 main sankey diagram (more can be added)
- **Tables**: 2 (economic impact and performance metrics)
- **References**: 3 (add more as needed)

## Additional Resources

- **Experiment Results**: `experiments/apple_variety/results/apple_realistic/`
- **Paper Statistics**: `experiments/apple_variety/paper_statistics.json`
- **Figure Captions**: `experiments/apple_variety/figure_captions.tex` (additional figures if needed)
- **Economic Analysis**: `experiments/apple_variety/results/apple_realistic/economic_impact_report.txt`

## Before Submission

1. Update placeholder author information in the LaTeX file
2. Add additional references as appropriate
3. Consider adding more figures from the experiment results
4. Review and polish abstract and conclusion
5. Ensure compliance with target venue formatting requirements