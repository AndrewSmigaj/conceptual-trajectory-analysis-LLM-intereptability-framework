#!/bin/bash
# Compile the Apple CTA paper

echo "Compiling Apple CTA Paper..."

# Check if figure exists
if [ ! -f "figures/apple_sankey_full_network.png" ]; then
    echo "WARNING: Main figure not found at figures/apple_sankey_full_network.png"
    echo "Please export the D3 sankey diagram first."
    echo "See experiments/apple_variety/MANUAL_EXPORT_INSTRUCTIONS.md"
    exit 1
fi

# Clean previous build files
rm -f *.aux *.bbl *.blg *.log *.out *.toc

# Compile paper with nonstopmode to continue past errors
pdflatex -interaction=nonstopmode apple_cta_paper.tex
if [ $? -ne 0 ]; then
    echo "WARNING: First pdflatex compilation had errors, continuing..."
fi

# Skip bibtex since bibliography is embedded in the document
echo "Skipping BibTeX (bibliography is embedded in document)"

# Compile twice more for references
pdflatex -interaction=nonstopmode apple_cta_paper.tex
pdflatex -interaction=nonstopmode apple_cta_paper.tex

# Check if PDF was created
if [ -f "apple_cta_paper.pdf" ]; then
    echo "SUCCESS: Paper compiled to apple_cta_paper.pdf"
    echo "File size: $(ls -lh apple_cta_paper.pdf | awk '{print $5}')"
else
    echo "ERROR: PDF not generated"
    exit 1
fi

# Clean intermediate files
rm -f *.aux *.bbl *.blg *.log *.out *.toc

echo "Compilation complete!"