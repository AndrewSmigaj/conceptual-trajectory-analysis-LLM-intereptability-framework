#!/bin/bash

echo "Compiling GPT-2 Pronoun Context Steering Paper"
echo "============================================="

# Clean up old files
rm -f *.aux *.bbl *.blg *.log *.out *.pdf

# First pass
echo "Running pdflatex (1st pass)..."
pdflatex pronoun_cta_paper.tex

# Bibliography
echo "Running bibtex..."
bibtex pronoun_cta_paper

# Second pass
echo "Running pdflatex (2nd pass)..."
pdflatex pronoun_cta_paper.tex

# Third pass (for references)
echo "Running pdflatex (3rd pass)..."
pdflatex pronoun_cta_paper.tex

# Check if PDF was created
if [ -f "pronoun_cta_paper.pdf" ]; then
    echo ""
    echo "Success! Paper compiled to: pronoun_cta_paper.pdf"
    echo "File size: $(ls -lh pronoun_cta_paper.pdf | awk '{print $5}')"
else
    echo ""
    echo "Error: PDF not generated. Check the log file for errors."
fi