#!/usr/bin/env python3
"""
Create arXiv submission package with fixed figures.
"""

import os
import shutil
import tarfile
from pathlib import Path

def main():
    print("Creating arXiv submission package...")
    
    # Change to arxiv_submission directory
    os.chdir("arxiv_submission")
    
    # Create tar.gz archive
    files_to_include = [
        "main_arxiv.tex",
        "references.bib",
        "trajectory_fan_plot.pdf",
        "layer_evolution.pdf", 
        "transformation_geometry.pdf",
        "context_similarity_dendrogram.pdf",
        "token_type_metrics.pdf",
        "main_arxiv.bbl"  # Bibliography
    ]
    
    # Check all files exist
    missing = []
    for f in files_to_include:
        if not Path(f).exists():
            missing.append(f)
    
    if missing:
        print(f"Warning: Missing files: {missing}")
    
    # Create archive
    archive_name = "arxiv_submission_fixed.tar.gz"
    with tarfile.open(archive_name, "w:gz") as tar:
        for f in files_to_include:
            if Path(f).exists():
                tar.add(f)
                print(f"Added: {f}")
    
    print(f"\nCreated: {archive_name}")
    print(f"Size: {Path(archive_name).stat().st_size / 1024:.1f} KB")
    
    # Also create a PDF copy in parent directory
    shutil.copy("main_arxiv.pdf", "../context_transformations_gpt2_fixed.pdf")
    print("\nCopied PDF to: ../context_transformations_gpt2_fixed.pdf")

if __name__ == "__main__":
    main()