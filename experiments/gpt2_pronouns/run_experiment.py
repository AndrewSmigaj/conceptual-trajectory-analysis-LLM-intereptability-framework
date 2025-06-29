#!/usr/bin/env python
"""
Run the GPT-2 pronoun context steering experiment.

This script:
1. Generates probing data
2. Runs the main experiment
3. Produces visualizations and analysis
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(description: str, command: list) -> bool:
    """Run a step of the experiment pipeline."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Run the complete experiment pipeline."""
    parser = argparse.ArgumentParser(description='Run GPT-2 Pronoun Context Steering Experiment')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation if already done')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Use the virtual environment Python
    python_exe = "../../venv311/Scripts/python.exe"
    
    print("GPT-2 Pronoun Context Steering Experiment")
    print("==========================================")
    
    # Step 1: Generate probing data
    if not args.skip_data_gen:
        if not run_step("Generate probing data", 
                       [python_exe, "data_generation.py"]):
            print("Failed to generate data. Exiting.")
            sys.exit(1)
    
    # Step 2: Run main experiment
    if not run_step("Run pronoun trajectory analysis",
                   [python_exe, "pronoun_experiment.py", "--config", args.config]):
        print("Failed to run experiment. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)
    
    # Print location of results
    results_dir = Path("results")
    if results_dir.exists():
        print(f"\nResults saved to: {results_dir.absolute()}")
        
        # List key output files
        print("\nKey outputs:")
        for file in results_dir.glob("*"):
            print(f"  - {file.name}")


if __name__ == "__main__":
    main()