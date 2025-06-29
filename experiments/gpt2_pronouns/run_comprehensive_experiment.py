#!/usr/bin/env python3
"""
Run comprehensive context effects experiment with expanded tokens and contexts.
"""

import sys
import logging
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))  # Add current directory first
sys.path.insert(0, str(current_dir.parent.parent))  # Add project root

from unified_context_experiment import UnifiedContextExperiment
from concept_fragmentation.experiments.base_experiment import ExperimentConfig

def main():
    """Run the comprehensive experiment."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_path = Path(__file__).parent / "config_comprehensive.yaml"
    config = ExperimentConfig.from_yaml(str(config_path))
    
    # Create output directory
    output_dir = Path(__file__).parent / config.output.base_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize experiment
    experiment = UnifiedContextExperiment(
        config=config,
        output_dir=str(output_dir)
    )
    
    # Run experiment
    print("=" * 80)
    print("COMPREHENSIVE CONTEXT EFFECTS EXPERIMENT")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Test cases: {config.data.test_cases_path}")
    print("=" * 80)
    
    try:
        # Setup
        experiment.setup()
        
        # Run
        results = experiment.run()
        
        # Save results
        experiment.save_results()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        print("Saving checkpoint...")
        experiment._save_checkpoint(experiment.last_processed_idx)
        print("Checkpoint saved. Run again to resume.")
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()