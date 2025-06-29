#!/usr/bin/env python3
"""
Run comprehensive context effects experiment with expanded tokens and contexts.
Simple version without complex imports.
"""

import sys
import logging
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))  # Add current directory first
sys.path.insert(0, str(current_dir.parent.parent))  # Add project root

from unified_context_experiment import UnifiedContextExperiment

def main():
    """Run the comprehensive experiment."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration path
    config_path = Path(__file__).parent / "config_comprehensive.yaml"
    
    # Initialize experiment with config path
    experiment = UnifiedContextExperiment(config_path=str(config_path))
    
    # Run experiment
    print("=" * 80)
    print("COMPREHENSIVE CONTEXT EFFECTS EXPERIMENT")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Test cases: expanded_test_cases.json")
    print("=" * 80)
    
    try:
        # Setup (if method exists)
        if hasattr(experiment, 'setup'):
            experiment.setup()
        
        # Run
        results = experiment.run()
        
        # Save results (if method exists)
        if hasattr(experiment, 'save_results'):
            experiment.save_results()
        
        print("\nExperiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        print("Results saved up to interruption point")
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()