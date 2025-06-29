#!/usr/bin/env python3
"""
Prepare a subset of results focused on the contexts used in the paper.
"""

import json
from pathlib import Path
from comprehensive_contexts import get_analysis_contexts

def prepare_paper_subset(full_results_dir: str = "results_comprehensive", 
                        output_dir: str = "results_paper_subset"):
    """Extract subset of results for paper analysis."""
    
    # Get the contexts we're focusing on for the paper
    paper_contexts = get_analysis_contexts()
    print(f"Paper contexts ({len(paper_contexts)}): {paper_contexts}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load full trajectories
    full_traj_path = Path(full_results_dir) / "unified_trajectories_k20.json"
    if full_traj_path.exists():
        print(f"Loading trajectories from {full_traj_path}...")
        with open(full_traj_path, 'r') as f:
            full_data = json.load(f)
        
        # Filter trajectories
        filtered_trajectories = {}
        for key, traj in full_data.get('trajectories', {}).items():
            # Check if this trajectory is for one of our paper contexts
            for ctx in paper_contexts:
                if ctx in key or (ctx == "baseline" and "_baseline" in key):
                    filtered_trajectories[key] = traj
                    break
        
        print(f"Filtered from {len(full_data.get('trajectories', {}))} to {len(filtered_trajectories)} trajectories")
        
        # Create filtered data
        filtered_data = {
            'metadata': full_data.get('metadata', {}),
            'trajectories': filtered_trajectories
        }
        
        # Save filtered trajectories
        output_file = output_path / "paper_trajectories_k20.json"
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Saved filtered trajectories to {output_file}")
    
    return paper_contexts

if __name__ == "__main__":
    prepare_paper_subset()