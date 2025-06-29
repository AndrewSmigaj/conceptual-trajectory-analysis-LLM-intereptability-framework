"""
Inspect trajectories to verify the 100% divergence finding.
"""

import json
from pathlib import Path

def inspect_trajectories():
    """Look at various tokens to understand the divergence pattern."""
    # Load trajectories
    with open("results_unified/unified_trajectories_k10.json", 'r') as f:
        data = json.load(f)
        trajectories = data['trajectories']
    
    # Group by token
    token_groups = {}
    for key, traj in trajectories.items():
        token_idx = traj['token_idx']
        if token_idx not in token_groups:
            token_groups[token_idx] = []
        token_groups[token_idx].append(traj)
    
    # Look at different types of tokens
    print("Inspecting trajectory divergence patterns:\n")
    
    # Check specific interesting tokens
    interesting_tokens = [
        0,    # " the" 
        10,   # some other common token
        100,  # mid-frequency
        500,  # another token
        999   # last token
    ]
    
    for token_idx in interesting_tokens:
        if token_idx in token_groups:
            print(f"\nToken index {token_idx}:")
            
            # Sort by context for consistent display
            token_trajs = sorted(token_groups[token_idx], key=lambda x: x['context_frame'])
            
            # Show all trajectories for this token
            for traj in token_trajs:
                print(f"  {traj['token_str']:10s} ({traj['context_frame']:15s}): {traj['path']}")
            
            # Calculate divergence between baseline and others
            baseline_path = None
            for traj in token_trajs:
                if traj['context_frame'] == 'baseline':
                    baseline_path = traj['path']
                    break
            
            if baseline_path:
                print(f"  Layer-wise divergence from baseline:")
                for traj in token_trajs:
                    if traj['context_frame'] != 'baseline':
                        diffs = []
                        for i in range(len(baseline_path)):
                            if baseline_path[i] != traj['path'][i]:
                                diffs.append(i)
                        print(f"    {traj['context_frame']:15s}: differs at layers {diffs}")
    
    # Statistical summary
    print("\n\nStatistical Summary:")
    
    # Count tokens with any stable trajectory
    stable_tokens = 0
    partial_stable = 0
    
    for token_idx, token_trajs in token_groups.items():
        # Find baseline
        baseline_path = None
        for traj in token_trajs:
            if traj['context_frame'] == 'baseline':
                baseline_path = traj['path']
                break
        
        if baseline_path:
            # Check each context
            has_stable = False
            has_partial = False
            
            for traj in token_trajs:
                if traj['context_frame'] != 'baseline':
                    # Count matching layers
                    matches = sum(1 for i in range(len(baseline_path)) 
                                 if baseline_path[i] == traj['path'][i])
                    
                    if matches == len(baseline_path):
                        has_stable = True
                    elif matches > 6:  # More than half
                        has_partial = True
            
            if has_stable:
                stable_tokens += 1
            elif has_partial:
                partial_stable += 1
    
    print(f"  Tokens with at least one stable context: {stable_tokens}")
    print(f"  Tokens with partially stable contexts (>6 layers): {partial_stable}")
    print(f"  Total unique tokens: {len(token_groups)}")

if __name__ == "__main__":
    inspect_trajectories()