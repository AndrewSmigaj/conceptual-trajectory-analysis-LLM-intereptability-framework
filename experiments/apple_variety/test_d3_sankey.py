"""Test script to verify D3SankeyGenerator works with apple experiment data."""

import json
from pathlib import Path
import sys

# Add parent directory to path to import concept_fragmentation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_fragmentation.visualization.d3_sankey import D3SankeyGenerator

def test_d3_sankey_generator():
    """Test the generic D3SankeyGenerator with apple experiment data."""
    
    # Load realistic apple experiment results
    results_path = Path("results/apple_realistic/apple_realistic_routing_results.json")
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return False
        
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract trajectory and cluster data
    trajectory_data = results.get('trajectory_analysis', {})
    cluster_results = results.get('cluster_results', {})
    
    # Define apple-specific cluster labels
    cluster_labels = {
        0: {  # Layer 0
            0: "Mixed Standard (192 samples)",
            1: "Premium-Leaning (68 samples)", 
            2: "Standard Dominant (220 samples)",
            3: "Balanced Quality (74 samples)",
            4: "Lower Quality (39 samples)",
            5: "Standard Focus (58 samples)",
            6: "Moderate Quality (58 samples)",
            7: "High Standard (56 samples)",
            8: "Premium Potential (36 samples)",
            9: "Mixed Quality (43 samples)"
        },
        1: {  # Layer 1
            0: "Minority Patterns (79 samples)",
            1: "Majority Flow (765 samples)"
        },
        2: {  # Layer 2
            0: "Minority Patterns (75 samples)",
            1: "Majority Flow (769 samples)"
        },
        3: {  # Layer 3 (Output)
            0: "Minority Patterns (75 samples)",
            1: "Majority Flow (769 samples)"
        }
    }
    
    # Define routing classes and colors
    routing_classes = ["fresh_premium", "fresh_standard", "juice"]
    routing_colors = {
        "fresh_premium": "#2ecc71",  # Green
        "fresh_standard": "#3498db",  # Blue
        "juice": "#ff8c00"           # Orange
    }
    
    # Create D3SankeyGenerator instance
    generator = D3SankeyGenerator(config={
        'width': 1600,
        'height': 800,
        'top_n_paths': 25,
        'font_size': 12
    })
    
    # Generate full network sankey
    try:
        output_path = Path("results/apple_realistic/test_full_network_d3_sankey.html")
        
        generator.generate(
            trajectory_data=trajectory_data,
            cluster_results=cluster_results,
            output_path=output_path,
            title="Apple Quality Routing Analysis - Full Network",
            subtitle="Neural Network Concept Trajectories through Quality Classifications",
            routing_classes=routing_classes,
            routing_colors=routing_colors,
            cluster_labels=cluster_labels,
            full_network=True,
            show_routing_composition=True,
            metadata={
                'experiment': 'apple_realistic',
                'accuracy': '92.8%',
                'total_loss': '$186.41'
            }
        )
        
        print(f"[SUCCESS] Full network D3 sankey created: {output_path}")
        
        # Also test windowed view
        window_output = Path("results/apple_realistic/test_window_d3_sankey.html")
        generator.generate(
            trajectory_data=trajectory_data,
            cluster_results=cluster_results,
            output_path=window_output,
            title="Apple Quality Routing - Early Layers",
            routing_classes=routing_classes,
            routing_colors=routing_colors,
            cluster_labels=cluster_labels,
            full_network=False,
            window=(0, 1),
            show_routing_composition=True
        )
        
        print(f"[SUCCESS] Windowed D3 sankey created: {window_output}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error generating sankey: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing D3SankeyGenerator with apple experiment data...")
    success = test_d3_sankey_generator()
    
    if success:
        print("\n[SUCCESS] All tests passed! D3SankeyGenerator works with apple data.")
    else:
        print("\n[ERROR] Tests failed!")