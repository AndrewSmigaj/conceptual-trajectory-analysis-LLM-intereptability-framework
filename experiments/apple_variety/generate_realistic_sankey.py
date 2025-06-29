"""Generate production D3 Sankey diagrams for realistic apple experiment."""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_fragmentation.visualization.d3_sankey import D3SankeyGenerator

def generate_realistic_sankey():
    """Generate the official full network sankey for realistic apple experiment."""
    
    # Load results
    results_path = Path("results/apple_realistic/apple_realistic_routing_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    trajectory_data = results.get('trajectory_analysis', {})
    cluster_results = results.get('cluster_results', {})
    
    # Get actual cluster labels from the data
    cluster_labels = {}
    
    # For each layer, create labels based on actual cluster compositions
    for layer_idx in range(4):  # 4 layers total
        layer_key = str(layer_idx)
        if layer_key not in cluster_results:
            continue
            
        n_clusters = cluster_results[layer_key]['n_clusters']
        cluster_labels[layer_idx] = {}
        
        # Get composition data
        comp = cluster_results[layer_key].get('composition', {})
        routing_comp = comp.get('routing', {})
        
        for cluster_id in range(n_clusters):
            # Count samples in this cluster
            cluster_samples = sum(1 for path in trajectory_data['paths'] 
                                if path[layer_idx] == cluster_id)
            
            # Determine label based on routing composition
            if str(cluster_id) in routing_comp:
                route_comp = routing_comp[str(cluster_id)]
                total = sum(route_comp.values())
                
                if total > 0:
                    # Calculate percentages
                    pct_premium = route_comp.get('fresh_premium', 0) / total * 100
                    pct_standard = route_comp.get('fresh_standard', 0) / total * 100
                    pct_juice = route_comp.get('juice', 0) / total * 100
                    
                    # Create descriptive label
                    if layer_idx == 0:
                        # More detailed labels for input layer
                        if pct_premium > 60:
                            label = f"Premium-Dominant ({cluster_samples})"
                        elif pct_standard > 60:
                            label = f"Standard-Dominant ({cluster_samples})"
                        elif pct_juice > 30:
                            label = f"Juice-Heavy ({cluster_samples})"
                        else:
                            label = f"Mixed Quality ({cluster_samples})"
                    else:
                        # Simpler labels for hidden layers
                        if cluster_samples > 500:
                            label = f"Majority Flow ({cluster_samples})"
                        else:
                            label = f"Minority Pattern ({cluster_samples})"
                else:
                    label = f"L{layer_idx}_C{cluster_id} ({cluster_samples})"
            else:
                label = f"L{layer_idx}_C{cluster_id} ({cluster_samples})"
                
            cluster_labels[layer_idx][cluster_id] = label
    
    # Apple routing classes and colors
    routing_classes = ["fresh_premium", "fresh_standard", "juice"]
    routing_colors = {
        "fresh_premium": "#2ecc71",    # Green - Premium apples
        "fresh_standard": "#3498db",   # Blue - Standard apples
        "juice": "#ff8c00"             # Orange - Juice apples
    }
    
    # Create generator
    generator = D3SankeyGenerator(config={
        'width': 1800,
        'height': 900,
        'top_n_paths': 25,
        'font_size': 12,
        'node_width': 30,
        'node_padding': 20,
        'margin': {'top': 50, 'right': 250, 'bottom': 50, 'left': 250}
    })
    
    # Generate full network sankey
    output_path = Path("results/apple_realistic/d3_sankey_full_network.html")
    
    generator.generate(
        trajectory_data=trajectory_data,
        cluster_results=cluster_results,
        output_path=output_path,
        title="Apple Quality Routing Analysis - Neural Network Concept Trajectories",
        subtitle="15 apple varieties tracked through 4-layer network (92.8% accuracy, $186.41 economic loss)",
        routing_classes=routing_classes,
        routing_colors=routing_colors,
        cluster_labels=cluster_labels,
        layer_names=["Input", "Hidden 1", "Hidden 2", "Output"],
        full_network=True,
        show_routing_composition=True,
        metadata={
            'experiment': 'apple_realistic',
            'n_varieties': 15,
            'n_samples': 1320,
            'test_accuracy': '92.8%',
            'economic_loss': '$186.41',
            'most_affected': 'Red Delicious (93.1% juice routing)'
        }
    )
    
    print(f"[SUCCESS] Full network D3 sankey created: {output_path}")
    
    # Also create JSON data file for embedding
    sankey_data = generator.create_sankey_data(
        paths=trajectory_data['paths'],
        cluster_results=cluster_results,
        layer_start=0,
        layer_end=3,
        cluster_labels=cluster_labels,
        routing_classes=routing_classes,
        show_routing_composition=True
    )
    
    # Add metadata
    sankey_data['metadata'] = {
        'total_samples': len(trajectory_data['paths']),
        'num_layers': 4,
        'routing_classes': routing_classes,
        'routing_colors': routing_colors,
        'layer_names': ["Input", "Hidden 1", "Hidden 2", "Output"]
    }
    
    json_path = Path("results/apple_realistic/d3_sankey_full_network_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sankey_data, f, indent=2)
    
    print(f"[SUCCESS] Sankey data JSON created: {json_path}")
    
    # Create standalone version with embedded data
    standalone_path = Path("results/apple_realistic/d3_sankey_full_network_standalone.html")
    
    # Read the generated HTML and embed the data
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # The HTML already has embedded data, so just copy it
    with open(standalone_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[SUCCESS] Standalone D3 sankey created: {standalone_path}")
    
    return True

if __name__ == "__main__":
    print("Generating production D3 Sankey diagrams for realistic apple experiment...")
    success = generate_realistic_sankey()
    
    if success:
        print("\n[SUCCESS] All sankey diagrams generated successfully!")
        print("\nFiles created:")
        print("  1. d3_sankey_full_network.html - Main visualization")
        print("  2. d3_sankey_full_network_data.json - Data file")
        print("  3. d3_sankey_full_network_standalone.html - Standalone version")