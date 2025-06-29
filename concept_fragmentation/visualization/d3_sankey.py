"""D3.js-based Sankey diagram generator for concept trajectory analysis.

This module provides a generic D3 Sankey visualization generator that can be used
with any dataset and neural network architecture.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class D3SankeyGenerator:
    """Generic D3.js Sankey diagram generator for concept trajectory analysis.
    
    This class generates interactive D3-based Sankey diagrams showing how
    data flows through neural network layers via cluster trajectories.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration.
        
        Args:
            config: Dict with visualization settings
                - width: Chart width (default: 1600)
                - height: Chart height (default: 800)
                - node_width: Width of nodes (default: 30)
                - node_padding: Vertical padding between nodes (default: 15)
                - top_n_paths: Number of top paths to show (default: 25)
                - font_size: Base font size (default: 12)
                - margin: Dict with top, right, bottom, left margins
        """
        self.config = config or {}
        self.width = self.config.get('width', 1600)
        self.height = self.config.get('height', 800)
        self.node_width = self.config.get('node_width', 30)
        self.node_padding = self.config.get('node_padding', 15)
        self.top_n_paths = self.config.get('top_n_paths', 25)
        self.font_size = self.config.get('font_size', 12)
        self.margin = self.config.get('margin', {
            'top': 50, 'right': 200, 'bottom': 50, 'left': 200
        })
        
    def generate(self,
                 trajectory_data: Dict[str, Any],
                 cluster_results: Dict[str, Any],
                 output_path: Optional[Union[str, Path]] = None,
                 title: str = "Concept Trajectory Analysis",
                 subtitle: Optional[str] = None,
                 routing_classes: Optional[List[str]] = None,
                 routing_colors: Optional[Dict[str, str]] = None,
                 cluster_labels: Optional[Dict[Union[int, str], Dict[int, str]]] = None,
                 path_descriptions: Optional[Dict[Tuple[int, ...], str]] = None,
                 layer_names: Optional[List[str]] = None,
                 full_network: bool = True,
                 window: Optional[Tuple[int, int]] = None,
                 show_routing_composition: bool = True,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate D3 Sankey visualization.
        
        Args:
            trajectory_data: Dict with 'paths' key containing list of paths
            cluster_results: Dict with cluster info per layer
            output_path: Where to save HTML (if None, returns HTML string)
            title: Chart title
            subtitle: Optional subtitle
            routing_classes: List of class names (e.g., ["class1", "class2"])
            routing_colors: Dict mapping classes to colors
            cluster_labels: Optional dict of custom cluster labels
                Format: {layer_idx: {cluster_idx: "Custom Label"}}
            path_descriptions: Optional dict mapping path tuples to descriptions
            layer_names: Optional list of layer names (default: ["L0", "L1", ...])
            full_network: If True, show all layers. If False, use window.
            window: Tuple of (start_layer, end_layer) if not full_network
            show_routing_composition: Whether to show routing percentages in tooltips
            metadata: Optional metadata to include in visualization
            
        Returns:
            HTML string if output_path is None, otherwise None
        """
        # Extract paths
        paths = trajectory_data.get('paths', trajectory_data.get('trajectory_analysis', {}).get('paths', []))
        if not paths:
            raise ValueError("No paths found in trajectory data")
            
        # Determine layers to visualize
        if full_network:
            num_layers = len(paths[0]) if paths else 0
            layer_start, layer_end = 0, num_layers - 1
        else:
            if window is None:
                raise ValueError("Window must be specified when full_network=False")
            layer_start, layer_end = window
            
        # Set default layer names if not provided
        if layer_names is None:
            layer_names = [f"L{i}" for i in range(len(paths[0]) if paths else 0)]
            
        # Create sankey data
        sankey_data = self.create_sankey_data(
            paths=paths,
            cluster_results=cluster_results,
            layer_start=layer_start,
            layer_end=layer_end,
            cluster_labels=cluster_labels,
            routing_classes=routing_classes,
            show_routing_composition=show_routing_composition
        )
        
        # Add metadata
        sankey_data['metadata'] = metadata or {}
        sankey_data['metadata'].update({
            'total_samples': len(paths),
            'num_layers': layer_end - layer_start + 1,
            'layer_names': layer_names[layer_start:layer_end+1] if layer_names else None
        })
        
        # Generate HTML
        html_content = self.generate_html(
            sankey_data=sankey_data,
            title=title,
            subtitle=subtitle,
            routing_colors=routing_colors,
            path_descriptions=path_descriptions,
            layer_start=layer_start,
            layer_end=layer_end,
            layer_names=layer_names
        )
        
        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"D3 Sankey diagram saved to: {output_path}")
            return None
        else:
            return html_content
            
    def create_sankey_data(self,
                          paths: List[List[int]],
                          cluster_results: Dict[str, Any],
                          layer_start: int,
                          layer_end: int,
                          cluster_labels: Optional[Dict] = None,
                          routing_classes: Optional[List[str]] = None,
                          show_routing_composition: bool = True) -> Dict[str, Any]:
        """Convert trajectory data to D3 sankey format.
        
        Returns dict with 'nodes' and 'links' for D3.
        """
        # Filter paths for the window
        window_paths = []
        path_counts = defaultdict(int)
        
        for path in paths:
            window_path = tuple(path[layer_start:layer_end+1])
            path_counts[window_path] += 1
            window_paths.append(window_path)
        
        # Get top paths
        top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:self.top_n_paths]
        
        # Build nodes
        nodes = []
        node_map = {}
        
        for layer_idx in range(layer_start, layer_end + 1):
            layer_key = str(layer_idx)
            if layer_key not in cluster_results:
                logger.warning(f"Layer {layer_idx} not found in cluster results")
                continue
                
            n_clusters = cluster_results[layer_key].get('n_clusters', 0)
            
            for cluster_id in range(n_clusters):
                # Determine node label
                if cluster_labels and layer_idx in cluster_labels:
                    label = cluster_labels[layer_idx].get(cluster_id, f"L{layer_idx}_C{cluster_id}")
                else:
                    # Count samples in this cluster from top paths
                    cluster_count = sum(count for path, count in top_paths 
                                      if layer_idx - layer_start < len(path) and 
                                      path[layer_idx - layer_start] == cluster_id)
                    label = f"L{layer_idx}_C{cluster_id} ({cluster_count} samples)"
                
                # Create node
                node_id = f"L{layer_idx}_C{cluster_id}"
                node_data = {
                    'id': node_id,
                    'name': label,
                    'layer': layer_idx,
                    'cluster': cluster_id,
                    'layer_position': layer_idx - layer_start
                }
                
                # Add routing composition if available
                if show_routing_composition and 'composition' in cluster_results[layer_key]:
                    comp = cluster_results[layer_key]['composition']
                    if 'routing' in comp and str(cluster_id) in comp['routing']:
                        routing_comp = comp['routing'][str(cluster_id)]
                        total = sum(routing_comp.values())
                        if total > 0 and routing_classes:
                            percentages = {}
                            for cls in routing_classes:
                                pct = routing_comp.get(cls, 0) / total * 100
                                percentages[cls] = pct
                            node_data['routing_distribution'] = percentages
                
                nodes.append(node_data)
                node_map[(layer_idx, cluster_id)] = node_id
        
        # Build links
        links = []
        link_map = {}  # To aggregate duplicate links
        
        for path_tuple, count in top_paths:
            for i in range(len(path_tuple) - 1):
                source_layer = layer_start + i
                target_layer = layer_start + i + 1
                source_cluster = path_tuple[i]
                target_cluster = path_tuple[i + 1]
                
                source_id = node_map.get((source_layer, source_cluster))
                target_id = node_map.get((target_layer, target_cluster))
                
                if source_id and target_id:
                    link_key = (source_id, target_id)
                    if link_key in link_map:
                        link_map[link_key]['value'] += count
                    else:
                        link_map[link_key] = {
                            'source': source_id,
                            'target': target_id,
                            'value': count,
                            'path_id': len(path_tuple)  # Can be used for coloring
                        }
        
        links = list(link_map.values())
        
        return {
            'nodes': nodes,
            'links': links
        }
        
    def generate_html(self,
                     sankey_data: Dict[str, Any],
                     title: str,
                     subtitle: Optional[str],
                     routing_colors: Optional[Dict[str, str]],
                     path_descriptions: Optional[Dict],
                     layer_start: int,
                     layer_end: int,
                     layer_names: Optional[List[str]]) -> str:
        """Generate the HTML with embedded D3 visualization."""
        
        # Default colors if not provided
        if routing_colors is None:
            routing_colors = {
                'default': '#95a5a6',
                'primary': '#3498db',
                'secondary': '#2ecc71',
                'tertiary': '#e74c3c'
            }
        
        # Build legend items from routing classes
        legend_items = []
        if routing_colors:
            for cls, color in routing_colors.items():
                if cls != 'default':
                    legend_items.append(f'<div class="legend-item"><div class="legend-color" style="background-color: {color};"></div><span>{cls}</span></div>')
        
        legend_html = f'<div id="legend">{" ".join(legend_items)}</div>' if legend_items else ''
        
        # Subtitle HTML
        subtitle_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ''
        
        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        #container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin: 0 auto;
            max-width: {self.width + 100}px;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        
        #legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        
        .node rect {{
            cursor: pointer;
        }}
        
        .node text {{
            font-size: {self.font_size}px;
            pointer-events: none;
        }}
        
        .link {{
            fill: none;
            stroke-opacity: 0.5;
        }}
        
        .link:hover {{
            stroke-opacity: 0.8;
        }}
        
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 10px;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .path-label {{
            font-size: 11px;
            fill: #666;
        }}
        
        .layer-label {{
            font-size: 14px;
            font-weight: bold;
            fill: #333;
        }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>
</head>
<body>
    <div id="container">
        <h1>{title}</h1>
        {subtitle_html}
        {legend_html}
        <svg id="sankey"></svg>
        <div class="tooltip"></div>
    </div>

    <script>
        // Configuration
        const config = {{
            width: {self.width},
            height: {self.height},
            nodeWidth: {self.node_width},
            nodePadding: {self.node_padding},
            margin: {json.dumps(self.margin)}
        }};
        
        // Data
        const data = {json.dumps(sankey_data)};
        
        // Create visualization
        createSankey(data);
        
        function createSankey(data) {{
            // Set up dimensions
            const width = config.width - config.margin.left - config.margin.right;
            const height = config.height - config.margin.top - config.margin.bottom;
            
            // Create SVG
            const svg = d3.select('#sankey')
                .attr('width', config.width)
                .attr('height', config.height);
                
            const g = svg.append('g')
                .attr('transform', `translate(${{config.margin.left}},${{config.margin.top}})`);
            
            // Create tooltip
            const tooltip = d3.select('.tooltip');
            
            // Create sankey generator
            const sankey = d3.sankey()
                .nodeId(d => d.id)
                .nodeAlign(d3.sankeyJustify)
                .nodeWidth(config.nodeWidth)
                .nodePadding(config.nodePadding)
                .extent([[0, 0], [width, height]]);
            
            // Generate layout
            const {{nodes, links}} = sankey(data);
            
            // Path colors for unique trajectories
            const pathColors = [
                '#FF6347', '#1E90FF', '#32CD32', '#FFD700', '#8A2BE2',
                '#FF8C00', '#00CED1', '#FF1493', '#9ACD32', '#DB7093',
                '#6495ED', '#FFB6C1', '#90EE90', '#FFA07A', '#B0C4DE',
                '#DC143C', '#4B0082', '#FF7F50', '#008080', '#F08080',
                '#20B2AA', '#FA8072', '#00BFFF', '#7FFF00', '#FF00FF'
            ];
            
            // Routing class colors
            const colors = {json.dumps(routing_colors) if routing_colors else '{}'};
            
            // Add unique path IDs to links
            const pathMap = new Map();
            let pathId = 0;
            links.forEach(d => {{
                const key = `${{d.source}}-${{d.target}}`;
                if (!pathMap.has(key)) {{
                    pathMap.set(key, pathId++);
                }}
                d.path_id = pathMap.get(key);
            }});
            
            // Add links with gradient colors
            const link = g.append('g')
                .attr('fill', 'none')
                .selectAll('.link')
                .data(links)
                .enter().append('path')
                .attr('class', 'link')
                .attr('d', d3.sankeyLinkHorizontal())
                .attr('stroke', d => pathColors[d.path_id % pathColors.length])
                .attr('stroke-width', d => Math.max(1, d.width))
                .attr('stroke-opacity', 0.5)
                .on('mouseover', function(event, d) {{
                    d3.select(this).attr('stroke-opacity', 0.8);
                    tooltip.transition()
                        .duration(200)
                        .style('opacity', .9);
                    
                    let tooltipContent = `<strong>Path Segment</strong><br/>`;
                    tooltipContent += `${{d.source.name}} → ${{d.target.name}}<br/>`;
                    tooltipContent += `Samples: ${{d.value}}<br/>`;
                    
                    tooltip.html(tooltipContent)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                }})
                .on('mouseout', function() {{
                    d3.select(this).attr('stroke-opacity', 0.5);
                    tooltip.transition()
                        .duration(500)
                        .style('opacity', 0);
                }});
            
            // Create nodes as stacked bars
            const node = g.append('g')
                .selectAll('.node')
                .data(nodes)
                .enter().append('g')
                .attr('class', 'node');
            
            // For each node, create stacked bars for routing distributions
            node.each(function(d) {{
                const nodeGroup = d3.select(this);
                
                // Check if we have routing distribution data
                if (d.routing_distribution && Object.keys(d.routing_distribution).length > 0) {{
                    const distribution = d.routing_distribution;
                    const routingClasses = Object.keys(distribution).sort();
                    
                    // Calculate cumulative heights
                    let cumHeight = 0;
                    const segments = routingClasses.map(cls => {{
                        const height = d.y1 - d.y0;
                        const pct = distribution[cls] / 100;  // Convert percentage to fraction
                        const segmentHeight = height * pct;
                        const segment = {{
                            class: cls,
                            y0: d.y0 + cumHeight,
                            y1: d.y0 + cumHeight + segmentHeight,
                            percentage: distribution[cls]
                        }};
                        cumHeight += segmentHeight;
                        return segment;
                    }});
                    
                    // Draw segments
                    nodeGroup.selectAll('.segment')
                        .data(segments)
                        .enter().append('rect')
                        .attr('class', 'segment')
                        .attr('x', d.x0)
                        .attr('y', seg => seg.y0)
                        .attr('height', seg => seg.y1 - seg.y0)
                        .attr('width', d.x1 - d.x0)
                        .attr('fill', seg => colors[seg.class] || '#95a5a6')
                        .on('mouseover', function(event, seg) {{
                            tooltip.transition().duration(200).style('opacity', .9);
                            let content = `<strong>${{d.name}}</strong><br/>`;
                            if (d.layer !== undefined) content += `Layer: ${{d.layer}}<br/>`;
                            if (d.total_samples !== undefined) content += `Total Samples: ${{d.total_samples}}<br/>`;
                            content += `<br/><strong>Routing Distribution:</strong><br/>`;
                            for (const [cls, pct] of Object.entries(distribution)) {{
                                content += `${{cls}}: ${{pct.toFixed(1)}}%<br/>`;
                            }}
                            tooltip.html(content)
                                .style('left', (event.pageX + 10) + 'px')
                                .style('top', (event.pageY - 28) + 'px');
                        }})
                        .on('mouseout', function() {{
                            tooltip.transition().duration(500).style('opacity', 0);
                        }});
                }} else {{
                    // Fallback to single rectangle if no distribution data
                    nodeGroup.append('rect')
                        .attr('x', d.x0)
                        .attr('y', d.y0)
                        .attr('height', d.y1 - d.y0)
                        .attr('width', d.x1 - d.x0)
                        .attr('fill', '#95a5a6')
                        .on('mouseover', function(event) {{
                            tooltip.transition()
                                .duration(200)
                                .style('opacity', .9);
                            tooltip.html(`<strong>${{d.name}}</strong>`)
                                .style('left', (event.pageX + 10) + 'px')
                                .style('top', (event.pageY - 28) + 'px');
                        }})
                        .on('mouseout', function() {{
                            tooltip.transition()
                                .duration(500)
                                .style('opacity', 0);
                        }});
                }}
                
                // Add node labels
                nodeGroup.append('text')
                    .attr('x', d.x0 - 6)
                    .attr('y', (d.y0 + d.y1) / 2)
                    .attr('dy', '0.35em')
                    .attr('text-anchor', 'end')
                    .attr('font-size', '11px')
                    .text(d => d.name)
                    .filter(d => d.x0 < width / 2)
                    .attr('x', d => d.x1 + 6)
                    .attr('text-anchor', 'start');
            }});
                
            // Add layer labels
            const layerLabels = {json.dumps(layer_names[layer_start:layer_end+1] if layer_names else [])} || 
                               d3.range({layer_end - layer_start + 1}).map(i => `Layer ${{i}}`);
            
            // Group nodes by layer position to find x coordinates
            const layerPositions = d3.rollup(
                nodes,
                v => v[0].x0 + config.nodeWidth / 2,
                d => d.layer_position
            );
            
            const layerData = Array.from(layerPositions, ([layer, x]) => ({{
                layer: layer,
                x: x
            }}));
                
            g.append('g')
                .selectAll('.layer-label')
                .data(layerData)
                .join('text')
                .attr('class', 'layer-label')
                .attr('x', d => d.x)
                .attr('y', -10)
                .attr('text-anchor', 'middle')
                .text(d => layerLabels[d.layer]);
        }}
    </script>
</body>
</html>"""
        
        return html_content