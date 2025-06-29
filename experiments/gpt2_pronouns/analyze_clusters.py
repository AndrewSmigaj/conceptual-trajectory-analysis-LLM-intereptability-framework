import json
from collections import defaultdict

# Load trajectories
with open('results/early_trajectories.json', 'r') as f:
    trajectories = json.load(f)

# Analyze cluster assignments by pronoun and context type
pronoun_clusters = defaultdict(lambda: defaultdict(list))

for traj in trajectories:
    label_parts = traj['label'].split('_')
    pronoun = label_parts[0]
    context = label_parts[1] if len(label_parts) > 1 else 'baseline'
    context_type = traj['context_type']
    
    # Look at layer 2 (where bifurcation typically occurs)
    layer2_cluster = traj['path'][2]
    
    pronoun_clusters[pronoun][context_type].append(layer2_cluster)

# Print analysis
print("Layer 2 Cluster Assignments by Pronoun and Context Type:")
print("=" * 60)

for pronoun in ['she', 'he', 'they', 'it', 'we', 'I']:
    print(f"\n{pronoun}:")
    for context_type in ['neutral', 'function', 'content']:
        if context_type in pronoun_clusters[pronoun]:
            clusters = pronoun_clusters[pronoun][context_type]
            unique_clusters = set(clusters)
            print(f"  {context_type:10} -> clusters: {unique_clusters}, consistency: {len(unique_clusters)}/{len(clusters)}")

# Check if function vs content words route to different clusters
print("\n\nCluster Separation Analysis:")
print("=" * 60)

for pronoun in ['she', 'he', 'they', 'it', 'we', 'I']:
    function_clusters = set(pronoun_clusters[pronoun]['function'])
    content_clusters = set(pronoun_clusters[pronoun]['content'])
    overlap = function_clusters & content_clusters
    
    print(f"{pronoun}: function->{function_clusters}, content->{content_clusters}, overlap->{overlap}")