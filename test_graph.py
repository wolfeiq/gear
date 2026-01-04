import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def build_test_graph():

    try:
        with open('test_results.json', 'r') as f:
            data = json.load(f)

    except FileNotFoundError:

        return
 
    G = nx.Graph()
    
    distortion_pairs = defaultdict(int)
    distortion_counts = defaultdict(int)

    
    for item in data:
        distortions = [d['type'] for d in item['distortions']]

        for d in distortions:
            distortion_counts[d] += 1

        for i, d1 in enumerate(distortions):
            for d2 in distortions[i+1:]:
                pair = tuple(sorted([d1, d2]))
                distortion_pairs[pair] += 1
    for dist, count in distortion_counts.items():
        G.add_node(dist, count=count)
    

    for (d1, d2), count in distortion_pairs.items():
        G.add_edge(d1, d2, weight=count)
    
    
    print("\nNode degrees (how connected each distortion is):")
    degrees = dict(G.degree())
    for node, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {degree} connections")
    
    print("\nMost common distortion:")
    max_dist = max(distortion_counts.items(), key=lambda x: x[1])
    print(f"  {max_dist[0]}: {max_dist[1]} occurrences")
    
    print("\nStrongest co-occurrence:")
    if distortion_pairs:
        max_pair = max(distortion_pairs.items(), key=lambda x: x[1])
        print(f"  {max_pair[0][0]} + {max_pair[0][1]}: {max_pair[1]} times")
    
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    node_sizes = [G.nodes[node]['count'] * 500 for node in G.nodes()]
    
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        alpha=0.7
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.5
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold'
    )
    
    plt.title("Test Distortion Co-occurrence Network", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = 'test_graph_viz.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    graph_data = {
        'nodes': [
            {'id': node, 'count': G.nodes[node]['count']}
            for node in G.nodes()
        ],
        'edges': [
            {'source': u, 'target': v, 'weight': G[u][v]['weight']}
            for u, v in G.edges()
        ]
    }
    
    with open('test_graph_data.json', 'w') as f:
        json.dump(graph_data, f, indent=2)


if __name__ == "__main__":
    build_test_graph()