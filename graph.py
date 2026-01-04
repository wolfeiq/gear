import json
import networkx as nx
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

class InterventionGraphBuilder:
    
    def __init__(self):
        self.personal_graphs = {}
        self.global_graph = nx.DiGraph()
        self.intervention_graph = nx.DiGraph()
    
    def build_personal_graph(self, user_journey: dict) -> nx.DiGraph:
        G = nx.DiGraph()
        user_id = user_journey['user_profile']['user_id']
        
        distortion_pairs = defaultdict(int)
        intervention_effectiveness = defaultdict(lambda: {'uses': 0, 'severity_changes': []})
        
        for i, entry in enumerate(user_journey['entries']):
            distortions_in_entry = [d['type'] for d in entry['distortions']]

            for dist in entry['distortions']:
                if not G.has_node(dist['type']):
                    G.add_node(dist['type'], node_type='distortion', occurrences=0, total_confidence=0.0)
                
                G.nodes[dist['type']]['occurrences'] += 1
                G.nodes[dist['type']]['total_confidence'] += dist['confidence']

            for idx, d1 in enumerate(distortions_in_entry):
                for d2 in distortions_in_entry[idx+1:]:
                    distortion_pairs[(d1, d2)] += 1

            if 'interventions_used' in entry and entry['interventions_used']:
                for intervention in entry['interventions_used']:
                    if not G.has_node(intervention):
                        G.add_node(intervention, node_type='intervention', uses=0)
                    
                    G.nodes[intervention]['uses'] += 1

                    for distortion in distortions_in_entry:
                        edge_key = (intervention, distortion)
                        if G.has_edge(intervention, distortion):
                            G[intervention][distortion]['weight'] += 1
                        else:
                            G.add_edge(intervention, distortion, weight=1, edge_type='addresses')
                    if i < len(user_journey['entries']) - 1:
                        severity_change = (
                            user_journey['entries'][i+1]['measured_severity'] - 
                            entry['measured_severity']
                        )
                        intervention_effectiveness[intervention]['severity_changes'].append(severity_change)
                    intervention_effectiveness[intervention]['uses'] += 1

        for (source, target), count in distortion_pairs.items():
            if count >= 2:
                G.add_edge(source, target, weight=count, edge_type='co_occurs')

        for intervention, data in intervention_effectiveness.items():
            if data['severity_changes']:
                avg_change = sum(data['severity_changes']) / len(data['severity_changes'])
                G.nodes[intervention]['avg_severity_change'] = avg_change
                G.nodes[intervention]['effectiveness_score'] = -avg_change
        
        G.graph.update({
            'user_id': user_id,
            'journey_type': user_journey['journey_type'],
            'initial_severity': user_journey['initial_severity'],
            'final_severity': user_journey['final_severity'],
            'improvement': user_journey['improvement'],
            'interventions_assigned': user_journey['user_profile'].get('assigned_interventions', [])
        })
        
        return G
    
    def build_global_intervention_graph(self, all_journeys: list):
        G = nx.DiGraph()
        
        distortion_cooccurrence = defaultdict(int)
        intervention_distortion_links = defaultdict(lambda: defaultdict(int))
        intervention_outcomes = defaultdict(lambda: {'total_improvement': 0, 'count': 0})
        distortion_counts = defaultdict(int)
        distortion_confidence = defaultdict(float)
        
        for journey in all_journeys:
            improvement = journey['improvement']
            
            for entry in journey['entries']:
                distortions = [d['type'] for d in entry['distortions']]
                interventions = entry.get('interventions_used', [])
                

                for d in entry['distortions']:
                    distortion_counts[d['type']] += 1
                    distortion_confidence[d['type']] += d['confidence']
                

                for i, d1 in enumerate(distortions):
                    for d2 in distortions[i+1:]:
                        edge = tuple(sorted([d1, d2]))
                        distortion_cooccurrence[edge] += 1
                
   
                for intervention in interventions:
                    for distortion in distortions:
                        intervention_distortion_links[intervention][distortion] += 1
                    
                    if improvement > 0:
                        intervention_outcomes[intervention]['total_improvement'] += improvement
                        intervention_outcomes[intervention]['count'] += 1
        
        all_distortions = set()
        all_interventions = set()
        
        for journey in all_journeys:
            for entry in journey['entries']:
                for d in entry['distortions']:
                    all_distortions.add(d['type'])
                if 'interventions_used' in entry:
                    all_interventions.update(entry['interventions_used'])

        for dist in all_distortions:
            G.add_node(
                dist, 
                node_type='distortion',
                total_occurrences=distortion_counts[dist],
                total_confidence=distortion_confidence[dist]
            )
  
        for intervention in all_interventions:
            avg_improvement = 0
            if intervention_outcomes[intervention]['count'] > 0:
                avg_improvement = (
                    intervention_outcomes[intervention]['total_improvement'] / 
                    intervention_outcomes[intervention]['count']
                )
            G.add_node(intervention, node_type='intervention', avg_improvement=avg_improvement)

        threshold = len(all_journeys) * 0.1
        for (d1, d2), count in distortion_cooccurrence.items():
            if count >= threshold:
                G.add_edge(d1, d2, weight=count, edge_type='co_occurs')
        
        for intervention, distortions in intervention_distortion_links.items():
            for distortion, count in distortions.items():
                if count >= 2:
                    G.add_edge(
                        intervention, 
                        distortion, 
                        weight=count, 
                        edge_type='targets',
                        effectiveness=intervention_outcomes[intervention]['total_improvement'] / 
                                     intervention_outcomes[intervention]['count'] 
                                     if intervention_outcomes[intervention]['count'] > 0 else 0
                    )
        
        self.global_graph = G
        return G
    
    def visualize_intervention_graph(self, G: nx.Graph, title: str, output_file: str):
        if len(G.nodes()) == 0:
            return
        
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        distortion_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'distortion']
        intervention_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'intervention']
        

        distortion_sizes = [G.nodes[n].get('occurrences', G.nodes[n].get('total_occurrences', 1)) * 200 for n in distortion_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=distortion_nodes,
            node_size=distortion_sizes,
            node_color='lightblue',
            alpha=0.7,
            label='Distortions'
        )

        intervention_sizes = [
            (1 + abs(G.nodes[n].get('avg_improvement', 0)) * 5) * 300 
            for n in intervention_nodes
        ]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=intervention_nodes,
            node_size=intervention_sizes,
            node_color='lightgreen',
            alpha=0.8,
            label='Interventions'
        )

        co_occur_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') == 'co_occurs']
        co_occur_weights = [G[u][v]['weight'] for u, v in co_occur_edges]
        
        if co_occur_weights:
            max_weight = max(co_occur_weights)
            co_occur_widths = [1 + (w / max_weight) * 4 for w in co_occur_weights]
            
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=co_occur_edges, 
                width=co_occur_widths,
                alpha=0.4, 
                edge_color='gray',
                arrows=False
            )

        targets_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') in ['targets', 'addresses']]
        target_weights = [G[u][v]['weight'] for u, v in targets_edges]
        
        if target_weights:
            max_target_weight = max(target_weights)
            target_widths = [1 + (w / max_target_weight) * 3 for w in target_weights]
            
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=targets_edges, 
                width=target_widths,
                alpha=0.7, 
                edge_color='green',
                arrows=True, 
                arrowsize=20,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1'
            )
        

        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

        if co_occur_edges:
            top_edges = sorted(co_occur_edges, key=lambda e: G[e[0]][e[1]]['weight'], reverse=True)[:15]
            edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in top_edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_color='red')
        
        plt.title(title, fontsize=18, fontweight='bold')
        plt.legend(loc='upper left', fontsize=12)

        explanation = """
        Node sizes: Distortions = occurrence frequency | Interventions = effectiveness
        Gray edges: Distortion co-occurrence (thicker = stronger relationship)
        Green arrows: Intervention targets distortion (thicker = more applications)
        Red numbers: Co-occurrence counts for strongest relationships
        """
        plt.text(0.02, 0.02, explanation, transform=plt.gcf().transFigure, 
                fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        

        if co_occur_edges:
            sorted_edges = sorted(co_occur_edges, key=lambda e: G[e[0]][e[1]]['weight'], reverse=True)[:10]
            for u, v in sorted_edges:
                print(f"  {u} ↔ {v}: {G[u][v]['weight']} times")
    
    def process_all_journeys(self, journeys: list):
 
        
        for journey in journeys:
            user_id = journey['user_profile']['user_id']
            personal_graph = self.build_personal_graph(journey)
            self.personal_graphs[user_id] = personal_graph
        
        self.build_global_intervention_graph(journeys)
    
    def export_for_gear(self, output_file: str):
        export_data = {
            'personal_graphs': {},
            'global_graph': {},
            'intervention_effectiveness': {},
            'metadata': {
                'created': datetime.now().isoformat(),
                'num_users': len(self.personal_graphs)
            }
        }

        for user_id, G in self.personal_graphs.items():
            export_data['personal_graphs'][user_id] = {
                'nodes': [
                    {
                        'id': node,
                        'type': G.nodes[node].get('node_type'),
                        **{k: v for k, v in G.nodes[node].items() if k != 'node_type'}
                    }
                    for node in G.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        **G[u][v]
                    }
                    for u, v in G.edges()
                ],
                'metadata': dict(G.graph)
            }

        export_data['global_graph'] = {
            'nodes': [
                {
                    'id': node,
                    'type': self.global_graph.nodes[node].get('node_type'),
                    **{k: v for k, v in self.global_graph.nodes[node].items() if k != 'node_type'}
                }
                for node in self.global_graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **self.global_graph[u][v]
                }
                for u, v in self.global_graph.edges()
            ]
        }

        for node in self.global_graph.nodes():
            if self.global_graph.nodes[node].get('node_type') == 'intervention':
                export_data['intervention_effectiveness'][node] = {
                    'avg_improvement': self.global_graph.nodes[node].get('avg_improvement', 0),
                    'targets': [
                        v for u, v in self.global_graph.edges() 
                        if u == node and self.global_graph[u][v].get('edge_type') == 'targets'
                    ]
                }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

if __name__ == "__main__":
    
    with open('synthetic_longitudinal_with_interventions.json', 'r') as f:
        journeys = json.load(f)
    
    
    builder = InterventionGraphBuilder()
    builder.process_all_journeys(journeys)
    sample_user_id = list(builder.personal_graphs.keys())[0]
    
    builder.visualize_intervention_graph(
        builder.personal_graphs[sample_user_id],
        f"Distortions + Interventions - {sample_user_id}",
        f"viz_interventions_{sample_user_id}.png"
    )
    
    builder.visualize_intervention_graph(
        builder.global_graph,
        "Global: Distortions, Interventions & Effectiveness",
        "viz_global_interventions.png"
    )
    
    builder.export_for_gear('graphs_with_interventions.json')