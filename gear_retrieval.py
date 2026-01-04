"""
GeAR: Graph-Enhanced Agent for Retrieval
Core retrieval algorithms for cognitive distortion patterns
"""

import json
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np

class GeARRetrieval:
    """
    Graph-enhanced retrieval for cognitive distortion interventions
    """
    
    def __init__(self, graphs_file: str):
        """Load graph data"""
        with open(graphs_file, 'r') as f:
            data = json.load(f)
        
        self.personal_graphs = {}
        self.global_graph = self._build_networkx_graph(
            data['global_graph']['nodes'],
            data['global_graph']['edges']
        )
        
        # Build personal graphs
        for user_id, user_data in data['personal_graphs'].items():
            G = self._build_networkx_graph(
                user_data['nodes'],
                user_data['edges']
            )
            G.graph.update(user_data['metadata'])
            self.personal_graphs[user_id] = G
    
    def _build_networkx_graph(self, nodes: List, edges: List) -> nx.DiGraph:
        """Convert JSON to NetworkX graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            if isinstance(node, dict):
                G.add_node(node['id'], **{k:v for k,v in node.items() if k != 'id'})
            else:
                G.add_node(node)
        
        # Add edges
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))
        
        return G
    
    def extract_user_pattern(self, distortions: List[str]) -> nx.DiGraph:
        """
        Create mini-graph from user's distortions
        """
        G = nx.DiGraph()
        
        # Add nodes
        for d in distortions:
            G.add_node(d)
        
        # Add edges from global graph
        for i, d1 in enumerate(distortions):
            for d2 in distortions[i+1:]:
                if self.global_graph.has_edge(d1, d2):
                    weight = self.global_graph[d1][d2]['weight']
                    G.add_edge(d1, d2, weight=weight)
        
        return G
    
    def find_similar_patterns(
        self, 
        query_distortions: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find users with similar distortion patterns
        Returns: [(user_id, similarity_score, metadata), ...]
        """
        query_graph = self.extract_user_pattern(query_distortions)
        
        similarities = []
        
        for user_id, user_graph in self.personal_graphs.items():
            # Multiple similarity metrics
            sim_score = self._compute_graph_similarity(query_graph, user_graph)
            
            similarities.append((
                user_id,
                sim_score,
                user_graph.graph  # metadata
            ))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _compute_graph_similarity(
        self, 
        G1: nx.DiGraph, 
        G2: nx.DiGraph
    ) -> float:
        """
        Multi-metric graph similarity
        Combines: node overlap, edge overlap, structural similarity
        """
        # Node overlap (Jaccard)
        nodes1 = set(G1.nodes())
        nodes2 = set(G2.nodes())
        
        if not nodes1 or not nodes2:
            return 0.0
        
        node_jaccard = len(nodes1 & nodes2) / len(nodes1 | nodes2)
        
        # Edge overlap
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        if edges1 and edges2:
            edge_jaccard = len(edges1 & edges2) / len(edges1 | edges2)
        else:
            edge_jaccard = 0.0
        
        # Centrality similarity for shared nodes
        shared = nodes1 & nodes2
        if shared:
            try:
                deg1 = nx.degree_centrality(G1)
                deg2 = nx.degree_centrality(G2)
                
                centrality_sim = np.mean([
                    1 - abs(deg1.get(n, 0) - deg2.get(n, 0))
                    for n in shared
                ])
            except:
                centrality_sim = 0.0
        else:
            centrality_sim = 0.0
        
        # Weighted combination
        similarity = (
            0.4 * node_jaccard +
            0.3 * edge_jaccard +
            0.3 * centrality_sim
        )
        
        return similarity
    
    def find_keystone_distortions(
        self, 
        distortions: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Identify which distortions are most central in the pattern
        These are intervention priorities
        """
        query_graph = self.extract_user_pattern(distortions)
        
        if len(query_graph.nodes()) == 0:
            return []
        
        # Multiple centrality measures
        try:
            betweenness = nx.betweenness_centrality(query_graph)
            degree = nx.degree_centrality(query_graph)
            
            # Combine scores
            scores = defaultdict(float)
            for node in query_graph.nodes():
                scores[node] = 0.6 * betweenness.get(node, 0) + 0.4 * degree.get(node, 0)
            
            # Sort by score
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return ranked
        
        except:
            # Fallback to simple degree
            return sorted(
                [(n, query_graph.degree(n)) for n in query_graph.nodes()],
                key=lambda x: x[1],
                reverse=True
            )
    
    def find_cascade_paths(
        self,
        distortions: List[str],
        max_length: int = 4
    ) -> List[List[str]]:
        """
        Find distortion cascade chains
        E.g., [catastrophizing → fortune_telling → hopelessness]
        """
        query_graph = self.extract_user_pattern(distortions)
        
        cascades = []
        
        # Find all simple paths
        for source in query_graph.nodes():
            for target in query_graph.nodes():
                if source != target:
                    try:
                        paths = nx.all_simple_paths(
                            query_graph,
                            source,
                            target,
                            cutoff=max_length
                        )
                        for path in paths:
                            if len(path) >= 2:
                                cascades.append(path)
                    except:
                        continue
        
        # Sort by path weight (sum of edge weights)
        cascades_with_weights = []
        for path in cascades:
            weight = sum(
                query_graph[path[i]][path[i+1]].get('weight', 1)
                for i in range(len(path)-1)
            )
            cascades_with_weights.append((path, weight))
        
        cascades_with_weights.sort(key=lambda x: x[1], reverse=True)
        
        return [path for path, _ in cascades_with_weights[:10]]
    
    def retrieve_interventions(
        self,
        distortions: List[str],
        top_k: int = 3
    ) -> Dict:
        """
        Main GeAR retrieval function
        Returns complete intervention recommendation
        """
        # Find similar users
        similar = self.find_similar_patterns(distortions, top_k=top_k)
        
        # Find keystone distortions
        keystones = self.find_keystone_distortions(distortions)
        
        # Find cascades
        cascades = self.find_cascade_paths(distortions)
        
        # Analyze outcomes from similar users
        outcomes = self._analyze_outcomes(similar)
        
        return {
            'query_distortions': distortions,
            'similar_users': [
                {
                    'user_id': uid,
                    'similarity': round(sim, 3),
                    'journey_type': meta.get('journey_type'),
                    'improvement': meta.get('improvement')
                }
                for uid, sim, meta in similar
            ],
            'keystone_distortions': [
                {'distortion': d, 'centrality': round(score, 3)}
                for d, score in keystones[:3]
            ],
            'cascade_patterns': [
                ' → '.join(cascade)
                for cascade in cascades[:5]
            ],
            'outcomes': outcomes
        }
    
    def _analyze_outcomes(self, similar_users: List) -> Dict:
        """
        Analyze what happened to similar users
        """
        if not similar_users:
            return {'success_rate': 0, 'avg_improvement': 0}
        
        improvements = []
        journey_types = defaultdict(int)
        
        for _, _, meta in similar_users:
            if 'improvement' in meta:
                improvements.append(meta['improvement'])
            if 'journey_type' in meta:
                journey_types[meta['journey_type']] += 1
        
        return {
            'num_similar_cases': len(similar_users),
            'avg_improvement': round(np.mean(improvements), 3) if improvements else 0,
            'most_common_trajectory': max(journey_types.items(), key=lambda x: x[1])[0] if journey_types else None,
            'success_rate': sum(1 for i in improvements if i > 0) / len(improvements) if improvements else 0
        }


# Example usage
if __name__ == "__main__":
    print("🔬 Initializing GeAR Retrieval System...\n")
    
    gear = GeARRetrieval('graphs_for_gear.json')
    
    print(f"✓ Loaded {len(gear.personal_graphs)} user graphs")
    print(f"✓ Global graph: {len(gear.global_graph.nodes())} distortion types\n")
    
    # Test query
    test_distortions = [
        'catastrophizing',
        'fortune_telling',
        'mind_reading'
    ]
    
    print(f"📊 Query: {test_distortions}\n")
    
    # Retrieve
    results = gear.retrieve_interventions(test_distortions, top_k=5)
    
    # Display results
    print("="*60)
    print("GEAR RETRIEVAL RESULTS")
    print("="*60)
    
    print("\n🎯 Keystone Distortions (Intervention Priorities):")
    for item in results['keystone_distortions']:
        print(f"   • {item['distortion']}: {item['centrality']:.3f}")
    
    print("\n⛓️  Cascade Patterns Detected:")
    for cascade in results['cascade_patterns']:
        print(f"   → {cascade}")
    
    print("\n👥 Similar Cases Found:")
    for user in results['similar_users']:
        print(f"   • {user['user_id']}")
        print(f"     Similarity: {user['similarity']:.3f}")
        print(f"     Journey: {user['journey_type']}")
        print(f"     Improvement: {user['improvement']:.3f}\n")
    
    print("📈 Outcome Analysis:")
    for key, value in results['outcomes'].items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ GeAR retrieval complete!")
    print("\nNext: Add intervention data to get recommendations")