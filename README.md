# GeAR: Graph-Enhanced Agent for Retrieval - Cognitive Distortion Network

GeAR applies graph-enhanced retrieval to mental health, mapping cognitive distortions (commonly present in depression, anxiety and other mental health disorders) as interconnected networks to provide personalized intervention recommendations.

## What the Code Does
- Extracts cognitive distortions from text using LLMs from a ([Kaggle Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health?resource=download))
- Maps distortion patterns as knowledge graphs: cascades, co-occurrences, centrality
- Generates synthetic longitudinal user journeys with intervention tracking: 2 entries per week with different journey types, such as improving, fluctuating or worsening for 12 weeks
- Retrieves interventions based on graph topology similarity, not just semantic matching
- Visualizes networks in interactive 3D (distortions + interventions + effectiveness)

## Why GeAR and not naive RAG?
GeAR matches structural patterns: users with similar distortion network topologies get interventions that worked for others with the same graph structure.

1. **Data Pipeline**: Kaggle mental health dataset → distortion extraction → synthetic longitudinal generation
2. **Graph Layer**: Personal graphs (per-user networks) + Global graph (population patterns) + Intervention graph (what targets what)
3. **Retrieval**: Multi-metric graph similarity (node overlap, edge overlap, centrality matching)
4. **Visualization**: 3D force-directed graph with intervention effectiveness highlighting

## Tech Stack
- Python (NetworkX, pandas, anthropic)
- Claude Sonnet 4 for distortion extraction
- Three.js for 3D visualization
- Graph algorithms: centrality analysis, cascade detection, subgraph matching


## Current Status
- 500 labeled statements with distortions from a Kaggle dataset
- 10 synthetic users with longitudinal data
- Intervention-enhanced graphs
- 3D visualization with effectiveness scoring
- ToDo: RAG interface 

## Use Case
Input: "I'll definitely fail this presentation, everyone will judge me"  
Output: Pattern matched to users with catastrophizing→mind_reading cascade. Recommended: Behavioral experiments (82% effectiveness, 0.5 avg improvement).
