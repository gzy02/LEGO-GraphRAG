from typing import List, Set
import igraph as ig


def personalized_pagerank(G: ig.Graph, vertices: Set[str], seed_nodes: List[str], restart_prob=0.8):
    """Return the PPR vector for the given seed nodes and restart prob using an igraph graph.

    Args:
        G: An igraph graph.
        vertices:
        seed_nodes: A list of seed node IDs.
        restart_prob: A scalar in [0, 1].

    Returns:
        ppr: A dictionary of node IDs to their PPR values.
    """

    vertices_list = list(vertices)
    # Calculate PPR using igraph's pagerank method
    ppr = G.personalized_pagerank(
        vertices=vertices,
        damping=restart_prob,
        reset_vertices=seed_nodes,
    )

    return dict(zip(vertices_list, ppr))


def rank_ppr_ents(G: ig.Graph, vertices: Set[str], seed_list: List[str], mode="fixed", max_ent=2000, min_ppr=0.005, restart_prob=0.8):
    # Run PPR on the subgraph
    ppr = personalized_pagerank(G, vertices, seed_list, restart_prob)

    # Extract entities based on mode
    if mode == "fixed":
        sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        extracted_ents = [node for node, _ in sorted_ppr[:max_ent]]
    else:
        extracted_ents = [node for node,
                          value in ppr.items() if value > min_ppr]

    return extracted_ents
