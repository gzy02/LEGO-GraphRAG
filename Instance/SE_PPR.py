import set_random
import json
import os
from utils.LLM import LLM
from utils.SemanticModel import EmbeddingModel, RandomModel, BM25Model, BGEModel
from pipeline import *
from retrieval import *
from post_retrieval import *
from tqdm import tqdm
import heapq
from utils.Tools import Query, construct_graph, MetaQATools
from typing import List, Set
import igraph as ig
G = MetaQATools().kg


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["name"]
        tail = G.vs[edge.target]["name"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


def personalized_pagerank(G: ig.Graph, seed_nodes: List[str], restart_prob=0.8):
    seed_nodes = [G.vs.find(name=name).index for name in seed_nodes]
    ppr = G.personalized_pagerank(
        damping=restart_prob,
        reset_vertices=seed_nodes,
    )
    return dict(zip(G.vs, ppr))


step0_file = "/back-up/gzy/dataset/AAAI/MultiHopExperiment/MetaQA/PPR2/test_name.jsonl"  # step0
ppr_file = "/back-up/gzy/dataset/VLDB/Rebuttal/R3/metaQA/subgraph/PPR.json"
basic_info = {
    "Dataset": "MetaQA",
    "ppr_file": ppr_file,
    "window": 1000,
}
query_info = []
with open(step0_file, 'r') as f:
    for line in f:
        ids = json.loads(line)
        ID = ids["id"]
        answers = ids["answers"]
        question = ids["question"]
        entities = ids["entities"]
        new_obj = {
            "id": ID,
            "answers": answers,
            "question": question,
            "entities": entities,
        }
        ppr = personalized_pagerank(G, entities)
        top_ppr = heapq.nlargest(1000, ppr.items(), key=lambda x: x[1])
        vertex_list = [v.index for v, _ in top_ppr]
        new_obj['subgraph'] = get_triples(G.subgraph(vertex_list))
        query_info.append({
            "basic_info": basic_info,
            "query_info": new_obj
        })
with open(ppr_file, 'w') as f:
    json.dump(
        query_info, f
    )
