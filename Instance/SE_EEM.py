import json
import os
import time
from tqdm import tqdm
import set_random
import igraph as ig
from utils.SemanticModel import EmbeddingModel, RandomModel, BM25Model
from utils.Tools import get_query_subgraph
from utils.LLM import LLM
from pre_retrieval import *

standard_ppr = 1000


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["name"]
        tail = G.vs[edge.target]["name"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


class QueryProcessor:
    def __init__(self, dataset, ppr_file, ModuleEEM):
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.queries = []
        # self.ModulePPR = PreRetrievalModulePPR(standard_ppr)
        self.ModuleEEM = ModuleEEM
        self.window = ModuleEEM.window

    def process_basic_info(self, window, structure_method, semantic_method):
        return {
            "Dataset": self.dataset,
            "ppr_file": self.ppr_file,
            "window": window,
            "structureMethod": str(structure_method),
            "semanticMethod": str(semantic_method)
        }

    def process_query_info(self, query):
        return {
            "id": query.qid,
            "question": query.question,
            "answers": query.answers,
            "entities": query.entities,
            "st_tokens": query.st_tokens,
            "subgraph": get_triples(query.subgraph)
        }

    def save_query(self, query_list, basic_info, query_info):
        query_list.append({
            "basic_info": basic_info,
            "query_info": query_info
        })

    def process(self, query):
        query = self.ModuleEEM.process(query)
        self.save_query(
            self.queries,
            self.process_basic_info(
                self.window, "PPR_1000", self.ModuleEEM
            ),
            self.process_query_info(query)
        )
        return query


if __name__ == '__main__':
    C = 32
    N = 1000
    window = 256
    print("PID =", os.getpid())

    models = {
        "qwen2-70b": "http://localhost:8000/v1/chat/completions"
    }
    rank_models = {
        # "BM25": BM25Model(),
        "EMB": EmbeddingModel(),
        # "random": RandomModel()
    }
    ModuleEEMs = {
        "triple": PreRetrievalModuleTriples,
        # "edge": PreRetrievalModuleEdge,
        # "node": PreRetrievalModuleNode,
    }

    for model, url in models.items():
        for reasoning_dataset in ["metaQA"]:
            ppr_file = "/back-up/gzy/dataset/VLDB/Rebuttal/R3/metaQA/subgraph/PPR.json"
            print("Dataset: ", reasoning_dataset)
            for rankModelName, rankModel in rank_models.items():
                print("Rank Model: ", rankModelName)
                for task, ModuleEEM in ModuleEEMs.items():
                    print("Task: ", task)
                    processor = QueryProcessor(
                        reasoning_dataset, ppr_file, ModuleEEM(window, rankModelName))
                    for query in get_query_subgraph(ppr_file):
                        query = processor.process(query)
                    json_path = f"/back-up/gzy/dataset/VLDB/Rebuttal/R3/metaQA/subgraph/EMB/triple.json"
                    if not os.path.exists(json_path):
                        os.makedirs(os.path.dirname(
                            json_path), exist_ok=True)

                    with open(json_path, "w") as fp:
                        json.dump(processor.queries, fp, indent=4)
