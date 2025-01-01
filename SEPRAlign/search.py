import igraph as ig
from pre_retrieval import *
from utils.Tools import get_query, Tools, get_triples
import json
import os
from time import time, sleep

ranked_info_path = "/back-up/gzy/dataset/VLDB/new250/ranked_info/{dataset}/{semantic_type}/{pr_type}.json"
STANDARD_PPR = 300
LARGE_PPR = 500


class QueryProcessor:
    def __init__(self, dataset, ppr_file, semantic_type):
        self.large_ppr = LARGE_PPR
        self.standard_ppr = STANDARD_PPR
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.semantic_type = semantic_type
        self.ppr_queries = []
        self.node_queries = []
        self.edge_queries = []
        self.triple_queries = []

    def process_basic_info(self, window, structure_method, semantic_method):
        return {
            "Dataset": self.dataset,
            "ppr_file": self.ppr_file,
            "window": window,
            "semanticType": self.semantic_type,
            "structureMethod": str(structure_method),
            "semanticMethod": str(semantic_method)
        }

    def process_query_info(self, query):
        return {
            "id": query.qid,
            "answers": query.answers,
            "question": query.question,
            "entities": query.entities,
            "subgraph": get_triples(query.subgraph)
        }

    def save_query(self, query_list, basic_info, query_info):
        query_list.append({
            "basic_info": basic_info,
            "query_info": query_info
        })

    def search(self, query, ranked_nodes, ranked_edges, ranked_triples):
        print("Question ID: ", query.qid)

        # Step 1: Initial PPR query
        ppr_query = PreRetrievalModulePPR(self.standard_ppr).process(query)
        self.save_query(
            self.ppr_queries,
            self.process_basic_info(
                self.standard_ppr, PreRetrievalModulePPR(
                    self.standard_ppr), PreRetrievalModuleNone()
            ),
            self.process_query_info(ppr_query)
        )
        target = len(ppr_query.subgraph.es)
        print("Target:", target)

        # Step 2: Refined queries
        refined_query = PreRetrievalModulePPR(self.large_ppr).process(query)
        t = time()

        nodes, node_query = PreRetrievalModuleNode(
            target).process(refined_query, ranked_nodes)

        edges, edge_query = PreRetrievalModuleEdge(
            target).process(refined_query, ranked_edges)

        triples, triple_query = PreRetrievalModuleTriple(
            target).process(refined_query, ranked_triples)
        print("Time:", time()-t)

        for module, query_list, semantic_param, query_instance in [
            (PreRetrievalModuleNode(target),
             self.node_queries, nodes, node_query),
            (PreRetrievalModuleEdge(target),
             self.edge_queries, edges, edge_query),
            (PreRetrievalModuleTriple(target),
             self.triple_queries, triples, triple_query),
        ]:
            self.save_query(
                query_list,
                self.process_basic_info(
                    semantic_param, PreRetrievalModulePPR(
                        self.large_ppr), module
                ),
                self.process_query_info(query_instance)
            )


tools = Tools()
kg = tools.kg
for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:
    ppr_file = f"/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
    for semantic_type in ["BGE", "BM25", "EMB"]:
        print("Dataset: ", reasoning_dataset)
        print("Semantic Type: ", semantic_type)
        ranked_info_dict = {
            "node": dict(),
            "edge": dict(),
            "triple": dict()
        }
        for pr_type in ["node", "edge", "triple"]:
            path = ranked_info_path.format(
                dataset=reasoning_dataset, semantic_type=semantic_type, pr_type=pr_type)
            with open(path, "r") as fp:
                ranked_info = json.load(fp)
            for data in ranked_info["eval_info"]:
                ranked_info_dict[pr_type][data["id"]] = data["ranked_corpus"]
        processor = QueryProcessor(reasoning_dataset, ppr_file, semantic_type)
        for query in get_query(kg, ppr_file):
            qid = query.qid
            processor.search(query, ranked_info_dict["node"][qid],
                             ranked_info_dict["edge"][qid], ranked_info_dict["triple"][qid])
        query_dict = {
            "ppr": processor.ppr_queries,
            "node": processor.node_queries,
            "edge": processor.edge_queries,
            "triple": processor.triple_queries
        }
        for query_type in query_dict:
            json_path = f"/back-up/gzy/dataset/VLDB/SE_new/{reasoning_dataset}/subgraph/EEMs/{STANDARD_PPR}/{semantic_type}/{query_type}.json"
            if not os.path.exists(json_path):
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as fp:
                json.dump(query_dict[query_type], fp,
                          indent=4, ensure_ascii=False)
