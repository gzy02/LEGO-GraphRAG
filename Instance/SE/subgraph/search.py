import psutil
from time import time, sleep
import os
import json
from utils.Tools import get_query, Tools
from pre_retrieval import *
import igraph as ig
import set_random


def fNode(window, query, semantic_module):
    semantic_module.window = window
    q = semantic_module.process(query)
    G = q.subgraph
    return len(G.es), q


def fEdge(window, query, semantic_module):
    semantic_module.window = window
    q = semantic_module.process(query)
    G = q.subgraph
    return len(G.es), q


def binary_search_closest(f, query, semantic_module, target, domain_start, domain_end):
    closest_x = None
    closest_f_val = None
    q = None
    min_diff = float('inf')

    while domain_start <= domain_end:
        mid = (domain_start + domain_end) // 2
        f_val, f_q = f(mid, query, semantic_module)
        diff = abs(f_val - target)

        # 更新最近的值
        if diff < min_diff:
            min_diff = diff
            closest_x = mid
            closest_f_val = f_val
            q = f_q

        # 根据二分逻辑调整范围
        if f_val < target:
            domain_start = mid + 1
        elif f_val > target:
            domain_end = mid - 1
        else:  # 找到完全匹配的值
            return mid, f_val, q

    return closest_x, closest_f_val, q


def SearchNode(target, query, semantic_module):
    domain_start = 2
    domain_end = len(query.subgraph.vs)
    t = time()
    x0, f_x0, q = binary_search_closest(
        fNode, query, semantic_module, target, domain_start, domain_end)
    print(f"最接近 {target} 的 fNode(x0) 是 {f_x0}，对应的 x0 是 {x0}")
    print("耗时：", time()-t)
    return x0, q


def SearchEdge(target, query, semantic_module):
    domain_start = 1
    domain_end = len({edge["name"] for edge in query.subgraph.es})
    t = time()
    x0, f_x0, q = binary_search_closest(
        fEdge, query, semantic_module, target, domain_start, domain_end)
    print(f"最接近 {target} 的 fEdge(x0) 是 {f_x0}，对应的 x0 是 {x0}")
    print("耗时：", time()-t)
    return x0, q


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["name"]
        tail = G.vs[edge.target]["name"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


class QueryProcessor:
    def __init__(self, dataset, ppr_file):
        self.large_ppr = 3000
        self.standard_ppr = 2500
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.ppr_queries = []
        self.node_queries = []
        self.edge_queries = []
        self.triple_queries = []

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

    def search(self, query, semantic_type):
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
        nodes, node_query = SearchNode(target, refined_query,
                                       PreRetrievalModuleNode(None, semantic_type))
        edges, edge_query = SearchEdge(target, refined_query,
                                       PreRetrievalModuleEdge(None, semantic_type))
        t = time()
        triples, triple_query = target, PreRetrievalModuleTriples(
            target, semantic_type).process(refined_query)
        print("Triples:", len(triple_query.subgraph.es))
        print("Triples Time:", time()-t)

        for module, query_list, semantic_param, query_instance in [
            (PreRetrievalModuleNode(nodes, semantic_type),
             self.node_queries, nodes, node_query),
            (PreRetrievalModuleEdge(edges, semantic_type),
             self.edge_queries, edges, edge_query),
            (PreRetrievalModuleTriples(triples, semantic_type),
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

        # return ppr_query, rw_query, self.node_queries[-1], self.edge_queries[-1], self.triple_queries[-1]


print("PID =", os.getpid())

tools = Tools()
kg = tools.kg
for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:  #
    ppr_file = f"/back-up/lzy/Dataset/{reasoning_dataset}/{reasoning_dataset}_250.jsonl"
    for semantic_type in ["EMB"]:  # ,
        print("Dataset: ", reasoning_dataset)
        print("Semantic Type: ", semantic_type)

        processor = QueryProcessor(reasoning_dataset, ppr_file)
        for query in get_query(kg, ppr_file):
            processor.search(query, semantic_type)
        query_dict = {
            "ppr": processor.ppr_queries,
            "node": processor.node_queries,
            "edge": processor.edge_queries,
            "triple": processor.triple_queries
        }
        for query_type in query_dict:
            json_path = f"/back-up/gzy/dataset/VLDB/ppr2500/{reasoning_dataset}/subgraph/{semantic_type}/{query_type}.json"
            if not os.path.exists(json_path):
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as fp:
                json.dump(query_dict[query_type], fp, indent=4)
