from time import time
import os
import json
from utils.Tools import get_query, Tools
from pre_retrieval import *
import igraph as ig
import set_random


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


def fRW(window, query, RW):
    RW.path_num = window
    q = RW.process2(query)
    G = q.subgraph
    return len(G.es), q


def SearchRW(target, query, RW):
    domain_start = 1
    domain_end = 256
    t = time()
    RW.prefill(query)
    print("fRW(x)预处理耗时：", time()-t)
    x0, f_x0, q = binary_search_closest(
        fRW, query, RW, target, domain_start, domain_end)
    print(f"最接近 {target} 的 fRW(x0) 是 {f_x0}，对应的 x0 是 {x0}")
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
        self.large_ppr = 500
        self.standard_ppr = 200
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.rw_queries = []
        self.ppr_queries = []

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

    def search(self, query):
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

        rw, rw_query = SearchRW(target, query,
                                PreRetrievalModuleRandomWalk(256))
        self.save_query(
            self.rw_queries,
            self.process_basic_info(
                rw, PreRetrievalModuleRandomWalk(
                    rw), PreRetrievalModuleNone()
            ),
            self.process_query_info(rw_query)
        )


print("PID =", os.getpid())
tools = Tools()
kg = tools.kg
for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:  #
    ppr_file = f"/back-up/lzy/Dataset/{reasoning_dataset}/{reasoning_dataset}_250.jsonl"
    print("Dataset: ", reasoning_dataset)

    processor = QueryProcessor(reasoning_dataset, ppr_file)
    for query in get_query(kg, ppr_file):
        processor.search(query,)
    query_dict = {
        "RandomWalk": processor.rw_queries,
        "PPR": processor.ppr_queries,
    }
    for query_type in query_dict:
        with open(f"/back-up/gzy/dataset/VLDB/SubgraphExtraction/{reasoning_dataset}/subgraph/{query_type}.json", "w") as fp:
            json.dump(query_dict[query_type], fp, indent=4)
