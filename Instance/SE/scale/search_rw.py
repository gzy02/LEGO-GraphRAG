from time import time
import os
import json
from utils.Tools import get_query, Tools, get_triples
from pre_retrieval import *
import igraph as ig
import set_random
from copy import deepcopy
print("PID =", os.getpid())
tools = Tools()
kg = tools.kg


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


def fPPR(window, query, PPR):
    PPR.max_ent = window
    q = PPR.process(query)
    G = q.subgraph
    return len(G.es), q


def SearchPPR(target, query, PPR, scale):
    domain_start = 2
    domain_end = scale
    t = time()
    x0, f_x0, q = binary_search_closest(
        fPPR, query, PPR, target, domain_start, domain_end)
    print(f"最接近 {target} 的 fPPR(x0) 是 {f_x0}，对应的 x0 是 {x0}")
    print("耗时：", time()-t)
    return x0, q


class QueryProcessor:
    def __init__(self, dataset, ppr_file, rw_path):
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.rw_file = rw_path
        self.rw_queries = []
        self.ppr_queries = []

        self.paths = {}
        with open(rw_path, "r") as fp:
            rw_info = json.load(fp)
            for info in rw_info["eval_info"]:
                self.paths[info["id"]] = info["path_dict"]

    def process_basic_info(self, window, structure_method, semantic_method):
        return {
            "Dataset": self.dataset,
            "ppr_file": self.ppr_file,
            "rw_file": self.rw_file,
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

    def post_process(self, query: Query) -> Query:
        # 将subgraph中的节点的name属性改为label属性
        for node in query.subgraph.vs:
            node["name"] = node["label"]

        query.answers = self.node_process(query.answers)
        query.entities = self.node_process(query.entities)
        return query

    def node_process(self, ents):
        new_answers = set()
        for ans in ents:
            if ans["text"] is None:
                new_answers.add(ans["kb_id"])
            elif ans["text"] != "":
                new_answers.add(ans["text"])
        return list(new_answers)

    def search(self, query, path_num):
        print("Question ID: ", query.qid)
        nodes = set()
        for mid_ent, paths in self.paths[query.qid].items():
            for path in paths[:path_num]:
                for entity in path:
                    nodes.add(entity["kb_id"])
        rw_subgraph = kg.subgraph(nodes)
        target = len(rw_subgraph.es)

        rw_query = deepcopy(query)
        rw_query.subgraph = rw_subgraph
        rw_query = self.post_process(rw_query)

        self.save_query(
            self.rw_queries,
            self.process_basic_info(
                path_num, PreRetrievalModuleRandomWalk(
                    path_num), PreRetrievalModuleNone()
            ),
            self.process_query_info(rw_query)
        )

        ppr_window, ppr_query = SearchPPR(
            target, query, PreRetrievalModulePPR(), len(rw_subgraph.vs)*2)
        self.save_query(
            self.ppr_queries,
            self.process_basic_info(
                ppr_window, PreRetrievalModulePPR(
                    ppr_window), PreRetrievalModuleNone()
            ),
            self.process_query_info(ppr_query)
        )


scales = [1, 2, 4, 8, 16, 32, 64, 128, 256]
scales.reverse()
for reasoning_dataset in ["WebQuestion", "GrailQA",]:
    for scale in scales:
        ppr_file = f"/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
        print("Dataset: ", reasoning_dataset)
        rw_path = f"/back-up/gzy/dataset/VLDB/new250/ranked_info/{reasoning_dataset}/RW/256.json"
        processor = QueryProcessor(reasoning_dataset, ppr_file, rw_path)
        for query in get_query(kg, ppr_file):
            processor.search(query, scale)
        query_dict = {
            "RandomWalk": processor.rw_queries,
            "PPR": processor.ppr_queries,
        }
        for query_type in query_dict:
            path = f"/back-up/gzy/dataset/VLDB/SE_new/{reasoning_dataset}/subgraph/SBE/{scale}/{query_type}.json"
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fp:
                json.dump(query_dict[query_type], fp, indent=4)
