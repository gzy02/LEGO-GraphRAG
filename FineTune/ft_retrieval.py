import json
import random
import igraph as ig
from typing import List, Dict

from retrieval import *
from utils.LLM import LLM
from pre_retrieval import *
from post_retrieval import *
from utils.Tools import get_query
from utils.PromptTemplate import *
from config import reasoning_model, local_models, emb_model_dir
from sentence_transformers import SentenceTransformer
from utils.Query import Query
from utils.ReasoningPath import ReasoningPath


class RetrievalModuleBeamSearch(RetrievalModule):
    def __init__(self, hop: int, beam_width: int,  llm_dir: str = None, thre: int = 24, window: int = 64, model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.beam_width = beam_width
        self.thre = thre
        self.window = window
        llm_dir = reasoning_model if llm_dir is None else llm_dir
        self.llm_dir = llm_dir
        self.llm_model = LLM(self.llm_dir)
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = SentenceTransformer(
            model_dir, device="cuda"
        )
        self.model.eval()

    def process(self, query: Query) -> Dict:
        G = query.subgraph
        self.query = query
        return self._process(
            G, query.entities)

    def _process(self, G: ig.Graph, entities: List[str]) -> Dict:
        reasoning_paths = []
        ans_paths = set()
        no_ans_paths = set()
        for entity in entities:
            try:
                start_vertex_id = G.vs.find(name=entity).index
            except:
                continue
            visited = {start_vertex_id: None}  # 记录访问过的节点及其父节点和边
            queue = [(start_vertex_id, None, 0)]  # 使用队列进行BFS，元组中还包含到达该节点的边

            while queue:
                next_queue = []

                for current_vertex_id, parent_edge, hop in queue:
                    if hop == self.hop or G.degree(current_vertex_id, mode="out") == 0:
                        # 构建路径
                        path = []
                        while current_vertex_id is not None:
                            parent_info = visited[current_vertex_id]
                            if parent_info is not None:
                                parent_vertex_id, edge_id = parent_info
                                path.append(
                                    (parent_vertex_id, current_vertex_id, edge_id))
                                current_vertex_id = parent_vertex_id
                            else:
                                break
                        path.reverse()

                        reasoning_path = ReasoningPath(entity)
                        for source_vertex_id, target_vertex_id, edge_id in path:
                            edge = G.es[edge_id]
                            source_vertex = G.vs[edge.source]["name"]
                            target_vertex = G.vs[edge.target]["name"]
                            relation = edge["name"]
                            reasoning_path.add_triple(
                                (source_vertex, relation, target_vertex))
                        reasoning_paths.append(reasoning_path)
                    else:
                        visited_set = set(visited.keys())
                        for neighbor in G.vs[current_vertex_id].neighbors(mode="out"):
                            neighbor_id = neighbor.index
                            if neighbor_id not in visited_set:
                                edge_id = G.get_eid(
                                    current_vertex_id, neighbor_id)
                                next_queue.append((neighbor_id, edge_id))
                                visited[neighbor_id] = (
                                    current_vertex_id, edge_id)

                if len(next_queue) == 0:
                    break

                # 拿到答案，构造答案路径
                ans_entities = set(self.query.answers)
                for neighbor_id, edge_id in next_queue:
                    edge = G.es[edge_id]
                    for entity in ans_entities:
                        if entity == G.vs[edge.source]["name"] or entity == G.vs[edge.target]["name"]:
                            vid = neighbor_id
                            while vid is not None:
                                parent_info = visited[vid]
                                if parent_info is not None:
                                    parent_vertex_id, e_id = parent_info
                                    ans_paths.add(
                                        G.vs[G.es[e_id].source]["name"] + ", " + G.es[e_id]["name"] + ", " + G.vs[G.es[e_id].target]["name"])
                                    vid = parent_vertex_id
                                else:
                                    break
                            ans_entities.discard(entity)
                            break
                # new_paths = self.extract_paths(G, next_queue, question, entity, ans_entities)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in next_queue]

        # 拿到推理路径，作为instruction一部分
        ans_entities = set(self.query.answers)
        for path in reasoning_paths[: self.window]:
            entities = set(path.entities_path)
            if len(entities.intersection(ans_entities)) == 0:
                paths = format_path(str(path))
                for p in paths:
                    no_ans_paths.add(p)
        # 把output加到instruction里面
        all_paths = list(no_ans_paths.union(ans_paths))
        random.shuffle(all_paths)

        output = '\n'.join(ans_paths)
        instruction = extract_prompt.format(beam_width=len(
            ans_paths), question=self.query.question, entity_name='\n'.join(entities), total_paths="\n".join(all_paths))
        return {
            "instruction": instruction,
            "input": "",
            "output": output
        }


def format_path(path):
    parts = path.split(" -> ")
    formatted_paths = []
    for i in range(0, len(parts) - 2, 2):
        formatted_paths.append(f"{parts[i]}, {parts[i+1]}, {parts[i+2]}")
    return formatted_paths


reasoning_dataset = "SimpleQuestion"
target_dir = "../sft_dataset/"
dataset_dir = "/back-up/gzy/dataset/graphrag/process_data/"
test_file = dataset_dir + reasoning_dataset + \
    "/SimpleQuestion_ppr1500_2_train.jsonl"
if __name__ == "__main__":
    pre = PreRetrievalModulePPR()
    re = RetrievalModuleBeamSearch(hop=2, beam_width=64, thre=96)
    info = []
    print("Dataset:", reasoning_dataset)
    id = 0
    for query in get_query(test_file):
        id += 1
        if id % 100 == 0:
            print(id)
        query = pre.process(query)
        item = re.process(query)
        info.append(item)

    # print(info)
    with open(target_dir+reasoning_dataset+"_4_32_train_retrieval_sft.json", "w") as fp:
        json.dump(info, fp)
