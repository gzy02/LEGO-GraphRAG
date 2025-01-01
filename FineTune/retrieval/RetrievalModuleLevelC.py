from retrieval.RetrievalModule import RetrievalModule
from utils.Query import Query
import igraph as ig
from utils.ReasoningPath import ReasoningPath
from typing import List, Tuple
from utils.PromptTemplate import extract_prompt
from utils.LLM import LLM
import random
from config import reasoning_model
import torch
from config import emb_model_dir
from sentence_transformers import SentenceTransformer


class RetrievalModuleBeamSearch(RetrievalModule):
    def __init__(self, hop: int, beam_width: int,  llm_dir: str = None, thre: int = 24, model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.beam_width = beam_width
        self.thre = thre
        llm_dir = reasoning_model if llm_dir is None else llm_dir
        self.llm_dir = llm_dir
        self.llm_model = LLM(llm_dir)
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = SentenceTransformer(
            model_dir, device="cuda"
        )
        self.model.eval()

    def process(self, query: Query) -> Query:
        G = query.subgraph
        question = query.question
        query.reasoning_paths = self.get_reasoning_paths(
            G, query.entities, question)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        reasoning_paths = []
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
                # # 对 next_queue 中的路径进行评分
                # neighbor_edge_pairs = [(neighbor_id, edge_id) for neighbor_id, edge_id in next_queue]
                # scores = self.score_paths(G, neighbor_edge_pairs, question, entity)

                # # 将评分与路径组合
                # scored_next_queue = [(score, neighbor_id, edge_id) for (neighbor_id, edge_id), score in zip(neighbor_edge_pairs, scores)]

                # # 保留分数最高的前 beam_width 个路径
                # top_scored_paths = heapq.nlargest(self.beam_width, scored_next_queue, key=lambda x: x[0])
                # queue = [(neighbor_id, edge_id, hop + 1) for _, neighbor_id, edge_id in top_scored_paths]

                # llm提取前 beam_width 个路径
                new_paths = self.extract_paths(G, next_queue, question, entity)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return list(set(reasoning_paths))

    # def score_paths(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str, entity_name: str) -> float:
    #     # use LLM to get score
    #     llm = self.llm_model
    #     relations = []
    #     for neighbor_id, edge_id in paths:
    #         relations.append(G.es[edge_id]["name"])
    #     score_prompt = self.construct_score_prompt(question, entity_name, relations)

    #     results = llm.invoke(score_prompt)
    #     scores = self.clean_scores(results, relations)
    #     return scores
    def extract_paths(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str, entity_name: str) -> List[Tuple[int, int]]:
        paths = [(neighbor_id, edge_id) for (neighbor_id, edge_id)
                 in paths]
        if len(paths) <= self.beam_width:
            return paths
        if len(paths) > self.thre:
            paths = self.emb_sort(G, paths, question)
        llm = self.llm_model
        total_paths = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            total_paths.append(G.vs[edge.source]["name"] + ", " +
                               edge["name"] + ", " + G.vs[edge.target]["name"])
        extract_prompt = self.construct_extract_prompt(
            beam_width=self.beam_width, question=question, entity_name=entity_name, total_paths=total_paths)

        results = llm.invoke(extract_prompt)
        # print("prompt: {}".format(extract_prompt))
        # print("results: {}".format(results))
        new_paths = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            if G.vs[edge.source]["name"] + ", " + edge["name"] + ", " + G.vs[edge.target]["name"] in results:
                new_paths.append((neighbor_id, edge_id))
        if len(new_paths) > self.beam_width:
            new_paths = random.sample(new_paths, self.beam_width)
        if len(new_paths) < self.beam_width:
            for neighbor_id, edge_id in paths:
                if (neighbor_id, edge_id) not in new_paths:
                    new_paths.append((neighbor_id, edge_id))
                if len(new_paths) == self.beam_width:
                    break
        return new_paths

    # def construct_score_prompt(self, question: str, entity_name: str, total_relations: List[str]) -> str:
    #     return score_prompt.format(question, entity_name) + ";".join(total_relations) + "\nScore: "

    def construct_extract_prompt(self, beam_width: int, question: str, entity_name: str, total_paths: List[str]) -> str:

        return extract_prompt.format(beam_width=beam_width, question=question, entity_name=entity_name, total_paths="\n".join(total_paths))

    # def clean_scores(self, result: str, relations: List[str]) -> List[float]:
    #     scores = re.findall(r'\d+\.\d+', result)
    #     scores = [float(number) for number in scores]
    #     if len(scores) == len(relations):
    #         return scores
    #     else:
    #         return [1/len(relations)] * len(relations)

    def encode(self, sentences, normalize_embeddings=False):
        with torch.no_grad():
            return self.model.encode(sentences, normalize_embeddings=normalize_embeddings)

    def emb_sort(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str) -> List[Tuple[int, int]]:
        all_chunks = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            all_chunks.append(G.vs[edge.source]["name"] + ", " +
                              edge["name"] + ", " + G.vs[edge.target]["name"])
        all_embeddings = self.encode(
            all_chunks, normalize_embeddings=True
        )
        query_embedding = self.encode(
            question, normalize_embeddings=True
        )[None, :]
        cosine_scores = (all_embeddings * query_embedding).sum(1)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, all_chunks), reverse=True)]
        new_paths = []
        for path in sorted_paths[:self.thre]:
            for neighbor_id, edge_id in paths:
                edge = G.es[edge_id]
                if G.vs[edge.source]["name"] + ", " + edge["name"] + ", " + G.vs[edge.target]["name"] == path:
                    new_paths.append((neighbor_id, edge_id))
                    break
        return new_paths
