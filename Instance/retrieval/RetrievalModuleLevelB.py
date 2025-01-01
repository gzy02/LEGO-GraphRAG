import torch
from config import emb_model_dir, rerank_model_dir
from copy import deepcopy
from retrieval.RetrievalModule import RetrievalModule
from utils.Query import Query
import igraph as ig
from typing import List, Tuple
from utils.ReasoningPath import ReasoningPath
from utils.SemanticModel import BGEModel, EmbeddingModel, BM25Model, RandomModel

from typing import List, Tuple
import igraph as ig


class RetrievalModuleSemanticModel(RetrievalModule):
    def __init__(self, hop: int = 4, top_k: int = -1, beam_width: int = 8, thre=32, semantic_type="BM25", model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        self.beam_width = beam_width
        self.thre = thre
        self.semantic_type = semantic_type

        if semantic_type == "BM25":
            self.model_dir = "BM25"
            self.model = BM25Model()
        elif semantic_type == "EMB":
            model_dir = emb_model_dir if model_dir is None else model_dir
            self.model = EmbeddingModel(model_dir)
        elif semantic_type == "BGE":
            model_dir = rerank_model_dir if model_dir is None else model_dir
            self.model = BGEModel(model_dir)
        else:
            model_dir = "Random"
            self.model = RandomModel()
        self.model_dir = model_dir

    async def aprocess(self, query: Query) -> Query:
        return self.process(query)

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self.get_reasoning_paths(
            query.subgraph, query.entities, query.question)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        reasoning_paths = []

        for entity in entities:
            try:
                start_vertex_id = G.vs.find(name=entity).index
            except:
                continue

            # Initialize beam with the starting entity
            beams = [([start_vertex_id], ReasoningPath(
                entity=G.vs[start_vertex_id]["name"]))]

            for step in range(self.hop):
                next_beams = []

                for path, reasoning_path in beams:
                    current_vertex_id = path[-1]
                    neighbors = G.vs[current_vertex_id].neighbors(mode="out")

                    for index, neighbor in enumerate(neighbors):
                        if index >= self.thre:
                            break
                        neighbor_id = neighbor.index
                        edge_id = G.get_eid(current_vertex_id, neighbor_id)
                        new_path = path + [neighbor_id]

                        cur_reasoning_path = deepcopy(reasoning_path)
                        edge = G.es[edge_id]
                        source_vertex = G.vs[edge.source]["name"]
                        target_vertex = G.vs[edge.target]["name"]
                        relation = edge["name"]
                        cur_reasoning_path.add_triple(
                            (source_vertex, relation, target_vertex))

                        next_beams.append((new_path, cur_reasoning_path))

                # Retain the top-k beams based on their scores
                path_dict = {str(reasoning_path): (path, reasoning_path)
                             for path, reasoning_path in next_beams}
                corpus = list(path_dict.keys())
                sorted_paths = self.model.top_k(
                    question, corpus, self.beam_width)
                next_beams = [path_dict[path] for path in sorted_paths]
                beams = next_beams

                # Add the top beams to the reasoning paths
                for _, reasoning_path in beams:
                    reasoning_paths.append(reasoning_path)

        path_dict = {str(path): path for path in reasoning_paths}
        corpus = list(path_dict.keys())
        sorted_paths = self.model.top_k(question, corpus, self.top_k)
        reasoning_paths = [path_dict[path] for path in sorted_paths]
        return reasoning_paths


class RetrievalModuleSemanticModelTriples(RetrievalModule):
    def __init__(self, hop: int = 4, top_k: int = 32, semantic_type="BM25", model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        self.semantic_type = semantic_type

        if semantic_type == "BM25":
            self.model_dir = "BM25"
            self.model = BM25Model()
        elif semantic_type == "EMB":
            model_dir = emb_model_dir if model_dir is None else model_dir
            self.model = EmbeddingModel(model_dir)
        else:
            model_dir = rerank_model_dir if model_dir is None else model_dir
            self.model = BGEModel(model_dir)
        self.model_dir = model_dir

    async def aprocess(self, query: Query) -> Query:
        return self.process(query)

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self.get_reasoning_paths(
            query.subgraph, query.entities, query.question)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        reasoning_paths = []
        for entity in entities:
            try:
                start_vertex_id = G.vs.find(name=entity).index
            except:
                continue
            visited = {start_vertex_id: None}
            queue = [(start_vertex_id, None, 0)]

            while queue:
                next_queue = []

                for current_vertex_id, parent_edge, hop in queue:
                    if hop == self.hop or G.degree(current_vertex_id, mode="out") == 0:
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

                new_paths = self.semantic_extract(G, next_queue, question)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return reasoning_paths

    def semantic_extract(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str) -> List[Tuple[int, int]]:
        paths = [(neighbor_id, edge_id) for (neighbor_id, edge_id) in paths]
        if len(paths) <= self.top_k:
            return paths
        corpus = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            corpus.append(G.vs[edge.source]["name"] + ", " +
                          edge["name"] + ", " + G.vs[edge.target]["name"])
        sorted_paths = self.model.top_k(question, corpus, self.top_k)
        new_paths = []
        for path in sorted_paths:
            for neighbor_id, edge_id in paths:
                edge = G.es[edge_id]
                if G.vs[edge.source]["name"] + ", " + edge["name"] + ", " + G.vs[edge.target]["name"] == path:
                    new_paths.append((neighbor_id, edge_id))
                    break
        return new_paths
