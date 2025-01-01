import torch
from config import emb_model_dir, rerank_model_dir
from copy import deepcopy
from retrieval.RetrievalModule import RetrievalModule
from utils.Query import Query
from rank_bm25 import BM25Okapi
import igraph as ig
from typing import List, Tuple
from utils.ReasoningPath import ReasoningPath

from torch import Tensor
from utils.Tools import get_k_hop_neighbors
from utils.SentenceModel import BGEModel, EmbeddingModel


class RetrievalModuleBM25(RetrievalModule):
    def __init__(self, hop: int = 3, top_k: int = 32):
        super().__init__()
        self.hop = hop
        self.top_k = top_k

    def process(self, query: Query) -> Query:
        """Process the query and return the updated query object"""
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

                # bm25 提取 top-k 路径
                new_paths = self.bm25_extract(G, next_queue, question)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return reasoning_paths

    def bm25_extract(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str) -> List[Tuple[int, int]]:
        paths = [(neighbor_id, edge_id) for (neighbor_id, edge_id)
                 in paths]
        if len(paths) <= self.top_k:
            return paths
        corpus = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            corpus.append(G.vs[edge.source]["name"] + ", " +
                          edge["name"] + ", " + G.vs[edge.target]["name"])
        bm25 = BM25Okapi(corpus)
        tokenized_question = question.split()
        bm25_scores = bm25.get_scores(tokenized_question)
        sorted_paths = [path for _, path in sorted(
            zip(bm25_scores, corpus), reverse=True)]
        new_paths = []
        for path in sorted_paths[:self.top_k]:
            for neighbor_id, edge_id in paths:
                edge = G.es[edge_id]
                if G.vs[edge.source]["name"] + ", " + edge["name"] + ", " + G.vs[edge.target]["name"] == path:
                    new_paths.append((neighbor_id, edge_id))
                    break
        return new_paths


class RetrievalModuleEmb(RetrievalModule):
    def __init__(self, hop: int = 3, top_k: int = 32, model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = EmbeddingModel(model_dir)
        # self.model.eval()

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

                # embbeding 提取 top-k 路径
                new_paths = self.emb_extract(G, next_queue, question)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return reasoning_paths

    def emb_extract(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str) -> List[Tuple[int, int]]:
        paths = [(neighbor_id, edge_id) for (neighbor_id, edge_id)
                 in paths]
        if len(paths) <= self.top_k:
            return paths
        all_chunks = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            all_chunks.append(G.vs[edge.source]["name"] + ", " +
                              edge["name"] + ", " + G.vs[edge.target]["name"])
        cosine_scores = self.model.get_scores(question, all_chunks)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, all_chunks), reverse=True)]
        new_paths = []
        for path in sorted_paths[:self.top_k]:
            for neighbor_id, edge_id in paths:
                edge = G.es[edge_id]
                if G.vs[edge.source]["name"] + ", " + edge["name"] + ", " + G.vs[edge.target]["name"] == path:
                    new_paths.append((neighbor_id, edge_id))
                    break
        return new_paths


class RetrievalModuleBGE(RetrievalModule):
    def __init__(self, hop: int = 3, top_k: int = 32, model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        model_dir = rerank_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = BGEModel(model_dir)

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

                # bge 提取 top-k 路径
                new_paths = self.bge_extract(G, next_queue, question)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return reasoning_paths

    def bge_extract(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str) -> List[Tuple[int, int]]:
        paths = [(neighbor_id, edge_id) for (neighbor_id, edge_id)
                 in paths]
        if len(paths) <= self.top_k:
            return paths
        contents = []
        for neighbor_id, edge_id in paths:
            edge = G.es[edge_id]
            contents.append(G.vs[edge.source]["name"] + ", " +
                            edge["name"] + ", " + G.vs[edge.target]["name"])
        scores = self.model.get_scores(question, contents)
        sorted_paths = [path for _, path in sorted(
            zip(scores, contents), reverse=True)]

        new_paths = []
        for path in sorted_paths[:self.top_k]:
            for neighbor_id, edge_id in paths:
                edge = G.es[edge_id]
                if G.vs[edge.source]["name"] + ", " + edge["name"] + ", " + G.vs[edge.target]["name"] == path:
                    new_paths.append((neighbor_id, edge_id))
                    break
        return new_paths


class RetrievalModuleTransE(RetrievalModule):
    def __init__(self, hop: int = 3, top_k: int = 32, agg=0, model_type: str = "emb", model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        self.agg = agg  # 聚合次数
        self.model_type = model_type
        if model_type == "emb":
            self.model_dir = emb_model_dir if model_dir is None else model_dir
            self.score_model = EmbeddingModel(self.model_dir)
        elif model_type == "rerank":
            self.model_dir = rerank_model_dir if model_dir is None else model_dir
            self.score_model = BGEModel(self.model_dir)
        else:
            self.model_dir = None
            self.score_model = None

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self.get_reasoning_paths(
            query.subgraph, query.entities, query.question)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        relations = {edge["name"] for edge in G.es}
        corpus = list(relations)
        cosine_scores = self.score_model.get_scores(question, corpus)
        scores = {name: score for name, score in zip(corpus, cosine_scores)}
        for edge in G.es:
            edge["weight"] = scores[edge["name"]]
        for _ in range(self.agg):
            """对所有边的weight加上边连接的两个节点的所有边softmax(weight)*weight"""
            tep_graph = deepcopy(G)
            for edge in tep_graph.es:
                source = tep_graph.vs[edge.source]
                target = tep_graph.vs[edge.target]
                source_edges = tep_graph.es.select(_target=source.index)
                target_edges = tep_graph.es.select(_source=target.index)
                source_weights = Tensor([edge["weight"]
                                        for edge in source_edges])
                target_weights = Tensor([edge["weight"]
                                        for edge in target_edges])
                softmax_source_weights = torch.nn.functional.softmax(
                    source_weights, dim=0)
                softmax_target_weights = torch.nn.functional.softmax(
                    target_weights, dim=0)
                source_weight = (softmax_source_weights@source_weights)
                target_weight = (softmax_target_weights@target_weights)
                g_edge = G.es.find(name=edge["name"])
                g_edge["weight"] += source_weight + target_weight
        reasoning_paths = []
        for entity in entities:
            reasoning_paths.extend(self.find_top_k_paths(
                get_k_hop_neighbors(G, [entity], self.hop), entity, question))
        return reasoning_paths

    def find_top_k_paths(self, graph: ig.Graph, entity: str, question: str):  # 寻找权值和最大的前k条路径
        try:
            source = graph.vs.find(name=entity)
        except:
            return []
        # try:
        #    out_paths = graph.get_shortest_paths(
        #        v=source, weights=graph.es["weight"], mode="out")
        #    print(out_paths)
        # except:
        #    out_paths = graph.get_all_simple_paths(
        #        v=source, cutoff=self.hop, mode="out")
        out_paths = graph.get_all_simple_paths(
            v=source, cutoff=self.hop, mode="out")

        paths = []
        for path in out_paths:
            tail = graph.vs[path[-1]]['name']
            if tail.startswith('m.') or tail.startswith('g.'):
                continue
            path_weight = 0
            for index in range(1, len(path)):
                edge_id = graph.get_eid(path[index-1], path[index])
                edge = graph.es[edge_id]
                path_weight += edge["weight"]
            paths.append((path, path_weight))

        # 排序并获取前k条路径
        paths = sorted(paths, key=lambda x: x[1], reverse=True)
        reasoning_paths = []
        for path, score in paths[:self.top_k]:
            if len(path) > 1:
                reasoning_path = ReasoningPath(entity)
                for i in range(len(path) - 1):
                    edge_id = graph.get_eid(path[i], path[i + 1])
                    edge = graph.es[edge_id]
                    triple = (graph.vs[path[i]]['name'],
                              edge['name'], graph.vs[path[i + 1]]['name'])
                    reasoning_path.add_triple(triple)
                reasoning_paths.append(reasoning_path)
        return reasoning_paths


class RetrievalModuleStructureTransE(RetrievalModule):
    def __init__(self, hop: int = 3, top_k: int = 32, agg=0, threshold=0, model_type: str = "emb", model_dir: str = None):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        self.agg = agg  # 聚合次数
        self.threshold = threshold  # 阈值

        self.model_type = model_type
        if model_type == "emb":
            self.model_dir = emb_model_dir if model_dir is None else model_dir
            self.score_model = EmbeddingModel(self.model_dir)
        elif model_type == "rerank":
            self.model_dir = rerank_model_dir if model_dir is None else model_dir
            self.score_model = BGEModel(self.model_dir)
        else:
            self.model_dir = None
            self.score_model = None

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self.get_reasoning_paths(
            query.subgraph, query.entities, query.question)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        relations = {edge["name"] for edge in G.es}
        corpus = list(relations)
        cosine_scores = self.score_model.get_scores(question, corpus)
        scores = {name: score for name, score in zip(corpus, cosine_scores)}
        for edge in G.es:
            edge["weight"] = scores[edge["name"]]
        for _ in range(self.agg):
            """对所有边的weight加上边连接的两个节点的所有边softmax(weight)*weight"""
            tep_graph = deepcopy(G)
            for edge in tep_graph.es:
                source = tep_graph.vs[edge.source]
                target = tep_graph.vs[edge.target]
                source_edges = tep_graph.es.select(_target=source.index)
                target_edges = tep_graph.es.select(_source=target.index)
                source_weights = Tensor([edge["weight"]
                                        for edge in source_edges])
                target_weights = Tensor([edge["weight"]
                                        for edge in target_edges])
                softmax_source_weights = torch.nn.functional.softmax(
                    source_weights, dim=0)
                softmax_target_weights = torch.nn.functional.softmax(
                    target_weights, dim=0)
                source_weight = (softmax_source_weights@source_weights)
                target_weight = (softmax_target_weights@target_weights)
                g_edge = G.es.find(name=edge["name"])
                g_edge["weight"] += source_weight + target_weight
        reasoning_paths = []
        for entity in entities:
            reasoning_paths.extend(self.find_top_k_paths(
                get_k_hop_neighbors(G, [entity], self.hop), entity, question))
        return reasoning_paths

    def find_top_k_paths(self, graph: ig.Graph, entity: str, question: str):  # 寻找权值和最大的前k条路径
        try:
            source = graph.vs.find(name=entity)
        except:
            return []
        out_paths = graph.get_shortest_paths(
            v=source, mode="out")
        paths = []
        for path in out_paths:
            tail = graph.vs[path[-1]]['name']
            if tail.startswith('m.') or tail.startswith('g.'):
                continue
            path_weight = 0
            for index in range(1, len(path)):
                edge_id = graph.get_eid(path[index-1], path[index])
                edge = graph.es[edge_id]
                path_weight += 1 if edge["weight"] > self.threshold else 0
            paths.append((path, path_weight))

        # 排序并获取前k条路径
        paths = sorted(paths, key=lambda x: x[1], reverse=True)
        reasoning_paths = []
        for path, score in paths[:self.top_k]:
            if len(path) > 1:
                reasoning_path = ReasoningPath(entity)
                for i in range(len(path) - 1):
                    edge_id = graph.get_eid(path[i], path[i + 1])
                    edge = graph.es[edge_id]
                    triple = (graph.vs[path[i]]['name'],
                              edge['name'], graph.vs[path[i + 1]]['name'])
                    reasoning_path.add_triple(triple)
                reasoning_paths.append(reasoning_path)
        return reasoning_paths
