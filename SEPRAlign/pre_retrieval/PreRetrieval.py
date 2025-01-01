from config import emb_model_dir, rerank_model_dir
from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.SemanticModel import BGEModel, EmbeddingModel, BM25Model
import igraph as ig
from typing import List, Dict, Set
from abc import abstractmethod
from copy import deepcopy


class PreRetrievalSubgraphAlign(PreRetrievalModule):
    def __init__(self, window):
        super().__init__()
        self.window = window

    @abstractmethod
    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], corpus: Set[str]) -> ig.Graph:
        return

    def get_subgraph(self, query: Query, ranked_corpus: List[str]):
        """二分查找获取三元组最接近window的子图"""
        l, r = 1, len(ranked_corpus)
        seed_list = query.entities
        while l < r:
            m = (l + r) // 2
            subgraph = self.filter_subgraph(
                query.subgraph, seed_list, set(ranked_corpus[:m]))
            if len(subgraph.es) > self.window:
                r = m
            else:
                l = m + 1
        return self.filter_subgraph(query.subgraph, seed_list, set(ranked_corpus[:l])), l

    def process(self, origin_query, ranked_list):
        query = deepcopy(origin_query)
        subgraph, window = self.get_subgraph(query, ranked_list)
        query.subgraph = subgraph
        return window, query


class PreRetrievalModuleEdge(PreRetrievalSubgraphAlign):
    def __init__(self, window: int = 32):
        super().__init__(window)

    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], relations: Set[str]) -> ig.Graph:
        if len(relations) != 0:
            relevant_edges = G.es.select(name_in=relations)
            filtered_subgraph = G.subgraph_edges(
                relevant_edges, delete_vertices=True)
            nodes = set(filtered_subgraph.vs["name"])
        else:
            filtered_subgraph = ig.Graph(directed=True)
            nodes = set()
        # Add seed nodes
        for e in seed_list:
            if e not in nodes:
                filtered_subgraph.add_vertex(e)
        return filtered_subgraph


class PreRetrievalModuleNode(PreRetrievalSubgraphAlign):
    def __init__(self, window: int = 1024):
        super().__init__(window)

    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], entities: Set[str]) -> ig.Graph:
        relevant_nodes = entities.union(seed_list)
        # 过滤掉不存在于图 G 中的节点
        relevant_nodes = {
            node for node in relevant_nodes if node in G.vs["name"]}
        filtered_subgraph = G.subgraph(relevant_nodes)
        return filtered_subgraph


class PreRetrievalModuleTriple(PreRetrievalSubgraphAlign):
    def __init__(self, window: int = 4096):
        super().__init__(window)

    def process(self, query: Query, ranked_triples):
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, ranked_triples[:self.window])
        query.subgraph = filtered_subgraph
        return len(filtered_subgraph.es), query

    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], triples: Set[str]) -> ig.Graph:
        filtered_subgraph = ig.Graph(directed=True)
        nodes = set()
        for triple in triples:
            source, relation, target = triple.split(" -> ")
            if source not in nodes:
                nodes.add(source)
                filtered_subgraph.add_vertex(source)
            if target not in nodes:
                nodes.add(target)
                filtered_subgraph.add_vertex(target)
            filtered_subgraph.add_edge(source, target, name=relation)

        # Add seed nodes
        for e in seed_list:
            if e not in nodes:
                filtered_subgraph.add_vertex(e)
        return filtered_subgraph
