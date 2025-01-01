from config import emb_model_dir, rerank_model_dir
from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.SemanticModel import BGEModel, EmbeddingModel, BM25Model
import igraph as ig
from typing import List, Dict, Set
from abc import abstractmethod
from copy import deepcopy


class PreRetrievalModuleSemanticModel(PreRetrievalModule):
    def __init__(self, window: int = 32,  semantic_type="BM25", model_dir: str = None):
        super().__init__()
        self.window = window
        self.semantic_type = semantic_type

        if semantic_type == "BM25":
            # print("Using BM25 model")
            model_dir = "BM25"
            self.model = BM25Model()
        elif semantic_type == "EMB":
            # print("Using EMB model")
            model_dir = emb_model_dir if model_dir is None else model_dir
            self.model = EmbeddingModel(model_dir)
        else:
            # print("using BGE model")
            model_dir = rerank_model_dir if model_dir is None else model_dir
            self.model = BGEModel(model_dir)
        self.model_dir = model_dir

    def process(self, query: Query) -> Query:
        new_query = deepcopy(query)
        new_query = self._process(new_query)
        return new_query

    @abstractmethod
    def _process(self, query: Query) -> Query:
        pass


class PreRetrievalModuleEdge(PreRetrievalModuleSemanticModel):
    def __init__(self, window: int = 32, semantic_type="BM25", model_dir: str = None):
        super().__init__(window, semantic_type, model_dir)

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question

        relations = self.model.top_k(question, corpus, self.window)
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, relations)
        query.subgraph = filtered_subgraph
        return query

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

    def filter_subgraph_nodes(self, G: ig.Graph, seed_list: List[str], relations: Set[str]) -> ig.Graph:
        entities = set()
        for entity in seed_list:
            try:
                start_vertex = G.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in G.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        return G.subgraph(entities)


class PreRetrievalModuleNode(PreRetrievalModuleSemanticModel):
    def __init__(self, window: int = 1024, semantic_type="BM25", model_dir: str = None):
        super().__init__(window, semantic_type, model_dir)

    def _process(self, query: Query) -> Query:
        entities = {vertex["name"]
                    for vertex in query.subgraph.vs}
        corpus = list(entities)
        question = query.question

        entities = self.model.top_k(question, corpus, self.window)
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, entities)
        query.subgraph = filtered_subgraph
        return query

    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], entities: Set[str]) -> ig.Graph:
        relevant_nodes = entities.union(seed_list)
        # 过滤掉不存在于图 G 中的节点
        relevant_nodes = {
            node for node in relevant_nodes if node in G.vs["name"]}
        filtered_subgraph = G.subgraph(relevant_nodes)
        return filtered_subgraph


class PreRetrievalModuleTriples(PreRetrievalModuleSemanticModel):
    def __init__(self, window: int = 4096, semantic_type="BM25", model_dir: str = None):
        super().__init__(window, semantic_type, model_dir)

    def _process(self, query: Query) -> Query:
        triples = {f'{query.subgraph.vs[edge.source]["name"]} -> {edge["name"]} -> {query.subgraph.vs[edge.target]["name"]}'
                   for edge in query.subgraph.es}
        corpus = list(triples)
        question = query.question

        triples = self.model.top_k(question, corpus, self.window)
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, triples)
        query.subgraph = filtered_subgraph
        return query

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
