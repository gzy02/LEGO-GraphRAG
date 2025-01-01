from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.LLM import LocalLLM
from utils.SemanticModel import EmbeddingModel, SemanticModel
from utils.PromptTemplate import PRERETIEVAL_EDGE_PERSONA, PRERETIEVAL_NODE_PERSONA, PRERETIEVAL_TRIPLE_PERSONA, preretrieval_edge_prompt, preretrieval_triples_prompt, preretrieval_node_prompt
import igraph as ig
from typing import List, Dict, Set, Tuple
from abc import ABC, abstractmethod


class PreRetrievalModuleLLMTokenAlign(PreRetrievalModule):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise ValueError("A tokenizer must be provided")
        super().__init__()

    def token_count(self, query):
        tokenized_prediction = self.tokenizer.encode(query)
        return len(tokenized_prediction)

    @abstractmethod
    def construct_prompt(self, corpus: List[str], question: str):
        return

    def get_user_input(self, ranked_corpus: List[str], question: str, target: int) -> Tuple[str, int]:
        l, r = 0, len(ranked_corpus)
        best_prompt = ""
        best_length = 0
        while l < r:
            m = (l + r) // 2
            prompt = self.construct_prompt(ranked_corpus[:m], question)
            length = self.token_count(prompt)
            if length > target:
                r = m
            else:
                l = m + 1
                if length > best_length:
                    best_prompt = prompt
                    best_length = length
        return best_prompt, l


class PreRetrievalModuleLLMEdgeTokenAlign(PreRetrievalModuleLLMTokenAlign):
    def __init__(self, tokenizer, window: int = 4096, rank_model: SemanticModel = None):
        super().__init__(tokenizer)
        self.rank_model = rank_model
        self.window = window - \
            self.token_count(PRERETIEVAL_EDGE_PERSONA)
        if self.window < 0:
            raise ValueError("The window size is too small")

    async def aprocess(self, query: Query) -> Query:
        return query

    def construct_prompt(self, relations: List[str], question: str):
        return preretrieval_edge_prompt.format(relations='\n'.join(relations), question=question)

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

    def pre_process(self, query: Query) -> List[str]:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question
        relations = self.rank_model.top_k(question, corpus, -1)
        return relations

    def process(self, query: Query, relations) -> Query:
        question = query.question
        user_input, _ = self.get_user_input(relations, question, self.window)
        return user_input

    def post_process(self, answer, query):
        query.output_tokens += self.token_count(answer)
        query.llm_output = answer
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        # export subgraph with new relations
        relations = {relation for relation in relations if relation in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, relations)
        query.subgraph = filtered_subgraph
        return query


class PreRetrievalModuleLLMNodeTokenAlign(PreRetrievalModuleLLMTokenAlign):
    def __init__(self, tokenizer, window: int = 4096, rank_model: SemanticModel = None):
        super().__init__(tokenizer)
        self.rank_model = rank_model
        self.window = window - \
            self.token_count(PRERETIEVAL_NODE_PERSONA)
        if self.window < 0:
            raise ValueError("The window size is too small")
        self.tokenizer = tokenizer

    def construct_prompt(self, relations: List[str], question: str):
        return preretrieval_node_prompt.format(entities='\n'.join(relations), question=question)

    async def aprocess(self, query: Query) -> Query:
        return query

    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], entities: Set[str]) -> ig.Graph:
        relevant_nodes = entities.union(seed_list)
        relevant_nodes = {
            node for node in relevant_nodes if node in G.vs["name"]}
        filtered_subgraph = G.subgraph(relevant_nodes)
        return filtered_subgraph

    def pre_process(self, query: Query) -> List[str]:
        entities = {vertex["name"] for vertex in query.subgraph.vs}
        corpus = list(entities)
        question = query.question
        entities = self.rank_model.top_k(question, corpus, -1)
        return entities

    def process(self, query, entities):
        user_input, _ = self.get_user_input(
            entities, query.question, self.window)
        return user_input

    def post_process(self, answer, query):
        query.output_tokens += self.token_count(answer)
        query.llm_output = answer
        entities = {vertex["name"] for vertex in query.subgraph.vs}
        entities = {entity for entity in entities if entity in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, entities)
        query.subgraph = filtered_subgraph
        return query


class PreRetrievalModuleLLMTripleTokenAlign(PreRetrievalModuleLLMTokenAlign):
    def __init__(self, tokenizer, window: int = 4096, rank_model: SemanticModel = None):
        super().__init__(tokenizer)
        self.rank_model = rank_model
        self.window = window - \
            self.token_count(PRERETIEVAL_TRIPLE_PERSONA)
        if self.window < 0:
            raise ValueError("The window size is too small")
        self.tokenizer = tokenizer

    def construct_prompt(self, relations: List[str], question: str):
        return preretrieval_triples_prompt.format(triples='\n'.join(relations), question=question)

    async def aprocess(self, query: Query) -> Query:
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
        for e in seed_list:
            if e not in nodes:
                filtered_subgraph.add_vertex(e)
        return filtered_subgraph

    def pre_process(self, query: Query) -> List[str]:
        triples = {
            f'{query.subgraph.vs[edge.source]["name"]} -> {edge["name"]} -> {query.subgraph.vs[edge.target]["name"]}' for edge in query.subgraph.es}
        corpus = list(triples)
        question = query.question
        triples = self.rank_model.top_k(question, corpus, -1)
        return triples

    def process(self, query: Query, triples) -> Query:
        question = query.question
        user_input, _ = self.get_user_input(triples, question, self.window)
        return user_input

    def post_process(self, answer, query):
        query.output_tokens += self.token_count(answer)
        query.llm_output = answer
        triples = {
            f'{query.subgraph.vs[edge.source]["name"]} -> {edge["name"]} -> {query.subgraph.vs[edge.target]["name"]}' for edge in query.subgraph.es}
        triples = {triple for triple in triples if triple in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, triples)
        query.subgraph = filtered_subgraph
        return query
