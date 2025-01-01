from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.LLM import LLM
from utils.SemanticModel import EmbeddingModel, SemanticModel
from utils.PromptTemplate import PRERETIEVAL_EDGE_PERSONA, PRERETIEVAL_NODE_PERSONA, PRERETIEVAL_TRIPLE_PERSONA, preretrieval_edge_prompt, preretrieval_triples_prompt, preretrieval_node_prompt
import igraph as ig
from typing import List, Dict, Set


class PreRetrievalModuleLLMEdge(PreRetrievalModule):
    def __init__(self, window: int = 32, rank_model: SemanticModel = None, llm: LLM = None):
        super().__init__()
        self.rank_model = rank_model
        self.window = window
        self.llm = llm

    async def aprocess(self, query: Query) -> Query:
        query = await self._process(query)
        return query

    async def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        if len(relations) == 0:
            return query
        corpus = list(relations)
        question = query.question
        relations = self.rank_model.top_k(question, corpus, self.window)
        user_input = preretrieval_edge_prompt.format(relations='\n'.join(
            relations), question=question)
        answer, prompt_tokens, completion_tokens, request_time = await self.llm.ainvoke(PRERETIEVAL_EDGE_PERSONA, user_input)
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        # export subgraph with new relations
        relations = {relation for relation in relations if relation in answer}
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

    def process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        if len(relations) == 0:
            return query
        corpus = list(relations)
        question = query.question
        relations = self.rank_model.top_k(question, corpus, self.window)
        user_input = preretrieval_edge_prompt.format(relations='\n'.join(
            relations), question=question)
        answer, prompt_tokens, completion_tokens, request_time = self.llm.invoke(
            PRERETIEVAL_EDGE_PERSONA, user_input)
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        # export subgraph with new relations
        relations = {relation for relation in relations if relation in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, relations)
        query.subgraph = filtered_subgraph
        return query


class PreRetrievalModuleLLMNode(PreRetrievalModule):
    def __init__(self, window: int = 1024, rank_model: SemanticModel = None, llm: LLM = None):
        super().__init__()
        self.rank_model = rank_model
        self.window = window
        self.llm = llm

    async def aprocess(self, query: Query) -> Query:
        query = await self._process(query)
        return query

    async def _process(self, query: Query) -> Query:
        entities = {vertex["name"] for vertex in query.subgraph.vs}
        if len(entities) == 0:
            return query
        corpus = list(entities)
        question = query.question
        entities = self.rank_model.top_k(question, corpus, self.window)
        user_input = preretrieval_node_prompt.format(
            entities='\n'.join(entities), question=question)
        answer, prompt_tokens, completion_tokens, request_time = await self.llm.ainvoke(PRERETIEVAL_NODE_PERSONA, user_input)
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        entities = {entity for entity in entities if entity in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, entities)
        query.subgraph = filtered_subgraph
        return query

    def filter_subgraph(self, G: ig.Graph, seed_list: List[str], entities: Set[str]) -> ig.Graph:
        relevant_nodes = entities.union(seed_list)
        relevant_nodes = {
            node for node in relevant_nodes if node in G.vs["name"]}
        filtered_subgraph = G.subgraph(relevant_nodes)
        return filtered_subgraph

    def process(self, query):
        entities = {vertex["name"] for vertex in query.subgraph.vs}
        if len(entities) == 0:
            return query
        corpus = list(entities)
        question = query.question
        entities = self.rank_model.top_k(question, corpus, self.window)
        user_input = preretrieval_node_prompt.format(
            entities='\n'.join(entities), question=question)
        answer, prompt_tokens, completion_tokens, request_time = self.llm.invoke(
            PRERETIEVAL_NODE_PERSONA, user_input)
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        entities = {entity for entity in entities if entity in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, entities)
        query.subgraph = filtered_subgraph
        return query


class PreRetrievalModuleLLMTriples(PreRetrievalModule):
    def __init__(self, window: int = 4096, rank_model: SemanticModel = None, llm: LLM = None):
        super().__init__()
        self.rank_model = rank_model
        self.window = window
        self.llm = llm

    async def aprocess(self, query: Query) -> Query:
        query = await self._process(query)
        return query

    async def _process(self, query: Query) -> Query:
        triples = {
            f'{query.subgraph.vs[edge.source]["name"]} -> {edge["name"]} -> {query.subgraph.vs[edge.target]["name"]}' for edge in query.subgraph.es}
        if len(triples) == 0:
            return query
        corpus = list(triples)
        question = query.question
        triples = self.rank_model.top_k(question, corpus, self.window)
        user_input = preretrieval_triples_prompt.format(
            triples='\n'.join(triples), question=question)
        answer, prompt_tokens, completion_tokens, request_time = await self.llm.ainvoke(PRERETIEVAL_TRIPLE_PERSONA, user_input)
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        triples = {triple for triple in triples if triple in answer}
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
        for e in seed_list:
            if e not in nodes:
                filtered_subgraph.add_vertex(e)
        return filtered_subgraph

    def process(self, query: Query) -> Query:
        triples = {
            f'{query.subgraph.vs[edge.source]["name"]} -> {edge["name"]} -> {query.subgraph.vs[edge.target]["name"]}' for edge in query.subgraph.es}
        if len(triples) == 0:
            return query
        corpus = list(triples)
        question = query.question
        triples = self.rank_model.top_k(question, corpus, self.window)
        user_input = preretrieval_triples_prompt.format(
            triples='\n'.join(triples), question=question)
        answer, prompt_tokens, completion_tokens, request_time = self.llm.invoke(
            PRERETIEVAL_TRIPLE_PERSONA, user_input)
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        triples = {triple for triple in triples if triple in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, triples)
        query.subgraph = filtered_subgraph
        return query
