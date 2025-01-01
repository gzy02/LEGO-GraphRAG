from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.LLM import LocalLLM
from utils.SemanticModel import EmbeddingModel, SemanticModel
from utils.PromptTemplate import PRERETIEVAL_EDGE_PERSONA, PRERETIEVAL_NODE_PERSONA, PRERETIEVAL_TRIPLE_PERSONA, preretrieval_edge_prompt, preretrieval_triples_prompt, preretrieval_node_prompt
import igraph as ig
from typing import List, Dict, Set


class PreRetrievalModuleLLMEdge(PreRetrievalModule):
    def __init__(self, window: int = 32, rank_model: SemanticModel = None, llm: LocalLLM = None):
        super().__init__()
        self.rank_model = rank_model
        self.window = window
        self.llm = llm

    async def aprocess(self, query: Query) -> Query:
        query = self.process(query)
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
        query.st_tokens += self.rank_model.token_count(question)+sum(
            [self.rank_model.token_count(r) for r in relations])
        resp = self.llm.invoke(
            PRERETIEVAL_EDGE_PERSONA, user_input)
        answer = resp["response"]
        prompt_tokens = resp["input_tokens"]
        completion_tokens = resp["output_tokens"]
        query.input_tokens += prompt_tokens
        query.output_tokens += completion_tokens
        # export subgraph with new relations
        relations = {relation for relation in relations if relation in answer}
        filtered_subgraph = self.filter_subgraph(
            query.subgraph, query.entities, relations)
        query.subgraph = filtered_subgraph
        return query
