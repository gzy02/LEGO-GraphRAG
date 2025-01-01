from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer
from config import reasoning_model, emb_model_dir, rerank_model_dir
from utils.PromptTemplate import preretrieval_prompt
from utils.LLM import LLM
from typing import List, Dict
import igraph as ig
from rank_bm25 import BM25Okapi
from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.SentenceModel import BGEModel, EmbeddingModel
# from utils.PPR import personalized_pagerank, rank_ppr_ents


class PreRetrievalModulePPR(PreRetrievalModule):
    def __init__(self, mode="fixed", max_ent=2000, min_ppr=0.005, restart_prob=0.8):
        super().__init__()
        self.mode = mode
        self.max_ent = max_ent
        self.min_ppr = min_ppr
        self.restart_prob = restart_prob

    def process(self, query: Query) -> Query:
        """使用PPR算法获取最相关的子图
        由于计算量过大，已经事先预处理输入数据，直接返回即可
        """
        # G=Tools().get_kg()
        # rank_ppr_ents(G,None,query.entities,self.mode,self.max_ent,self.min_ppr,self.restart_prob)
        # query.subgraph=G.subgraph(entities)
        return query


class PreRetrievalModuleBM25(PreRetrievalModule):
    def __init__(self, window: int = 32):
        super().__init__()
        self.window = window

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question
        # Create BM25 object
        bm25 = BM25Okapi(corpus)

        # Tokenize the question
        tokenized_question = question.split()

        # Calculate BM25 scores
        cosine_scores = bm25.get_scores(tokenized_question)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        relations = sorted_paths[:self.window]
        entities = set()
        for entity in query.entities:
            try:
                start_vertex = query.subgraph.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in query.subgraph.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        query.subgraph = query.subgraph.subgraph(entities)
        return query


class PreRetrievalModuleEmb(PreRetrievalModule):
    def __init__(self, window: int = 32, model_dir: str = None):
        super().__init__()
        self.window = window
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.emb = EmbeddingModel(model_dir)

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question
        cosine_scores = self.emb.get_scores(question, corpus)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        relations = sorted_paths[:self.window]
        entities = set()
        for entity in query.entities:
            try:
                start_vertex = query.subgraph.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in query.subgraph.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        query.subgraph = query.subgraph.subgraph(entities)
        return query


class PreRetrievalModuleBGE(PreRetrievalModule):
    def __init__(self, window: int = 32, model_dir: str = None):
        super().__init__()

        self.window = window
        model_dir = rerank_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.bge = BGEModel(model_dir)

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question
        cosine_scores = self.bge.get_scores(question, corpus)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        relations = sorted_paths[:self.window]
        # export subgraph with new relations
        entities = set()
        for entity in query.entities:
            try:
                start_vertex = query.subgraph.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in query.subgraph.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        query.subgraph = query.subgraph.subgraph(entities)
        return query
