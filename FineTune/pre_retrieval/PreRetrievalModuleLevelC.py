from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from typing import List, Dict
from utils.LLM import LLM
from utils.PromptTemplate import preretrieval_prompt
from config import reasoning_model, emb_model_dir
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    def __init__(self, model_dir: str = None):
        super().__init__()
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model = SentenceTransformer(
            model_dir, device="cuda"
        )
        self.model.eval()

    def encode(self, sentences, normalize_embeddings=False):
        with torch.no_grad():
            return self.model.encode(sentences, normalize_embeddings=normalize_embeddings)


class PreRetrievalModuleLLM(PreRetrievalModule):
    def __init__(self, window: int = 32, llm_path: str = None):
        super().__init__()
        self.emb = EmbeddingModel()
        self.window = window

        if llm_path is None:
            self.llm_model = LLM(reasoning_model)
            self.llm_path = reasoning_model
        else:
            self.llm_model = LLM(llm_path)
            self.llm_path = llm_path

    def process(self, query: Query) -> Query:
        """RoG Not FT
        """
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question
        question_embedding = self.emb.encode(
            question, normalize_embeddings=True)[None, :]
        relation_embeddings = self.emb.encode(
            corpus, normalize_embeddings=True)
        cosine_scores = (relation_embeddings * question_embedding).sum(1)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        relations = sorted_paths[:self.window]
        llm_input = preretrieval_prompt.format(relations='\n'.join(
            relations), question=question)
        answer = self.llm_model.invoke(llm_input)
        relations = {relation for relation in relations if relation in answer}
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
