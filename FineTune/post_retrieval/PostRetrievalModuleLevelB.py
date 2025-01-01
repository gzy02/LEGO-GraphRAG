import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
from rank_bm25 import BM25Okapi
from config import emb_model_dir, rerank_model_dir
from utils.ReasoningPath import ReasoningPath
from typing import List, Dict
from utils.SentenceModel import EmbeddingModel, BGEModel


def find_answer(sorted_paths, entitys):
    ans = []
    for index, path in enumerate(sorted_paths):
        for entity in entitys:
            if entity in path:
                ans.append(index)
    if len(ans) == 0:
        print("Answer entity NO FOUND")
        return
    print("Answer entity position: ", end=" ")
    for index in ans[:-1]:
        print(index, "/", len(sorted_paths), sep='', end=";")
    print(ans[-1], "/", len(sorted_paths), sep='')
    return ans


class PostRetrievalModuleBM25(PostRetrievalModule):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def process(self, query: Query) -> Query:
        if len(query.reasoning_paths) == 0:
            return query
        query.reasoning_paths = self._process(query)
        return query

    def _process(self, query: Query) -> List[ReasoningPath]:
        path_dict = {str(path): path for path in query.reasoning_paths}
        question = query.question
        corpus = list(path_dict.keys())
        # Create BM25 object
        bm25 = BM25Okapi(corpus)

        # Tokenize the question
        tokenized_question = question.split()

        # Calculate BM25 scores
        bm25_scores = bm25.get_scores(tokenized_question)

        # Sort the reasoning paths based on BM25 scores
        sorted_paths = [path for _, path in sorted(
            zip(bm25_scores, corpus), reverse=True)]
        # find_answer(sorted_paths, query.answers)
        # Update the reasoning paths in the query object
        return [path_dict[path] for path in sorted_paths[:self.top_k]]


class PostRetrievalModuleEmb(PostRetrievalModule):
    def __init__(self, top_k: int = 5, model_dir: str = None):
        super().__init__()
        self.top_k = top_k
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = EmbeddingModel(model_dir)

    def _process(self, query: Query) -> List[ReasoningPath]:
        path_dict = {str(path): path for path in query.reasoning_paths}
        question = query.question
        all_chunks = list(path_dict.keys())

        cosine_scores = self.model.get_scores(question, all_chunks)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, all_chunks), reverse=True)]
        # find_answer(sorted_paths, query.answers)
        return [path_dict[path] for path in sorted_paths[:self.top_k]]

    def process(self, query: Query) -> Query:
        if len(query.reasoning_paths) == 0:
            return query
        query.reasoning_paths = self._process(query)
        return query


class PostRetrievalModuleBGE(PostRetrievalModule):
    def __init__(self, top_k: int = 5, model_dir: str = None):
        super().__init__()
        self.top_k = top_k
        model_dir = rerank_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = BGEModel(model_dir)

    def process(self, query: Query) -> Query:
        if len(query.reasoning_paths) == 0:
            return query
        query.reasoning_paths = self._process(query)
        return query

    def _process(self, query: Query) -> List[ReasoningPath]:
        path_dict = {str(path): path for path in query.reasoning_paths}
        question = query.question
        contents = list(path_dict.keys())
        scores = self.model.get_scores(question, contents)

        sorted_paths = [path for _, path in sorted(
            zip(scores, contents), reverse=True)]
        # find_answer(sorted_paths, query.answers)
        return [path_dict[path] for path in sorted_paths[:self.top_k]]
