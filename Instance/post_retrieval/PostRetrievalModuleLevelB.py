from typing import List
from copy import deepcopy
from utils.Query import Query
from config import emb_model_dir, rerank_model_dir
from utils.ReasoningPath import ReasoningPath
from utils.SemanticModel import EmbeddingModel, BGEModel, BM25Model, RandomModel
from post_retrieval.PostRetrievalModule import PostRetrievalModule


class PostRetrievalModuleSemanticModel(PostRetrievalModule):
    def __init__(self, semantic_type="BM25", model_dir: str = None, window: int = 32):
        super().__init__()
        self.window = window
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

    def process(self, origin_query: Query) -> Query:
        query = deepcopy(origin_query)
        if len(query.reasoning_paths) == 0:
            return query
        query.reasoning_paths = self._process(query)
        return query

    def _process(self, query: Query) -> List[ReasoningPath]:
        path_dict = {str(path): path for path in query.reasoning_paths}
        question = query.question
        corpus = list(path_dict.keys())
        sorted_paths = self.model.top_k(question, corpus, self.window)
        return [path_dict[path] for path in sorted_paths]
