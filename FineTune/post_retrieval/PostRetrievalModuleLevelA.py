from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
import random
from typing import List
from utils.ReasoningPath import ReasoningPath


class PostRetrievalModuleSimple(PostRetrievalModule):
    def __init__(self, top_k=32):
        super().__init__()
        self.top_k = top_k

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self._process(query)
        return query

    def _process(self, query: Query) -> List[ReasoningPath]:
        return self.randomFilter(query.reasoning_paths)

    def randomFilter(self, reasoning_paths: List[ReasoningPath]):
        return random.sample(reasoning_paths, min(len(reasoning_paths), self.top_k))
