from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
import random
from typing import List
from utils.ReasoningPath import ReasoningPath
from copy import deepcopy


class PostRetrievalModuleNone(PostRetrievalModule):
    def __init__(self):
        super().__init__()

    def process(self, query: Query) -> Query:
        return query

    async def aprocess(self, query: Query) -> Query:
        return query


class PostRetrievalModuleSimple(PostRetrievalModule):
    def __init__(self, top_k=32):
        super().__init__()
        self.top_k = top_k

    def process(self, origin_query: Query) -> Query:
        query = deepcopy(origin_query)
        query.reasoning_paths = self._process(query)
        return query

    async def aprocess(self, query: Query) -> Query:
        return self.process(query)

    def _process(self, query: Query) -> List[ReasoningPath]:
        return self.randomFilter(query.reasoning_paths)

    def randomFilter(self, reasoning_paths: List[ReasoningPath]):
        if self.top_k == -1:
            return reasoning_paths
        elif len(reasoning_paths) == 0:
            return []
        elif len(reasoning_paths) <= self.top_k:
            return reasoning_paths
        return random.sample(reasoning_paths, self.top_k)
