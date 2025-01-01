from .PostRetrievalModuleLevelC import PostRetrievalModuleLLM
from utils.LLM import LLM
from utils.Query import Query


class PostRetrievalModuleLLM_FT(PostRetrievalModuleLLM):
    def __init__(self, top_k: int = 32, llm_model: str = None):
        super().__init__()
        self.top_k = top_k
        if llm_model is None:
            raise ValueError("Post Retrieval Model NO FOUND")
        else:
            self.llm_model = LLM(llm_model)

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self._process(query)
        return query
