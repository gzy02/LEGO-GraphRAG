from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
from utils.LLM import LLM
from config import reasoning_model
from utils.PromptTemplate import filter_prompt
from utils.ReasoningPath import ReasoningPath
from typing import List, Dict
import random
from .PostRetrievalModuleLevelB import PostRetrievalModuleEmb
from .PostRetrievalModuleLevelA import PostRetrievalModuleSimple


class PostRetrievalModuleLLM(PostRetrievalModule):
    def __init__(self, top_k: int = 32, llm_model: str = None):
        super().__init__()
        self.top_k = top_k
        if llm_model is None:
            self.llm_model = LLM(reasoning_model)
        else:
            self.llm_model = LLM(llm_model)

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self._process(query)
        return query

    def _process(self, query: Query) -> List[ReasoningPath]:
        question = query.question
        entities = query.entities

        reasoning_paths = self.filter(query)
        path_dict = {str(path): path for path in reasoning_paths}
        corpus = list(path_dict.keys())

        prompt_input = filter_prompt.format(
            question=question, entities='; '.join(entities), corpus='\n'.join(corpus))
        answer = self.llm_model.invoke(prompt_input)
        # print("Post Answer\n", answer)
        return self.parse_answer(answer, path_dict)

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path_dict[path] for path in path_dict if path in answer]

    def filter(self, query: Query) -> List[ReasoningPath]:
        if self.top_k == -1:
            return query.reasoning_paths
        if len(query.reasoning_paths) == 0:
            return []
        return self.embFilter(query)

    def randomFilter(self, query: Query) -> List[ReasoningPath]:
        return PostRetrievalModuleSimple(self.top_k)._process(query)

    def embFilter(self, query: Query) -> List[ReasoningPath]:
        return PostRetrievalModuleEmb(self.top_k)._process(query)
