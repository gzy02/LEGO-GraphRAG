from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
from utils.LLM import LLM
from config import reasoning_model
from utils.PromptTemplate import filter_prompt, FILTER_PERSONA
from utils.ReasoningPath import ReasoningPath
from typing import List, Dict
from copy import deepcopy
from utils.SemanticModel import SemanticModel


class PostRetrievalModuleLLM(PostRetrievalModule):
    def __init__(self, llm_model: LLM = None, rank_model: SemanticModel = None, top_k: int = 32):
        super().__init__()
        self.top_k = top_k
        self.llm_model = llm_model
        self.rank_model = rank_model

    def process(self, origin_query: Query) -> Query:
        query = deepcopy(origin_query)
        reasoning_paths = self.filter(query)
        path_dict = {str(path): path for path in reasoning_paths}
        corpus = list(path_dict.keys())

        prompt_input = filter_prompt.format(
            question=query.question, entities='; '.join(query.entities), corpus='\n'.join(corpus))
        answer = self.llm_model.invoke(FILTER_PERSONA, prompt_input)[0]
        filter_reasoning_paths = self.parse_answer(answer, path_dict)
        # if len(filter_reasoning_paths) == 0:
        #    filter_reasoning_paths = reasoning_paths
        query.reasoning_paths = filter_reasoning_paths
        return query

    async def aprocess(self, origin_query: Query) -> Query:
        query = deepcopy(origin_query)
        query.reasoning_paths = await self._process(query)
        return query

    async def _process(self, query: Query) -> List[ReasoningPath]:
        reasoning_paths = self.filter(query)
        path_dict = {str(path): path for path in reasoning_paths}
        corpus = list(path_dict.keys())

        prompt_input = filter_prompt.format(
            question=query.question, entities='; '.join(query.entities), corpus='\n'.join(corpus))
        resp = await self.llm_model.ainvoke(FILTER_PERSONA, prompt_input)
        answer = resp[0]
        filter_reasoning_paths = self.parse_answer(answer, path_dict)
        # if len(filter_reasoning_paths) == 0:
        #    filter_reasoning_paths = reasoning_paths
        return filter_reasoning_paths

    def prepare_prompt(self, question: str, entities: List[str], corpus: List[str]) -> str:
        return filter_prompt.format(question=question, entities='; '.join(entities), corpus='\n'.join(corpus))

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path_dict[path] for path in path_dict if path in answer]

    def filter(self, query: Query) -> List[ReasoningPath]:
        if self.top_k == -1:
            return query.reasoning_paths
        if len(query.reasoning_paths) == 0:
            return []
        corpus = [str(path) for path in query.reasoning_paths]
        question = query.question
        top_k_paths = self.rank_model.top_k(question, corpus, self.top_k)
        return [path for path in query.reasoning_paths if str(path) in top_k_paths]
