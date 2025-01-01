from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
from utils.LLM import LLM
from config import reasoning_model
from utils.PromptTemplate import filter_prompt, FILTER_PERSONA
from utils.ReasoningPath import ReasoningPath
from typing import List, Dict
from copy import deepcopy
from utils.SemanticModel import SemanticModel


class PostRetrievalModuleAgentLLM(PostRetrievalModule):
    def __init__(self, llm_model: LLM = None, window: int = 64):
        super().__init__()
        self.window = window
        self.llm_model = llm_model

    def process(self, origin_query: Query) -> Query:
        query = deepcopy(origin_query)
        return query

    async def aprocess(self, query: Query) -> Query:
        query.reasoning_paths = await self._process(query)
        return query

    async def _process(self, query: Query) -> List[ReasoningPath]:
        final_reasoning_paths = []
        for i in range(0, len(query.reasoning_paths), self.window):
            reasoning_paths = query.reasoning_paths[i:i+self.window]
            path_dict = {str(path): path for path in reasoning_paths}
            corpus = list(path_dict.keys())

            prompt_input = filter_prompt.format(
                question=query.question, entities='; '.join(query.entities), corpus='\n'.join(corpus))
            resp = await self.llm_model.ainvoke(FILTER_PERSONA, prompt_input)
            answer = resp[0]
            query.input_tokens += resp[1]
            query.output_tokens += resp[2]
            query.llm_call += 1
            filter_reasoning_paths = self.parse_answer(answer, path_dict)
            final_reasoning_paths.extend(filter_reasoning_paths)

        # 最后问一次
        path_dict = {str(path): path for path in final_reasoning_paths}
        corpus = list(path_dict.keys())
        prompt_input = filter_prompt.format(question=query.question, entities='; '.join(
            query.entities), corpus='\n'.join(corpus))
        resp = await self.llm_model.ainvoke(FILTER_PERSONA, prompt_input)
        answer = resp[0]
        query.input_tokens += resp[1]
        query.output_tokens += resp[2]
        query.llm_call += 1
        final_reasoning_paths = self.parse_answer(answer, path_dict)

        return final_reasoning_paths

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path_dict[path] for path in path_dict if path in answer]


class PostRetrievalModuleLLMFilter(PostRetrievalModule):
    def __init__(self, llm_model: LLM = None):
        super().__init__()
        self.llm_model = llm_model

    def process(self, origin_query: Query) -> Query:
        query = deepcopy(origin_query)
        return query

    async def aprocess(self, query: Query) -> Query:
        query.reasoning_paths = await self._process(query)
        return query

    async def _process(self, query: Query) -> List[ReasoningPath]:
        # 最后问一次
        path_dict = {str(path): path for path in query.reasoning_paths}
        corpus = list(path_dict.keys())
        prompt_input = filter_prompt.format(question=query.question, entities='; '.join(
            query.entities), corpus='\n'.join(corpus))
        resp = await self.llm_model.ainvoke(FILTER_PERSONA, prompt_input)
        answer = resp[0]
        query.input_tokens += resp[1]
        query.output_tokens += resp[2]
        query.llm_call += 1
        query.user_input = FILTER_PERSONA+prompt_input
        query.llm_output = answer
        final_reasoning_paths = self.parse_answer(answer, path_dict)

        return final_reasoning_paths

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path_dict[path] for path in path_dict if path in answer]
