from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
from utils.LLM import LocalLLM
from config import reasoning_model
from utils.PromptTemplate import filter_prompt, FILTER_PERSONA
from utils.ReasoningPath import ReasoningPath
from typing import List, Dict
from copy import deepcopy
from utils.SemanticModel import SemanticModel


class PostRetrievalModuleLLM(PostRetrievalModule):
    def __init__(self, llm_model: LocalLLM = None, rank_model: SemanticModel = None, top_k: int = 32):
        super().__init__()
        self.top_k = top_k
        self.llm_model = llm_model
        self.rank_model = rank_model

    def process(self, query: Query) -> Query:
        reasoning_paths = self.filter(query)
        path_dict = {str(path): path for path in reasoning_paths}
        corpus = list(path_dict.keys())
        prompt_input = filter_prompt.format(
            question=query.question, entities='; '.join(query.entities), corpus='\n'.join(corpus))
        resp = self.llm_model.invoke(FILTER_PERSONA, prompt_input)
        query.input_tokens += resp["input_tokens"]
        query.output_tokens += resp["output_tokens"]
        answer = resp['response']
        filter_reasoning_paths = self.parse_answer(answer, path_dict)
        # if len(filter_reasoning_paths) == 0:
        #    filter_reasoning_paths = reasoning_paths
        query.reasoning_paths = filter_reasoning_paths
        return query

    async def aprocess(self, origin_query: Query) -> Query:
        return self.process(origin_query)

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
        query.st_tokens += self.rank_model.token_count(question)+sum(
            [self.rank_model.token_count(path) for path in corpus])
        return [path for path in query.reasoning_paths if str(path) in top_k_paths]
