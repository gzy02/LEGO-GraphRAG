from post_retrieval.PostRetrievalModule import PostRetrievalModule
from utils.Query import Query
from utils.PromptTemplate import filter_prompt, FILTER_PERSONA
from utils.ReasoningPath import ReasoningPath
from typing import List, Dict, Tuple
from copy import deepcopy
from utils.SemanticModel import SemanticModel


class PostRetrievalModuleLLM(PostRetrievalModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.target = self.tokenizer.model_max_length//2 - \
            self.token_count(FILTER_PERSONA)

    def token_count(self, query):
        tokenized_prediction = self.tokenizer.encode(query)
        return len(tokenized_prediction)

    def get_user_input(self, ranked_corpus: List[str], question: str, entities, target: int) -> Tuple[str, int]:
        l, r = 0, len(ranked_corpus)
        best_prompt = ""
        best_length = 0
        while l < r:
            m = (l + r) // 2
            prompt = self.prepare_prompt(question, entities, ranked_corpus[:m])
            length = self.token_count(prompt)
            if length > target:
                r = m
            else:
                l = m + 1
                if length > best_length:
                    best_prompt = prompt
                    best_length = length
        return best_prompt, l

    def prepare_prompt(self, question: str, entities: List[str], corpus: List[str]) -> str:
        return filter_prompt.format(question=question, entities='; '.join(entities), corpus='\n'.join(corpus))

    def process(self, query: Query, reasoning_paths):
        path_dict = {str(path) for path in reasoning_paths}
        prompt_input, path_num = self.get_user_input(
            reasoning_paths, query.question, query.entities, self.target)
        return prompt_input, path_dict

    def post_process(self, answer, path_dict):
        filter_reasoning_paths = self.parse_answer(answer, path_dict)
        return filter_reasoning_paths

    def prepare_prompt(self, question: str, entities: List[str], corpus: List[str]) -> str:
        return filter_prompt.format(question=question, entities='; '.join(entities), corpus='\n'.join(corpus))

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path for path in path_dict if path in answer]

    async def aprocess(self, query):
        return await super().aprocess(query)
