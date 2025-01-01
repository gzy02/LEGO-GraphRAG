import igraph as ig

from typing import List, Tuple, Dict
from utils.ReasoningPath import ReasoningPath


class Query:
    def __init__(self, info: Dict):
        self.qid: str = info['id']
        self.question: str = info['question']
        self.answers: List[str] | List[Dict[str, str]] = info['answers']
        self.entities: List[str] | List[Dict[str, str]] = info['entities']
        self.ppr_list: List[Dict[str, float]
                            ] = None if "ppr" not in info else info["ppr"]

        self.subgraph: ig.Graph = None
        self.reasoning_paths: List[ReasoningPath] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.window = 0
        self.user_input = ""
        self.llm_output = ""
