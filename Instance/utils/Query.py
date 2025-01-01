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
        self.input_tokens = 0 if "input_tokens" not in info else info["input_tokens"]
        self.output_tokens = 0 if "output_tokens" not in info else info["output_tokens"]
        self.llm_call = 0 if "llm_call" not in info else info["llm_call"]
        self.window = 0 if "window" not in info else info["window"]
        self.user_input = ""
        self.llm_output = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.qid,
            "question": self.question,
            "answers": self.answers,
            "entities": self.entities,
            # "ppr": self.ppr_list,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_call": self.llm_call,
            "window": self.window,
            "user_input": self.user_input,
            "llm_output": self.llm_output,
            "ReasoningPaths": "\n".join([str(path) for path in self.reasoning_paths])
        }
