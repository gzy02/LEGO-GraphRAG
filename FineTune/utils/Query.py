import igraph as ig

from typing import List, Tuple, Dict
from utils.ReasoningPath import ReasoningPath


class Query:
    def __init__(self, info: Dict):
        self.qid: str = info['id']
        self.question: str = info['question']
        self.answers: List[str] = info['answers']
        self.entities: List[str] = info['entities']
        self.subgraph: ig.Graph = info['subgraph']
        self.reasoning_paths: List[ReasoningPath] = []
