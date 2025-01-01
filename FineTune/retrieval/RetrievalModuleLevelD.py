from .RetrievalModuleLevelC import RetrievalModuleBeamSearch
from utils.Query import Query
from utils.LLM import LLM


class RetrievalModuleD(RetrievalModuleBeamSearch):
    def __init__(self, beam_width: int, hop: int, llm_model: str = None, thre: int = 24):
        super().__init__()
        self.beam_width = beam_width
        self.thre = thre
        if llm_model is None:
            raise ValueError("Retrieval Model NO FOUND")
        else:
            self.llm_model = LLM(llm_model)
        self.hop = hop

    def process(self, query: Query) -> Query:
        G = query.subgraph
        question = query.question
        query.reasoning_paths = self.get_reasoning_paths(
            G, query.entities, question)
        return query
