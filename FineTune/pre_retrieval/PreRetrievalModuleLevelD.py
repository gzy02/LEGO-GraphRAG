from .PreRetrievalModuleLevelC import PreRetrievalModuleLLM, PreRetrievalModule
from utils.LLM import LLM
from utils.Query import Query


class PreRetrievalModuleLLM_FT(PreRetrievalModuleLLM):
    def __init__(self, window: int = 32, llm_model: str = None):
        super().__init__()
        self.window = window
        if llm_model is None:
            raise ValueError("Pre Retrieval Model NO FOUND")
        else:
            self.llm_model = LLM(llm_model)

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query


class PreRetrievalModuleRoG(PreRetrievalModule):
    def __init__(self, predict_path):
        super().__init__()
        self.predict_path = predict_path
        import json
        with open(predict_path, "r") as f:
            predictions = {json.loads(line)["id"]: json.loads(line)[
                "prediction"] for line in f}
        self.predictions = predictions

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        prediction = self.predictions[query.qid]
        relations = set()
        for pred in prediction:
            for relation in pred:
                relations.add(relation)
        entities = set()
        for entity in query.entities:
            try:
                start_vertex = query.subgraph.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in query.subgraph.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        query.subgraph = query.subgraph.subgraph(entities)
        return query
