from .PreRetrievalModuleLevelC import PreRetrievalModule
from utils.Query import Query


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
