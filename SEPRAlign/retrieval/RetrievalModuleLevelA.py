from utils.Tools import get_k_hop_neighbors
from retrieval.RetrievalModule import RetrievalModule
from utils.Query import Query
import igraph as ig
from utils.ReasoningPath import ReasoningPath
from typing import List


class RetrievalModuleBFS(RetrievalModule):
    def __init__(self, hop: int = 4):
        super().__init__()
        self.hop = hop

    async def aprocess(self, query: Query) -> Query:
        return self.process(query)

    def process(self, query: Query) -> Query:
        G = query.subgraph
        query.reasoning_paths = self.get_reasoning_paths(
            G, query.entities)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str]) -> List[ReasoningPath]:
        reasoning_paths = []
        for entity in entities:
            try:
                source = G.vs.find(name=entity)
            except:
                continue
            out_paths = G.get_all_simple_paths(
                v=source, cutoff=self.hop, mode="out")
            for path in out_paths:
                if G.vs[path[-1]]['name'].startswith('m.') or G.vs[path[-1]]['name'].startswith('g.'):
                    continue
                reasoning_path = ReasoningPath(entity)
                for i in range(len(path) - 1):
                    edge_id = G.get_eid(path[i], path[i + 1])
                    edge = G.es[edge_id]
                    triple = (G.vs[path[i]]['name'],
                              edge['name'], G.vs[path[i + 1]]['name'])
                    reasoning_path.add_triple(triple)
                reasoning_paths.append(reasoning_path)
        return reasoning_paths


class RetrievalModuleDij(RetrievalModule):
    def __init__(self, hop: int = 4):
        super().__init__()
        self.hop = hop

    async def aprocess(self, query: Query) -> Query:
        return self.process(query)

    def process(self, query: Query) -> Query:
        G = query.subgraph
        query.reasoning_paths = self.get_reasoning_paths(
            G, query.entities)
        return query

    def get_reasoning_paths(self, graph: ig.Graph, entities: List[str]) -> List[ReasoningPath]:
        reasoning_paths = []
        for entity in entities:
            try:
                source = graph.vs.find(name=entity)
            except:
                continue
            G = get_k_hop_neighbors(graph, [entity], self.hop)
            source = G.vs.find(name=entity)
            out_paths = G.get_shortest_paths(
                v=source, mode="out")
            for path in out_paths:
                if G.vs[path[-1]]['name'].startswith('m.') or G.vs[path[-1]]['name'].startswith('g.'):
                    continue
                reasoning_path = ReasoningPath(entity)
                for i in range(len(path) - 1):
                    edge_id = G.get_eid(path[i], path[i + 1])
                    edge = G.es[edge_id]
                    triple = (G.vs[path[i]]['name'],
                              edge['name'], G.vs[path[i + 1]]['name'])
                    reasoning_path.add_triple(triple)
                reasoning_paths.append(reasoning_path)
        return reasoning_paths


class RetrievalModuleDFSRoG(RetrievalModule):
    def __init__(self, predict_path):
        super().__init__()
        self.predict_path = predict_path
        import json
        with open(predict_path, "r") as f:
            predictions = {json.loads(line)["id"]: json.loads(line)[
                "prediction"] for line in f}
        self.predictions = predictions

    def process(self, query: Query) -> Query:
        G = query.subgraph
        entities = query.entities
        preds = self.predictions[query.qid]
        reasoning_paths = []

        for pred in preds:
            "沿着pred的路径DFS找paths"
            reasoning_paths.extend(self.get_reasoning_paths(G, entities, pred))
        query.reasoning_paths = reasoning_paths
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], pred: List[str]) -> List[ReasoningPath]:
        reasoning_paths = []
        for entity in entities:
            try:
                source = G.vs.find(name=entity)
            except:
                continue

            def dfs(current_vertex, current_path, current_pred_index):
                if current_pred_index == len(pred):
                    reasoning_path = ReasoningPath(entity)
                    for i in range(len(current_path) - 1):
                        edge_id = G.get_eid(
                            current_path[i], current_path[i + 1])
                        edge = G.es[edge_id]
                        triple = (G.vs[current_path[i]]['name'],
                                  edge['name'], G.vs[current_path[i + 1]]['name'])
                        reasoning_path.add_triple(triple)
                    reasoning_paths.append(reasoning_path)
                    return

                for neighbor in G.neighbors(current_vertex, mode="out"):
                    edge_id = G.get_eid(current_vertex, neighbor)
                    edge = G.es[edge_id]
                    if edge['name'] == pred[current_pred_index]:
                        dfs(neighbor, current_path +
                            [neighbor], current_pred_index + 1)

            dfs(source.index, [source.index], 0)

        return reasoning_paths
