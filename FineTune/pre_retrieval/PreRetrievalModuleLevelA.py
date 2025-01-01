from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
import igraph as ig
from typing import List


class PreRetrievalModuleRandomWalk(PreRetrievalModule):
    def __init__(self, path_num: int = 128, steps: int = 4):
        super().__init__()
        self.path_num = path_num
        self.steps = steps

    def process(self, query: Query) -> Query:
        """Get k paths from the graph using random walk."""
        query.subgraph = self.get_subgraph(query.subgraph, query.entities)
        return query

    def get_subgraph(self, G: ig.Graph, entities: List[str]) -> ig.Graph:
        nodes = set()

        for entity in entities:
            try:
                start_vertex = G.vs.find(name=entity)
            except:
                continue
            for _ in range(self.path_num):
                path = G.random_walk(start=start_vertex,
                                     steps=self.steps, mode="all")
                nodes.update(path)
        subgraph = G.subgraph(nodes)
        return subgraph


if __name__ == "__main__":
    # Create a sample graph
    G = ig.Graph(directed=True)
    G.add_vertices(["A", "B", "C", "D"])
    G.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")])

    # Create a sample query
    query = Query(subgraph=G, entities=["A", "B"])

    # Instantiate the PreRetrievalModuleRandomWalk
    prm_random_walk = PreRetrievalModuleRandomWalk(num=3, steps=2)

    # Process the query to get the subgraph
    processed_query = prm_random_walk.process(query)

    # Access the resulting subgraph
    resulting_subgraph = processed_query.subgraph
    print(resulting_subgraph)
