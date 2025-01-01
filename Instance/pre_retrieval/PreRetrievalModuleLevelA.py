from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
import igraph as ig
from typing import List
from utils.Tools import Tools, get_k_hop_neighbors
from abc import abstractmethod
from copy import deepcopy


class PreRetrievalModuleNone(PreRetrievalModule):
    def __init__(self):
        super().__init__()

    def process(self, query: Query) -> Query:
        # new_query = deepcopy(query)
        return query


class PreRetrievalModuleStructure(PreRetrievalModule):
    def process(self, query: Query) -> Query:
        new_query = deepcopy(query)
        new_query = self._process(new_query)
        new_query = self.post_process(new_query)
        return new_query

    def post_process(self, query: Query) -> Query:
        # 将subgraph中的节点的name属性改为label属性
        for node in query.subgraph.vs:
            node["name"] = node["label"]

        query.answers = self.node_process(query.answers)
        query.entities = self.node_process(query.entities)
        return query

    def node_process(self, ents):
        new_answers = set()
        for ans in ents:
            if ans["text"] is None:
                new_answers.add(ans["kb_id"])
            elif ans["text"] != "":
                new_answers.add(ans["text"])
        return list(new_answers)

    @abstractmethod
    def _process(self, query: Query) -> Query:
        pass


class PreRetrievalModuleRandomWalk(PreRetrievalModuleStructure):
    def __init__(self, path_num: int = 128, steps: int = 4):
        super().__init__()
        self.path_num = path_num
        self.steps = steps
        self.tools = Tools()
        self.kg = self.tools.kg
        self.path_list = None

    def _process(self, query: Query) -> Query:
        nodes = set()
        for entity in query.entities:
            start_vertex = self.kg.vs.find(name=entity["kb_id"])

            for _ in range(self.path_num):
                path = self.kg.random_walk(start=start_vertex,
                                           steps=self.steps, mode="out")
                nodes.update(path)
        query.subgraph = self.kg.subgraph(nodes)
        return query

    def prefill(self, query: Query):
        if self.path_list is not None:
            return
        path_list = []
        vertexs = [self.kg.vs.find(name=entity["kb_id"])
                   for entity in query.entities]
        for _ in range(self.path_num):
            paths = []
            for start_vertex in vertexs:
                paths.append(self.kg.random_walk(start=start_vertex,
                                                 steps=self.steps, mode="out"))
            path_list.append(paths)
        self.path_list = path_list

    def process2(self, query: Query) -> Query:
        new_query = deepcopy(query)
        nodes = set()
        for paths in self.path_list[:self.path_num]:
            for path in paths:
                nodes.update(path)

        new_query.subgraph = self.kg.subgraph(nodes)
        new_query = self.post_process(new_query)
        return new_query


class PreRetrievalModulePPR(PreRetrievalModuleStructure):
    def __init__(self, max_ent=2000):
        super().__init__()
        self.max_ent = max_ent
        self.tools = Tools()
        self.kg = self.tools.kg

    def _process(self, query: Query) -> Query:
        """使用PPR算法获取最相关的子图
        由于计算量过大，已经事先预处理输入数据，直接返回即可
        """
        kb_id_list = [node["kb_id"] for node in query.ppr_list[:self.max_ent]]
        query.subgraph = self.kg.subgraph(kb_id_list)
        return query


class PreRetrievalModuleEgo(PreRetrievalModuleStructure):
    def __init__(self, hop=2):
        super().__init__()
        self.hop = hop
        self.tools = Tools()
        self.kg = self.tools.kg

    def _process(self, query: Query) -> Query:
        seed_list = [
            self.kg.vs.find(name=entity["kb_id"]).index for entity in query.entities
        ]

        query.subgraph = get_k_hop_neighbors(self.kg, seed_list, self.hop)
        return query
