from math import inf
import igraph as ig
from utils.Query import Query
from typing import Generator
import json
from typing import List, Set


def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("type.type.") or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation or "sameas" in relation:
        return True


def get_k_hop_neighbors(G: ig.Graph, seed_list: List[str], hop: int) -> ig.Graph:
    if hop == -1:
        return G
    visited = set(seed_list)
    current_layer = set(seed_list)
    for _ in range(hop):
        next_layer = set()
        for node in current_layer:
            neighbors = G.neighbors(node, mode="out")
            new_neighbors = set(neighbors) - visited
            next_layer.update(new_neighbors)
            visited.update(new_neighbors)
        current_layer = next_layer
    return G.subgraph(visited)


def get_query_nojudge(path: str) -> Generator[Query, None, None]:
    with open(path) as fp:
        for line in fp:
            data = json.loads(line)
            G = ig.Graph(directed=True)
            edges = []
            nodes = set()
            for triple in data["subgraph"]:
                head, rel, tail = triple
                nodes.add(head)
                nodes.add(tail)
                edges.append((head, tail, rel))
            G.add_vertices(list(nodes))
            G.add_edges([(edge[0], edge[1]) for edge in edges])
            G.es["name"] = [edge[2] for edge in edges]
            data["subgraph"] = G
            yield Query(data)


def get_query(path: str) -> Generator[Query, None, None]:
    with open(path) as fp:
        for line in fp:
            data = json.loads(line)
            if None in data["answers"] or None in data["entities"]:
                continue
            G = ig.Graph(directed=True)
            edges = []
            nodes = set()
            for triple in data["subgraph"]:
                head, rel, tail = triple
                if abandon_rels(rel):
                    continue
                nodes.add(head)
                nodes.add(tail)
                edges.append((head, tail, rel))
            ents = []
            for ent in (set(data["entities"]) & nodes):
                if not (ent.startswith("m.") or ent.startswith("g.")):
                    ents.append(ent)
            if len(ents) == 0 or len(data["answers"]) == 0 or len(data["question"]) == 0:
                continue

            data["entities"] = ents
            # data["answers"] = list(set(data["answers"])-nodes)
            # if len(set(data["answers"])-nodes) == 0:
            #    continue

            G.add_vertices(list(nodes))
            G.add_edges([(edge[0], edge[1]) for edge in edges])

            G.es["name"] = [edge[2] for edge in edges]
            data["subgraph"] = G
            if len(set(data["answers"]) & nodes) == 0:
                yield Query(data)  # 答案不在图中，PPR未覆盖的情况
                continue
            matrix = G.distances(data["entities"],
                                 (set(data["answers"]) & nodes))
            for line in matrix:
                if any(item != inf for item in line):  # 只要有一个路径不是无穷大，即可达
                    yield Query(data)
                    break  # 找到一个可达路径后立即退出


class Tools:
    """单例模式工具类，用于加载 id2name 映射文件，并提供 id2name 和 name2id 两个方法
    此后只要全生命流程中只需要一个实例的功能，都可以在此类中实现
    """
    _instance = None
    path_id2name = "/back-up/gzy/dataset/graphrag/process_data/id2name.txt"
    path_kg = "/back-up/gzy/dataset/graphrag/process_data/CWQ/subgraph/subgraph_hop2.txt"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tools, cls).__new__(cls)
            cls.id2name_dict = {}
            cls.name2id_dict = {}
            with open(cls.path_id2name, "r") as fp:
                for line in fp:
                    mid, rel, name = line.strip().split('\t')
                    cls.id2name_dict[mid] = name
                    cls.name2id_dict[name] = mid

            # cls.kg = cls._get_kg()
        return cls._instance

    def _get_kg(self):
        G = ig.Graph(directed=True)
        with open(self.path_kg, "r") as fp:
            edges = []
            nodes = set()
            for line in fp:
                head, rel, tail = line.strip().split('\t')
                nodes.add(head)
                nodes.add(tail)
                edges.append((head, tail, rel))
        G.add_vertices(list(nodes))
        G.add_edges([(edge[0], edge[1]) for edge in edges])
        G.es["relation"] = [edge[2] for edge in edges]
        return G

    def id2name(self, mid):
        if mid in self.id2name_dict:
            return self.id2name_dict[mid]
        else:
            return mid

    def name2id(self, name):
        if name in self.name2id_dict:
            return self.name2id_dict[name]
        else:
            return name


if __name__ == "__main__":
    from time import time
    import json
    # 使用示例
    t = time()
    singleton_dict = Tools()
    print("第一次加载 id2name 映射文件耗时：", time()-t)
    name = singleton_dict.id2name("m.0w0015s")
    print(name)

    t = time()
    another_instance = Tools()
    print("第二次加载 id2name 映射文件耗时：", time()-t)
    name = another_instance.id2name("m.05ghm98")
    print(name)

    # 安全地打开和读取文件
    with open("/home/gzy/graphrag/data/demo.json", "r") as fp:
        data = json.load(fp)

    # 遍历 data["subgraph"]，修改 triple 中的元素
    # 这里直接修改 triple 元素，因为 triple 是列表的引用，所以会直接影响 data 中的内容
    for i, triple in enumerate(data["subgraph"]):
        data["subgraph"][i][0] = singleton_dict.id2name(triple[0])
        data["subgraph"][i][2] = singleton_dict.id2name(triple[2])

    # 使用 with 语句安全地写入修改后的数据到新文件
    with open("/home/gzy/graphrag/data/demo2.json", "w") as fp:
        json.dump(data, fp)
