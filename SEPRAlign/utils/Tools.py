from tqdm import tqdm
import pickle
import igraph as ig
from utils.Query import Query
from typing import Generator
import json
from typing import List, Set
from time import time
import psutil
import gc


def token_count(tokenizer, query):
    tokenized_prediction = tokenizer.encode(query)
    return len(tokenized_prediction)


def wait_for_pid(pid):
    try:
        # 获取进程对象
        process = psutil.Process(pid)
        while process.is_running():
            print(f"Waiting for process {pid} to finish...")
            time.sleep(30)
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist.")


def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("type.type.") or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation or "sameas" in relation:
        return True


def get_k_hop_neighbors(G: ig.Graph, seed_list: List[int], hop: int) -> ig.Graph:
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


def get_query(kg: ig.Graph, path: str) -> Generator[Query, None, None]:
    with open(path) as fp:
        for line in fp:
            data = json.loads(line)
            valid_entities_id = set()
            valid_entities = []
            for entity in data["entities"]:
                if entity["kb_id"] in valid_entities_id:
                    continue
                try:
                    start_vertex = kg.vs.find(name=entity["kb_id"])
                    valid_entities_id.add(entity["kb_id"])
                    valid_entities.append(entity)
                except:
                    continue
            data["entities"] = valid_entities

            valid_entities_id = set()
            valid_entities = []
            for answer in data["answers"]:
                if answer["kb_id"] not in valid_entities_id:
                    valid_entities_id.add(answer["kb_id"])
                    valid_entities.append(answer)
            data["answers"] = valid_entities
            yield Query(data)


def construct_graph(triples: List[List[str]]) -> ig.Graph:
    G = ig.Graph(directed=True)
    if len(triples) == 0:
        return G
    nodes = set()
    for triple in triples:
        head, rel, tail = triple
        nodes.add(head)
        nodes.add(tail)
    G.add_vertices(list(nodes))
    G.add_edges([(triple[0], triple[2]) for triple in triples])
    G.es["name"] = [triple[1] for triple in triples]
    return G


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["name"]
        tail = G.vs[edge.target]["name"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


def get_query_subgraph(subgraph_file):
    with open(subgraph_file, "r") as f:
        infos = json.load(f)
        for index, info in enumerate(infos):
            triples = info["query_info"]["subgraph"]
            query = Query(info["query_info"])
            query.subgraph = construct_graph(triples)
            yield query


"""
class Tools:
    _instance = None
    redis_host = '127.0.0.1'
    redis_port = 16379

    path_id2name = "/back-up/gzy/id2name.txt"
    path_id2name_pkl = "/back-up/gzy/id2name.pkl"
    path_name2id_pkl = "/back-up/gzy/name2id.pkl"
    path_kg = "/back-up/gzy/rel_filter.txt"
    path_kg_pkl = "/back-up/gzy/rel_filter_name.pkl"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tools, cls).__new__(cls)
            cls.redis_client = redis.StrictRedis(
                host=cls.redis_host, port=cls.redis_port, decode_responses=False)

            cls.id2name_dict, cls.name2id_dict = cls._get_id2name()
            cls.kg = cls._get_kg()
        return cls._instance

    @classmethod
    def _get_id2name(self):
        print("开始读入 id2name 映射文件 loading...")
        id2name_dict = {}
        name2id_dict = {}
        try:
            data = self.load_from_redis(self.redis_client, 'id2name_data')
            id2name_dict, name2id_dict = data['id2name_dict'], data['name2id_dict']
            print("id2name loaded from Redis.")
        except (TypeError, KeyError, pickle.UnpicklingError):
            try:
                id2name_dict = pickle.load(open(self.path_id2name_pkl, "rb"))
                name2id_dict = pickle.load(open(self.path_name2id_pkl, "rb"))
                print("id2name loaded from pickle file.")
            except (FileNotFoundError, EOFError):
                id2name_dict, name2id_dict = self._load_id2name()
                pickle.dump(id2name_dict, open(self.path_id2name_pkl, "wb"))
                pickle.dump(name2id_dict, open(self.path_name2id_pkl, "wb"))
                print("id2name saved to pickle file.")

            self.store_to_redis(self.redis_client, 'id2name_data', pickle.dumps({
                'id2name_dict': id2name_dict,
                'name2id_dict': name2id_dict
            }))
            print("id2name saved to Redis.")
        return id2name_dict, name2id_dict

    @classmethod
    def _load_id2name(self):
        id2name_dict = {}
        name2id_dict = {}
        with open(self.path_id2name, "r") as fp:
            for line in fp:
                mid, rel, name = line.strip().split('\t')
                id2name_dict[mid] = name
                name2id_dict[name] = mid
        return id2name_dict, name2id_dict

    @classmethod
    def _get_kg(self):
        print("开始读入大图 loading...")
        try:
            G = self.load_from_redis(self.redis_client, 'kg')
            print("Graph loaded from Redis.")
        except (TypeError, KeyError, pickle.UnpicklingError):
            try:
                G = pickle.load(open(self.path_kg_pkl, "rb"))
                print("Graph loaded from pickle file.")
            except (FileNotFoundError, EOFError):
                G = ig.Graph(directed=True)
                with open(self.path_kg, "r") as fp:
                    edges = []
                    nodes = set()
                    for line in tqdm(fp):
                        head, rel, tail = line.strip().split('\t')
                        nodes.add(head)
                        nodes.add(tail)
                        edges.append((head, tail, rel))
                G.add_vertices(list(nodes))
                G.add_edges([(edge[0], edge[1]) for edge in edges])
                G.es["name"] = [edge[2] for edge in edges]

                for v in tqdm(G.vs):
                    v["label"] = self.id2name(v["name"])
                pickle.dump(G, open(self.path_kg_pkl, "wb"))
                print("Graph saved to pickle file.")

            self.store_to_redis(self.redis_client, 'kg', pickle.dumps(G))
            print("Graph saved to Redis.")
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

    @staticmethod
    def store_to_redis(redis_client, key_prefix, data, chunk_size=128*1024*1024):
        serialized_data = pickle.dumps(data)
        chunk_count = 0
        for i in range(0, len(serialized_data), chunk_size):
            redis_client.set(f"{key_prefix}:{chunk_count}",
                             serialized_data[i:i + chunk_size])
            chunk_count += 1
        redis_client.set(f"{key_prefix}:num_chunks", chunk_count)

    @staticmethod
    def load_from_redis(redis_client, key_prefix):
        num_chunks = int(redis_client.get(f"{key_prefix}:num_chunks"))
        t = time()
        data = b"".join(redis_client.get(
            f"{key_prefix}:{i}") for i in range(num_chunks))
        print("Redis load time:", time()-t)
        return pickle.loads(pickle.loads(data))
"""


class Tools:
    _instance = None
    path_id2name = "/back-up/gzy/id2name.txt"
    path_id2name_pkl = "/back-up/gzy/id2name.pkl"
    path_name2id_pkl = "/back-up/gzy/name2id.pkl"
    path_kg = "/back-up/gzy/rel_filter.txt"
    path_kg_pkl = "/back-up/gzy/rel_filter_name.pkl"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tools, cls).__new__(cls)

            cls.id2name_dict, cls.name2id_dict = cls._get_id2name()
            cls.kg = cls._get_kg()
        return cls._instance

    @classmethod
    def _get_id2name(self):
        print("开始读入 id2name 映射文件 loading...")
        try:
            t = time()
            id2name_dict = pickle.load(open(self.path_id2name_pkl, "rb"))
            name2id_dict = pickle.load(open(self.path_name2id_pkl, "rb"))
            print("id2name loaded from pickle file.")
            print("耗时：", time()-t)
        except (FileNotFoundError, EOFError):
            id2name_dict, name2id_dict = self._load_id2name()
            pickle.dump(id2name_dict, open(self.path_id2name_pkl, "wb"))
            pickle.dump(name2id_dict, open(self.path_name2id_pkl, "wb"))
            print("id2name saved to pickle file.")
        finally:
            gc.collect()
        return id2name_dict, name2id_dict

    @classmethod
    def _load_id2name(self):
        id2name_dict = {}
        name2id_dict = {}
        with open(self.path_id2name, "r") as fp:
            for line in fp:
                mid, rel, name = line.strip().split('\t')
                id2name_dict[mid] = name
                name2id_dict[name] = mid
        return id2name_dict, name2id_dict

    @classmethod
    def _get_kg(self):
        print("开始读入大图 loading...")
        try:
            t = time()
            G = pickle.load(open(self.path_kg_pkl, "rb"))
            print("Graph loaded from pickle file.")
            print("耗时：", time()-t)
        except (FileNotFoundError, EOFError):
            G = ig.Graph(directed=True)
            with open(self.path_kg, "r") as fp:
                edges = []
                nodes = set()
                for line in tqdm(fp):
                    head, rel, tail = line.strip().split('\t')
                    nodes.add(head)
                    nodes.add(tail)
                    edges.append((head, tail, rel))
            G.add_vertices(list(nodes))
            G.add_edges([(edge[0], edge[1]) for edge in edges])
            G.es["name"] = [edge[2] for edge in edges]

            for v in tqdm(G.vs):
                v["label"] = self.id2name(v["name"])
            pickle.dump(G, open(self.path_kg_pkl, "wb"))

            print("Graph saved to pickle file.")
        finally:
            gc.collect()
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
