from tqdm import tqdm
from time import time
import json
import igraph as ig
from typing import List, Dict, Set
import pickle


HOP = 4
MAX_ENTITIES = 2000


def personalized_pagerank(G: ig.Graph, vertices: Set[str], seed_nodes: List[str], restart_prob=0.8):
    vertices_list = list(vertices)
    seed_nodes = [G.vs.find(name=name).index for name in seed_nodes]
    # Calculate PPR using igraph's pagerank method
    ppr = G.personalized_pagerank(
        vertices=vertices_list,
        directed=False,
        damping=restart_prob,
        reset_vertices=seed_nodes,
    )

    return dict(zip(vertices_list, ppr))


def rank_ppr_ents(G: ig.Graph, vertices: Set[str], seed_list: List[str], mode="fixed", max_ent=MAX_ENTITIES, min_ppr=0.005):
    # Run PPR on the subgraph
    ppr = personalized_pagerank(
        G, vertices, seed_list, restart_prob=0.8)

    # Extract entities based on mode
    if mode == "fixed":
        sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        extracted_ents = [node for node, _ in sorted_ppr[:max_ent]]
    else:
        extracted_ents = [node for node,
                          value in ppr.items() if value > min_ppr]

    return extracted_ents


def get_k_hop_neighbors_optimized(G: ig.Graph, seed_list: List[str], hop: int) -> Set:
    seed_list = [G.vs.find(name=name).index for name in seed_list]
    visited = set(seed_list)
    current_layer = set(seed_list)
    for _ in range(hop):
        next_layer = set()
        for node in current_layer:
            neighbors = G.neighbors(node, mode="all")
            new_neighbors = set(neighbors) - visited
            next_layer.update(new_neighbors)
            visited.update(new_neighbors)
        current_layer = next_layer
    return visited


def process(G, info: Dict, nodeNum):
    try:
        seed_list = [entity["kb_id"] for entity in info["entities"]]
        t = time()
        subgraph_nodes = get_k_hop_neighbors_optimized(G, seed_list, HOP)
        preTime = time()-t
        print("get neighbors:", preTime)
        if len(subgraph_nodes) == 0:
            return None  # Skip if no subgraph nodes
        t = time()
        entities = rank_ppr_ents(
            G, subgraph_nodes, seed_list, max_ent=MAX_ENTITIES)
        pprTime = time()-t
        print("PPR:", pprTime)
        t = time()
        ppr_subgraph = G.subgraph(entities)
        postTime = time()-t
        print("get subgraph:", postTime)
        return preTime, pprTime, postTime, len(G.vs), len(seed_list), len(G.es)
    except Exception as e:
        print("Error:", e)
        print(info)
        return 0, 0, 0, 0, 0, 0


def process_data_chunk(G, filter_lines):
    # 在这里计算每个dataset的时间
    datasetlist = ["CWQ", "GrailQA", "webqsp", "WebQuestion"]

    dataJsondata = []
    node_nums = [int(10**i) for i in range(3, 10)]
    for nodeNum in node_nums:
        print("Num:", nodeNum)
        dataset_count = 0
        ppr_info = []
        alltime = []
        VecticesNum = 0
        SeedNum = 0
        count = 0
        Edges = 0
        past = time()
        for index, line in enumerate(filter_lines):
            try:
                info = json.loads(line)
                preTime, spendtime, postTime, vectices_len, seed_len, edges_len = process(
                    G, info, nodeNum)
                if spendtime != 0:
                    alltime.append((preTime, spendtime, postTime))
                    VecticesNum += vectices_len
                    SeedNum += seed_len
                    Edges += edges_len
                    onedata = {
                        "spendtime": spendtime,
                        "preTime": preTime,
                        "postTime": postTime,
                        "vectices": vectices_len,
                        "seed": seed_len,
                        "edges": edges_len
                    }
                    ppr_info.append(onedata)
                count += 1
                if count % 20 == 0:  # 到20个数据换一个数据集
                    averagepreTime = sum([x[0] for x in alltime])/len(alltime)
                    averagetime = sum([x[1] for x in alltime])/len(alltime)
                    averagepostTime = sum([x[2] for x in alltime])/len(alltime)
                    averageVectices = VecticesNum/(len(alltime))
                    averageSeed = SeedNum/(len(alltime))
                    averageEdges = Edges/(len(alltime))
                    onehop = {
                        "nodeNum": nodeNum,
                        "dataset": datasetlist[dataset_count],
                        "average preTime": averagepreTime,
                        "average time": averagetime,
                        "average postTime": averagepostTime,
                        "average Vectices": averageVectices,
                        "average Seed": averageSeed,
                        "average Edges": averageEdges,
                        "ppr_info": ppr_info,
                    }
                    dataset_count += 1
                    dataJsondata.append(onehop)

                    ppr_info = []
                    alltime = []
                    VecticesNum = 0
                    SeedNum = 0
                    count = 0
                    Edges = 0
            except:
                with open('./PPR-time.json', 'w', encoding='utf-8') as json_file:
                    json.dump(dataJsondata, json_file,
                              ensure_ascii=False, indent=4)

                print("error happened!")

            cur = time()
            print(index, cur-past)
            past = cur

    with open('./PPR-time.json', 'w', encoding='utf-8') as json_file:
        # ensure_ascii=False 保证中文正常写入，indent=4 美化输出
        json.dump(dataJsondata, json_file, ensure_ascii=False, indent=4)


def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)


def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    kb_file = "/back-up/gzy/rel_filter.txt"  # filter_1.txt
    in_file = "/home/lzy/PPR/dataset_step0.jsonl"
    graph_file = "/back-up/gzy/rel_filter.pkl"

    t = time()
    print("开始读入大图 loading...")

    try:
        G = load_graph(graph_file)
        print("Graph loaded from pickle file.")
    except (FileNotFoundError, EOFError):
        G = ig.Graph(directed=True)
        with open(kb_file, "r") as fp:
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
            v["label"] = v["name"]

        save_graph(G, graph_file)
        print("Graph saved to pickle file.")

    filter_lines = []
    with open(in_file, "r") as fp:
        for line in fp.readlines():
            filter_lines.append(line)
    print("prepare:", time()-t, "s")

    process_data_chunk(G, filter_lines)
    print("done!")
