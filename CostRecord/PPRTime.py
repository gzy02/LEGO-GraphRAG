from tqdm import tqdm
from time import time
from copy import deepcopy
import json
import igraph as ig
from typing import List, Dict, Set


def personalized_pagerank(G: ig.Graph, vertices: Set[str], seed_nodes: List[str], restart_prob=0.8):
    """Return the PPR vector for the given seed nodes and restart prob using an igraph graph.

    Args:
        G: An igraph graph.
        vertices:
        seed_nodes: A list of seed node IDs.
        restart_prob: A scalar in [0, 1].

    Returns:
        ppr: A dictionary of node IDs to their PPR values.
    """

    nowtime = time()
    ppr = G.personalized_pagerank(
        vertices=vertices,
        damping=restart_prob,
        reset_vertices=seed_nodes,
    )
    aftertime = time()
    spendtime = aftertime-nowtime
    return spendtime


def get_k_hop_neighbors_optimized(G: ig.Graph, seed_list: List[str], hop: int) -> Set:
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
    return visited


def process(G, info: Dict, hopnum):
    try:
        seed_list = [entity["kb_id"]
                     for entity in info["entities"]]
        #    t=time()
        subgraph_nodes = get_k_hop_neighbors_optimized(G, seed_list, hopnum)
        subgraph_hopnum = G.subgraph(subgraph_nodes)
        #    print("get neighbors:",time()-t)
        if len(subgraph_nodes) == 0:
            return 0, 0, 0, 0  # Skip if no subgraph nodes
        #    t=time()
        spendtime = personalized_pagerank(
            subgraph_hopnum, None, seed_list)

        return spendtime, len(subgraph_hopnum.vs), len(seed_list), len(subgraph_hopnum.es)
    except Exception as e:
        print(e)
        return 0, 0, 0, 0


def process_data_chunk(G, filter_lines):
    # 在这里计算每个dataset的时间
    datasetlist = ["CWQ", "GrailQA", "webqsp", "WebQuestion"]

    dataJsondata = []
    for hopnum in range(2, 5):
        dataset_count = 0
        ppr_info = []
        alltime = []
        VecticesNum = 0
        SeedNum = 0
        count = 0
        Edges = 0
        for index, line in enumerate(filter_lines):
            try:
                info = json.loads(line)
                spendtime, vectices_len, seed_len, edges_len = process(
                    G, info, hopnum)
                if spendtime != 0:
                    alltime.append(spendtime)
                    # Vectices.append(vectices_len)
                    VecticesNum += vectices_len
                    SeedNum += seed_len
                    Edges += edges_len
                    # count += 1
                    onedata = {
                        "time": spendtime,
                        "vectices": vectices_len,
                        "seed": seed_len,
                        "edges": edges_len
                    }
                    ppr_info.append(onedata)
                count += 1
                if count % 20 == 0:  # 到20个数据换一个数据集
                    averagetime = sum(alltime)/len(alltime)
                    averageVectices = VecticesNum/(len(alltime))
                    averageSeed = SeedNum/(len(alltime))
                    averageEdges = Edges/(len(alltime))
                    onehop = {
                        "hop": hopnum,
                        "dataset": datasetlist[dataset_count],
                        "ppr_info": ppr_info,
                        "average time": averagetime,
                        "average Vectices": averageVectices,
                        "average Seed": averageSeed,
                        "average Edges": averageEdges
                    }
                    dataset_count += 1
                    dataJsondata.append(onehop)
            except:
                print("error happened!")

    with open('./PPR-time.json', 'w', encoding='utf-8') as json_file:
        # ensure_ascii=False 保证中文正常写入，indent=4 美化输出
        json.dump(dataJsondata, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # kb_file = "/back-up/gzy/rel_filter.txt"
    # in_file = "/home/lzy/PPR/dataset_step0.jsonl"
    # map_file = "/back-up/gzy/id2name.txt"
    in_file = "/home/lzy/PPR/dataset_step0.jsonl"
    kb_file = "/back-up/gzy/rel_filter.txt"
    map_file = "/back-up/gzy/id2name.txt"

    t = time()
    print("开始读入大图 loading...")
    id2name_dict = {}
    with open(map_file, "r") as fp:
        for line in tqdm(fp):
            mid, rel, name = line.strip().split('\t')
            id2name_dict[mid] = name

    def id2name(mid):
        if mid in id2name_dict:
            return id2name_dict[mid]
        else:
            return mid

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
        v["label"] = id2name(v["name"])

    filter_lines = []
    with open(in_file, "r") as fp:
        for line in fp.readlines():
            filter_lines.append(line)
    print("prepare:", time()-t, "s")
    process_data_chunk(G, filter_lines)
    print("done!")
