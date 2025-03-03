from multiprocessing import Pool
from time import time
from copy import deepcopy
import json
import igraph as ig
from typing import List, Dict, Set

HOP = 2
MAX_ENTITIES = 2000


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
    vertices_list = list(vertices)
    seed_nodes = [G.vs.find(name=name).index for name in seed_nodes]
    # Calculate PPR using igraph's pagerank method
    beforetime = time()
    ppr = G.personalized_pagerank(
        vertices=vertices_list,
        directed=True,
        damping=restart_prob,
        reset_vertices=seed_nodes,
    )
    aftertime = time()
    alltime = aftertime - beforetime
    return dict(zip(vertices_list, ppr)), alltime


def rank_ppr_ents(G: ig.Graph, vertices: Set[str], seed_list: List[str], mode="fixed", max_ent=MAX_ENTITIES, min_ppr=0.005):
    # Run PPR on the subgraph

    ppr, alltime = personalized_pagerank(
        G, vertices, seed_list, restart_prob=0.8)
    # Extract entities based on mode
    if mode == "fixed":
        sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        extracted_ents = [node for node, _ in sorted_ppr[:max_ent]]
    else:
        extracted_ents = [node for node,
                          value in ppr.items() if value > min_ppr]

    return extracted_ents, alltime


def get_k_hop_neighbors_optimized(G: ig.Graph, seed_list: List[str], hop: int) -> Set:
    seed_list = [G.vs.find(name=name).index for name in seed_list]
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

    return visited, seed_list


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["label"]
        tail = G.vs[edge.target]["label"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


def chunkify(lst, n):
    """将列表 lst 分割成 n 个尽可能长度相近的子列表"""
    # 计算每个子列表的基本长度
    base_chunk_size = len(lst) // n
    # 计算需要额外一个元素的子列表的数量
    num_chunks_with_extra = len(lst) % n
    # 初始化结果列表
    chunks = []
    # 初始化起始索引
    start_index = 0

    for i in range(n):
        # 如果当前子列表索引小于需要额外一个元素的子列表数量
        # 则当前子列表长度为基本长度加一
        if i < num_chunks_with_extra:
            chunk_size = base_chunk_size + 1
        else:
            chunk_size = base_chunk_size
        # 从 lst 中切片获取当前子列表，并添加到结果列表中
        chunks.append(lst[start_index:start_index + chunk_size])
        # 更新起始索引为下一个子列表的起始位置
        start_index += chunk_size

    return chunks


def process(info: Dict, id):
    try:
        seed_list = [entity["kb_id"] for entity in info["entities"]]
        # print(id,"seed_list name",seed_list)
        #    t=time()
        # subgraph_nodes,seed_list_index = get_k_hop_neighbors_optimized(G, seed_list, HOP)
        # print(id,"seed_list index:",seed_list_index)
        #    print("get neighbors:",time()-t)
        # if len(subgraph_nodes) == 0:
        #     print(id,"No subgraph nodes")
        #     return None,0  # Skip if no subgraph nodes
        #    t=time()
        # 对G里面所有node进行PPR
        allnodes = G.vs.indices
        entities, alltime = rank_ppr_ents(
            G, allnodes, seed_list, max_ent=MAX_ENTITIES)
        # print(id,"entities:",entities)
        #    print("PPR:",time()-t)
        #    t=time()
        ppr_subgraph = G.subgraph(entities)
        #    print("get subgraph:",time()-t)
        obj = deepcopy(info)
        obj["answers"] = [ans["text"] if ans["text"] else ans["kb_id"]
                          for ans in obj["answers"]]
        obj["entities"] = [ent["text"] if ent["text"] else ent["kb_id"]
                           for ent in obj["entities"]]
        obj["subgraph"] = get_triples(ppr_subgraph)
        obj["nodes"] = G.vs[entities]["name"]
        print(id, "done!")
        return obj, alltime
    except Exception as e:
        print(id, e)
        return None, 0


def process_data_chunk(args):
    data_chunk, out_file_path = args  # (list, str)
    t = time()
    Alltime = {}
    with open(out_file_path, "w") as output_file:
        for index, line in enumerate(data_chunk):
            info = json.loads(line)
            id = info["id"]
            obj, alltime = process(info, id)
            Alltime[id] = alltime
            if obj is None:
                continue
            output_file.write(json.dumps(obj) + "\n")
            if index % 20 == 0:
                print(out_file_path, index, ":", time()-t, "s")
    # 将Alltime 写入json文件
    with open("/home/gzy/kg_rag/kg_rag_preprocess/Freebase/DifferentPPR/OurPPRtime.json", "w") as f:
        json.dump(Alltime, f)
    print("OurPPRtime.json", "done!")


if __name__ == "__main__":
    import os
    # 获取当前程序的 PID
    pid = os.getpid()
    print(f"当前程序的 PID 是: {pid}")
    kb_file = "/raid/gzy/dataset/graphrag/process_data/rel_filter.txt"
    in_file = "/home/gzy/kg_rag/kg_rag_preprocess/Freebase/DifferentPPR/webqsp_step0_first_200.jsonl"  # 前100个问题
    PPRtype = "OurPPR"  # OurPPR,TopPPR,fora,kPAR
    # 输出文件
    out_file_ppr = f"DifferentPPR/{PPRtype}_webqsp_first_200_output.jsonl"
    out_time = f"DifferentPPR/{PPRtype}_time.txt"
    map_file = "/raid/gzy/dataset/graphrag/process_data/id2name.txt"
    threads = 1  # 线程数量

    t = time()
    id2name_dict = {}
    with open(map_file, "r") as fp:
        for line in fp:
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
        for line in fp:
            head, rel, tail = line.strip().split('\t')
            nodes.add(head)
            nodes.add(tail)
            edges.append((head, tail, rel))

    G.add_vertices(list(nodes))
    G.add_edges([(edge[0], edge[1]) for edge in edges])
    G.es["name"] = [edge[2] for edge in edges]
    for v in G.vs:
        v["label"] = id2name(v["name"])

    filter_lines = []
    with open(in_file, "r") as fp:
        for line in fp.readlines():
            filter_lines.append(line)
    print("prepare:", time()-t, "s")

    # info = json.loads(lines[0])
    # obj=process(info)
    # raise Exception()
    data_chunks = chunkify(filter_lines, threads)

    pool = Pool(processes=threads)

    temp_files = []
    args = []
    for i in range(threads):
        temp_file = f"tmp/ppr_{i}.tmp"  # 为每个线程创建一个临时文件
        temp_files.append(temp_file)
        args.append((data_chunks[i], temp_file))  # [(list, str), ...]

    pool.map(process_data_chunk, args)
    pool.close()
    pool.join()

    # 合并临时文件到最终文件
    with open(out_file_ppr, "w") as final_file:
        for temp_file in temp_files:
            with open(temp_file, "r") as f:
                for line in f:
                    final_file.write(line)
    # 记录时间
    print("done!")
