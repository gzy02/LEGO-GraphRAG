from multiprocessing import Pool
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

    vertices_list = list(vertices)
    # Calculate PPR using igraph's pagerank method
    ppr = G.personalized_pagerank(
        vertices=vertices,
        damping=restart_prob,
        reset_vertices=seed_nodes,
    )

    return dict(zip(vertices_list, ppr))


def rank_ppr_ents(G: ig.Graph, vertices: Set[str], seed_list: List[str], mode="fixed", max_ent=2000, min_ppr=0.005):
    # Run PPR on the subgraph
    ppr = personalized_pagerank(G, vertices, seed_list, restart_prob=0.8)

    # Extract entities based on mode
    if mode == "fixed":
        sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        extracted_ents = [node for node, _ in sorted_ppr[:max_ent]]
    else:
        extracted_ents = [node for node,
                          value in ppr.items() if value > min_ppr]

    return extracted_ents


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


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["label"]
        tail = G.vs[edge.target]["label"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


def chunkify(lst, n):
    base_chunk_size = len(lst) // n
    num_chunks_with_extra = len(lst) % n
    chunks = []
    start_index = 0

    for i in range(n):
        if i < num_chunks_with_extra:
            chunk_size = base_chunk_size + 1
        else:
            chunk_size = base_chunk_size
        chunks.append(lst[start_index:start_index + chunk_size])
        start_index += chunk_size

    return chunks


def process(info: Dict):
    try:
        seed_list = [entity["kb_id"] for entity in info["entities"]]
        #    t=time()
        subgraph_nodes = get_k_hop_neighbors_optimized(G, seed_list, 2)
        #    print("get neighbors:",time()-t)
        if len(subgraph_nodes) == 0:
            return None  # Skip if no subgraph nodes
        #    t=time()
        entities = rank_ppr_ents(G, subgraph_nodes, seed_list, max_ent=2000)
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
        return obj
    except Exception as e:
        print(e)
        return None


def process_data_chunk(args):
    data_chunk, out_file_path = args
    t = time()
    with open(out_file_path, "w") as output_file:
        for index, line in enumerate(data_chunk):
            info = json.loads(line)
            obj = process(info)
            if obj is None:
                continue
            output_file.write(json.dumps(obj) + "\n")
            if index % 20 == 0:
                print(out_file_path, index, ":", time()-t, "s")
    print(out_file_path, ":", time()-t, "s")


if __name__ == "__main__":
    kb_file = "rel_filter.txt"
    in_file = "CWQ/CWQ_step0.jsonl"
    out_file_ppr = "dataset/AAAI/MainExperiment/CWQ/PPR2/test_name.jsonl"
    map_file = "id2name.txt"
    threads = 64

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
    # G.vs["name"] = G.vs["name"]

    for v in G.vs:
        v["label"] = id2name(v["name"])

    print("prepare:", time()-t, "s")

    with open(in_file, "r") as fp:
        lines = fp.readlines()

    # info = json.loads(lines[0])
    # obj=process(info)
    # raise Exception()
    data_chunks = chunkify(lines, threads)

    pool = Pool(processes=threads)

    temp_files = []
    args = []
    for i in range(threads):
        temp_file = f"CWQ/tmp_ppr2000_2/ppr_{i}.tmp"
        temp_files.append(temp_file)
        args.append((data_chunks[i], temp_file))

    pool.map(process_data_chunk, args)

    with open(out_file_ppr, "w") as final_file:
        for temp_file in temp_files:
            with open(temp_file, "r") as f:
                for line in f:
                    final_file.write(line)
