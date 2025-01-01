from time import time
import os
import json
from utils.Tools import get_query, Tools
from pre_retrieval import *
import igraph as ig
import set_random

tools = Tools()
kg = tools.kg
path_num = 256
steps = 4

for dataset in ["GrailQA", "WebQuestion"]:  # "webqsp", "CWQ",
    input_path = f"/back-up/gzy/dataset/VLDB/new250/{dataset}_250_new.jsonl"
    eval_info = []
    for query in get_query(kg, input_path):
        path_dict = {}
        t = time()
        for entity in query.entities:
            path_dict[entity["kb_id"]] = []
            start_vertex = kg.vs.find(name=entity["kb_id"])
            for _ in range(path_num):
                path = kg.random_walk(start=start_vertex,
                                      steps=steps, mode="out")
                # 转为kb_id, text形式

                new_paths = [{"kb_id": kg.vs[node_id]["name"],
                              "text": kg.vs[node_id]["label"]} for node_id in path]

                path_dict[entity["kb_id"]].append(new_paths)
        eval_info.append({
            "id": query.qid,
            "answers": query.answers,
            "question": query.question,
            "entities": query.entities,
            "path_dict": path_dict
        })
        print(time()-t, "s")
        print(len(eval_info), "/", 250)

    basic_info = {
        "Dataset": dataset,
        "path_num": path_num,
        "steps": steps,
        "input_path": input_path
    }
    output_path = f"/back-up/gzy/dataset/VLDB/new250/ranked_info/{dataset}/RW/256.json"
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(
            {
                "basic_info": basic_info,
                "eval_info": eval_info
            }, fp, indent=4)
