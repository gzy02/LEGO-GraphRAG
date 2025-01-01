import json
import torch
from utils.LLM import LocalLLM
import psutil
import os
from pipeline import *
from pre_retrieval import *
from utils.SemanticModel import EmbeddingModel
from utils.Tools import Query, construct_graph
pid = os.getpid()
print("pid:", pid)
process = psutil.Process(pid)

# 未初始化时的内存使用情况
memory0_gpu = torch.cuda.memory_allocated()
memory0_sys = process.memory_info().rss

dataset_list = ["webqsp", "CWQ", "GrailQA", "WebQuestion"]
subgraph_list = [
    "PPR"
]

llm_path = "/back-up/LLMs/qwen/Qwen2-72B-Instruct-AWQ/"
device = "cuda"
# llm = LocalLLM(llm_path, device=device)
# emb = EmbeddingModel()
retrievalPipeline = {
    # "PPR": SEPipeline(PreRetrievalModuleNone()),
    "EMB/edge": SEPipeline(PreRetrievalModuleEdge(semantic_type="EMB")),
    # "LLM/qwen2-70b/EMB/ppr_1000_edge_64": SEPipeline(PreRetrievalModuleLLMEdge(llm=llm, rank_model=emb, window=64)),
}

se_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/subgraph/"
# "/back-up/gzy/dataset/VLDB/new25/SubgraphExtraction/"
target_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/Cost/SubgraphExtraction/"
for reasoning_dataset in dataset_list:
    jsonl_25 = f"/back-up/gzy/dataset/VLDB/new25/{reasoning_dataset}_25_new.jsonl"
    qid_25 = set()
    with open(jsonl_25, "r") as f:
        for line in f:
            qid_25.add(json.loads(line)["id"])
    for subgraph_type in subgraph_list:
        subgraph_path = se_base_url + \
            f"{reasoning_dataset}/subgraph/{subgraph_type}.json"
        for retrievaltype, pipeline in retrievalPipeline.items():
            with open(subgraph_path, "r") as f:
                infos = json.load(f)
            print(f"Processing {subgraph_path}...")
            basic_info = {
                "Dataset": reasoning_dataset,
                "subgraph_file": subgraph_path,
                "postRetrievalMethod": str(pipeline.semanticMethod),
                "memory0_sys": memory0_sys,  # 未初始化时的内存使用情况
                "memory0_gpu": memory0_gpu,
                "memory_sys_allocation": process.memory_info().rss,  # 初始化时的内存使用情况
                "memory_allocation": torch.cuda.memory_allocated(),
            }
            eval_info = []
            for info in infos:
                # if info["query_info"]["id"] not in qid_25:
                #    continue
                triples = info["query_info"]["subgraph"]
                query = Query(info["query_info"])
                query.subgraph = construct_graph(triples)
                query, res_dict = pipeline.run(query)
                res_dict["memory_sys"] = process.memory_info().rss
                res_dict["memory_gpu"] = torch.cuda.memory_allocated()
                eval_info.append(res_dict)
            output_path = target_base_url + \
                f"{reasoning_dataset}/{retrievaltype}.json"
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print("output_path:", output_path)
            with open(output_path, "w") as fp:
                json.dump(
                    {
                        "basic_info": basic_info,
                        "eval_info": eval_info
                    }, fp, indent=4)
