from pipeline import PRPipeline
from retrieval import RetrievalModuleDij
from post_retrieval import PostRetrievalModuleLLMFilter
import set_random
import json
import os
from utils.LLM import LLM
import asyncio
from tqdm import tqdm
from utils.Tools import Query, construct_graph
from utils.ReasoningPath import ReasoningPath
se_base_url = "/back-up/gzy/dataset/VLDB/new250/SubgraphExtraction/"
pr_base_url = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/"

max_concurrent_requests = 64


async def process_query(sem, pipeline, info, pbar):
    async with sem:
        query = Query(info)
        paths = info["ReasoningPaths"].split("\n")
        paths = [path for path in paths if "->" in path]
        query.reasoning_paths = [ReasoningPath(
            "", path=path) for path in paths]
        query = await pipeline.aprocess(query)
        pbar.update(1)
        return query.to_dict()


async def run(module, infos):
    sem = asyncio.Semaphore(max_concurrent_requests)
    if True:
        tasks = []
        total = len(infos)
        with tqdm(total=total) as pbar:
            for index, info in enumerate(infos):
                if index == total:
                    break
                task = asyncio.ensure_future(
                    process_query(sem, module, info, pbar)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

    eval_info = [res for res in results if res is not None]

    return eval_info

model_name = "qwen2-70b"
llm_url = 'http://localhost:8000/v1/chat/completions'
llm = LLM(model=model_name, url=llm_url)
src_json_type = [
    "SPR/LLM/qwen2-70b/Agent16"
]
dataset_list = ["webqsp", "CWQ"]
# "PPR","EMB/edge", "LLM/qwen2-70b/EMB/ppr_1000_edge_64"
subgraph_list = ["PPR"]

for reasoning_dataset in dataset_list:
    for subgraph_type in subgraph_list:
        for retrievaltype in src_json_type:
            input_path = pr_base_url + \
                f"{reasoning_dataset}/{subgraph_type}/{retrievaltype}.json"
            with open(input_path, "r") as f:
                infos = json.load(f)
            basic_info = infos["basic_info"]

            print(f"Processing {input_path}...")
            eval_info = asyncio.run(
                run(PostRetrievalModuleLLMFilter(llm), infos["eval_info"]))

            output_path = pr_base_url + \
                f"{reasoning_dataset}/{subgraph_type}/{retrievaltype}_v2.json"
            print(f"Saving to {output_path}...")
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as fp:
                json.dump(
                    {
                        "basic_info": basic_info,
                        "eval_info": eval_info
                    }, fp, indent=4)
