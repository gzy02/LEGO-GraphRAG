from pipeline import PRPipeline
from retrieval import RetrievalModuleDij
from post_retrieval import PostRetrievalModuleAgentLLM
import set_random
import json
import os
from utils.LLM import LLM
import asyncio
from tqdm import tqdm
from utils.Tools import Query, construct_graph
se_base_url = "/back-up/gzy/dataset/VLDB/new250/SubgraphExtraction/"
pr_base_url = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/"
max_concurrent_requests = 64


async def process_query(sem, pipeline, info, pbar):
    async with sem:
        triples = info["query_info"]["subgraph"]
        if len(triples) == 0:
            return None
        query = Query(info["query_info"])
        query.subgraph = construct_graph(triples)

        query, res_dict = await pipeline.arun(query)
        pbar.update(1)
        return res_dict


async def run(pipeline, infos):
    sem = asyncio.Semaphore(max_concurrent_requests)
    if True:
        tasks = []
        total = len(infos)
        with tqdm(total=total) as pbar:
            for index, info in enumerate(infos):
                if index == total:
                    break
                task = asyncio.ensure_future(
                    process_query(sem, pipeline, info, pbar)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

    eval_info = [res for res in results if res is not None]
    basic_info = {
        "Dataset": reasoning_dataset,
        "subgraph_file": subgraph_path,
        "retrievalMethod": str(pipeline.structureMethod),
        "postRetrievalMethod": str(pipeline.semanticMethod)
    }

    return basic_info, eval_info


model_name = "qwen2-70b"
llm_url = 'http://localhost:8000/v1/chat/completions'
llm = LLM(model=model_name, url=llm_url)
retrievalPipeline = {
    "SPR/LLM/qwen2-70b/Agent_v2_16": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleAgentLLM(llm, window=64)),
}
dataset_list = ["webqsp",]
# "PPR","EMB/edge", "LLM/qwen2-70b/EMB/ppr_1000_edge_64"
subgraph_list = ["PPR"]

for reasoning_dataset in dataset_list:
    for subgraph_type in subgraph_list:
        subgraph_path = se_base_url + \
            f"{reasoning_dataset}/subgraph/{subgraph_type}.json"
        for retrievaltype in retrievalPipeline:
            pipeline = retrievalPipeline[retrievaltype]

            with open(subgraph_path, "r") as f:
                infos = json.load(f)
            print(f"Processing {subgraph_path}...")
            basic_info, eval_info = asyncio.run(run(pipeline, infos))

            output_path = pr_base_url + \
                f"{reasoning_dataset}/{subgraph_type}/{retrievaltype}.json"
            print(f"Saving to {output_path}...")
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as fp:
                json.dump(
                    {
                        "basic_info": basic_info,
                        "eval_info": eval_info
                    }, fp, indent=4)
