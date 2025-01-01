import set_random
import json
import os
from utils.LLM import LLM
from utils.SemanticModel import EmbeddingModel, RandomModel, BM25Model, BGEModel
from pipeline import *
from retrieval import *
from post_retrieval import *
import asyncio
from tqdm import tqdm
from utils.Tools import Query, construct_graph
from config import llm_url, reasoning_model,  subgraph_list, reasoning_dataset, subgraph_path
se_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/subgraph/"
pr_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/"
max_concurrent_requests = 32


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

llm = LLM(model=reasoning_model, url=llm_url)
retrievalPipeline = {
    "SPR": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleNone()),
    "SPR/EMB": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleSemanticModel(window=-1, semantic_type="EMB")),
    f"SPR/LLM/{reasoning_model}/EMB": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleLLM(llm, EmbeddingModel())),
    "BeamSearch/EMB": PRPipeline(RetrievalModuleSemanticModel(semantic_type="EMB"), PostRetrievalModuleNone()),
    f"BeamSearch/LLM/{reasoning_model}/EMB": PRPipeline(RetrievalModuleLLM(llm, EmbeddingModel()),
                                                        PostRetrievalModuleNone()),
}
if __name__ == "__main__":
    for subgraph_type in subgraph_list:
        subgraph_path = se_base_url + \
            f"{reasoning_dataset}/subgraph/{subgraph_type}.json"
        for retrievaltype, pipeline in retrievalPipeline.items():
            with open(subgraph_path, "r") as f:
                infos = json.load(f)
            print(f"Processing {subgraph_path}...")
            basic_info, eval_info = asyncio.run(run(pipeline, infos))

            output_path = pr_base_url + \
                f"{reasoning_dataset}/{subgraph_type}/{retrievaltype}.json"
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as fp:
                json.dump(
                    {
                        "basic_info": basic_info,
                        "eval_info": eval_info
                    }, fp, indent=4)
