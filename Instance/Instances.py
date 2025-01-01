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


model_name = "qwen2-70b"
llm = LLM(model=model_name, url='http://localhost:8002/v1/chat/completions')
retrievalPipeline = {
    # "SPR": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleNone()),
    # "SPR/EMB": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleSemanticModel(window=-1, semantic_type="EMB")),
    # f"SPR/LLM/{model_name}/EMB": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleLLM(llm, EmbeddingModel())),
    # "BeamSearch/EMB": PRPipeline(RetrievalModuleSemanticModel(semantic_type="EMB"), PostRetrievalModuleNone()),
    # "BeamSearch/BGE": PRPipeline(RetrievalModuleSemanticModel(semantic_type="BGE"), PostRetrievalModuleNone()),
    # f"BeamSearch/LLM/{model_name}/EMB": PRPipeline(RetrievalModuleLLM(llm, EmbeddingModel()),
    #                                                PostRetrievalModuleNone()),
}

llm_pipeline = {
    # f"SPR/LLM/{model_name}/EMB": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleLLM(llm, EmbeddingModel())),
    f"BeamSearch/LLM/{model_name}/EMB_v2": PRPipeline(RetrievalModuleLLM(llm, EmbeddingModel()), PostRetrievalModuleNone()),
    f"BeamSearch/LLM/{model_name}/BGE_v2": PRPipeline(RetrievalModuleLLM(llm, BGEModel()), PostRetrievalModuleNone()),
    # f"SPR/LLM/{model_name}/Random": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleLLM(llm, RandomModel())),
    # f"BeamSearch/LLM/{model_name}/Random": PRPipeline(RetrievalModuleLLM(llm, RandomModel()), PostRetrievalModuleNone()),
}

cpu_pipeline = {
    # "SPR": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleNone()),
    # "SPR/Random": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleSemanticModel(window=-1, semantic_type="Random")),
    # "SPR/BM25": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleSemanticModel(window=-1, semantic_type="BM25")),
    # "EPR": PRPipeline(RetrievalModuleBFS(2), PostRetrievalModuleNone()),
    # "BeamSearch/Random": PRPipeline(RetrievalModuleSemanticModel(semantic_type="Random"), PostRetrievalModuleNone()),
    # "BeamSearch/BM25": PRPipeline(RetrievalModuleSemanticModel(semantic_type="BM25"), PostRetrievalModuleNone()),
}

gpu_pipeline = {
    # "SPR/BGE": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleSemanticModel(window=-1, semantic_type="BGE")),
    # "SPR/EMB": PRPipeline(RetrievalModuleDij(), PostRetrievalModuleSemanticModel(window=-1, semantic_type="EMB")),

    # "BeamSearch/BGE": PRPipeline(RetrievalModuleSemanticModel(semantic_type="BGE"), PostRetrievalModuleNone()),
    # "BeamSearch/EMB": PRPipeline(RetrievalModuleSemanticModel(semantic_type="EMB"), PostRetrievalModuleNone()),
}

# "webqsp","CWQ", "GrailQA","WebQuestion"
dataset_list = ["webqsp", "CWQ", "GrailQA", "WebQuestion"]
subgraph_list = ["PPR", "EMB/edge", f"LLM/{model_name}/EMB/ppr_1000_edge_64"]

for reasoning_dataset in dataset_list:
    for subgraph_type in subgraph_list:
        subgraph_path = se_base_url + \
            f"{reasoning_dataset}/subgraph/{subgraph_type}.json"
        for retrievaltype, pipeline in llm_pipeline.items():
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
