from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import aiohttp
import asyncio
import time
from tqdm import tqdm
import set_random
import igraph as ig
from utils.Evaluation import eval_cover
from utils.SemanticModel import EmbeddingModel, RandomModel, BM25Model
from utils.Tools import get_query, Tools, get_query_subgraph, get_triples
from utils.LLM import LLM
from pre_retrieval import *

standard_ppr = 200


class QueryProcessor:
    def __init__(self, dataset, ppr_file, ModuleLLM):
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.queries = []
        # PreRetrievalModulePPR(standard_ppr)
        self.ModulePPR = "PreRetrievalModulePPR(max_ent=200)"
        self.ModuleLLM = ModuleLLM
        self.window = ModuleLLM.window

    def process_basic_info(self, window, structure_method, semantic_method):
        return {
            "Dataset": self.dataset,
            "ppr_file": self.ppr_file,
            "scale": window,
            "structureMethod": str(structure_method),
            "semanticMethod": str(semantic_method)
        }

    def process_query_info(self, query):
        return {
            "id": query.qid,
            "question": query.question,
            "answers": query.answers,
            "entities": query.entities,
            "window": query.window,
            "user_input": query.user_input,
            "llm_output": query.llm_output,
            "input_token": query.input_tokens,
            "output_token": query.output_tokens,
            "subgraph": get_triples(query.subgraph)
        }

    def save_query(self, query_list, basic_info, query_info):
        query_list.append({
            "basic_info": basic_info,
            "query_info": query_info
        })

    async def process(self, query):
        # print("Question ID: ", query.qid)
        # query = self.ModulePPR.process(query)
        query = await self.ModuleLLM.aprocess(query)
        self.save_query(
            self.queries,
            self.process_basic_info(
                self.window, self.ModulePPR, self.ModuleLLM
            ),
            self.process_query_info(query)
        )
        return query


async def bound_fetch(sem, query, processor, pbar):
    # 使用信号量 sem 来限制并发请求的数量，确保不会超过最大并发请求数
    async with sem:
        t = time.time()
        query = await processor.process(query)
        request_time = time.time()-t
        pbar.update(1)
        return query, request_time


async def run(reasoning_dataset, ppr_file, url, model, rankModel, ModuleLLM, tokenizer, window, max_concurrent_requests, total_requests):
    # 创建 Semaphore 来限制并发请求的数量
    sem = asyncio.Semaphore(max_concurrent_requests)
    if True:
        tasks = []
        llm = LLM(url, model)
        module = ModuleLLM(tokenizer, window, rankModel, llm)
        processor = QueryProcessor(reasoning_dataset, ppr_file, module)
        # 创建一个进度条来可视化请求的进度
        with tqdm(total=total_requests) as pbar:
            # 循环创建任务，直到达到总请求数
            for query in get_query_subgraph(ppr_file):
                # 为每个请求创建一个任务，确保它遵守信号量的限制
                task = asyncio.ensure_future(
                    bound_fetch(sem, query, processor, pbar))
                tasks.append(task)

            # 等待所有任务完成
            results = await asyncio.gather(*tasks)
    cover = sum(eval_cover(query.subgraph, query.answers) != 0
                for query, _ in results) / total_requests
    # 计算所有结果中的完成token总数
    completion_tokens = sum(query.output_tokens for query, _ in results)

    # 从所有结果中提取响应时间
    response_times = [result[-1] for result in results]

    # 返回完成token的总数和响应时间的列表
    return processor, cover, completion_tokens, response_times


def get_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
    C = 16
    N = 250
    print("PID =", os.getpid())

    urls = {
        "qwen2-70b": "http://localhost:8002/v1/chat/completions",
        # "llama3-70b": "http://localhost:8001/v1/chat/completions",
    }
    paths = {
        "qwen2-70b": "/back-up/LLMs/qwen/Qwen2-72B-Instruct-AWQ/",
        # "llama3-70b": "/back-up/LLMs/llama3/Meta-Llama-3-70B-Instruct-AWQ/",
    }
    rank_models = {
        # "BM25": BM25Model(),
        "EMB": EmbeddingModel(),
        # "random": RandomModel()
    }
    ModuleLLMs = {
        "triple": PreRetrievalModuleLLMTriplesTokenAlign,
        # "edge": PreRetrievalModuleLLMEdgeTokenAlign,
        # "node": PreRetrievalModuleLLMNodeTokenAlign,
    }
    scales = [1000, 2000, 4000, 8000, 16000]
    scales.reverse()
    if True:
        for scale in scales:
            for model, url in urls.items():
                tokenizer = get_tokenizer(paths[model])
                for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:
                    # f"/back-up/gzy/dataset/VLDB/SE/{reasoning_dataset}/subgraph/PPR.json"
                    ppr_file = f"/back-up/gzy/dataset/VLDB/Pipeline/subgraph/{reasoning_dataset}/subgraph/PPR.json"
                    print("Dataset: ", reasoning_dataset)
                    for rankModelName, rankModel in rank_models.items():
                        print("Rank Model: ", rankModelName)
                        for task, ModuleLLM in ModuleLLMs.items():
                            print("Task: ", task)
                            start_time = time.time()
                            processor, cover, completion_tokens, response_times = asyncio.run(
                                run(
                                    reasoning_dataset, ppr_file, url, model, rankModel, ModuleLLM, tokenizer, scale, C, N
                                ))
                            end_time = time.time()
                            json_path = f"/back-up/gzy/dataset/VLDB/SE/{reasoning_dataset}/subgraph/LLM_token_scale/{model}/{rankModelName}/{task}/{scale}.json"
                            if not os.path.exists(json_path):
                                os.makedirs(os.path.dirname(
                                    json_path), exist_ok=True)

                            with open(json_path, "w") as fp:
                                json.dump(processor.queries, fp, indent=4)

                            # 计算总时间
                            total_time = end_time - start_time
                            # 计算每个请求的平均时间
                            avg_time_per_request = sum(
                                response_times) / len(response_times)
                            # 计算每秒生成的 token 数量
                            tokens_per_second = completion_tokens / total_time

                            print(f'Performance Results:')
                            print(f'  Total requests            : {N}')
                            print(f'  Max concurrent requests   : {C}')
                            print(f'  Cover Rate                : {cover}')
                            print(
                                f'  Total time                : {total_time:.2f} seconds')
                            print(
                                f'  Average time per request  : {avg_time_per_request:.2f} seconds')
                            print(
                                f'  Tokens per second         : {tokens_per_second:.2f}')
