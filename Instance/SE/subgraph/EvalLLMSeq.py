import json
import os
from utils.Tools import get_query, Tools
from utils.LLM import LLM
import set_random
from pre_retrieval import *
import aiohttp
import asyncio
import time
from tqdm import tqdm
from utils.Evaluation import eval_cover
from utils.SemanticModel import EmbeddingModel, RandomModel
import igraph as ig

standard_ppr = 200


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["name"]
        tail = G.vs[edge.target]["name"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


class QueryProcessor:
    def __init__(self, dataset, ppr_file, ModuleLLM):
        self.dataset = dataset
        self.ppr_file = ppr_file
        self.queries = []
        self.ModulePPR = PreRetrievalModulePPR(standard_ppr)
        self.ModuleLLM = ModuleLLM
        self.window = ModuleLLM.window

    def process_basic_info(self, window, structure_method, semantic_method):
        return {
            "Dataset": self.dataset,
            "ppr_file": self.ppr_file,
            "window": window,
            "structureMethod": str(structure_method),
            "semanticMethod": str(semantic_method)
        }

    def process_query_info(self, query):
        return {
            "id": query.qid,
            "answers": query.answers,
            "question": query.question,
            "entities": query.entities,
            "subgraph": get_triples(query.subgraph)
        }

    def save_query(self, query_list, basic_info, query_info):
        query_list.append({
            "basic_info": basic_info,
            "query_info": query_info
        })

    async def process(self, query):
        print("Question ID: ", query.qid)
        query = self.ModulePPR.process(query)
        query = await self.ModuleLLM.process(query)
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


async def run(session, kg, reasoning_dataset, ppr_file, url, model, rankModel, ModuleLLM, window, max_concurrent_requests, total_requests):
    # 创建 Semaphore 来限制并发请求的数量
    sem = asyncio.Semaphore(max_concurrent_requests)
    results = []
    llm = LLM(session, url, model)

    rank_model = rankModel
    module = ModuleLLM(window, rank_model, llm)
    processor = QueryProcessor(reasoning_dataset, ppr_file, module)
    # 创建一个进度条来可视化请求的进度
    with tqdm(total=total_requests) as pbar:
        # 循环创建任务，直到达到总请求数
        for query in get_query(kg, ppr_file):
            # 为每个请求创建一个任务，确保它遵守信号量的限制
            result = await bound_fetch(sem, query, processor, pbar)
            results.append(result)

    cover = sum(eval_cover(query.subgraph, query.answers)
                for query, _ in results) / total_requests
    # 计算所有结果中的完成token总数
    completion_tokens = sum(query.output_tokens for query, _ in results)

    # 从所有结果中提取响应时间
    response_times = [result[-1] for result in results]

    # 返回完成token的总数和响应时间的列表
    return processor, cover, completion_tokens, response_times

if __name__ == '__main__':
    C = 25
    N = 250

    print("PID =", os.getpid())
    tools = Tools()
    kg = tools.kg

    vllm_api = {
        # "qwen2-70b": "http://localhost:8001/v1/chat/completions",
        "llama3-70b": "http://localhost:8000/v1/chat/completions"
    }
    rank_models = {
        "EMB": EmbeddingModel(),
        "random": RandomModel()
    }
    ModuleLLMs = {
        "edge": [PreRetrievalModuleLLMEdge, 32],
        "node": [PreRetrievalModuleLLMNode, 64],
        "triple": [PreRetrievalModuleLLMTriples, 1024]
    }

    async def main():
        async with aiohttp.ClientSession() as session:
            for model, url in vllm_api.items():
                for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:
                    ppr_file = f"/back-up/lzy/Dataset/{reasoning_dataset}/{reasoning_dataset}_250.jsonl"
                    print("Dataset: ", reasoning_dataset)
                    for rankModelName, rankModel in rank_models.items():
                        print("Rank Model: ", rankModelName)
                        for task, [ModuleLLM, window] in ModuleLLMs.items():
                            print("Task: ", task)
                            start_time = time.time()
                            processor, cover, completion_tokens, response_times = await run(
                                session, kg, reasoning_dataset, ppr_file, url, model, rankModel, ModuleLLM, window, C, N
                            )
                            end_time = time.time()

                            with open(f"/back-up/gzy/dataset/VLDB/SubgraphExtraction/{reasoning_dataset}/subgraph/LLM/{model}/{rankModelName}/{task}.json", "w") as fp:
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

    asyncio.run(main())
