from utils.PromptTemplate import PRERETIEVAL_NODE_PERSONA, PRERETIEVAL_EDGE_PERSONA, PRERETIEVAL_TRIPLE_PERSONA
from utils.Tools import get_query_subgraph, get_triples
from pre_retrieval import *
import json
from utils.LLM import LocalLLM
from utils.Evaluation import eval_cover
from config import model_paths, reasoning_model, dataset_list, subgraph_list
import os
from utils.SemanticModel import EmbeddingModel, BGEModel, BM25Model
from transformers import AutoTokenizer
from tqdm import tqdm

ppr_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/subgraph/{reasoning_dataset}/subgraph/PPR.json"
id_250_base_url = "/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
output_url = "/back-up/gzy/dataset/VLDB/new250/ranked_info/{reasoning_dataset}/{semantic_type}/{pr_type}.json"


# emb = EmbeddingModel()
bge = BGEModel()
# bm25 = BM25Model()
tokenizer = AutoTokenizer.from_pretrained(model_paths[reasoning_model])
preRetrievalPipeline = [
    # ("edge", "EMB", emb, PreRetrievalModuleLLMEdgeTokenAlign),
    # ("triple", "EMB", emb, PreRetrievalModuleLLMTripleTokenAlign),
    # ("node", "EMB", emb, PreRetrievalModuleLLMNodeTokenAlign),
    ("edge", "BGE", bge, PreRetrievalModuleLLMEdgeTokenAlign),
    # ("triple", "BGE", bge, PreRetrievalModuleLLMTripleTokenAlign),
    ("node", "BGE", bge, PreRetrievalModuleLLMNodeTokenAlign),
    # ("edge", "BM25", bm25, PreRetrievalModuleLLMEdgeTokenAlign),
    # ("triple", "BM25", bm25, PreRetrievalModuleLLMTripleTokenAlign),
    # ("node", "BM25", bm25, PreRetrievalModuleLLMNodeTokenAlign),
]
# WebQSP全做完，CWQ做完edge，

print("Reasoning Model:", reasoning_model)
DEBUG = False
if __name__ == "__main__":
    all_questions = []
    for reasoning_dataset in dataset_list:
        ppr_file = ppr_base_url.format(reasoning_dataset=reasoning_dataset)

        id_250_file = id_250_base_url.format(
            reasoning_dataset=reasoning_dataset)
        id_250 = set()
        with open(id_250_file, "r") as f:
            for line in f:
                id_250.add(json.loads(line)["id"])
        for pr_type, semantic_type, model, module in preRetrievalPipeline:
            pre = module(tokenizer, 32000, model)
            if True:
                json_datas = {
                    "base_info": {"Dataset": reasoning_dataset,
                                  "ppr_file": ppr_file},
                    "eval_info": [],
                }
                for query in tqdm(get_query_subgraph(ppr_file), total=1500):
                    if query.qid not in id_250:
                        continue
                    lst = pre.pre_process(query)
                    res_dict = {"id": query.qid, "question": query.question,
                                "answers": query.answers, "entities": query.entities}
                    res_dict["ranked_corpus"] = lst
                    json_datas["eval_info"].append(res_dict)
                    if DEBUG:
                        break

                answers_file = output_url.format(
                    reasoning_dataset=reasoning_dataset, semantic_type=semantic_type, pr_type=pr_type)
                if not os.path.exists(answers_file):
                    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
                with open(answers_file, "w") as fp:
                    json.dump(json_datas, fp, indent=4, ensure_ascii=False)
                print(f"Finish {answers_file}")
