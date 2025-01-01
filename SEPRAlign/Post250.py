from utils.PromptTemplate import filter_prompt, FILTER_PERSONA
from utils.Tools import get_query_subgraph
from post_retrieval import *
import json
from utils.LLM import LocalLLM
from utils.Evaluation import eval_f1, eval_hr_topk
from config import model_paths, reasoning_model, dataset_list, subgraph_list, pr_base_url, se_base_url
import os
from utils.SemanticModel import EmbeddingModel, BGEModel
from retrieval import RetrievalModuleDij
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained(model_paths[reasoning_model])
postRetrievalPipeline = [
    ("SPR/LLM/qwen2-70b/BGE_new",
     "/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/{dataset}/{subgraph_type}/SPR/BGE.json", PostRetrievalModuleLLM(tokenizer)),
    # ("SPR/LLM/qwen2-70b/EMB_new",
    # "/back-up/gzy/dataset/VLDB/Pipeline/PathRetrieval/{dataset}/PPR/SPR/EMB.json", PostRetrievalModuleLLM(tokenizer))
]
pr_base_url = "/back-up/gzy/dataset/VLDB/new250/PathRetrieval/"
se_base_url = "/back-up/gzy/dataset/VLDB/new250/SubgraphExtraction/"


def token_count(tokenizer, query):
    tokenized_prediction = tokenizer.encode(query)
    return len(tokenized_prediction)


print("Reasoning Model:", reasoning_model)
DEBUG = False
if __name__ == "__main__":
    all_questions = []
    all_path_dict = []
    for reasoning_dataset in dataset_list:
        jsonl_250 = f"/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
        qid_250 = set()
        with open(jsonl_250, "r") as f:
            for line in f:
                qid_250.add(json.loads(line)["id"])
        for subgraph_type in subgraph_list:
            for retrievaltype, json_path, module in postRetrievalPipeline:
                test_file = se_base_url + \
                    f"{reasoning_dataset}/subgraph/{subgraph_type}.json"
                with open(json_path.format(dataset=reasoning_dataset, subgraph_type=subgraph_type), "r") as f:
                    datas = json.load(f)["eval_info"]
                    id2paths = {}
                    for data in datas:
                        id2paths[data["id"]] = data["ReasoningPaths"]
                cnt = 0
                print(f"processing {test_file}")
                for query in tqdm(get_query_subgraph(test_file), total=250):
                    if query.qid not in qid_250 or query.qid not in id2paths:
                        continue
                    paths = id2paths[query.qid]
                    paths = paths.split('\n')
                    paths = [path for path in paths if "->" in path]
                    prompt_input, path_dict = module.process(query, paths)
                    all_questions.append(prompt_input)
                    all_path_dict.append(path_dict)
                    cnt += 1
                    if DEBUG:
                        break

    questions = []
    for user_question in all_questions:
        allm_question = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": FILTER_PERSONA},
                {"role": "user", "content": user_question}
            ],
            tokenize=False
        )
        questions.append(allm_question)
    reasoning_llm = LocalLLM(model_paths[reasoning_model])
    # 统一进行推理
    all_answers = reasoning_llm.batch_invoke(questions)

    # 将结果写回各自的文件
    answer_idx = 0
    for reasoning_dataset in dataset_list:
        jsonl_250 = f"/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
        qid_250 = set()
        with open(jsonl_250, "r") as f:
            for line in f:
                qid_250.add(json.loads(line)["id"])
        for subgraph_type in subgraph_list:
            for retrievaltype, json_path, module in postRetrievalPipeline:
                test_file = se_base_url + \
                    f"{reasoning_dataset}/subgraph/{subgraph_type}.json"
                json_info = json.load(open(test_file))
                basic_info = {
                    "Dataset": reasoning_dataset,
                    "subgraph_file": test_file,
                    "retrievalMethod": "RetrievalModuleDij(hop=4)",
                    "postRetrievalMethod": retrievaltype,
                }
                with open(json_path.format(dataset=reasoning_dataset, subgraph_type=subgraph_type), "r") as f:
                    datas = json.load(f)["eval_info"]
                    id2paths = {}
                    for data in datas:
                        id2paths[data["id"]] = data["ReasoningPaths"]
                eval_info = []
                for info in json_info:
                    info = info["query_info"]
                    if info["id"] not in qid_250 or info["id"] not in id2paths:
                        continue
                    paths = module.post_process(
                        all_answers[answer_idx], all_path_dict[answer_idx])

                    res_dict = {
                        "id": info["id"], "question": info["question"],
                        "answers": info["answers"], "entities": info["entities"]
                    }
                    f1, acc, recall = eval_f1(paths, info["answers"])
                    res_dict["semanticMethodRetrievalModuleACC"] = acc
                    res_dict["semanticMethodRetrievalModuleF1"] = f1
                    res_dict["semanticMethodRetrievalModuleRecall"] = recall
                    hr_1 = eval_hr_topk(paths, info["answers"], 1)
                    hr_all = eval_hr_topk(paths, info["answers"], len(paths))
                    res_dict["semanticMethodRetrievalModuleHR@1"] = hr_1
                    res_dict["semanticMethodRetrievalModuleHR@All"] = hr_all
                    res_dict["input_token"] = token_count(
                        tokenizer, questions[answer_idx])
                    res_dict["output_token"] = token_count(
                        tokenizer, all_answers[answer_idx])
                    res_dict["user_input"] = questions[answer_idx]
                    res_dict["llm_output"] = all_answers[answer_idx]

                    ReasoningPaths = '\n'.join(paths)
                    res_dict["ReasoningPaths"] = ReasoningPaths
                    eval_info.append(res_dict)

                    answer_idx += 1
                    if DEBUG:
                        break

                answers_file = pr_base_url + \
                    f"{reasoning_dataset}/{subgraph_type}/{retrievaltype}.json"
                if not os.path.exists(answers_file):
                    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
                with open(answers_file, "w", encoding="utf-8") as fp:
                    json.dump(
                        {
                            "basic_info": basic_info,
                            "eval_info": eval_info,
                        }, fp, indent=4, ensure_ascii=False)
                print(f"Finish {answers_file}")
