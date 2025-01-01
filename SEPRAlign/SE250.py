from utils.PromptTemplate import PRERETIEVAL_NODE_PERSONA, PRERETIEVAL_EDGE_PERSONA, PRERETIEVAL_TRIPLE_PERSONA
from utils.Tools import get_query_subgraph, get_triples
from pre_retrieval import *
import json
from utils.LLM import LocalLLM
from utils.Evaluation import eval_cover
from config import model_paths, reasoning_model, dataset_list, subgraph_list
import os
from utils.SemanticModel import EmbeddingModel
from transformers import AutoTokenizer
from tqdm import tqdm

ppr_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/subgraph/{reasoning_dataset}/subgraph/PPR.json"
id_250_base_url = "/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
output_url = "/back-up/gzy/dataset/VLDB/SE_new/{reasoning_dataset}/subgraph/{scale}.json"


tokenizer = AutoTokenizer.from_pretrained(model_paths[reasoning_model])
preRetrievalPipeline = [
    ("LLM_token_scale/qwen2-70b/EMB/triple/", PRERETIEVAL_TRIPLE_PERSONA,
     PreRetrievalModuleLLMEdgeTokenAlign),
    ("LLM_token_scale/qwen2-70b/EMB/edge/", PRERETIEVAL_EDGE_PERSONA,
     PreRetrievalModuleLLMTripleTokenAlign),
    ("LLM_token_scale/qwen2-70b/EMB/node/",
     PRERETIEVAL_NODE_PERSONA, PreRetrievalModuleLLMNodeTokenAlign),
]
scales = [1000, 2000, 4000, 8000, 16000]
emb = EmbeddingModel()


def token_count(tokenizer, query):
    tokenized_prediction = tokenizer.encode(query)
    return len(tokenized_prediction)


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
        for preRetrievalType, persona, module in preRetrievalPipeline:
            for scale in scales:
                pre = module(tokenizer, scale, emb)
                for query in tqdm(get_query_subgraph(ppr_file), total=1500):
                    if query.qid not in id_250:
                        continue
                    user_input = pre.process
                    (query)
                    llm_question = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": user_input}
                        ],
                        tokenize=False
                    )
                    all_questions.append(llm_question)
                    if DEBUG:
                        break
    reasoning_llm = LocalLLM(model_paths[reasoning_model])
    # 统一进行推理
    all_answers = reasoning_llm.batch_invoke(all_questions)

    # 将结果写回各自的文件
    answer_idx = 0
    for reasoning_dataset in dataset_list:
        ppr_file = ppr_base_url.format(reasoning_dataset=reasoning_dataset)

        id_250_file = id_250_base_url.format(
            reasoning_dataset=reasoning_dataset)
        id_250 = set()
        with open(id_250_file, "r") as f:
            for line in f:
                id_250.add(json.loads(line)["id"])
        for retrievaltype, persona, module in preRetrievalPipeline:
            for scale in scales:
                post = module(tokenizer, scale, emb)
                json_datas = {
                    "base_info": {"Dataset": reasoning_dataset,
                                  "ppr_file": ppr_file, "scale": scale},
                    "eval_info": [],
                }
                for query in tqdm(get_query_subgraph(ppr_file), total=1500):
                    if query.qid not in id_250:
                        continue
                    query.input_tokens = token_count(all_questions[answer_idx])
                    query.user_input = all_questions[answer_idx]
                    query.window = scale

                    res_dict = {"id": query.qid, "question": query.question,
                                "answers": query.answers, "entities": query.entities}

                    found_count = eval_cover(query.subgraph, query.answers)
                    print("Eval-structureMethodPreRetrievalModuleACC:",
                          f"{found_count}/{len(query.subgraph.vs)}")
                    res_dict[
                        "structureMethodPreRetrievalModuleACC"] = f"{found_count}/{len(query.subgraph.vs)}"
                    res_dict["afterStructureMethodPreRetrievalModule"] = str(
                        {"nodes": len(query.subgraph.vs), "edges": len(query.subgraph.es)})

                    query = post.post_process(
                        all_answers[answer_idx], query)

                    found_count = eval_cover(query.subgraph, query.answers)
                    print("Eval-semanticMethodPreRetrievalModuleACC:",
                          f"{found_count}/{len(query.subgraph.vs)}")
                    res_dict[
                        "semanticMethodPreRetrievalModuleACC"] = f"{found_count}/{len(query.subgraph.vs)}"
                    res_dict["afterSemanticMethodPreRetrievalModule"] = str(
                        {"nodes": len(query.subgraph.vs), "edges": len(query.subgraph.es)})
                    res_dict["output_token"] = query.output_tokens
                    res_dict["input_token"] = query.input_tokens
                    res_dict["user_input"] = all_questions[answer_idx]
                    res_dict["llm_output"] = all_answers[answer_idx]
                    json_datas["eval_info"].append(res_dict)

                    answer_idx += 1
                    if DEBUG:
                        break

                answers_file = output_url.format(
                    reasoning_dataset=reasoning_dataset, scale=scale)
                if not os.path.exists(answers_file):
                    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
                with open(answers_file, "w") as fp:
                    json.dump(json_datas, fp, indent=4, ensure_ascii=False)
                print(f"Finish {answers_file}")
