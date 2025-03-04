import json
from utils.LLM import LLM
from utils.Evaluation import eval_f1,  eval_hit
from utils.PromptTemplate import REASONING_TEMPLATE, PERSONA
import os
path_num = 32
model_name = "qwen2-70b"
generation_base_url = "/back-up/gzy/dataset/VLDB/Rebuttal/R3/metaQA/Generation/"
pr_base_url = "/back-up/gzy/dataset/VLDB/Rebuttal/R3/metaQA/PathRetrieval/"

reasoning_llm = LLM(
    model=model_name, url='http://localhost:8000/v1/chat/completions')

retrievalPipeline = [
    "SPR",
    "SPR/EMB",
    f"SPR/LLM/{model_name}/EMB",
    "BeamSearch/EMB",
    f"BeamSearch/LLM/{model_name}/EMB",
]
dataset_list = ["metaQA"]
subgraph_list = ["PPR", "EMB/triple",
                 f"LLM/{model_name}/EMB/ppr_1000_triple_256"]
if __name__ == "__main__":
    for reasoning_dataset in dataset_list:
        for subgraph_type in subgraph_list:
            for retrievaltype in retrievalPipeline:
                test_file = pr_base_url + \
                    f"{subgraph_type}/{retrievaltype}.json"
                print("Dataset:", reasoning_dataset)
                print(test_file)
                print("Reasoning Model:", model_name)
                json_info = json.load(open(test_file))
                basic_info = json_info["basic_info"]
                basic_info["llm"] = model_name
                eval_info = []
                questions = []
                sample = 0
                hits = 0

                for info in json_info["eval_info"]:
                    sample += 1
                    reasoning_paths = info["ReasoningPaths"]
                    reasoning_paths = reasoning_paths.split("\n")
                    reasoning_paths = reasoning_paths[:path_num]
                    reasoning_paths = "\n".join(reasoning_paths)
                    # print(reasoning_paths)
                    llm_question = REASONING_TEMPLATE.format(
                        paths=reasoning_paths, question=info["question"])
                    questions.append(llm_question)

                answers = reasoning_llm.batch_invoke(PERSONA, questions)
                for info in json_info["eval_info"]:
                    llm_answer = answers.pop(0)
                    info["llm_answer"] = llm_answer

                    f1, acc, recall = eval_f1([llm_answer], info["answers"])
                    hit = eval_hit(llm_answer, info["answers"])
                    info["ACC"] = acc
                    info["F1"] = f1
                    info["Recall"] = recall
                    info["HR"] = hit
                    hits += hit

                    eval_info.append(info)

                print(hits, "/", sample)
                answers_file = f"{generation_base_url}{subgraph_type}/{retrievaltype}/{model_name}_{path_num}_zero_shot_answers.json"
                if not os.path.exists(answers_file):
                    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
                with open(answers_file, "w") as fp:
                    json.dump(
                        {
                            "basic_info": basic_info,
                            "eval_info": eval_info,
                        }, fp, indent=4, ensure_ascii=False)
