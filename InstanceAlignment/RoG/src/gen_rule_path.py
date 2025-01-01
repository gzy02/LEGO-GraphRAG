import re
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import datasets
from datasets import load_dataset
import utils
import argparse
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

datasets.disable_progress_bar()

N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]
        ) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
PATH_RE = r"<PATH>(.*)<\/PATH>"
INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def parse_prediction(prediction):
    """
    Parse a list of predictions to a list of rules

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for p in prediction:
        path = re.search(PATH_RE, p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = []
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.append(rel)
        results.append(rules)
    return results


def generate_seq(
    model, input_text, tokenizer, num_beam=3, do_sample=False, max_new_tokens=100
):
    # tokenize the question
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    # generate sequences
    output = model.generate(
        input_ids=input_ids,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=False,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
    prediction = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1]:], skip_special_tokens=True
    )
    prediction = [p.strip() for p in prediction]

    if num_beam > 1:
        scores = output.sequences_scores.tolist()
        norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()
    else:
        scores = [1]
        norm_scores = [1]

    return {"paths": prediction, "scores": scores, "norm_scores": norm_scores}


def gen_prediction(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if args.lora or os.path.exists(args.model_path + "/adapter_config.json"):
        print("Load LORA model")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=True,
        )

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.reason_type,
                              args.model_name, args.split)
    print("Save results to: ", output_dir)

    # Load dataset
    # , split=args.split)
    dataset = load_dataset('json', data_files={"test": input_file})["test"]

    # Load prompt template
    prompter = utils.InstructFormater(args.prompt_path)

    def prepare_dataset(sample):

        # Prepare input prompt
        sample["text"] = prompter.format(
            instruction=INSTRUCTION, message=sample["question"]
        )
        # Find ground-truth paths for each Q-P pair
        graph = utils.build_graph(sample["subgraph"])
        paths = utils.get_truth_paths(
            sample["entities"], sample["answers"], graph)
        ground_paths = set()
        for path in paths:
            # extract relation path
            ground_paths.add(tuple([p[1] for p in path]))
        sample["ground_paths"] = list(ground_paths)
        return sample

    dataset = dataset.map(
        prepare_dataset,
        num_proc=N_CPUS,
    )

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(
        output_dir, f"predictions_{args.n_beam}_{args.do_sample}.jsonl"
    )
    f, processed_results = get_output_file(prediction_file, force=args.force)
    for data in tqdm(dataset):
        question = data["question"]
        input_text = data["text"]
        qid = data["id"]
        if qid in processed_results:
            continue
        raw_output = generate_seq(
            model,
            input_text,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beam=args.n_beam,
            do_sample=args.do_sample,
        )
        rel_paths = parse_prediction(raw_output["paths"])
        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Prediction: ", rel_paths)
        # prediction = outputs[0]["generated_text"].strip()
        data = {
            "id": qid,
            "question": question,
            "prediction": rel_paths,
            "ground_paths": data["ground_paths"],
            "input": input_text,
            "raw_output": raw_output,
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
    f.close()

    return prediction_file


model_dict = {
    'llama2': "/back-up/LLMs/models/Llama-2-7b-chat-hf/",
    'RoG': "/back-up/gzy/ft/RoG/",
    "llama3": "/back-up/LLMs/llama3/Meta-Llama-3-8B-Instruct/",
    "rog-llama3": "/back-up/gzy/models/RoG-llama3-8B-Instruct/"
}

if __name__ == "__main__":
    reasoning_dataset = "webqsp"
    reasoning_type = "PPR2"
    model_name = "RoG"
    task_type = "MainExperiment"  # "MultiHopExperiment"
    model_path = model_dict[model_name]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=f"/back-up/gzy/dataset/AAAI/{task_type}/{reasoning_dataset}/{reasoning_type}/"
    )
    parser.add_argument("--d", "-d", type=str, default="test_name.jsonl")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument("--output_path", type=str,
                        default="results/gen_rule_path_main")
    parser.add_argument("--reason_type", type=str,
                        default=f"{reasoning_dataset}-{reasoning_type}")
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default=model_name,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model_name for save results",
        default=model_path,
    )
    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="../prompts/llama2.txt"
    )
    parser.add_argument(
        "--rel_dict",
        nargs="+",
        default=["datasets/KG/fbnet/relations.dict"],
        help="relation dictionary",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--lora", action="store_true",
                        help="load lora weights")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true", help="do sampling")

    args = parser.parse_args()

    gen_path = gen_prediction(args)
