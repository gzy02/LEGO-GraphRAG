from config import hr_top_k, supported_datasets, max_reasoning_paths

from utils.Tools import get_query_nojudge, get_query
from pre_retrieval import *
from retrieval import *
from post_retrieval import *
from pipeline.RecordPipeline import RecordPipeline
import json
import os
print("PID =", os.getpid())
# import FlagEmbedding.baai_general_embedding.finetune.run

for reasoning_dataset in ["CWQ", "webqsp", "GrailQA", "WebQuestion"]:
    version = "v10"
    type = "pre"

    model_dir = f"/back-up/gzy/ft/minilm/{type}/{reasoning_dataset}/{version}/"

    test_file = supported_datasets[reasoning_dataset]

    if type == "pre":
        module_pipe = RecordPipeline(PreRetrievalModuleEmb(64, model_dir=model_dir),
                                     RetrievalModuleDij(4), PostRetrievalModuleBGE(max_reasoning_paths))
    elif type == "re":
        module_pipe = RecordPipeline(PreRetrievalModulePPR(),
                                     RetrievalModuleEmb(4, model_dir=model_dir), PostRetrievalModuleBGE(max_reasoning_paths))
    elif type == "post":
        module_pipe = RecordPipeline(PreRetrievalModulePPR(),
                                     RetrievalModuleDij(4), PostRetrievalModuleEmb(max_reasoning_paths, model_dir=model_dir))
    print(module_pipe)
    basic_info = {"Dataset": reasoning_dataset,
                  "test_file": test_file, "PreRetrievalModule": str(module_pipe.preRetrieval), "RetrievalModule": str(module_pipe.retrieval), "PostRetrievalModule": str(module_pipe.postRetrieval)}
    eval_info = []
    sample = 0
    for query in get_query(test_file):
        sample += 1

        query, res_dict = module_pipe.run(query)
        eval_info.append(res_dict)
        # if sample == 5:
        #    break
    print(sample)
    print(
        f"HR@{hr_top_k}=", sum([res[f"PostRetrievalModuleHR@{hr_top_k}"] for res in eval_info])/len(eval_info))
    print(
        f"F1@{hr_top_k}=", sum([res[f'PostRetrievalModuleF1@{hr_top_k}'] for res in eval_info])/len(eval_info))
    with open(f"./ft_test/v10_2/{reasoning_dataset}_{type}_paths.json", "w") as fp:
        json.dump(
            {
                "basic_info": basic_info,
                "eval_info": eval_info
            }, fp, indent=4)
    print()
    print()
    print("-----------------------------------")
    print()
    print()
