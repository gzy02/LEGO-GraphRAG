import os
import json
from utils.Tools import get_query, Tools
from pipeline import SEPipeline
from pre_retrieval import *
print("PID =", os.getpid())
kg = Tools().kg

windows = {
    "edge": [4, 8, 16, 32, 64, 128],
    "node": [4, 8, 16, 32, 64, 128],
    "triple": [32, 64, 128, 256, 512, 1024]
}
configs = {
    "edge": PreRetrievalModuleEdge,
    "node": PreRetrievalModuleNode,
    "triple": PreRetrievalModuleTriples
}
for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:  #
    ppr_file = f"/back-up/lzy/Dataset/{reasoning_dataset}/{reasoning_dataset}_250.jsonl"
    for semantic_type in ["BM25"]:  # ,"BGE",  "BM25"
        for module_type in ["edge", "node", "triple"]:
            for window in windows[module_type]:
                module_pipe = SEPipeline(
                    PreRetrievalModulePPR(200),
                    configs[module_type](window, semantic_type)
                )
                print(module_pipe)

                basic_info = {
                    "Dataset": reasoning_dataset,
                    "ppr_file": ppr_file,
                    "window": window,
                    "structureMethod": str(module_pipe.structureMethod),
                    "semanticMethod": str(module_pipe.semanticMethod)
                }
                eval_info = []
                cnt = 0

                for query in get_query(kg, ppr_file):
                    query, res_dict = module_pipe.run(kg, query)
                    eval_info.append(res_dict)
                    cnt += 1

                json_path = f"/back-up/gzy/dataset/VLDB/SubgraphExtraction/{reasoning_dataset}/Scale/{semantic_type}/{module_type}/{str(window)}.json"
                if not os.path.exists(json_path):
                    os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as fp:
                    json.dump(
                        {
                            "basic_info": basic_info,
                            "eval_info": eval_info
                        }, fp, indent=4)
