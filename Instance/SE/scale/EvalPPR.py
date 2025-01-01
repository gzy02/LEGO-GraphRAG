import os
import json
from utils.Tools import get_query, Tools
from pipeline import SEPipeline
from pre_retrieval import *
print("PID =", os.getpid())
tools = Tools()
kg = tools.kg

ppr_nodes = [200, 500, 1000, 2000, 5000, 10000]
for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:
    ppr_file = f"/back-up/lzy/Dataset/{reasoning_dataset}/{reasoning_dataset}_250.jsonl"
    for node in ppr_nodes:
        module_pipe = SEPipeline(
            PreRetrievalModulePPR(node),
            PreRetrievalModuleNone()
        )
        print(module_pipe)

        basic_info = {
            "Dataset": reasoning_dataset,
            "ppr_file": ppr_file,
            "structureMethod": str(module_pipe.structureMethod),
            "semanticMethod": str(module_pipe.semanticMethod)
        }
        eval_info = []
        cnt = 0

        for query in get_query(kg, ppr_file):
            query, res_dict = module_pipe.run(kg, query)
            eval_info.append(res_dict)
            cnt += 1

        with open(f"/back-up/gzy/dataset/VLDB/SubgraphExtraction/{reasoning_dataset}/subgraph/PPR/{str(node)}.json", "w") as fp:
            json.dump(
                {
                    "basic_info": basic_info,
                    "eval_info": eval_info
                }, fp, indent=4)
