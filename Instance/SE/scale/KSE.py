from pre_retrieval import PreRetrievalModuleEgo, PreRetrievalModuleNone
from utils.Tools import get_query, Tools
from pipeline import SEPipeline
import os
import json
kg = Tools().kg
if __name__ == "__main__":
    for hop in [1, 2, 3, 4]:
        module_pipe = SEPipeline(
            PreRetrievalModuleEgo(hop),
            PreRetrievalModuleNone()
        )
        print(module_pipe)

        for reasoning_dataset in ["webqsp", "CWQ", "GrailQA", "WebQuestion"]:
            ppr_file = f"/back-up/gzy/dataset/VLDB/new250/{reasoning_dataset}_250_new.jsonl"
            basic_info = {
                "Dataset": reasoning_dataset,
                "ppr_file": ppr_file,
                "structureMethod": str(module_pipe.structureMethod),
                "semanticMethod": str(module_pipe.semanticMethod)
            }
            eval_info = []
            for query in get_query(kg, ppr_file):
                query, res_dict = module_pipe.run(kg, query)
                eval_info.append(res_dict)
            # module_pipe.process(query)
            json_path = f"/back-up/gzy/dataset/VLDB/SE_new/{reasoning_dataset}/Scale/Ego/{str(hop)}.json"
            if not os.path.exists(json_path):
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as fp:
                json.dump(
                    {
                        "basic_info": basic_info,
                        "eval_info": eval_info
                    }, fp, indent=4)
