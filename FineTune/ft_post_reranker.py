from utils.Tools import get_query
from pre_retrieval import *
from post_retrieval import *
from retrieval import *
import json
from utils.ReasoningPath import ReasoningPath

train_ppr = {
    "WebQuestion": "/back-up/gzy/dataset/graphrag/process_data/WebQuestion/WebQuestion_ppr2000_2_train.jsonl",
    "CWQ": "/back-up/gzy/dataset/graphrag/WSDM2021/CWQ/train_name.jsonl",
    "webqsp": "/back-up/gzy/dataset/graphrag/WSDM2021/webqsp/train_name.jsonl",
    "GrailQA": "/back-up/gzy/dataset/graphrag/process_data/GrailQA/GrailQA_train_ppr2000_2.jsonl"
}

test_ppr = {
    "CWQ": "/back-up/gzy/dataset/AAAI/MainExperiment/CWQ/PPR2/test_name.jsonl",
    "webqsp": "/back-up/gzy/dataset/AAAI/MainExperiment/webqsp/PPR2/test_name.jsonl",
    "WebQuestion": "/back-up/gzy/dataset/graphrag/process_data/WebQuestion/WebQuestion_ppr2000_2_test.jsonl",
    "GrailQA": "/back-up/gzy/dataset/graphrag/process_data/GrailQA/GrailQA_test_ppr2000_2.jsonl"
}


def get_reasoning_paths(paths):
    reasoning_paths = []
    for path in paths:
        reasoning_path = ReasoningPath(path[0])
        for i in range(len(path) - 1):
            edge_id = G.get_eid(path[i], path[i + 1])
            edge = G.es[edge_id]
            triple = (G.vs[path[i]]['name'],
                      edge['name'], G.vs[path[i + 1]]['name'])
            reasoning_path.add_triple(triple)
        reasoning_paths.append(reasoning_path)
    return reasoning_paths


reasoning_dataset = "CWQ"
print(reasoning_dataset)
sft_dir = "/back-up/gzy/dataset/sft_data/reranker_sft_test/post/"
fine_tune_jsonl_path = sft_dir + reasoning_dataset + \
    "_train_post_retrieval_sft.jsonl"
fine_tune_data = []
for query in get_query(test_ppr[reasoning_dataset]):
    answers = query.answers
    entities = query.entities
    G = query.subgraph
    paths = []
    for entity in entities:
        for answer in answers:
            try:
                paths.extend(G.get_all_simple_paths(entity, answer, 4))
            except:
                pass
    reasoning_paths = get_reasoning_paths(paths)
    all_paths = []
    for entity in entities:
        try:
            all_paths.extend(G.get_all_simple_paths(entity, cutoff=2))
        except:
            pass
    neg_paths = set(map(tuple, all_paths)) - set(map(tuple, paths))
    neg_paths = get_reasoning_paths(neg_paths)
    if len(reasoning_paths) == 0 or len(neg_paths) == 0:
        continue
    fine_tune_data.append({
        "query": query.question,
        "pos": [str(path) for path in reasoning_paths],
        "neg": [str(path) for path in neg_paths]
    })
with open(fine_tune_jsonl_path, "w") as fp:
    for item in fine_tune_data:
        json.dump(item, fp)
        fp.write("\n")
    fp.close()
