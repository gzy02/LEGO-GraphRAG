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


def get_triples(paths, G):
    triples = set()
    for path in paths:
        for i in range(len(path) - 1):
            edge_id = G.get_eid(path[i], path[i + 1])
            edge = G.es[edge_id]
            triple = G.vs[path[i]]['name'] + ", " + \
                edge['name'] + ", " + G.vs[path[i + 1]]['name']
            triples.add(triple)
    return triples


reasoning_dataset = "CWQ"
print(reasoning_dataset)
sft_dir = "/back-up/gzy/dataset/sft_data/reranker_sft_test/re/"
fine_tune_jsonl_path = sft_dir + reasoning_dataset + \
    "_train_re_retrieval_sft.jsonl"
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
    triples = get_triples(paths, G)
    all_triples = set()
    for e in G.es:
        all_triples.add(G.vs[e.source]['name'] +
                        ", " + e['name'] + ", " + G.vs[e.target]['name'])
    neg_triples = all_triples - triples
    if len(triples) == 0 or len(neg_triples) == 0:
        continue
    fine_tune_data.append({
        "query": query.question,
        "pos": list(triples),
        "neg": list(neg_triples)
    })
with open(fine_tune_jsonl_path, "w") as fp:
    for item in fine_tune_data:
        json.dump(item, fp)
        fp.write("\n")
    fp.close()
