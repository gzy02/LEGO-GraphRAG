from utils.Tools import get_query
from pre_retrieval import *
from post_retrieval import *
from retrieval import *
import json
import os


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


# {"query": "A man pulls two women down a city street in a rickshaw.", "pos": ["A man is in a city."], "neg": ["A man is a pilot of an airplane.", "It is boring and mundane.", "The morning sunlight was shining brightly and it was warm. ", "Two people jumped off the dock.", "People watching a spaceship launch.", "Mother Teresa is an easy choice.", "It's worth being able to go at a pace you prefer."]}


reasoning_dataset = "CWQ"
sft_dir = "/back-up/gzy/dataset/sft_data/reranker_sft_test/pre/"
fine_tune_jsonl_path = sft_dir + reasoning_dataset + \
    "_train_pre_retrieval_sft.jsonl"
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
    relations = set()
    for path in paths:
        for i in range(len(path) - 1):
            edge_id = G.get_eid(path[i], path[i + 1])
            edge = G.es[edge_id]
            relations.add(edge['name'])

    all_relations = {
        edge["name"]
        for edge in query.subgraph.es
    }
    negative_relations = all_relations - relations
    if len(relations) == 0 or len(negative_relations) == 0:
        continue
    fine_tune_data.append({
        "query": query.question,
        "pos": list(relations),
        "neg": list(negative_relations)
    })
with open(fine_tune_jsonl_path, "w") as fp:
    for item in fine_tune_data:
        json.dump(item, fp)
        fp.write("\n")
    fp.close()
