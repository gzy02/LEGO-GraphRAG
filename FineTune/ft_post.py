import json
from pre_retrieval import *
from retrieval import *
from post_retrieval import *
from utils.Tools import get_query
from utils.PromptTemplate import *
from pipeline import ModulePipeline
import random
reasoning_dataset = "SimpleQuestion"
target_dir = "../sft_dataset/"
dataset_dir = "/back-up/gzy/dataset/graphrag/process_data/"
test_file = dataset_dir + reasoning_dataset + \
    "/"+"SimpleQuestion_ppr1500_2_train.jsonl"
if __name__ == "__main__":
    pre = PreRetrievalModulePPR()
    re = RetrievalModuleBFS(2)
    post = PostRetrievalModuleEmb(32)
    info = []
    print("Dataset:", reasoning_dataset)
    id = 0
    for query in get_query(test_file):
        id += 1
        if id % 100 == 0:
            print(id)

        query = pre.process(query)
        query = re.process(query)
        all_paths = query.reasoning_paths
        # 拿到答案，构造答案路径
        ans_paths = set()
        no_ans_paths = set()
        ans_entities = set(query.answers)
        for path in all_paths:
            entities = set(path.entities_path)
            for entitiy in ans_entities:
                if entitiy in entities:
                    ans_paths.add(str(path))
                    ans_entities.discard(entitiy)
                    break
        # 拿到推理路径，作为instruction的一部分
        query = post.process(query)
        ans_entities = set(query.answers)
        for path in query.reasoning_paths:
            entities = set(path.entities_path)
            if len(entities.intersection(ans_entities)) == 0:
                no_ans_paths.add(str(path))
        # 把output加到instruction里面
        all_paths = list(no_ans_paths.union(ans_paths))
        random.shuffle(all_paths)

        instruction = filter_prompt.format(question=query.question,
                                           entities='; '.join(query.entities), corpus='\n'.join(all_paths))
        output = '\n'.join(ans_paths)
        info.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })
    with open(target_dir+reasoning_dataset+"_32_post_retrieval_sft.json", "w") as fp:
        json.dump(info, fp)
