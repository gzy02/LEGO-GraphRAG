import json
from utils.Tools import get_query
from utils.PromptTemplate import *
from post_retrieval import *
from retrieval import *
from pre_retrieval import *
from utils.Query import Query
from utils.Tools import abandon_rels
from utils.PromptTemplate import preretrieval_prompt
from config import reasoning_model, emb_model_dir
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    def __init__(self, model_dir: str = None):
        super().__init__()
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model = SentenceTransformer(
            model_dir, device="cuda"
        )
        self.model.eval()

    def encode(self, sentences, normalize_embeddings=False):
        with torch.no_grad():
            return self.model.encode(sentences, normalize_embeddings=normalize_embeddings)


class PreRetrievalModuleRoG():
    def __init__(self, window: int = 64):
        super().__init__()
        self.emb = EmbeddingModel()
        self.window = window

    def process(self, query: Query) -> Query:
        """RoG Not FT
        """
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        G = query.subgraph
        relations = {edge["name"]
                     for edge in G.es if not abandon_rels(edge["name"])}
        corpus = list(relations)
        question = query.question
        question_embedding = self.emb.encode(
            question, normalize_embeddings=True)[None, :]
        relation_embeddings = self.emb.encode(
            corpus, normalize_embeddings=True)
        cosine_scores = (relation_embeddings * question_embedding).sum(1)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        # 拿到推理路径，作为instruction的一部分
        relations = set(sorted_paths[:self.window])
        entities = set()
        for entity in query.entities:
            entities.add(G.vs.find(name=entity).index)
        answers = set()
        for answer in query.answers:
            answers.add(G.vs.find(name=answer).index)
        # 拿到答案，构造答案路径
        answer_relations = set()
        all_paths = set()
        for entity in entities:
            for answer in answers:
                all_paths.update(
                    map(tuple, G.get_all_simple_paths(entity, answer, 3)))
        for path in all_paths:
            for i in range(len(path) - 1):
                edge_id = G.get_eid(path[i], path[i + 1])
                edge = G.es[edge_id]
                answer_relations.add(edge['name'])
        # 把output加到instruction里面
        relations.update(answer_relations)
        instruction = preretrieval_prompt.format(relations='\n'.join(
            relations), question=question)
        output = '\n'.join(answer_relations)

        return {
            "instruction": instruction,
            "input": "",
            "output": output
        }


reasoning_dataset = "SimpleQuestion"
target_dir = "../sft_dataset/"
dataset_dir = "/back-up/gzy/dataset/graphrag/process_data/"
test_file = dataset_dir + reasoning_dataset + \
    "/SimpleQuestion_ppr1500_2_train.jsonl"
if __name__ == "__main__":
    pre = PreRetrievalModuleRoG()
    info = []
    print("Dataset:", reasoning_dataset)
    id = 0
    for query in get_query(test_file):
        id += 1
        try:
            item = pre.process(query)
            info.append(item)
        except:
            pass
        if id % 100 == 0:
            print(id)

    with open(target_dir+reasoning_dataset+"_train_pre_retrieval_sft.json", "w") as fp:
        json.dump(info, fp)
