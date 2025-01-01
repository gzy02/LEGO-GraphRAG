from utils.Evaluation import eval_acc, eval_f1, eval_hr_topk, eval_hit
from pre_retrieval import PreRetrievalModule
from retrieval import RetrievalModule
from post_retrieval import PostRetrievalModule
from utils.Query import Query
import igraph as ig
from time import time
from dataclasses import dataclass
from config import hr_top_k, max_reasoning_paths
from utils.LLM import LLM
from utils.PromptTemplate import REASONING_INPUT


@dataclass
class RecordPipeline:
    preRetrieval: PreRetrievalModule
    retrieval: RetrievalModule
    postRetrieval: PostRetrievalModule

    def run(self, query: Query):
        print("Question ID: ", query.qid)
        res_dict = {"id": query.qid, "question": query.question,
                    "answers": query.answers, "entities": query.entities}
        t = time()
        query = self.preRetrieval.process(query)
        res_dict["PreRetrievalModuleTime"] = time()-t

        # region Eval-PreRetrieval
        found_count = 0
        for ans in query.answers:
            try:
                query.subgraph.vs.find(name=ans)
                found_count += 1
            except:
                pass
        print("Eval-PreRetrieval: ACC =",
              f"{found_count}/{len(query.subgraph.vs)}")
        res_dict["PreRetrievalModuleACC"] = f"{found_count}/{len(query.subgraph.vs)}"
        # endregion

        t = time()
        query = self.retrieval.process(query)
        res_dict["RetrievalModuleTime"] = time()-t
        # region Eval-Retrieval
        paths = [str(path) for path in query.reasoning_paths]
        acc = eval_acc(paths, query.answers)

        f1, acc, recall = eval_f1(paths, query.answers)
        print("Eval-Retrieval: ACC =", acc)
        print("Eval-Retrieval: F1 =", f1)
        print("Eval-Retrieval: Recall =", recall)
        res_dict["RetrievalModuleACC"] = acc
        res_dict["RetrievalModuleF1"] = f1
        res_dict["RetrievalModuleRecall"] = recall
        # endregion
        t = time()
        query = self.postRetrieval.process(query)
        res_dict["PostRetrievalModuleTime"] = time()-t
        # region Eval-PostRetrieval
        paths = [str(path) for path in query.reasoning_paths]
        f1, acc, recall = eval_f1(paths, query.answers)
        hr1 = eval_hr_topk(paths, query.answers, 1)
        hr = eval_hr_topk(paths, query.answers, hr_top_k)

        print("Eval-PostRetrieval: ACC =", acc)
        print("Eval-PostRetrieval: F1 =", f1)
        print("Eval-PostRetrieval: Recall =", recall)
        print(f"Eval-PostRetrieval: HR@{hr_top_k} =", hr)
        print(f"Eval-PostRetrieval: HR@1 =", hr1)
        res_dict["PostRetrievalModuleACC"] = acc
        res_dict["PostRetrievalModuleF1"] = f1
        res_dict["PostRetrievalModuleRecall"] = recall
        res_dict[f"PostRetrievalModuleHR@{hr_top_k}"] = hr
        res_dict["PostRetrievalModuleHR@1"] = hr1
        res_dict[f"PostRetrievalModuleF1@{hr_top_k}"] = eval_f1(paths[:hr_top_k],
                                                                query.answers)[0]
        # endregion

        reasoning_paths = '\n'.join(
            str(path) for path in query.reasoning_paths[:max_reasoning_paths])
        res_dict["ReasoningPaths"] = reasoning_paths
        return query, res_dict

    def __str__(self):
        return f"Pipeline: {self.preRetrieval} -> {self.retrieval} -> {self.postRetrieval}"
