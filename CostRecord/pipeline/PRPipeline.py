from utils.Query import Query
from config import hr_top_k
from dataclasses import dataclass
from retrieval import RetrievalModule
from post_retrieval import PostRetrievalModule
from time import time
from utils.Evaluation import eval_acc, eval_f1, eval_hr_topk


@dataclass
class PRPipeline:
    structureMethod: RetrievalModule
    semanticMethod: PostRetrievalModule

    def run(self, query: Query):
        # print("Question ID: ", query.qid)

        res_dict = {"id": query.qid, "question": query.question,
                    "answers": query.answers, "entities": query.entities}

        t = time()
        query = self.structureMethod.process(query)
        res_dict["structureMethodRetrievalModuleTime"] = time()-t

        # region Eval-Retrieval
        paths = [str(path) for path in query.reasoning_paths]
        acc = eval_acc(paths, query.answers)

        f1, acc, recall = eval_f1(paths, query.answers)
        # print("Eval-structureMethodRetrieval: ACC =", acc)
        # print("Eval-structureMethodRetrieval: F1 =", f1)
        # print("Eval-structureMethodRetrieval: Recall =", recall)
        res_dict["structureMethodRetrievalModuleACC"] = acc
        res_dict["structureMethodRetrievalModuleF1"] = f1
        res_dict["structureMethodRetrievalModuleRecall"] = recall
        # endregion

        t = time()
        query = self.semanticMethod.process(query)
        res_dict["semanticMethodRetrievalModuleTime"] = time()-t

        # region Eval-Retrieval
        paths = [str(path) for path in query.reasoning_paths]
        acc = eval_acc(paths, query.answers)

        f1, acc, recall = eval_f1(paths, query.answers)
        # print("Eval-semanticMethodRetrieval: ACC =", acc)
        # print("Eval-semanticMethodRetrieval: F1 =", f1)
        # print("Eval-semanticMethodRetrieval: Recall =", recall)
        res_dict["semanticMethodRetrievalModuleACC"] = acc
        res_dict["semanticMethodRetrievalModuleF1"] = f1
        res_dict["semanticMethodRetrievalModuleRecall"] = recall
        paths = [str(path) for path in query.reasoning_paths]

        hr_1 = eval_hr_topk(paths, query.answers, 1)
        hr_k = eval_hr_topk(paths, query.answers, hr_top_k)
        hr_all = eval_hr_topk(paths, query.answers, len(paths))
        # print("Eval-semanticMethodRetrieval: HR@1 =", hr_1)
        # print(f"Eval-semanticMethodRetrieval: HR@{hr_top_k} =", hr_k)
        # print("Eval-semanticMethodRetrieval: HR@All =", hr_all)
        res_dict["semanticMethodRetrievalModuleHR@1"] = hr_1
        res_dict[f"semanticMethodRetrievalModuleHR@{hr_top_k}"] = hr_k
        res_dict["semanticMethodRetrievalModuleHR@All"] = hr_all

        res_dict["input_tokens"] = query.input_tokens
        res_dict["output_tokens"] = query.output_tokens
        res_dict["st_tokens"] = query.st_tokens

        reasoning_paths = '\n'.join(
            str(path) for path in query.reasoning_paths)
        res_dict["ReasoningPaths"] = reasoning_paths
        # endregion

        return query, res_dict

    def __str__(self):
        return f"{self.structureMethod} -> {self.semanticMethod}"
