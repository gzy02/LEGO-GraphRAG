from utils.Query import Query
from utils.Tools import Tools
from dataclasses import dataclass
from pre_retrieval import PreRetrievalModule
from time import time
from utils.Evaluation import eval_acc, eval_f1, eval_hr_topk, eval_cover
import igraph as ig


@dataclass
class SEPipeline:
    semanticMethod: PreRetrievalModule

    def run(self, query: Query):
        kg = query.subgraph
        res_dict = {"id": query.qid, "question": query.question,
                    "answers": query.answers, "entities": query.entities}

        t = time()
        query = self.semanticMethod.process(query)
        res_dict["semanticMethodPreRetrievalModuleTime"] = time()-t

        # region Eval-PreRetrieval
        found_count = eval_cover(query.subgraph, query.answers)
        # print("Eval-semanticMethodPreRetrievalModuleACC:",
        #      f"{found_count}/{len(query.subgraph.vs)}")
        res_dict["semanticMethodPreRetrievalModuleACC"] = f"{found_count}/{len(query.subgraph.vs)}"
        res_dict["afterSemanticMethodPreRetrievalModule"] = str(
            {"nodes": len(query.subgraph.vs), "edges": len(query.subgraph.es)})

        res_dict["input_tokens"] = query.input_tokens
        res_dict["output_tokens"] = query.output_tokens
        res_dict["st_tokens"] = query.st_tokens

        found_count = eval_cover(kg, query.answers)
        res_dict["Eval-beforeRetrievalACC"] = f"{found_count}/{len(kg.vs)}"
        res_dict["beforeStructureMethodPreRetrievalModule"] = str(
            {"nodes": len(kg.vs), "edges": len(kg.es)})
        # endregion

        return query, res_dict

    def __str__(self):
        return f"{self.semanticMethod}"
