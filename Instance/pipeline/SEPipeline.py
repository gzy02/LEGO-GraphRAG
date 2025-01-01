from utils.Query import Query
from utils.Tools import Tools
from dataclasses import dataclass
from pre_retrieval import PreRetrievalModule
from time import time
from utils.Evaluation import eval_acc, eval_f1, eval_hr_topk, eval_cover
import igraph as ig


def get_triples(G: ig.Graph):
    ans = []
    for edge in G.es:
        head = G.vs[edge.source]["name"]
        tail = G.vs[edge.target]["name"]
        rel = edge["name"]
        ans.append([head, rel, tail])
    return ans


@dataclass
class SEPipeline:
    structureMethod: PreRetrievalModule
    semanticMethod: PreRetrievalModule

    def run(self, kg: ig.Graph, query: Query):
        print("Question ID: ", query.qid)

        res_dict = {"id": query.qid, "question": query.question,
                    "answers": query.answers, "entities": query.entities}

        t = time()
        query = self.structureMethod.process(query)
        res_dict["structureMethodPreRetrievalModuleTime"] = time()-t

        # region Eval-PreRetrieval
        found_count = eval_cover(query.subgraph, query.answers)
        print("Eval-structureMethodPreRetrievalModuleACC:",
              f"{found_count}/{len(query.subgraph.vs)}")
        res_dict["structureMethodPreRetrievalModuleACC"] = f"{found_count}/{len(query.subgraph.vs)}"
        res_dict["afterStructureMethodPreRetrievalModule"] = str(
            {"nodes": len(query.subgraph.vs), "edges": len(query.subgraph.es)})
        # endregion
        # res_dict["ppr_subgraph"]=get_triples(query.subgraph)
        t = time()
        query = self.semanticMethod.process(query)
        res_dict["semanticMethodPreRetrievalModuleTime"] = time()-t
        # region Eval-PreRetrieval
        found_count = eval_cover(query.subgraph, query.answers)
        print("Eval-semanticMethodPreRetrievalModuleACC:",
              f"{found_count}/{len(query.subgraph.vs)}")
        res_dict["semanticMethodPreRetrievalModuleACC"] = f"{found_count}/{len(query.subgraph.vs)}"
        res_dict["afterSemanticMethodPreRetrievalModule"] = str(
            {"nodes": len(query.subgraph.vs), "edges": len(query.subgraph.es)})

        found_count = eval_cover(kg, query.answers)
        res_dict["Eval-beforeRetrievalACC"] = f"{found_count}/{len(kg.vs)}"
        res_dict["beforeStructureMethodPreRetrievalModule"] = str(
            {"nodes": len(kg.vs), "edges": len(kg.es)})
        # res_dict["subgraph"] = get_triples(query.subgraph)
        # endregion

        return query, res_dict

    def __str__(self):
        return f"{self.structureMethod} -> {self.semanticMethod}"
