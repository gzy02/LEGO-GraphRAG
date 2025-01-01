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
class EvalPipeline:
    preRetrieval: PreRetrievalModule
    retrieval: RetrievalModule
    postRetrieval: PostRetrievalModule
    reasoning_llm: LLM

    def run(self, query: Query):
        # try:
        print("Question ID: ", query.qid)
        t = time()
        query = self.process(query)
        print("Full Time:", time()-t)
        # except Exception as e:
        #    print("Error: ", e)
        return query

    def process(self, query: Query):
        query = self.preRetrieval.process(query)

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
        # endregion

        query = self.retrieval.process(query)

        # region Eval-Retrieval
        paths = [str(path) for path in query.reasoning_paths]
        acc = eval_acc(paths, query.answers)
        if found_count != 0:
            f1, acc, recall = eval_f1(paths, query.answers)
            print("Eval-Retrieval: ACC =", acc)
            print("Eval-Retrieval: F1 =", f1)
            print("Eval-Retrieval: Recall =", recall)
        # endregion

        query = self.postRetrieval.process(query)

        # region Eval-PostRetrieval
        if found_count != 0 and acc != 0:
            paths = [str(path) for path in query.reasoning_paths]
            f1, acc, recall = eval_f1(paths, query.answers)
            hr = eval_hr_topk(paths, query.answers, hr_top_k)
            # llm_window = self.postRetrieval.top_k
            # hr_llm = eval_hr_topk(paths, query.answers, llm_window)
            print("Eval-PostRetrieval: ACC =", acc)
            print("Eval-PostRetrieval: F1 =", f1)
            print("Eval-PostRetrieval: Recall =", recall)
            print(f"Eval-PostRetrieval: HR@{hr_top_k} =", hr)
            # print(f"Eval-PostRetrieval: HR@{llm_window} =", hr_llm)
        # endregion

        reasoning_paths = '\n'.join(
            str(path) for path in query.reasoning_paths[:max_reasoning_paths])
        llm_question = REASONING_INPUT.format(
            paths=reasoning_paths, question=query.question)
        print("Question: ", query.question)
        print("Answers:", query.answers)
        llm_answer = self.reasoning_llm.invoke(llm_question)
        print("LLM Answer:", llm_answer)

        # region Eval-Reasoning
        f1, acc, recall = eval_f1([llm_answer], query.answers)
        hit = eval_hit(llm_answer, query.answers)
        print("Eval-Reasoning: ACC =", acc)
        print("Eval-Reasoning: F1 =", f1)
        print("Eval-Reasoning: Recall =", recall)
        print("Eval-Reasoning: HR =", hit)
        # endregion

    def __str__(self):
        return f"Pipeline: {self.preRetrieval} -> {self.retrieval} -> {self.postRetrieval}"
