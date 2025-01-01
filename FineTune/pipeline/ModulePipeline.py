from pre_retrieval import PreRetrievalModule
from retrieval import RetrievalModule
from post_retrieval import PostRetrievalModule
from utils.Query import Query

from dataclasses import dataclass


@dataclass
class ModulePipeline:
    preRetrieval: PreRetrievalModule
    retrieval: RetrievalModule
    postRetrieval: PostRetrievalModule

    def run(self, query: Query):
        try:
            query = self.preRetrieval.process(query)
        except Exception as e:
            print("Error: PreRetrieval", e)
        try:
            query = self.retrieval.process(query)
        except Exception as e:
            print("Error: Retrieval", e)
        try:
            query = self.postRetrieval.process(query)
        except Exception as e:
            print("Error: PostRetrieval", e)
        return query

    def __str__(self):
        return f"Pipeline: {self.preRetrieval} -> {self.retrieval} -> {self.postRetrieval}"
