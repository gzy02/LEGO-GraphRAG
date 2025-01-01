
from config import emb_model_dir, rerank_model_dir
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import heapq
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from functools import lru_cache
import random
from nltk.tokenize import word_tokenize


class SemanticModel:
    def __init__(self):
        pass

    def get_scores(self, question: str, corpus: Tuple[str]):
        pass

    def top_k(self, question: str, corpus: List[str], k: int):
        if len(corpus) == 0:
            return []
        corpus = tuple(set(corpus))
        if k < 0 or k >= len(corpus):
            # return all with ranking
            scores = self.get_scores(question, corpus)
            sorted_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [corpus[i] for i in sorted_indices]
        scores = self.get_scores(question, corpus)
        top_k_indices = heapq.nlargest(
            k, range(len(scores)), scores.__getitem__)
        return [corpus[i] for i in top_k_indices]

    def __str__(self) -> str:
        # 打印子类的名称
        return self.__class__.__name__


class RandomModel(SemanticModel):
    def __init__(self):
        super().__init__()

    def get_scores(self, question, corpus):
        return [0.0] * len(corpus)

    def top_k(self, question, corpus, k):
        if k < 0:
            return set(corpus)
        return random.sample(corpus, min(k, len(corpus)))


class BM25Model(SemanticModel):
    def __init__(self, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.tokenizer = tokenizer if tokenizer is not None else word_tokenize
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        return text.split()

    # @lru_cache(maxsize=16)
    def get_scores(self, question, corpus):
        corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(corpus,
                              k1=self.k1, b=self.b, epsilon=self.epsilon)
        tokenized_question = self.tokenizer(question)
        return self.bm25.get_scores(tokenized_question)


class EmbeddingModel(SemanticModel):
    def __init__(self, model_dir=None):
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = SentenceTransformer(
            model_dir, device="cuda"
        )
        self.batch_size = 65536
        self.model.eval()

    def encode(self, sentences, normalize_embeddings=False):
        with torch.no_grad():
            return self.model.encode(sentences, normalize_embeddings=normalize_embeddings, batch_size=self.batch_size)

    # @lru_cache(maxsize=16)
    def get_scores(self, question, all_chunks):
        all_embeddings = self.encode(
            all_chunks  # , normalize_embeddings=True
        )
        query_embedding = self.encode(
            question  # , normalize_embeddings=True
        )[None, :]
        cosine_scores = (all_embeddings * query_embedding).sum(1)

        # 释放未使用的显存
        # del all_embeddings
        # del query_embedding
        # torch.cuda.empty_cache()
        return cosine_scores


class BGEModel(SemanticModel):
    def __init__(self, model_dir: str = None):
        super().__init__()
        model_dir = rerank_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, device_map="auto"
        )
        self.model.eval()
        self.batch_size = 1024

    def get_scores_pairs(self, pairs):
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            scores = self.model(
                **inputs, return_dict=True).logits.view(-1, ).float().tolist()
        # 释放未使用的显存
        # del inputs
        # torch.cuda.empty_cache()

        return scores

    # @lru_cache(maxsize=16)
    def get_scores(self, query, page_list):
        scores = []
        for i in range(0, len(page_list), self.batch_size):
            pairs = [(query, page) for page in page_list[i:i+self.batch_size]]
            scores += self.get_scores_pairs(pairs)
        return scores
