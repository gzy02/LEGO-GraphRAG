
from config import emb_model_dir, rerank_model_dir
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


class EmbeddingModel():
    def __init__(self, model_dir=None):
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.model = SentenceTransformer(
            model_dir, device="cuda"
        )
        self.batch_size = 128
        self.model.eval()

    def encode(self, sentences, normalize_embeddings=False):
        with torch.no_grad():
            return self.model.encode(sentences, normalize_embeddings=normalize_embeddings, batch_size=self.batch_size)

    def get_scores(self, question, all_chunks):
        all_embeddings = self.encode(
            all_chunks  # , normalize_embeddings=True
        )
        query_embedding = self.encode(
            question  # , normalize_embeddings=True
        )[None, :]
        cosine_scores = (all_embeddings * query_embedding).sum(1)
        return cosine_scores


class BGEModel:
    def __init__(self, model_dir: str = None):
        super().__init__()
        model_dir = rerank_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            device_map="auto"
        )
        self.model.eval()
        self.batch_size = 128

    def get_scores_pairs(self, pairs):
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            scores = self.model(
                **inputs, return_dict=True).logits.view(-1, ).float().tolist()

        return scores

    def get_scores(self, query, page_list):
        scores = []
        for i in range(0, len(page_list), self.batch_size):
            pairs = [(query, page) for page in page_list[i:i+self.batch_size]]
            scores += self.get_scores_pairs(pairs)
        return scores
