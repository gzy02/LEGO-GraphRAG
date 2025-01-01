
############################################
# region LLM config
llm_url = 'http://localhost:8000/v1/chat/completions'
reasoning_model = "llama3-70b"
temperature = 0
max_tokens = 256
# endregion

# region SentenceModel config
emb_model_dir = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

rerank_model_dir = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/bge-reranker-v2-m3"
# endregion

############################################
# region Datasets config
reasoning_dataset = "webqsp"
ppr_file = f"/back-up/gzy/dataset/graphrag/process_data/{reasoning_dataset}/rank_all.jsonl"
# endregion

############################################
# region Evaluation config
hr_top_k = 10
# endregion
