
############################################
# region LLM config
llm_url = 'http://localhost:8000/v1/chat/completions'
reasoning_model = "qwen2-70b"
temperature = 0
max_tokens = 512


dataset_list = ["webqsp", "CWQ", "GrailQA", "WebQuestion"]
subgraph_list = ["PPR", "EMB/edge",
                 f"LLM/{reasoning_model}/EMB/ppr_1000_edge_64"]

paths = {
    "qwen2-70b": "/back-up/LLMs/qwen/Qwen2-72B-Instruct-AWQ/",
    "llama3-70b": "/back-up/LLMs/llama3/Meta-Llama-3-70B-Instruct-AWQ/",
}
# endregion

# region SentenceModel config
emb_model_dir = "sentence-transformers/all-MiniLM-L6-v2/"
rerank_model_dir = "sentence-transformers/bge-reranker-v2-m3"
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
