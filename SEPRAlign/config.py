path_num = 32
reasoning_model = "qwen2-70b"
se_base_url = "/back-up/gzy/dataset/VLDB/Instance/SubgraphExtraction/"
pr_base_url = "/back-up/gzy/dataset/VLDB/Instance/PathRetrieval/"
# generation_base_url = "/back-up/gzy/dataset/VLDB/Pipeline/Generation/"

emb_model_dir = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

rerank_model_dir = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/bge-reranker-v2-m3"
# vllm
temperature = 0
max_tokens = 4000
few_shot = False
one_shot = False
tensor_parallel_size = 1
gpu_memory_utilization = 0.97
quantization = None  # "fp8"
dtype = "auto"  # "bfloat16"
enforce_eager = True  # 省空间

# Different models
model_paths = {
    "qwen2-70b": "/back-up/LLMs/qwen/Qwen2-72B-Instruct-AWQ/",
    "llama3-70b": "/back-up/LLMs/llama3/Meta-Llama-3-70B-Instruct-AWQ/",

    "qwen2-7b": "/back-up/LLMs/qwen/Qwen2-7B-Instruct/",
    "llama3-8b": "/back-up/LLMs/llama3/Meta-Llama-3-8B-Instruct/",

    "llama2-7b": "/back-up/LLMs/llama2/Llama-2-7b-chat-hf/",
    "llama2-13b": "/back-up/LLMs/llama2/Llama-2-13b-chat-hf/",
    "vicuna-7b": "/back-up/LLMs/vicuna-7b-v1.1/",
    "t5-11b": "/back-up/LLMs/t5-11b/",
    "mistral-7b": "/back-up/LLMs/mistral/Mistral-7B-Instruct-v0.2/",
    "glm4-9b": "/back-up/LLMs/chatglm/glm-4-9b-chat/"
}

# Different datasets and subgraph types
# "webqsp", "CWQ", "GrailQA", "WebQuestion"
dataset_list = ["CWQ"]
# "PPR", "EMB/edge", f"LLM/qwen2-70b/EMB/ppr_1000_edge_64"
subgraph_list = ["PPR", "EMB/edge", f"LLM/qwen2-70b/EMB/ppr_1000_edge_64"]
