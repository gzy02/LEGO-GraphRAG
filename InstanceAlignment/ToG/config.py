
local_models = {
    # vanilla LLMs
    "llama3-8b": "/back-up/LLMs/llama3/Meta-Llama-3-8B-Instruct/",
    "llama2-7b": "/back-up/LLMs/llama2/Llama-2-7b-chat-hf/",
    "llama2-13b": "/back-up/LLMs/llama2/Llama-2-13b-chat-hf/",
    "qwen2-7b": "/back-up/LLMs/qwen/Qwen2-7B-Instruct/",
    "vicuna-7b": "/back-up/LLMs/vicuna-7b-v1.1/",
    "t5-11b": "/back-up/LLMs/t5-11b/",
    "mistral-7b": "/back-up/LLMs/mistral/Mistral-7B-Instruct-v0.2/",
    "glm4-9b": "/back-up/LLMs/chatglm/glm-4-9b-chat/"
}

reasoning_model = "llama2-7b"
hop_pred_model = reasoning_model
tensor_parallel_size = 1
gpu_memory_utilization = 0.45
temperature = 0  # 0.01
max_tokens = 256
# ["\n","<|eot_id|>", ]
stop_tokens = None  # ["</s>",]
quantization = None  # "fp8"
dtype = "auto"  # "bfloat16"
enforce_eager = True  # 省空间
