CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model /back-up/LLMs/llama3/Meta-Llama-3-70B-Instruct-AWQ/ \
    --served-model-name llama3-70b \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --tensor_parallel_size 1 \
    --port 8000

# /back-up/LLMs/llama3/Meta-Llama-3-70B-Instruct-AWQ/
# /back-up/LLMs/qwen/Qwen2-72B-Instruct-AWQ/