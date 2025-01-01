from langchain_community.llms.ollama import Ollama
import vllm.engine
import vllm.engine.arg_utils
from config import reasoning_model, commercial_models, ollama_models, local_models
import config
from langchain_openai import OpenAI
from langchain_community.llms.moonshot import Moonshot
from utils.TokenCount import token_count
import vllm
from time import time
from vllm.engine.arg_utils import EngineArgs


class LocalLLM:
    def __init__(self, model_path):
        self.sampling_params = vllm.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=config.stop_tokens
        )

        self.llm = vllm.LLM(
            model=model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            quantization=config.quantization
        )

    def invoke(self, query: str):
        return self.llm.generate(query, self.sampling_params, use_tqdm=False)[0].outputs[0].text

    def batch_invoke(self, queries: list):
        outputs = self.llm.generate(
            queries, self.sampling_params)  # , use_tqdm=False)
        return [output.outputs[0].text for output in outputs]


class LLM:
    def __init__(self, model: str = None):
        if model is None:
            model = str(reasoning_model)
        self.model = model
        if model in local_models:
            self.llm = LocalLLM(local_models[model])

        elif model in commercial_models:
            if model.startswith("gpt"):
                self.llm = OpenAI(model_name=model, temperature=0)
            else:
                self.llm = Moonshot(model_name=model, temperature=0)

        elif model in ollama_models:
            self.llm = Ollama(
                model=model,
                # num_ctx=8192,
                temperature=config.temperature,
                num_predict=config.max_tokens
            )

        else:
            raise ValueError("Model not supported")

    def batch_invoke(self, queries):
        if self.model in local_models:
            answers = self.llm.batch_invoke(queries)
        else:
            answers = [self.llm.invoke(query) for query in queries]
        return answers

    def invoke(self, query):
        t = time()
        input_token_count = token_count(query)
        answer = self.llm.invoke(query)
        output_token_count = token_count(answer)
        print(
            f"LLM Call: Input token count: {input_token_count}, Output token count: {output_token_count}, Time taken: {time()-t}")
        return answer


if __name__ == "__main__":
    llm = LLM(model="llama3")

    answer = llm.invoke("Tell me a joke")
    print(answer)
