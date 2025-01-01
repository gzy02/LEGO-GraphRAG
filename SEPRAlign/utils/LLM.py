import vllm.engine
import vllm.engine.arg_utils
import vllm
import config
from transformers import AutoTokenizer


def get_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)


class LocalLLM:
    def __init__(self, model_path):
        self.tokenizer = get_tokenizer(model_path)
        self.sampling_params = vllm.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens
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

    def __str__(self):
        return f"LocalLLM({self.llm}, {self.sampling_params})"
