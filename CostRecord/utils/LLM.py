import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config


class LocalLLM:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)

    def calculate_tokens(self, query: str, response: str):
        input_tokens = len(self.tokenizer.encode(query))
        output_tokens = len(self.tokenizer.encode(response))
        return input_tokens, output_tokens

    def invoke(self, sys_query: str, user_query: str, max_new_tokens=config.max_tokens, temperature=config.temperature):
        query = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_query},
                {"role": "user", "content": user_query}
            ],
            tokenize=False
        )
        # Tokenize input and generate response
        inputs = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        if temperature == 0:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=None,
                top_p=None,
                top_k=None,
                do_sample=False
            ).to(self.device)
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            ).to(self.device)  # Ensure outputs are on the same device

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Calculate tokens
        input_tokens, output_tokens = self.calculate_tokens(
            query, response)

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def __str__(self):
        return f"LocalLLM(Model: {self.model.name_or_path}, Device: {self.device})"
