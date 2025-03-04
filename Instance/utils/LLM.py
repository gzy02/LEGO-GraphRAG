import aiohttp
import config
import time
import requests
import traceback


class LLM:
    def __init__(self, url: str = None, model: str = None):
        model = model if model else str(config.reasoning_model)
        url = url if url else config.llm_url
        self.model = model
        self.url = url

    async def ainvoke(self, sys_prompt, user_prompt, temperature=None, max_tokens=None):
        temperature = temperature or config.temperature
        max_tokens = max_tokens or config.max_tokens

        json_payload = self._build_payload(
            sys_prompt, user_prompt, temperature, max_tokens)
        max_tries = 3
        while True:
            try:
                max_tries -= 1
                if max_tries == 0:
                    return "", -1, -1, -1
                return await self._async_request(json_payload)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()  # 输出详细的异常信息和堆栈跟踪
                time.sleep(1)

    async def _async_request(self, json_payload):
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=json_payload) as response:
                response_json = await response.json()
                request_time = time.time() - start_time
                return self._parse_response(response_json, request_time)

    def batch_invoke(self, sys_prompt, user_prompts, temperature=None, max_tokens=None):
        from tqdm import tqdm
        temperature = temperature or config.temperature
        max_tokens = max_tokens or config.max_tokens
        answers = []
        for user_prompt in tqdm(user_prompts):
            json_payload = self._build_payload(
                sys_prompt, user_prompt, temperature, max_tokens)
            response_json, request_time = self._sync_request(json_payload)
            answers.append(self._parse_response(
                response_json, request_time)[0])
        return answers

    def invoke(self, sys_prompt, user_prompt, temperature=None, max_tokens=None):
        temperature = temperature or config.temperature
        max_tokens = max_tokens or config.max_tokens

        json_payload = self._build_payload(
            sys_prompt, user_prompt, temperature, max_tokens)
        response_json, request_time = self._sync_request(json_payload)
        return self._parse_response(response_json, request_time)

    def _sync_request(self, json_payload):
        start_time = time.time()
        response = requests.post(self.url, json=json_payload)
        request_time = time.time() - start_time
        return response.json(), request_time

    def _build_payload(self, sys_prompt, user_prompt, temperature, max_tokens):
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _parse_response(response_json, request_time):
        answer = response_json['choices'][0]['message']['content']
        prompt_tokens = response_json['usage']['prompt_tokens']
        completion_tokens = response_json['usage']['completion_tokens']
        return answer, prompt_tokens, completion_tokens, request_time
