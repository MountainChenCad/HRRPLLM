from openai import OpenAI
import time
import json 

class GPTCaller:
    def __init__(self, model_name, api_key, base_url=None, temperature=0.1, top_p=1.0, 
                 max_tokens_completion=200, frequency_penalty=0.0, presence_penalty=0.0,
                 api_retry_delay=5, max_retries=3):
        
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens_completion = max_tokens_completion
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_retry_delay = api_retry_delay
        self.max_retries = max_retries
        
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError("OpenAI API key 未配置或无效。")

        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url
        self.client = OpenAI(**client_args)
        
        # print(f"GPTCaller 初始化: 模型={self.model_name}, Temp={self.temperature}, MaxTokens={self.max_tokens_completion}")
        # if self.base_url: print(f"Base URL: {self.base_url}")

    def get_completion(self, prompt_content):
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_content}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens_completion,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=False 
                )
                result_text = response.choices[0].message.content.strip()
                return result_text
            except Exception as e:
                retries += 1; error_message = f"LLM API 请求错误 (尝试 {retries}/{self.max_retries}): {str(e)}"
                try: 
                    if hasattr(e, 'response') and e.response: error_message += f" | API 错误: {json.dumps(e.response.json())}"
                except: pass 
                print(error_message)
                if retries < self.max_retries: time.sleep(self.api_retry_delay)
                else: print("已达到最大重试次数。"); return None 
        return None 

if __name__ == "__main__":
    try:
        from config import OPENAI_API_KEY, OPENAI_PROXY_BASE_URL, LLM_CALLER_PARAMS
        if OPENAI_API_KEY == "YOUR_API_KEY_HERE": print("API Key 未配置，跳过 GPTCaller 测试。")
        else:
            caller = GPTCaller(model_name=LLM_CALLER_PARAMS.get("model_name", "gpt-4o-mini"), api_key=OPENAI_API_KEY, base_url=OPENAI_PROXY_BASE_URL, temperature=0.1, max_tokens_completion=50)
            response = caller.get_completion("你好，请介绍一下你自己。")
            if response: print("\nLLM 回复:\n", response)
            else: print("\n未能从LLM获取回复。")
    except ImportError:
        print("无法导入config，跳过GPTCaller的独立测试。")