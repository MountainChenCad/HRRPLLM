# src/api_callers/openai_caller.py
import time
import json
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError, APITimeoutError
from .base_caller import BaseCaller


class OpenAICaller(BaseCaller):
    def __init__(self, model_name, api_key, base_url=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = base_url

        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url

        try:
            self.client = OpenAI(**client_args)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    def get_completion(self, prompt_content):
        retries = 0
        # --- MODIFICATION: Determine correct max_tokens parameter name ---
        max_tokens_param_name = "max_tokens"  # Default
        # Check for models that might use 'max_completion_tokens'
        # This includes "o1" as per the error, and potentially "o1-mini", "o1-preview"
        # Also, some newer models or endpoints might adopt this.
        # A more general check could be for "o1" or "oone" or specific model IDs.
        if self.model_name.startswith("o1") or "oone" or self.model_name.startswith("o3") in self.model_name.lower() or \
                "gpt-4.5" in self.model_name.lower() or "gpt-4.1" in self.model_name.lower():  # Add other prefixes if known
            max_tokens_param_name = "max_completion_tokens"
            print(f"  Note: Using '{max_tokens_param_name}' for model: {self.model_name}")
        # --- END MODIFICATION ---

        while retries < self.max_retries:
            try:
                api_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt_content}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    # "max_tokens": self.max_tokens_completion, # Original line
                    max_tokens_param_name: self.max_tokens_completion,  # MODIFIED: Use dynamic param name
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "stream": False
                }

                if "qwen" in self.model_name.lower():  # For Qwen models
                    if "extra_body" not in api_params:
                        api_params["extra_body"] = {}
                    api_params["extra_body"]["enable_thinking"] = False
                    # print(f"  Note: Added 'enable_thinking=False' to 'extra_body' for Qwen model: {self.model_name}")

                response = self.client.chat.completions.create(**api_params)

                result_text = response.choices[0].message.content.strip()
                return result_text
            except (APIConnectionError, RateLimitError, APIStatusError, APITimeoutError) as e:
                self._handle_error(e, retries)
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print("Max retries reached for OpenAI API call.")
                    return None
            except Exception as e:
                self._handle_error(e, retries)

                # --- MODIFICATION: Specific check for the max_tokens error ---
                error_str = str(e).lower()
                if "unsupported parameter" in error_str and (
                        "max_tokens" in error_str or "max_completion_tokens" in error_str):
                    print(f"  Error regarding max_tokens parameter name for model {self.model_name} persists: {e}")
                    # This might indicate the initial model name check wasn't exhaustive enough,
                    # or the API changed. For now, we stop retrying for this specific error if it recurs.
                    return f"ERROR_MAX_TOKENS_PARAM: {e}"
                # --- END MODIFICATION ---

                if "unexpected keyword argument" in error_str and "qwen" in self.model_name.lower():
                    print(
                        f"  Error regarding 'unexpected keyword argument' persists for Qwen model {self.model_name} even with 'extra_body'.")
                    return None

                print(
                    f"An unexpected error occurred with OpenAI API call ({type(e).__name__}). Stopping retries for this call.")
                return None
        return None