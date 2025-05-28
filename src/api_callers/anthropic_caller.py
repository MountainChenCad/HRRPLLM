# src/api_callers/anthropic_caller.py
import time
from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError, APITimeoutError  # Anthropic errors
from .base_caller import BaseCaller


class AnthropicCaller(BaseCaller):
    def __init__(self, model_name, api_key, base_url=None, **kwargs):  # base_url for proxy
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = base_url

        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url

        try:
            self.client = Anthropic(**client_args)
        except Exception as e:
            raise ValueError(f"Failed to initialize Anthropic client: {e}")
        # print(f"AnthropicCaller initialized: Model={self.model_name}, BaseURL={self.base_url}, Temp={self.temperature}")

    def get_completion(self, prompt_content):
        # Anthropic's API might benefit from a system prompt for overall instructions.
        # For now, we assume prompt_content contains everything the user role should say.
        # If prompt_constructor_sc.py can split context_header into a system prompt,
        # that would be ideal.
        # system_prompt = "You are a radar target recognition expert..." (from context_header)
        # user_query = "Analyze the following scattering centers..." (the rest of the prompt)

        retries = 0
        while retries < self.max_retries:
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens_completion,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_content,  # Pass the full prompt content here
                        }
                    ],
                    temperature=self.temperature,
                    # top_p=self.top_p, # Anthropic also supports top_p
                    # system=system_prompt # If you decide to use a system prompt
                )

                if message.content and isinstance(message.content, list) and \
                        len(message.content) > 0 and hasattr(message.content[0], 'text'):
                    return message.content[0].text.strip()
                else:
                    print(f"Anthropic API returned unexpected content structure: {message.content}")
                    return None

            except (APIConnectionError, RateLimitError, APIStatusError, APITimeoutError) as e:
                self._handle_error(e, retries)
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print("Max retries reached for Anthropic API call.")
                    return None
            except Exception as e:
                self._handle_error(e, retries)
                print("An unexpected error occurred with Anthropic API call. Stopping retries for this call.")
                return None
        return None