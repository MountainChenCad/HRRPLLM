# src/api_callers/base_caller.py
from abc import ABC, abstractmethod

class BaseCaller(ABC):
    def __init__(self, model_name, api_key, temperature, top_p, max_tokens_completion,
                 frequency_penalty, presence_penalty, api_retry_delay, max_retries, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens_completion = max_tokens_completion
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_retry_delay = api_retry_delay
        self.max_retries = max_retries
        # Store any additional kwargs that might be specific to subclasses
        self.kwargs = kwargs

        if not self.api_key or "YOUR_API_KEY" in self.api_key: # Basic check
            raise ValueError(f"API key for {self.__class__.__name__} is not configured or is a placeholder.")

    @abstractmethod
    def get_completion(self, prompt_content):
        pass

    def _handle_error(self, e, current_retry):
        error_message = f"LLM API Error ({self.__class__.__name__}, model: {self.model_name}, attempt {current_retry+1}/{self.max_retries}): {str(e)}"
        # Attempt to get more details from the exception if available (specific to SDKs)
        # For example, OpenAI's exception might have a 'response' attribute
        try:
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'):
                error_message += f" | API Error Details: {e.response.json()}"
            elif hasattr(e, 'message'): # For Anthropic or other SDKs
                 error_message += f" | SDK Message: {e.message}"

        except Exception: # noqa
            pass # Avoid errors within error handling
        print(error_message)