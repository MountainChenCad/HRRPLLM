# src/api_callers/google_caller.py
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions  # For more specific error handling
from .base_caller import BaseCaller


class GoogleCaller(BaseCaller):
    def __init__(self, model_name, api_key, google_api_endpoint=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_endpoint = google_api_endpoint

        try:
            genai.configure(
                api_key=self.api_key,
                transport="rest",  # As requested
                client_options={"api_endpoint": self.api_endpoint} if self.api_endpoint else None,
            )
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize Google GenAI client or model: {e}")
        # print(f"GoogleCaller initialized: Model={self.model_name}, Endpoint={self.api_endpoint}, Temp={self.temperature}")

    def get_completion(self, prompt_content):
        retries = 0
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens_completion,
            temperature=self.temperature,
            top_p=self.top_p
            # top_k can also be set if needed
        )
        # Define safety settings to be less restrictive if needed, or use defaults
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        while retries < self.max_retries:
            try:
                response = self.model.generate_content(
                    prompt_content,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                # Check for empty candidates or blocked prompts
                if not response.candidates:
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    print(f"Google API call blocked or returned no candidates. Reason: {block_reason}")
                    # If blocked due to safety, it might not be worth retrying with the same prompt.
                    # Depending on the reason, you might decide to return None immediately.
                    # For now, let's allow retry for other potential transient issues.
                    if block_reason != "SAFETY":  # Only retry if not a safety block
                        raise Exception(
                            f"Blocked by API or no candidates. Reason: {block_reason}")  # Force retry for non-safety blocks
                    return None  # Do not retry for safety blocks.

                return response.text.strip() if hasattr(response, 'text') else None
            # More specific Google API errors
            except (google_exceptions.DeadlineExceeded,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.ResourceExhausted,  # For quota/rate limits
                    google_exceptions.InternalServerError) as e:
                self._handle_error(e, retries)
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print("Max retries reached for Google API call.")
                    return None
            except Exception as e:  # Catch other google genai errors or general errors
                self._handle_error(e, retries)
                # If the error is due to invalid API key or specific client setup, retrying won't help.
                print(
                    f"An unexpected error occurred with Google API call: {type(e).__name__}. Stopping retries for this call.")
                return None
        return None