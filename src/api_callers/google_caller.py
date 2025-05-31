# src/api_callers/google_caller.py
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from .base_caller import BaseCaller


class GoogleCaller(BaseCaller):
    def __init__(self, model_name, api_key, google_api_endpoint=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_endpoint = google_api_endpoint

        try:
            genai.configure(
                api_key=self.api_key,
                transport="rest",
                client_options={"api_endpoint": self.api_endpoint} if self.api_endpoint else None,
            )
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize Google GenAI client or model: {e}")

    def get_completion(self, prompt_content):
        retries = 0
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens_completion,
            temperature=self.temperature,
            top_p=self.top_p
        )

        # More permissive safety settings for technical content
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

                # Handle different finish reasons
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)

                    # Debug log
                    if finish_reason:
                        print(f"Finish reason: {finish_reason}")

                    # Check finish reason
                    # 1 = STOP (normal completion)
                    # 2 = SAFETY (blocked by safety filters)
                    # 3 = MAX_TOKENS
                    # 4 = RECITATION
                    # 5 = OTHER

                    if finish_reason == 2:  # SAFETY
                        print(f"Content blocked by safety filters for model {self.model_name}")
                        # Try to get more details
                        if hasattr(response, 'prompt_feedback'):
                            print(f"Prompt feedback: {response.prompt_feedback}")

                        # For technical content like radar data, safety blocks are often false positives
                        # Try with a modified prompt
                        if retries == 0:
                            # Add a prefix to clarify this is technical content
                            modified_prompt = (
                                    "This is a technical analysis task for radar signal processing research. "
                                    "The following content contains technical data only:\n\n" + prompt_content
                            )
                            retries += 1
                            print("Retrying with clarified technical context...")
                            # Recursive call with modified prompt
                            response = self.model.generate_content(
                                modified_prompt,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                            )
                            # Continue to process this response
                        else:
                            return None

                    # Try to get content from the candidate
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts = candidate.content.parts
                        if parts and len(parts) > 0:
                            # Get text from first part
                            text = parts[0].text if hasattr(parts[0], 'text') else None
                            if text:
                                return text.strip()

                    # If no content but finish_reason is STOP, it might be empty response
                    if finish_reason == 1:  # STOP
                        print(f"Model returned empty response with STOP finish reason")
                        return None

                # If response.text accessor doesn't work, try manual extraction
                # This handles the "requires valid Part" error
                if hasattr(response, '_result') and hasattr(response._result, 'candidates'):
                    for cand in response._result.candidates:
                        if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                            for part in cand.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    return part.text.strip()

                # No valid content found
                print(f"No valid content in response from {self.model_name}")
                return None

            except ValueError as e:
                # Handle the specific "response.text" error
                if "response.text" in str(e) and "finish_reason" in str(e):
                    print(f"Response blocked or invalid. Details: {e}")

                    # Try to handle based on what we know about the error
                    if "finish_reason] is 2" in str(e):
                        print("Content was blocked by safety filters")
                        if retries == 0:
                            retries += 1
                            print("Retrying with modified approach...")
                            continue

                    return None
                else:
                    # Other ValueError, re-raise
                    raise

            except (google_exceptions.ResourceExhausted,
                    google_exceptions.DeadlineExceeded,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.InternalServerError) as e:
                self._handle_error(e, retries)
                retries += 1
                if retries < self.max_retries:
                    wait_time = self.api_retry_delay * (2 ** (retries - 1))  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached for Google API call.")
                    return None

            except Exception as e:
                self._handle_error(e, retries)

                # Check for specific errors
                error_msg = str(e).lower()
                if "not found" in error_msg:
                    print(f"Model '{self.model_name}' not found. Please check the model name.")
                    print("Available Gemini models: gemini-1.5-pro, gemini-1.5-flash, gemini-pro")
                    return None

                # For other errors, retry might help
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print(f"Unexpected error with Google API: {type(e).__name__}: {e}")
                    return None

        return None