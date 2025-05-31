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

        # Determine correct max_tokens parameter name
        max_tokens_param_name = "max_tokens"  # Default

        # Models that use reasoning tokens (o1, o3, o4 series)
        reasoning_models = ["o1", "o3", "o4"]
        uses_reasoning = False

        # Special models that need v1/responses endpoint
        response_endpoint_models = ["o1-pro"]
        uses_response_endpoint = False

        # Check model type
        model_lower = self.model_name.lower()

        # Check if it's a response endpoint model
        for special_model in response_endpoint_models:
            if special_model in model_lower:
                uses_response_endpoint = True
                print(f"  Note: Model {self.model_name} requires special API endpoint")
                break

        # Check if current model needs max_completion_tokens
        for model_prefix in reasoning_models:
            if model_prefix in model_lower:
                max_tokens_param_name = "max_completion_tokens"
                uses_reasoning = True
                print(f"  Note: Model {self.model_name} uses reasoning. Using '{max_tokens_param_name}'")
                break

        # For reasoning models, we need MUCH more tokens
        effective_max_tokens = self.max_tokens_completion
        if uses_reasoning:
            # Reasoning models need extra tokens for thinking
            effective_max_tokens = max(2000, self.max_tokens_completion * 8)
            print(f"  Increased max_completion_tokens to {effective_max_tokens} for reasoning model")

        while retries < self.max_retries:
            try:
                # Handle special response endpoint models
                if uses_response_endpoint:
                    # For o1-pro models, we need to use a different approach
                    # Try using the completions API instead of chat completions
                    print(f"  Using completions API for {self.model_name}")

                    # Convert chat format to completion format
                    # Add a simple instruction format
                    formatted_prompt = f"Instructions: You are a helpful assistant.\n\nUser: {prompt_content}\n\nAssistant:"

                    try:
                        # Try the older completions API
                        response = self.client.completions.create(
                            model=self.model_name,
                            prompt=formatted_prompt,
                            max_tokens=effective_max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            frequency_penalty=self.frequency_penalty,
                            presence_penalty=self.presence_penalty
                        )

                        if response.choices and len(response.choices) > 0:
                            content = response.choices[0].text
                            if content:
                                return content.strip()
                            else:
                                print(f"Warning: Empty content from {self.model_name}")
                                return None
                        else:
                            print(f"Warning: No choices in completions response")
                            return None

                    except Exception as completion_error:
                        # If completions API also fails, log and skip
                        print(f"  Completions API also failed: {completion_error}")
                        print(f"  Model {self.model_name} may not be accessible via this proxy")
                        return None

                # Regular chat completions API
                else:
                    api_params = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt_content}],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        max_tokens_param_name: effective_max_tokens,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty,
                        "stream": False
                    }

                    # For o1/o3/o4 models, we might need special handling
                    if uses_reasoning:
                        # These models might need specific parameters
                        # Reduce temperature for more consistent output
                        api_params["temperature"] = min(0.3, self.temperature)

                    # Special handling for Qwen models
                    if "qwen" in model_lower:
                        if "extra_body" not in api_params:
                            api_params["extra_body"] = {}
                        api_params["extra_body"]["enable_thinking"] = False

                    # Make the API call
                    response = self.client.chat.completions.create(**api_params)

                    # Handle response
                    if response.choices and len(response.choices) > 0:
                        choice = response.choices[0]

                        # Debug info for reasoning models
                        if uses_reasoning and hasattr(response, 'usage'):
                            reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
                            if reasoning_tokens > 0:
                                print(f"  Reasoning tokens used: {reasoning_tokens}")

                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            content = choice.message.content

                            if content:
                                # For reasoning models, the content might have special formatting
                                # Strip any potential thinking markers
                                if uses_reasoning:
                                    # Remove common thinking patterns if present
                                    content = content.strip()
                                    # Some models might wrap the answer
                                    if content.startswith("<answer>") and content.endswith("</answer>"):
                                        content = content[8:-9].strip()

                                return content
                            else:
                                print(f"Warning: Empty content from {self.model_name}")
                                print(f"  Finish reason: {getattr(choice, 'finish_reason', 'unknown')}")
                                print(f"  Usage: {getattr(response, 'usage', 'N/A')}")

                                # For reasoning models with empty content, it might be a token limit issue
                                if uses_reasoning and getattr(choice, 'finish_reason', '') == 'length':
                                    print(f"  Token limit reached. Consider increasing max_completion_tokens further.")
                                    # Retry with even more tokens
                                    if retries == 0 and effective_max_tokens < 4000:
                                        effective_max_tokens = 4000
                                        print(f"  Retrying with {effective_max_tokens} tokens...")
                                        continue

                                return None
                        else:
                            print(f"Warning: No message.content in response")
                            return None
                    else:
                        print(f"Warning: No choices in response")
                        return None

            except (APIConnectionError, RateLimitError, APIStatusError, APITimeoutError) as e:
                self._handle_error(e, retries)

                # Check if it's the specific endpoint error
                error_str = str(e).lower()
                if "v1/responses" in error_str or "not in v1/chat/completions" in error_str:
                    print(f"  Model {self.model_name} requires different API endpoint")
                    # Add to response endpoint models list for next retry
                    if self.model_name not in response_endpoint_models:
                        response_endpoint_models.append(self.model_name)
                        uses_response_endpoint = True
                        print(f"  Added {self.model_name} to special endpoint list, retrying...")
                        continue

                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print("Max retries reached for OpenAI API call.")
                    return None

            except Exception as e:
                self._handle_error(e, retries)

                error_str = str(e).lower()
                if "maximum context length" in error_str:
                    print(f"  Context length exceeded. Try reducing prompt or max_tokens.")
                    return None

                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    return None

        return None