"""Interface for interacting with OpenAI language model API."""

from typing import List
import time
import os
import json
from tqdm import tqdm
import concurrent.futures
import openai
from openai import AzureOpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError

delay_time = 0.5
decay_rate = 0.8
max_attempts = 10

def apply_arg_defaults(args: dict):
    assert args["max_tokens"]
    args["temperature"] = args.get("temperature", 0.0)
    args["n"] = args.get("n", 1)

class OpenAIClient:
    def __init__(self, use_azure_client: bool):
        if use_azure_client:
            openai.api_type = "azure"
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        else:
            openai.api_type = "openai"
            self.client = openai

    def get_batched_responses(self, prompts: List[str], model: str, batch_size: int, generation_args: dict,
                            system_message: str = None, histories: List[str] = None, show_progress: bool = False):
        apply_arg_defaults(generation_args)
        # Load model's response cache
        use_cache = histories is None
        cache_filename = f"openai_cache_{model}.json"
        if use_cache:
            if os.path.exists(cache_filename):
                with open(cache_filename) as cache_file:
                    cache: dict = json.load(cache_file)
            else:
                cache = {}
            top_key = f"#ARGS-{generation_args}#SYS-{system_message}#"
            sm_cache = cache.setdefault(top_key, {})
            uncached_prompts = list({prompt for prompt in prompts if prompt not in sm_cache})
        else:
            uncached_prompts = prompts
        print(f"{len(prompts)} prompts, sending {len(uncached_prompts)} new requests")

        # Batch parallel requests to API
        responses = []
        it = range(0, len(uncached_prompts), batch_size)
        if show_progress:
            it = tqdm(it)
        try:
            for batch_start_idx in it:
                batch = uncached_prompts[batch_start_idx : batch_start_idx + batch_size]
                histories_batch = histories[batch_start_idx : batch_start_idx + batch_size] if histories else None
                batch_responses = self._get_parallel_responses(batch, model, generation_args,
                                                        system_message=system_message, histories=histories_batch)
                if use_cache:
                    for prompt, response in zip(batch, batch_responses):
                        sm_cache[prompt] = response
                else:
                    responses.extend(batch_responses)
        finally:
            # Update model's response cache
            if use_cache:
                print(f"Saving response cache for {model}")
                with open(cache_filename, "w") as cache_file:
                    json.dump(cache, cache_file)

        # Return responses
        if use_cache:
            return [sm_cache[prompt] for prompt in prompts]
        return responses

    def _get_parallel_responses(self, prompts: List[str], model: str, generation_args: dict,
                            system_message: str = None, histories: List[dict] = None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            # Submit requests to threads
            futures = [
                executor.submit(self._get_responses, [prompt], model, generation_args,
                                system_message=system_message, histories=[histories[prompt_idx]] if histories else None)
                for prompt_idx, prompt in enumerate(prompts)
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            # Accumulate results
            results = [future.result()[0] for future in futures]
            return results

    def _get_responses(self, prompts: List[str], model: str, generation_args: dict,
                    system_message: str = None, histories: List[dict] = None, attempt: int = 1):
        global delay_time

        # Wait for rate limit
        time.sleep(delay_time)

        # Send request
        try:
            results = []
            for prompt_idx, prompt in enumerate(prompts):
                history = histories[prompt_idx] if histories else []
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message or "You are a helpful assistant."
                        },
                        *history,
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    **generation_args,
                    timeout=45
                )
                if generation_args["n"] == 1:
                    results.append(response.choices[0].message.content)
                else:
                    results.append([choice.message.content for choice in response.choices])
            delay_time = max(delay_time * decay_rate, 0.1)
        except (RateLimitError, APITimeoutError, APIError, APIConnectionError) as exc:
            print(openai.api_key, exc)
            delay_time = min(delay_time * 2, 30)
            if attempt >= max_attempts:
                print("Max attempts reached, prompt:")
                print(prompt)
                raise exc
            return self._get_responses(prompts, model, generation_args, system_message=system_message,
                                histories=histories, attempt=attempt + 1)
        except Exception as exc:
            print(exc)
            raise exc

        return results
