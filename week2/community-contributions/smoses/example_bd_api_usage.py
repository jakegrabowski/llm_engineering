"""
MIGRATION NOTE - bd_api Library Updated

BREAKING CHANGE:
    OLD API (no longer works):
        client.retrieve.create(question="...", knowledge_base_id="KB123")
        client.ask.create(..., knowledge_base_id="KB123")

    NEW API (required):
        # Uses server default KB ID
        client.retrieve.create(question="...")

        # Or override KB ID with retrieval parameter
        client.retrieve.create(question="...", retrieval={"knowledge_base_id": "KB123"})

NEW FEATURES - Advanced retrieval options:
    retrieval={
        "knowledge_base_id": "KB123",    # Optional: override default KB
        "retrieval_mode": "rerank",      # Optional: "standard" or "rerank"
        "candidate_count": 20,           # Optional: 1-100 initial results
        "result_count": 5                # Optional: 1-100 final results
    }

All examples below work unchanged - they use server defaults.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from bd_api import BdApiClient, AsyncBdApiClient
from bd_api.errors import BdApiRequestError, BdApiStreamError

# Load variables from .env file
load_dotenv(override=True)

# Fetch the API endpoint (supports both http and https)
BD_API_BASE_URL = os.getenv("BD_API_BASE_URL")
if not BD_API_BASE_URL:
    raise ValueError("BD_API_BASE_URL environment variable is missing. Please set it in your .env file.")

# Fetch the API Key
BD_API_KEY = os.getenv("BD_API_KEY")
if not BD_API_KEY:
    print("Warning: BD_API_KEY environment variable is missing. Requests may fail if the server requires an API key.")

print(f"Using API Base URL: {BD_API_BASE_URL}")

# Shared Request Parameters
QUESTION = "How to organize medical practice in 3 paragraphs?"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM_INSTRUCTIONS = "Answer question using the supplied context. Do not invent citations."
USER_MESSAGE = "Question:\n```{{question}}```\n\nContext:\n```{{context}}```"
INFERENCE_CONFIG = {
    "maxTokens": 1024,
    "temperature": 0.2
}


# ==========================================
# Example 1: Custom Mode (Sync)
# ==========================================
def example_1_custom_mode_sync():
    print("\n--- Custom Mode (Sync) ---")
    with BdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY) as client:
        try:
            response = client.ask.create(
                question=QUESTION,
                model_id=MODEL_ID,
                system_instructions=SYSTEM_INSTRUCTIONS,
                user_message=USER_MESSAGE,
                inference_config=INFERENCE_CONFIG
            )

            print("Final Answer:", response.text)
            print("Sources used:", response.sources)
            print("Total Cost:", response.cost.get("total", {}).get("dollars"))

        except BdApiRequestError as e:
            print(f"Request failed before streaming. HTTP {e.status_code}")
            print(f"Server response: {e.response_body}")
        except BdApiStreamError as e:
            print(f"Streaming failed midway! Message: {e.message}")
            print(f"Code: {e.code}, Retryable: {e.retryable}")
        except Exception as e:
            print(f"Exception occurred. Ensure the server is running on {BD_API_BASE_URL}.", e)


# ==========================================
# Example 2: Retrieval Mode - Raw JSON (Sync)
# ==========================================
def example_2_retrieval_raw_sync():
    print("\n--- Retrieval Mode - Raw JSON (Sync) ---")
    with BdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY) as client:
        try:
            retrieve_response = client.retrieve.create(question=QUESTION)

            print("Complete Raw API Response:")
            raw_json = json.dumps(retrieve_response.raw_json, indent=2)
            print(raw_json)

            print("\nNormalized Retrieved Items JSON (first item):")
            if retrieve_response.retrieved_items:
                print(json.dumps(retrieve_response.retrieved_items[0], indent=2))

            print("\nPrepared Prompt Input JSON:")
            prepared_json = json.dumps(retrieve_response.prepared_prompt_input, indent=2)
            print(prepared_json)

        except BdApiRequestError as e:
            print(f"Retrieve Request failed. HTTP {e.status_code}")
        except Exception as e:
            print("Exception occurred.", e)


# ==========================================
# Example 3: Retrieval Mode - Formatted (Sync)
# ==========================================
def example_3_retrieval_formatted_sync():
    print("\n--- Retrieval Mode - Formatted (Sync) ---")
    with BdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY) as client:
        try:
            retrieve_response = client.retrieve.create(question=QUESTION)

            print(f"Total retrieved items: {len(retrieve_response.retrieved_items)}")

            for item in retrieve_response.retrieved_items:
                print(f"\n- Item {item.get('index')}: Score {item.get('score')}")
                print(f"  Content: {item.get('text', '')}")

            prompt_vars = retrieve_response.prepared_prompt_input.get("promptVariables", {})
            print(f"\nPrepared Context Variable:\n{prompt_vars.get('context', '')}\n")
            print(f"Retrieval Estimated Cost: {retrieve_response.cost.get('retrievalEstimated', {}).get('dollars')} dollars")

        except BdApiRequestError as e:
            print(f"Retrieve Request failed. HTTP {e.status_code}")
        except Exception as e:
            print("Exception occurred.", e)


# ==========================================
# Example 4: Asynchronous Usage (Ask & Retrieve)
# ==========================================
async def example_4_async_usage():
    print("\n--- Asynchronous Usage (Async) ---")
    async with AsyncBdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY) as async_client:
        try:
            response = await async_client.ask.create(
                question=QUESTION,
                model_id=MODEL_ID,
                system_instructions=SYSTEM_INSTRUCTIONS,
                user_message=USER_MESSAGE,
                inference_config=INFERENCE_CONFIG
            )
            print("Async Final Answer:", response.text)
            print("Async Answer ID:", response.answer_id)
        except BdApiRequestError as e:
            print(f"Async Request failed before streaming. HTTP {e.status_code}")
        except BdApiStreamError as e:
            print(f"Async stream error: {e.message}")
        except Exception as e:
            print("Exception occurred in async call.", e)

        print("\n--- Asynchronous Retrieval Mode ---")
        try:
            retrieve_response = await async_client.retrieve.create(question=QUESTION)
            print("Async Retrieved Items:", len(retrieve_response.retrieved_items))
        except BdApiRequestError as e:
            print(f"Async Request failed. HTTP {e.status_code}")
        except Exception as e:
            print("Exception occurred in async retrieve.", e)


# ==========================================
# Example 5: NEW - Advanced Retrieval with Reranking
# ==========================================
def example_5_advanced_retrieval():
    """Demonstrates new retrieval features: reranking, custom counts, and metadata."""
    print("\n--- Advanced Retrieval with Reranking ---")
    with BdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY) as client:
        try:
            # Use reranking for better relevance
            retrieve_response = client.retrieve.create(
                question=QUESTION,
                retrieval={
                    "retrieval_mode": "rerank",
                    "candidate_count": 20,
                    "result_count": 5
                }
            )

            print("Complete Raw API Response:")
            raw_json = json.dumps(retrieve_response.raw_json, indent=2)
            print(raw_json)

            print("\nNormalized Retrieved Items JSON (first item):")
            if retrieve_response.retrieved_items:
                print(json.dumps(retrieve_response.retrieved_items[0], indent=2))

            print("\nPrepared Prompt Input JSON:")
            prepared_json = json.dumps(retrieve_response.prepared_prompt_input, indent=2)
            print(prepared_json)

            # NEW: Access retrieval metadata
            print("\n--- NEW: Retrieval Metadata ---")
            print(f"Requested mode: {retrieve_response.retrieval.requested_mode}")
            print(f"Effective mode: {retrieve_response.retrieval.effective_mode}")
            print(f"Fallback used: {retrieve_response.retrieval.fallback_used}")
            if retrieve_response.retrieval.fallback_used:
                print(f"Fallback reason: {retrieve_response.retrieval.fallback_reason}")
            print(f"Candidates retrieved: {retrieve_response.retrieval.initial_retrieved_count}")
            print(f"Final results: {retrieve_response.retrieval.final_retrieved_count}")

            # Cost breakdown
            retrieval_cost = retrieve_response.cost.get('retrievalEstimated', {}).get('dollars', 0)
            rerank_cost = retrieve_response.cost.get('rerankingEstimated', {}).get('dollars', 0)
            print(f"\nRetrieval Cost: ${retrieval_cost:.6f}")
            print(f"Reranking Cost: ${rerank_cost:.6f}")

        except BdApiRequestError as e:
            print(f"Retrieve Request failed. HTTP {e.status_code}")
        except Exception as e:
            print("Exception occurred.", e)


if __name__ == "__main__":
    # -------------------------------------------------------------
    # Uncomment the specific example(s) below that you want to run!
    # -------------------------------------------------------------

    # example_1_custom_mode_sync()

    # example_2_retrieval_raw_sync()

    # example_3_retrieval_formatted_sync()

    # asyncio.run(example_4_async_usage())

    # NEW: Advanced retrieval with reranking
    example_5_advanced_retrieval()
