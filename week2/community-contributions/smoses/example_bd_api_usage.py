import os
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


def run_sync_examples():
    # Initialize the synchronous client
    client = BdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY)

    # ==========================================
    # Example 1: Non-streaming Custom Mode (Sync)
    # ==========================================
    print("\n--- Non-streaming Custom Mode (Sync) ---")
    try:
        response = client.ask.create(
            question=QUESTION,
            model_id=MODEL_ID,
            system_instructions=SYSTEM_INSTRUCTIONS,
            user_message=USER_MESSAGE,
            inference_config=INFERENCE_CONFIG,
            stream=False
        )

        # The non-streaming response collects all parts of the SSE stream
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
    # Example 2: Streaming Custom Mode (Sync)
    # ==========================================
    print("\n--- Streaming Custom Mode (Sync) ---")
    try:
        stream = client.ask.create(
            question=QUESTION,
            model_id=MODEL_ID,
            system_instructions=SYSTEM_INSTRUCTIONS,
            user_message=USER_MESSAGE,
            inference_config=INFERENCE_CONFIG,
            stream=True
        )

        for event in stream:
            if event.type == "start":
                print(f"Stream started. Answer ID: {event.data.get('answerId')}")
            elif event.type == "sources":
                print(f"Found {len(event.data.get('sources', []))} sources.")
            elif event.type == "delta":
                print(event.data.get("text", ""), end="", flush=True)
            elif event.type == "done":
                print(f"\nStream complete. Stop reason: {event.data.get('stopReason')}")
            elif event.type == "error":
                print(f"\nError encountered: {event.data.get('message')}")

    except BdApiRequestError as e:
        print(f"Request failed before streaming. HTTP {e.status_code}")
    except Exception as e:
        print("Exception occurred.", e)


# ==========================================
# Example 3: Asynchronous Usage
# ==========================================
async def run_async_example():
    print("\n--- Asynchronous Usage (Async) ---")
    # Recommended to use via 'async with' context manager
    async with AsyncBdApiClient(base_url=BD_API_BASE_URL, api_key=BD_API_KEY) as async_client:
        try:
            response = await async_client.ask.create(
                question=QUESTION,
                model_id=MODEL_ID,
                system_instructions=SYSTEM_INSTRUCTIONS,
                user_message=USER_MESSAGE,
                inference_config=INFERENCE_CONFIG,
                stream=False
            )
            print("Async Final Answer:", response.text)
            print("Async Answer ID:", response.answer_id)
        except BdApiRequestError as e:
            print(f"Async Request failed before streaming. HTTP {e.status_code}")
        except BdApiStreamError as e:
            print(f"Async stream error: {e.message}")
        except Exception as e:
            print("Exception occurred in async call.", e)


if __name__ == "__main__":
    # Run the synchronous examples
    run_sync_examples()

    # Run the asynchronous example (requires asyncio.run for normal python execution)
    asyncio.run(run_async_example())
