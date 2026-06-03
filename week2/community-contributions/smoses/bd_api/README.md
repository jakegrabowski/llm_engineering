# bd_api Client

This library provides a synchronous and asynchronous Python client for the `bd_api`. It natively parses Server-Sent Events (SSE) and exposes Custom Mode functionality.

## Requirements

- Python 3.8+
- `httpx` (If not installed, run `pip install httpx`)

## Initialization

```python
from bd_api import BdApiClient, AsyncBdApiClient

# Synchronous client (with optional API key)
client = BdApiClient(base_url="http://localhost:3000", api_key="your-api-key-here")

# Asynchronous client (with optional API key)
async_client = AsyncBdApiClient(base_url="http://localhost:3000", api_key="your-api-key-here")
```

Both clients can be used with context managers (`with` and `async with`) to ensure the underlying HTTP connections are closed properly.

## Using Custom Mode

Custom mode allows you to pass a specific `model_id`, `system_instructions`, `user_message`, and `inference_config`.

### Synchronous

The client automatically aggregates the streamed SSE events from the API into a single `BdApiResponse` object.

```python
response = client.ask.create(
    question="What are the reporting obligations?",
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    system_instructions="Answer using the supplied context. Do not invent citations.",
    user_message="Question:\n{{question}}\n\nContext:\n{{context}}",
    inference_config={
        "maxTokens": 1024,
        "temperature": 0.2
    }
)

print("Answer:", response.text)
```

### Asynchronous

You can use the `AsyncBdApiClient` to perform requests without blocking the event loop.

```python
async def get_answer():
    response = await async_client.ask.create(
        question="What are the reporting obligations?",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_instructions="Answer using the supplied context. Do not invent citations.",
        user_message="Question:\n{{question}}\n\nContext:\n{{context}}"
    )
    print("Cost:", response.cost)
```

## Return Types

`.create()` returns a `BdApiResponse` object with the following attributes:
- `text`: (str) The aggregated generated answer.
- `sources`: (list) A list of sources retrieved from the knowledge base.
- `cost`: (dict) Cost information (inference, retrieval, total).
- `usage`: (dict) Token usage details.
- `metrics`: (dict) Generation metrics.
- `stop_reason`: (str) E.g., `end_turn`.
- `persisted_to_s3`: (bool) True if successfully persisted.
- `answer_id`: (str) The UUID of the request.

When calling `client.retrieve.create()`, it returns a `BdApiRetrieveResponse` object with the following attributes:
- `knowledge_base_id`: (str) The ID of the knowledge base queried.
- `raw_retrieval_response`: (dict) The raw AWS SDK retrieve response payload.
- `retrieved_items`: (list) A list of normalized sources retrieved.
- `prepared_prompt_input`: (dict) Contains variable names and fully prepared `promptVariables` as they would be sent to the prompt.
- `cost`: (dict) Estimated retrieval cost.
