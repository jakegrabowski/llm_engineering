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

### Synchronous (Non-Streaming)

When `stream=False`, the client automatically aggregates the streamed SSE events into a single `BdApiResponse` object.

```python
response = client.ask.create(
    question="What are the reporting obligations?",
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    system_instructions="Answer using the supplied context. Do not invent citations.",
    user_message="Question:\n{{question}}\n\nContext:\n{{context}}",
    inference_config={
        "maxTokens": 1024,
        "temperature": 0.2
    },
    stream=False
)

print("Answer:", response.text)
```

### Synchronous (Streaming)

When `stream=True`, the client returns a generator of `BdApiEvent` objects.

```python
stream = client.ask.create(
    question="What are the reporting obligations?",
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    system_instructions="Answer using the supplied context. Do not invent citations.",
    user_message="Question:\n{{question}}\n\nContext:\n{{context}}",
    stream=True
)

for event in stream:
    if event.type == "start":
        print(f"Answer ID: {event.data.get('answerId')}")
    elif event.type == "delta":
        print(event.data.get("text", ""), end="")
    elif event.type == "done":
        print(f"\nCost: {event.data.get('cost')}")
```

### Asynchronous

You can use the `AsyncBdApiClient` to perform requests without blocking the event loop.

```python
async def get_answer():
    response = await async_client.ask.create(
        question="What are the reporting obligations?",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_instructions="Answer using the supplied context. Do not invent citations.",
        user_message="Question:\n{{question}}\n\nContext:\n{{context}}",
        stream=False
    )
    print("Answer:", response.text)
    print("Sources:", response.sources)
```

## Using Retrieval Mode

Retrieval mode (`POST /retrieve`) exposes the Knowledge Base retrieval step directly without generating an answer via an LLM. It's useful for previewing how a prompt is going to be prepared.

### Synchronous Retrieval

```python
response = client.retrieve.create(
    question="What are the reporting obligations?"
)

print("Knowledge Base ID:", response.knowledge_base_id)
print("Retrieved Items:", len(response.retrieved_items))
print("Prepared Prompt variables:", response.prepared_prompt_input.get("promptVariables"))
```

### Asynchronous Retrieval

```python
async def get_retrieval():
    response = await async_client.retrieve.create(
        question="What are the reporting obligations?"
    )
    print("Cost:", response.cost)
```

## Return Types

When `stream=False`, `.create()` returns a `BdApiResponse` object with the following attributes:
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
