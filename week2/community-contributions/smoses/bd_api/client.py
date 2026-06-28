import json
import httpx
from typing import Optional, Dict, Any, Generator, AsyncGenerator
from .types import BdApiEvent, BdApiResponse, BdApiRetrieveResponse, RetrievalConfig, RetrievalMetadata
from .errors import BdApiRequestError, BdApiStreamError

class _RetrieveEndpoints:
    def __init__(self, client: httpx.Client, base_url: str):
        self._client = client
        self._base_url = base_url.rstrip("/")

    def create(
        self,
        question: str,
        retrieval: Optional[RetrievalConfig] = None
    ) -> BdApiRetrieveResponse:
        url = f"{self._base_url}/retrieve"
        payload = {"question": question}
        if retrieval:
            payload["retrieval"] = retrieval

        response = self._client.post(url, json=payload)
        if response.status_code >= 400:
            raise BdApiRequestError(status_code=response.status_code, response_body=response.text)

        data = response.json()
        retrieval_data = data.get("retrieval", {})

        return BdApiRetrieveResponse(
            knowledge_base_id=data.get("knowledgeBaseId"),
            retrieval=RetrievalMetadata(
                requested_mode=retrieval_data.get("requestedMode", ""),
                effective_mode=retrieval_data.get("effectiveMode", ""),
                fallback_used=retrieval_data.get("fallbackUsed", False),
                candidate_count=retrieval_data.get("candidateCount", 0),
                result_count=retrieval_data.get("resultCount", 0),
                initial_retrieved_count=retrieval_data.get("initialRetrievedCount", 0),
                final_retrieved_count=retrieval_data.get("finalRetrievedCount", 0),
                fallback_reason=retrieval_data.get("fallbackReason"),
                rerank_model_arn=retrieval_data.get("rerankModelArn"),
                deduplication=retrieval_data.get("deduplication")
            ),
            raw_retrieval_response=data.get("rawRetrievalResponse", {}),
            retrieved_items=data.get("retrievedItems", []),
            prepared_prompt_input=data.get("preparedPromptInput", {}),
            cost=data.get("cost", {}),
            raw_json=data
        )

class _AskEndpoints:
    def __init__(self, client: httpx.Client, base_url: str):
        self._client = client
        self._base_url = base_url.rstrip("/")

    def create(
        self,
        question: str,
        model_id: str,
        system_instructions: str,
        user_message: str,
        inference_config: Optional[Dict[str, Any]] = None,
        retrieval: Optional[RetrievalConfig] = None
    ) -> BdApiResponse:
        url = f"{self._base_url}/ask"
        payload: Dict[str, Any] = {
            "question": question,
            "model_id": model_id,
            "system_instructions": system_instructions,
            "user_message": user_message
        }

        if inference_config:
            payload["inference_config"] = inference_config
        if retrieval:
            payload["retrieval"] = retrieval

        return self._sync_request(url, payload)

    def _parse_sse(self, response: httpx.Response) -> Generator[BdApiEvent, None, None]:
        event_type = None
        data_lines = []
        for line in response.iter_lines():
            if not line:
                if data_lines:
                    data_str = "\n".join(data_lines)
                    yield BdApiEvent(type=event_type or "message", data=json.loads(data_str))
                    event_type = None
                    data_lines = []
                continue

            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())

    def _stream_request(self, url: str, payload: dict) -> Generator[BdApiEvent, None, None]:
        with self._client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                response.read()
                raise BdApiRequestError(status_code=response.status_code, response_body=response.text)
            yield from self._parse_sse(response)

    def _sync_request(self, url: str, payload: dict) -> BdApiResponse:
        text = ""
        sources = []
        cost = {}
        usage = {}
        metrics = {}
        stop_reason = None
        persisted_to_s3 = False
        answer_id = None
        retrieval_metadata = None

        for event in self._stream_request(url, payload):
            if event.type == "start":
                answer_id = event.data.get("answerId")
                retrieval_data = event.data.get("retrieval")
                if retrieval_data:
                    retrieval_metadata = RetrievalMetadata(
                        requested_mode=retrieval_data.get("requestedMode", ""),
                        effective_mode=retrieval_data.get("effectiveMode", ""),
                        fallback_used=retrieval_data.get("fallbackUsed", False),
                        candidate_count=retrieval_data.get("candidateCount", 0),
                        result_count=retrieval_data.get("resultCount", 0),
                        initial_retrieved_count=retrieval_data.get("initialRetrievedCount", 0),
                        final_retrieved_count=retrieval_data.get("finalRetrievedCount", 0),
                        fallback_reason=retrieval_data.get("fallbackReason"),
                        rerank_model_arn=retrieval_data.get("rerankModelArn"),
                        deduplication=retrieval_data.get("deduplication")
                    )
            elif event.type == "sources":
                sources = event.data.get("sources", [])
            elif event.type == "delta":
                text += event.data.get("text", "")
            elif event.type == "done":
                stop_reason = event.data.get("stopReason")
                usage = event.data.get("usage", {})
                metrics = event.data.get("metrics", {})
                cost = event.data.get("cost", {})
                persisted_to_s3 = event.data.get("persistedToS3", False)
            elif event.type == "error":
                raise BdApiStreamError(
                    message=event.data.get("message", event.data.get("error", "Unknown error")),
                    code=event.data.get("code"),
                    retryable=event.data.get("retryable", False)
                )

        return BdApiResponse(
            text=text, sources=sources, cost=cost, usage=usage,
            metrics=metrics, stop_reason=stop_reason,
            persisted_to_s3=persisted_to_s3, answer_id=answer_id,
            retrieval=retrieval_metadata
        )

class _AsyncRetrieveEndpoints:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def create(
        self,
        question: str,
        retrieval: Optional[RetrievalConfig] = None
    ) -> BdApiRetrieveResponse:
        url = f"{self._base_url}/retrieve"
        payload = {"question": question}
        if retrieval:
            payload["retrieval"] = retrieval

        response = await self._client.post(url, json=payload)
        if response.status_code >= 400:
            await response.aread()
            raise BdApiRequestError(status_code=response.status_code, response_body=response.text)

        data = response.json()
        retrieval_data = data.get("retrieval", {})

        return BdApiRetrieveResponse(
            knowledge_base_id=data.get("knowledgeBaseId"),
            retrieval=RetrievalMetadata(
                requested_mode=retrieval_data.get("requestedMode", ""),
                effective_mode=retrieval_data.get("effectiveMode", ""),
                fallback_used=retrieval_data.get("fallbackUsed", False),
                candidate_count=retrieval_data.get("candidateCount", 0),
                result_count=retrieval_data.get("resultCount", 0),
                initial_retrieved_count=retrieval_data.get("initialRetrievedCount", 0),
                final_retrieved_count=retrieval_data.get("finalRetrievedCount", 0),
                fallback_reason=retrieval_data.get("fallbackReason"),
                rerank_model_arn=retrieval_data.get("rerankModelArn"),
                deduplication=retrieval_data.get("deduplication")
            ),
            raw_retrieval_response=data.get("rawRetrievalResponse", {}),
            retrieved_items=data.get("retrievedItems", []),
            prepared_prompt_input=data.get("preparedPromptInput", {}),
            cost=data.get("cost", {}),
            raw_json=data
        )

class _AsyncAskEndpoints:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def create(
        self,
        question: str,
        model_id: str,
        system_instructions: str,
        user_message: str,
        inference_config: Optional[Dict[str, Any]] = None,
        retrieval: Optional[RetrievalConfig] = None
    ) -> BdApiResponse:
        url = f"{self._base_url}/ask"
        payload: Dict[str, Any] = {
            "question": question,
            "model_id": model_id,
            "system_instructions": system_instructions,
            "user_message": user_message
        }

        if inference_config:
            payload["inference_config"] = inference_config
        if retrieval:
            payload["retrieval"] = retrieval

        return await self._sync_request(url, payload)

    async def _parse_sse(self, response: httpx.Response) -> AsyncGenerator[BdApiEvent, None]:
        event_type = None
        data_lines = []
        async for line in response.aiter_lines():
            if not line:
                if data_lines:
                    data_str = "\n".join(data_lines)
                    yield BdApiEvent(type=event_type or "message", data=json.loads(data_str))
                    event_type = None
                    data_lines = []
                continue

            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())

    async def _stream_request(self, url: str, payload: dict) -> AsyncGenerator[BdApiEvent, None]:
        async with self._client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                await response.aread()
                raise BdApiRequestError(status_code=response.status_code, response_body=response.text)
            async for event in self._parse_sse(response):
                yield event

    async def _sync_request(self, url: str, payload: dict) -> BdApiResponse:
        text = ""
        sources = []
        cost = {}
        usage = {}
        metrics = {}
        stop_reason = None
        persisted_to_s3 = False
        answer_id = None
        retrieval_metadata = None

        async for event in self._stream_request(url, payload):
            if event.type == "start":
                answer_id = event.data.get("answerId")
                retrieval_data = event.data.get("retrieval")
                if retrieval_data:
                    retrieval_metadata = RetrievalMetadata(
                        requested_mode=retrieval_data.get("requestedMode", ""),
                        effective_mode=retrieval_data.get("effectiveMode", ""),
                        fallback_used=retrieval_data.get("fallbackUsed", False),
                        candidate_count=retrieval_data.get("candidateCount", 0),
                        result_count=retrieval_data.get("resultCount", 0),
                        initial_retrieved_count=retrieval_data.get("initialRetrievedCount", 0),
                        final_retrieved_count=retrieval_data.get("finalRetrievedCount", 0),
                        fallback_reason=retrieval_data.get("fallbackReason"),
                        rerank_model_arn=retrieval_data.get("rerankModelArn"),
                        deduplication=retrieval_data.get("deduplication")
                    )
            elif event.type == "sources":
                sources = event.data.get("sources", [])
            elif event.type == "delta":
                text += event.data.get("text", "")
            elif event.type == "done":
                stop_reason = event.data.get("stopReason")
                usage = event.data.get("usage", {})
                metrics = event.data.get("metrics", {})
                cost = event.data.get("cost", {})
                persisted_to_s3 = event.data.get("persistedToS3", False)
            elif event.type == "error":
                raise BdApiStreamError(
                    message=event.data.get("message", event.data.get("error", "Unknown error")),
                    code=event.data.get("code"),
                    retryable=event.data.get("retryable", False)
                )

        return BdApiResponse(
            text=text, sources=sources, cost=cost, usage=usage,
            metrics=metrics, stop_reason=stop_reason,
            persisted_to_s3=persisted_to_s3, answer_id=answer_id,
            retrieval=retrieval_metadata
        )

class BdApiClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        self._http_client = httpx.Client(headers=headers, timeout=60.0)
        self.ask = _AskEndpoints(self._http_client, self.base_url)
        self.retrieve = _RetrieveEndpoints(self._http_client, self.base_url)

    def close(self):
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class AsyncBdApiClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        self._http_client = httpx.AsyncClient(headers=headers, timeout=60.0)
        self.ask = _AsyncAskEndpoints(self._http_client, self.base_url)
        self.retrieve = _AsyncRetrieveEndpoints(self._http_client, self.base_url)

    async def close(self):
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
