from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

class RetrievalConfig(TypedDict, total=False):
    """Retrieval configuration for Knowledge Base queries.

    Attributes:
        knowledge_base_id: Bedrock Knowledge Base ID (required)
        retrieval_mode: Retrieval strategy - 'standard' or 'rerank' (optional)
        candidate_count: Initial retrieval count from KB, 1-100 (optional)
        result_count: Final result count returned, 1-100, must be <= candidate_count (optional)
    """
    knowledge_base_id: str
    retrieval_mode: Literal["standard", "rerank"]
    candidate_count: int
    result_count: int

@dataclass
class RetrievalMetadata:
    """Detailed retrieval behavior metadata returned by the API.

    This shows what actually happened during retrieval, including whether
    reranking was successful or fell back to standard mode.
    """
    requested_mode: str
    effective_mode: str
    fallback_used: bool
    candidate_count: int
    result_count: int
    initial_retrieved_count: int
    final_retrieved_count: int
    fallback_reason: Optional[str] = None
    rerank_model_arn: Optional[str] = None
    deduplication: Optional[Dict[str, Any]] = None

@dataclass
class BdApiEvent:
    type: str
    data: Dict[str, Any]

@dataclass
class BdApiResponse:
    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    persisted_to_s3: bool = False
    answer_id: Optional[str] = None
    retrieval: Optional[RetrievalMetadata] = None

@dataclass
class BdApiRetrieveResponse:
    knowledge_base_id: str
    retrieval: RetrievalMetadata
    raw_retrieval_response: Dict[str, Any] = field(default_factory=dict)
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    prepared_prompt_input: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    raw_json: Dict[str, Any] = field(default_factory=dict)
