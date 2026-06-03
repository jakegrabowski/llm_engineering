from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

@dataclass
class BdApiRetrieveResponse:
    knowledge_base_id: str
    raw_retrieval_response: Dict[str, Any] = field(default_factory=dict)
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    prepared_prompt_input: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    raw_json: Dict[str, Any] = field(default_factory=dict)
