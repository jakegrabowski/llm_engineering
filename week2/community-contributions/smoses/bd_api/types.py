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
