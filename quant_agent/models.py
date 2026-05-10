from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HypothesisRecord:
    hypothesis: str
    reason: str
    action: str | None = None
    concise_reason: str | None = None
    concise_observation: str | None = None
    concise_justification: str | None = None
    concise_knowledge: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackRecord:
    observations: str
    hypothesis_evaluation: str
    decision: bool
    new_hypothesis: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceRecord:
    hist: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TraceRecord":
        if not payload:
            return cls()
        return cls(hist=list(payload.get("hist", [])))

    def to_dict(self) -> dict[str, Any]:
        return {"hist": self.hist}
