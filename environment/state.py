from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class RelationshipState:
    david_chen: float = 0.75
    sundar: float = 0.80
    priya: float = 0.90
    rahul: float = 0.88
    arjun: float = 0.82

    def clamped(self) -> "RelationshipState":
        return RelationshipState(
            david_chen=clamp01(self.david_chen),
            sundar=clamp01(self.sundar),
            priya=clamp01(self.priya),
            rahul=clamp01(self.rahul),
            arjun=clamp01(self.arjun),
        )


@dataclass
class EpisodeState:
    day: int = 0
    energy: float = 0.87
    sprint_health: float = 0.80
    staging_fixed: bool = False
    leave_status: str = "pending"
    appraisal_done: bool = False
    oncall_accepted: bool = False
    relationships: RelationshipState = field(default_factory=RelationshipState)
    decisions: List[dict] = field(default_factory=list)
    history: List[str] = field(default_factory=list)

    def normalized(self) -> "EpisodeState":
        return replace(
            self,
            energy=clamp01(self.energy),
            sprint_health=clamp01(self.sprint_health),
            relationships=self.relationships.clamped(),
        )

    def to_observation(self) -> Dict:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        return {
            "day": days[self.day],
            "energy_pct": round(self.energy * 100),
            "sprint_health_pct": round(self.sprint_health * 100),
            "staging_fixed": self.staging_fixed,
            "leave_status": self.leave_status,
            "appraisal_done": self.appraisal_done,
            "relationships": asdict(self.relationships),
            "history": list(self.history),
        }
