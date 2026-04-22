"""Autonomy taxonomy: mapping between Int 0-100 scale and 6 marketing phases.

The canonical representation of autonomy is `Int 0-100`. The 6 phases
(Foundation / Assistance / Automation / Optimization / Intelligence / Autonomy)
are UI-only projections of the integer.

All functions here are pure — no DB, no IO — so Community can use them
without any Business dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


class AutonomyPhase:
    """Phase string constants (do not use Enum to keep this pure and simple)."""

    FOUNDATION = "foundation"
    ASSISTANCE = "assistance"
    AUTOMATION = "automation"
    OPTIMIZATION = "optimization"
    INTELLIGENCE = "intelligence"
    AUTONOMY = "autonomy"


@dataclass(frozen=True)
class PhaseDescriptor:
    key: str
    label: str
    tagline: str
    min_level: int
    max_level: int
    midpoint: int


# Ordered list of phases — the band edges are inclusive: [min_level, max_level].
PHASES: List[PhaseDescriptor] = [
    PhaseDescriptor(AutonomyPhase.FOUNDATION,   "Foundation",   "Show me",              0,  16,  8),
    PhaseDescriptor(AutonomyPhase.ASSISTANCE,   "Assistance",   "Suggest to me",        17, 33,  25),
    PhaseDescriptor(AutonomyPhase.AUTOMATION,   "Automation",   "Do it, I'll approve",  34, 50,  42),
    PhaseDescriptor(AutonomyPhase.OPTIMIZATION, "Optimization", "Do it, ask sometimes", 51, 67,  59),
    PhaseDescriptor(AutonomyPhase.INTELLIGENCE, "Intelligence", "Anticipate my needs",  68, 83,  75),
    PhaseDescriptor(AutonomyPhase.AUTONOMY,     "Autonomy",     "Run my company",       84, 100, 92),
]

_PHASES_BY_KEY: Dict[str, PhaseDescriptor] = {p.key: p for p in PHASES}


# Onboarding Chapter 3.2 answers → autonomy level midpoints.
# Matches existing 4-level comfort choice captured in PersonalAgentConfig.user_context.
_COMFORT_TO_LEVEL: Dict[str, int] = {
    "observe": 8,
    "suggest": 25,
    "supervised": 42,
    "autonomous": 85,
}


# Static per-node-type risk scores [0.0, 1.0].
# Dynamic boost (RiskAssessmentEngine) is applied on top for {apiCall, code,
# browserAutomation, iam} in the Business gate.
NODE_TYPE_RISK: Dict[str, float] = {
    "code": 0.7,
    "apiCall": 0.6,
    "browserAutomation": 0.8,
    "iam": 0.9,
    "mcp": 0.6,
    "aiProcess": 0.5,
    "loop": 0.3,
    "notification": 0.2,
    "dataTransform": 0.1,
    "condition": 0.1,
    "approval": 0.0,
}

DEFAULT_NODE_RISK = 0.4  # unknown node type — middle-of-the-road


SYSTEM_DEFAULT_LEVEL = 25  # Assistance phase — safe default when nothing else resolves


def _clamp_level(level: int) -> int:
    if level < 0:
        return 0
    if level > 100:
        return 100
    return level


def level_to_phase(level: int) -> str:
    """Map an autonomy level (0-100) to its phase key."""
    level = _clamp_level(level)
    for phase in PHASES:
        if phase.min_level <= level <= phase.max_level:
            return phase.key
    return AutonomyPhase.AUTONOMY  # unreachable given clamp, defensive


def phase_to_level(phase_key: str) -> int:
    """Map a phase key to its band midpoint integer. Raises KeyError on unknown phase."""
    return _PHASES_BY_KEY[phase_key].midpoint


def phase_descriptor(phase_key: str) -> PhaseDescriptor:
    return _PHASES_BY_KEY[phase_key]


def comfort_to_level(comfort: Optional[str]) -> Optional[int]:
    """Map the onboarding comfort_level string to an autonomy level midpoint.

    Returns None for unknown/empty comfort strings so the resolver can fall
    through to the next precedence tier.
    """
    if not comfort:
        return None
    return _COMFORT_TO_LEVEL.get(comfort.strip().lower())


def level_to_auto_approve_threshold(level: int) -> float:
    """Compute the risk threshold below which a node auto-approves at this level.

    Linear map: threshold(level) = (level / 100) * 0.95

    At level 32 (Assistance→Automation boundary) → threshold ≈ 0.30,
    matching the marketing claim that risk < 0.3 = auto-approve.
    At level 0  → 0.00 (gates everything).
    At level 100 → 0.95 (gates only the very riskiest).
    """
    return _clamp_level(level) / 100.0 * 0.95


def node_risk(node_type: str, risk_override: Optional[float] = None) -> float:
    """Return the static risk score for a node type, honoring per-node override.

    `risk_override` is the value of `node.parameters.risk_override` when set.
    """
    if risk_override is not None:
        try:
            value = float(risk_override)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            if value < 0.0:
                return 0.0
            if value > 1.0:
                return 1.0
            return value
    return NODE_TYPE_RISK.get(node_type, DEFAULT_NODE_RISK)
