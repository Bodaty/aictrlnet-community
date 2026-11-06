"""Preview runs: a run that shows what a workflow would do before anything acts.

A run started with ``trigger_metadata["preview"]`` asks the registered preview
policy, before each step, whether the step runs for real, is simulated, or
stops the run. An edition registers its policy with ``set_preview_policy``;
with none registered every step stops, so a preview can never cause a side
effect. A stop ends the whole run: the step is skipped, nothing after it
starts, and the run completes with the stop recorded.
"""

from typing import Any, Dict, List, Optional

RUN = "run"
SIMULATE = "simulate"
STOP = "stop"


class PreviewStop(Exception):
    """Raised by a step, or by the gate before it, to end a preview run there."""

    def __init__(self, node_id: str, reason: str, detail: Optional[Dict] = None):
        super().__init__(f"Preview stopped before {node_id}: {reason}")
        self.node_id = node_id
        self.reason = reason
        self.detail = detail or {}

    def as_record(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "reason": self.reason, "detail": self.detail}


class StopEveryStep:
    """The policy when no edition registered one: no step acts."""

    def classify(self, effective_type: str, node_config: Any, real_steps: List[str]) -> str:
        return STOP

    def simulate(self, effective_type: str, node_config: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise PreviewStop(getattr(node_config, "id", ""), f"stopped_before_{effective_type}")


_default_policy = StopEveryStep()
_registered_policy: Any = None


def set_preview_policy(policy: Any) -> None:
    """Register the preview policy; None restores the default (stop every step)."""
    global _registered_policy
    _registered_policy = policy


def get_preview_policy() -> Any:
    return _registered_policy if _registered_policy is not None else _default_policy
