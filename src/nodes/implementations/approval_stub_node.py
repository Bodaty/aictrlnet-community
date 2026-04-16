"""Community-edition stub for ApprovalNode.

Approval workflows are a Business Edition feature. Community ships this
stub so that templates referencing `approval` / NodeType.APPROVAL fail
with a clear error rather than a generic "no implementation" message
or a silent fallback to TaskNode.

Business and Enterprise editions overwrite this registration via
`nodes/edition_nodes.py` with the real ApprovalNode implementation.
"""

from typing import Any, Dict

from nodes.base_node import BaseNode


class ApprovalNode(BaseNode):
    """Stub approval node — raises if executed in Community edition."""

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise RuntimeError(
            "Approval nodes require Business Edition or higher. "
            "This deployment is running Community Edition only."
        )
