"""
Workflow Diff & Merge Service

Provides functionality to compare workflows and merge changes:
- compare_workflows: Find differences between two workflows
- apply_changes: Apply a set of changes to a base workflow
- validate_connectivity: Ensure workflow remains valid after edits
"""

import logging
import copy
from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of changes that can be made to a workflow."""
    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    NODE_MODIFIED = "node_modified"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    EDGE_MODIFIED = "edge_modified"
    PROPERTY_CHANGED = "property_changed"


class WorkflowChange:
    """Represents a single change to a workflow."""

    def __init__(
        self,
        change_type: ChangeType,
        element_id: str,
        element_type: str,  # 'node' or 'edge'
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.change_type = change_type
        self.element_id = element_id
        self.element_type = element_type
        self.old_value = old_value
        self.new_value = new_value
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "element_id": self.element_id,
            "element_type": self.element_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata
        }


class WorkflowDiffService:
    """Service for comparing and merging workflows."""

    def __init__(self):
        """Initialize the diff service."""
        pass

    def compare_workflows(
        self,
        original: Dict[str, Any],
        modified: Dict[str, Any]
    ) -> List[WorkflowChange]:
        """
        Compare two workflows and return the list of changes.

        Args:
            original: The original workflow structure
            modified: The modified workflow structure

        Returns:
            List of WorkflowChange objects describing the differences
        """
        changes = []

        # Compare nodes
        original_nodes = {n.get('id'): n for n in original.get('nodes', [])}
        modified_nodes = {n.get('id'): n for n in modified.get('nodes', [])}

        # Find added nodes
        for node_id, node in modified_nodes.items():
            if node_id not in original_nodes:
                changes.append(WorkflowChange(
                    change_type=ChangeType.NODE_ADDED,
                    element_id=node_id,
                    element_type='node',
                    new_value=node
                ))

        # Find removed nodes
        for node_id, node in original_nodes.items():
            if node_id not in modified_nodes:
                changes.append(WorkflowChange(
                    change_type=ChangeType.NODE_REMOVED,
                    element_id=node_id,
                    element_type='node',
                    old_value=node
                ))

        # Find modified nodes
        for node_id in original_nodes:
            if node_id in modified_nodes:
                if self._nodes_differ(original_nodes[node_id], modified_nodes[node_id]):
                    changes.append(WorkflowChange(
                        change_type=ChangeType.NODE_MODIFIED,
                        element_id=node_id,
                        element_type='node',
                        old_value=original_nodes[node_id],
                        new_value=modified_nodes[node_id]
                    ))

        # Compare edges
        original_edges = {self._edge_key(e): e for e in original.get('edges', [])}
        modified_edges = {self._edge_key(e): e for e in modified.get('edges', [])}

        # Find added edges
        for edge_key, edge in modified_edges.items():
            if edge_key not in original_edges:
                changes.append(WorkflowChange(
                    change_type=ChangeType.EDGE_ADDED,
                    element_id=edge.get('id', edge_key),
                    element_type='edge',
                    new_value=edge
                ))

        # Find removed edges
        for edge_key, edge in original_edges.items():
            if edge_key not in modified_edges:
                changes.append(WorkflowChange(
                    change_type=ChangeType.EDGE_REMOVED,
                    element_id=edge.get('id', edge_key),
                    element_type='edge',
                    old_value=edge
                ))

        logger.info(f"Workflow comparison found {len(changes)} changes")
        return changes

    def apply_changes(
        self,
        base_workflow: Dict[str, Any],
        changes: List[WorkflowChange]
    ) -> Dict[str, Any]:
        """
        Apply a list of changes to a base workflow.

        Args:
            base_workflow: The workflow to modify
            changes: List of changes to apply

        Returns:
            The modified workflow
        """
        result = copy.deepcopy(base_workflow)
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])

        for change in changes:
            if change.change_type == ChangeType.NODE_ADDED:
                # Add new node
                new_node = change.new_value
                if not any(n.get('id') == new_node.get('id') for n in nodes):
                    nodes.append(new_node)
                    logger.debug(f"Added node: {new_node.get('id')}")

            elif change.change_type == ChangeType.NODE_REMOVED:
                # Remove node and its connected edges
                node_id = change.element_id
                nodes = [n for n in nodes if n.get('id') != node_id]
                edges = [e for e in edges if e.get('source') != node_id and e.get('target') != node_id]
                edges = [e for e in edges if e.get('from') != node_id and e.get('to') != node_id]
                logger.debug(f"Removed node: {node_id}")

            elif change.change_type == ChangeType.NODE_MODIFIED:
                # Update existing node
                node_id = change.element_id
                for i, node in enumerate(nodes):
                    if node.get('id') == node_id:
                        # Merge changes, preserving position
                        old_position = node.get('position', {})
                        nodes[i] = change.new_value
                        if 'position' not in nodes[i]:
                            nodes[i]['position'] = old_position
                        logger.debug(f"Modified node: {node_id}")
                        break

            elif change.change_type == ChangeType.EDGE_ADDED:
                # Add new edge
                new_edge = change.new_value
                edge_key = self._edge_key(new_edge)
                if not any(self._edge_key(e) == edge_key for e in edges):
                    edges.append(new_edge)
                    logger.debug(f"Added edge: {edge_key}")

            elif change.change_type == ChangeType.EDGE_REMOVED:
                # Remove edge
                edge_id = change.element_id
                old_edge = change.old_value
                if old_edge:
                    edge_key = self._edge_key(old_edge)
                    edges = [e for e in edges if self._edge_key(e) != edge_key]
                else:
                    edges = [e for e in edges if e.get('id') != edge_id]
                logger.debug(f"Removed edge: {edge_id}")

        result['nodes'] = nodes
        result['edges'] = edges

        return result

    def merge_workflows(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any],
        edit_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Intelligently merge an AI-generated workflow modification with the original.

        This is the main entry point for edit mode - it uses the edit intent
        to determine how to merge the AI's suggestions with the existing workflow.

        Args:
            original: The original workflow
            ai_generated: The AI-generated workflow/changes
            edit_intent: The analyzed edit intent

        Returns:
            Merged workflow
        """
        intent = edit_intent.get('intent', 'unknown')
        result = copy.deepcopy(original)

        if intent == 'add':
            result = self._merge_add(result, ai_generated, edit_intent)
        elif intent == 'remove':
            result = self._merge_remove(result, ai_generated, edit_intent)
        elif intent == 'modify':
            result = self._merge_modify(result, ai_generated, edit_intent)
        elif intent == 'replace':
            result = self._merge_replace(result, ai_generated, edit_intent)
        elif intent == 'reorder':
            result = self._merge_reorder(result, ai_generated, edit_intent)
        else:
            # Unknown intent - try to intelligently merge
            result = self._merge_intelligent(result, ai_generated)

        # Validate and fix connectivity
        result = self.validate_and_fix_connectivity(result)

        return result

    def _merge_add(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any],
        edit_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge when intent is to add new nodes."""
        result = copy.deepcopy(original)
        orig_nodes = result.get('nodes', [])
        orig_edges = result.get('edges', [])
        orig_node_ids = {n.get('id') for n in orig_nodes}

        # Get new nodes from AI-generated workflow
        ai_nodes = ai_generated.get('nodes', [])
        ai_edges = ai_generated.get('edges', [])

        # Find nodes that are truly new (not in original)
        new_nodes = [n for n in ai_nodes if n.get('id') not in orig_node_ids]

        if not new_nodes:
            logger.warning("No new nodes found in AI-generated workflow")
            return result

        # Determine position for new nodes based on edit_intent
        position = edit_intent.get('position', {})
        pos_type = position.get('type', 'last')

        # Find insertion point
        if pos_type == 'after':
            reference = position.get('reference', '').lower()
            insert_after_node = self._find_node_by_name(orig_nodes, reference)
        elif pos_type == 'before':
            reference = position.get('reference', '').lower()
            insert_before_node = self._find_node_by_name(orig_nodes, reference)
        else:
            insert_after_node = None
            insert_before_node = None

        # Add new nodes with proper positioning
        for new_node in new_nodes:
            # Ensure unique ID
            if new_node.get('id') in orig_node_ids:
                new_node['id'] = f"{new_node['id']}_{uuid4().hex[:8]}"

            # Position the node
            if insert_after_node:
                ref_pos = insert_after_node.get('position', {'x': 400, 'y': 200})
                new_node['position'] = {
                    'x': ref_pos.get('x', 400) + 200,
                    'y': ref_pos.get('y', 200)
                }
            else:
                # Default: add at the end, before end node
                new_node['position'] = {
                    'x': 400 + len(orig_nodes) * 150,
                    'y': 200
                }

            orig_nodes.append(new_node)
            orig_node_ids.add(new_node.get('id'))

        # Add edges connecting new nodes
        for edge in ai_edges:
            source = edge.get('source') or edge.get('from')
            target = edge.get('target') or edge.get('to')

            # Only add edges that connect to existing or new nodes
            if source in orig_node_ids and target in orig_node_ids:
                # Check if edge already exists
                edge_key = f"{source}->{target}"
                existing_keys = {f"{e.get('source') or e.get('from')}->{e.get('target') or e.get('to')}" for e in orig_edges}

                if edge_key not in existing_keys:
                    new_edge = {
                        'id': edge.get('id', f"edge_{uuid4().hex[:8]}"),
                        'source': source,
                        'target': target,
                        'type': edge.get('type', 'default')
                    }
                    orig_edges.append(new_edge)

        result['nodes'] = orig_nodes
        result['edges'] = orig_edges

        return result

    def _merge_remove(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any],
        edit_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge when intent is to remove nodes."""
        result = copy.deepcopy(original)

        # Get target node from intent
        target = edit_intent.get('target_node', {})
        target_id = target.get('id')

        if not target_id:
            # Try to find by name from details
            details = edit_intent.get('details', {})
            element_to_remove = details.get('element_to_remove', '')
            if element_to_remove:
                node = self._find_node_by_name(result.get('nodes', []), element_to_remove)
                if node:
                    target_id = node.get('id')

        if target_id:
            # Remove the node
            result['nodes'] = [n for n in result.get('nodes', []) if n.get('id') != target_id]

            # Remove connected edges and reconnect
            result = self._reconnect_after_removal(result, target_id)

        return result

    def _merge_modify(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any],
        edit_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge when intent is to modify existing nodes."""
        result = copy.deepcopy(original)

        target = edit_intent.get('target_node', {})
        target_id = target.get('id')

        if target_id:
            # Find the node and update it with AI-generated changes
            for i, node in enumerate(result.get('nodes', [])):
                if node.get('id') == target_id:
                    # Find corresponding AI-generated node
                    ai_node = next(
                        (n for n in ai_generated.get('nodes', []) if n.get('id') == target_id),
                        None
                    )

                    if ai_node:
                        # Preserve position, update other properties
                        position = node.get('position', {})
                        result['nodes'][i] = {**ai_node, 'position': position}
                    break

        return result

    def _merge_replace(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any],
        edit_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge when intent is to replace a node."""
        result = copy.deepcopy(original)

        details = edit_intent.get('details', {})
        old_element = details.get('old_element', '')

        # Find the node to replace
        old_node = self._find_node_by_name(result.get('nodes', []), old_element)

        if old_node:
            old_id = old_node.get('id')
            old_position = old_node.get('position', {'x': 400, 'y': 200})

            # Get new node from AI-generated
            new_nodes = [n for n in ai_generated.get('nodes', [])
                        if n.get('id') not in {n.get('id') for n in result.get('nodes', [])}]

            if new_nodes:
                new_node = new_nodes[0]
                new_node['position'] = old_position
                new_id = new_node.get('id')

                # Replace the node
                result['nodes'] = [n for n in result.get('nodes', []) if n.get('id') != old_id]
                result['nodes'].append(new_node)

                # Update edges to point to new node
                for edge in result.get('edges', []):
                    if edge.get('source') == old_id or edge.get('from') == old_id:
                        if 'source' in edge:
                            edge['source'] = new_id
                        if 'from' in edge:
                            edge['from'] = new_id
                    if edge.get('target') == old_id or edge.get('to') == old_id:
                        if 'target' in edge:
                            edge['target'] = new_id
                        if 'to' in edge:
                            edge['to'] = new_id

        return result

    def _merge_reorder(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any],
        edit_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge when intent is to reorder nodes."""
        result = copy.deepcopy(original)

        # For reorder, we primarily update edges to reflect new order
        # The AI should have generated the new edge structure

        ai_edges = ai_generated.get('edges', [])
        if ai_edges:
            # Use AI-generated edges but validate they connect existing nodes
            node_ids = {n.get('id') for n in result.get('nodes', [])}
            valid_edges = []

            for edge in ai_edges:
                source = edge.get('source') or edge.get('from')
                target = edge.get('target') or edge.get('to')

                if source in node_ids and target in node_ids:
                    valid_edges.append(edge)

            if valid_edges:
                result['edges'] = valid_edges

        return result

    def _merge_intelligent(
        self,
        original: Dict[str, Any],
        ai_generated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback intelligent merge when intent is unknown."""
        # Use diff-based merge
        changes = self.compare_workflows(original, ai_generated)

        # Filter to only include additions and modifications (safer)
        safe_changes = [c for c in changes if c.change_type in [
            ChangeType.NODE_ADDED,
            ChangeType.NODE_MODIFIED,
            ChangeType.EDGE_ADDED
        ]]

        return self.apply_changes(original, safe_changes)

    def validate_and_fix_connectivity(
        self,
        workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate workflow connectivity and fix issues.

        Ensures:
        - All nodes are reachable from start
        - End node is reachable
        - No orphaned nodes
        """
        result = copy.deepcopy(workflow)
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])

        if not nodes:
            return result

        # Build adjacency list
        node_ids = {n.get('id') for n in nodes}
        outgoing = {nid: [] for nid in node_ids}
        incoming = {nid: [] for nid in node_ids}

        for edge in edges:
            source = edge.get('source') or edge.get('from')
            target = edge.get('target') or edge.get('to')

            if source in node_ids and target in node_ids:
                outgoing[source].append(target)
                incoming[target].append(source)

        # Find start node(s) - nodes with no incoming edges or type='start'
        start_nodes = [n for n in nodes if n.get('type') == 'start' or n.get('id') == 'start']
        if not start_nodes:
            start_nodes = [n for n in nodes if not incoming.get(n.get('id'), [])]

        # Find end node(s)
        end_nodes = [n for n in nodes if n.get('type') == 'end' or n.get('id') == 'end']

        # Find orphaned nodes (not reachable from start)
        reachable = set()
        to_visit = [n.get('id') for n in start_nodes]

        while to_visit:
            current = to_visit.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            to_visit.extend(outgoing.get(current, []))

        # Connect orphaned nodes
        orphaned = [n for n in nodes if n.get('id') not in reachable
                   and n.get('type') not in ['start', 'end']]

        for orphan in orphaned:
            orphan_id = orphan.get('id')
            # Connect to the node with highest outgoing degree (hub)
            best_connector = None
            best_degree = -1

            for nid in reachable:
                if len(outgoing.get(nid, [])) > best_degree:
                    best_degree = len(outgoing.get(nid, []))
                    best_connector = nid

            if best_connector:
                new_edge = {
                    'id': f"edge_fix_{uuid4().hex[:8]}",
                    'source': best_connector,
                    'target': orphan_id,
                    'type': 'default',
                    'metadata': {'auto_connected': True}
                }
                edges.append(new_edge)
                reachable.add(orphan_id)
                logger.debug(f"Auto-connected orphaned node: {orphan_id}")

        result['edges'] = edges
        return result

    def _find_node_by_name(
        self,
        nodes: List[Dict[str, Any]],
        name: str
    ) -> Optional[Dict[str, Any]]:
        """Find a node by name (case-insensitive partial match)."""
        name_lower = name.lower()

        for node in nodes:
            node_label = node.get('data', {}).get('label', '') or node.get('name', '')
            if node_label and name_lower in node_label.lower():
                return node

        return None

    def _reconnect_after_removal(
        self,
        workflow: Dict[str, Any],
        removed_node_id: str
    ) -> Dict[str, Any]:
        """Reconnect edges after a node is removed."""
        edges = workflow.get('edges', [])

        # Find incoming and outgoing edges of removed node
        incoming_sources = []
        outgoing_targets = []

        for edge in edges:
            source = edge.get('source') or edge.get('from')
            target = edge.get('target') or edge.get('to')

            if target == removed_node_id:
                incoming_sources.append(source)
            if source == removed_node_id:
                outgoing_targets.append(target)

        # Remove edges connected to removed node
        edges = [e for e in edges
                if (e.get('source') or e.get('from')) != removed_node_id
                and (e.get('target') or e.get('to')) != removed_node_id]

        # Reconnect: connect each incoming source to each outgoing target
        for source in incoming_sources:
            for target in outgoing_targets:
                new_edge = {
                    'id': f"edge_reconnect_{uuid4().hex[:8]}",
                    'source': source,
                    'target': target,
                    'type': 'default',
                    'metadata': {'auto_reconnected': True}
                }
                edges.append(new_edge)

        workflow['edges'] = edges
        return workflow

    def _nodes_differ(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
        """Check if two nodes are different (ignoring position)."""
        # Compare relevant properties, excluding position
        keys_to_compare = ['type', 'data', 'name', 'config']

        for key in keys_to_compare:
            if node1.get(key) != node2.get(key):
                return True

        return False

    def _edge_key(self, edge: Dict[str, Any]) -> str:
        """Create a unique key for an edge."""
        source = edge.get('source') or edge.get('from')
        target = edge.get('target') or edge.get('to')
        return f"{source}->{target}"


# Singleton instance
workflow_diff_service = WorkflowDiffService()
