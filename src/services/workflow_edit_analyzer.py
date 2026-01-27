"""
Workflow Edit Intent Analyzer

Analyzes user prompts in edit mode to determine the type of workflow modification:
- ADD: Adding new nodes/steps to the workflow
- REMOVE: Removing existing nodes/steps
- MODIFY: Changing properties of existing nodes
- REORDER: Changing the sequence of steps
- REPLACE: Replacing one node with another
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EditIntent(str, Enum):
    """Types of workflow edit operations."""
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    REORDER = "reorder"
    REPLACE = "replace"
    UNKNOWN = "unknown"


class WorkflowEditIntentAnalyzer:
    """Analyzes natural language to detect workflow edit intents."""

    # Keywords that indicate different edit intents
    ADD_KEYWORDS = [
        'add', 'insert', 'include', 'create', 'new', 'put', 'append',
        'introduce', 'attach', 'integrate', 'incorporate', 'place'
    ]

    REMOVE_KEYWORDS = [
        'remove', 'delete', 'drop', 'eliminate', 'get rid of', 'take out',
        'exclude', 'discard', 'cut', 'strip', 'omit', 'skip'
    ]

    MODIFY_KEYWORDS = [
        'change', 'modify', 'update', 'edit', 'alter', 'adjust', 'configure',
        'set', 'make', 'convert', 'transform', 'rename', 'switch'
    ]

    REORDER_KEYWORDS = [
        'move', 'reorder', 'rearrange', 'shift', 'relocate', 'swap',
        'before', 'after', 'between', 'first', 'last', 'earlier', 'later'
    ]

    REPLACE_KEYWORDS = [
        'replace', 'substitute', 'swap out', 'exchange', 'instead of',
        'use ... instead', 'switch to', 'change to'
    ]

    # Position indicators for targeting specific nodes
    POSITION_PATTERNS = {
        'after': r'after\s+(?:the\s+)?(.+?)(?:\s+step|\s+node|\s*$)',
        'before': r'before\s+(?:the\s+)?(.+?)(?:\s+step|\s+node|\s*$)',
        'between': r'between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)(?:\s+step|\s+node|\s*$)',
        'first': r'^(?:at\s+the\s+)?(?:beginning|start|first)',
        'last': r'(?:at\s+the\s+)?(?:end|last|final)',
    }

    def __init__(self):
        """Initialize the analyzer."""
        self._compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        for name, pattern in self.POSITION_PATTERNS.items():
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)

    def analyze(
        self,
        prompt: str,
        current_workflow: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the user's edit request and extract intent.

        Args:
            prompt: User's natural language edit request
            current_workflow: The current workflow structure (for context)

        Returns:
            Dict containing:
            - intent: EditIntent enum value
            - confidence: Float 0-1 indicating confidence
            - target_node: Node being affected (if identifiable)
            - position: Where to place new node (for ADD)
            - details: Additional extracted information
        """
        prompt_lower = prompt.lower().strip()

        # Detect primary intent
        intent, confidence = self._detect_intent(prompt_lower)

        # Extract target node(s) if identifiable
        target_node = self._extract_target_node(prompt_lower, current_workflow)

        # Extract position information (for ADD/REORDER)
        position = self._extract_position(prompt_lower)

        # Extract what's being added/modified
        operation_details = self._extract_operation_details(prompt_lower, intent)

        result = {
            "intent": intent.value,
            "confidence": confidence,
            "target_node": target_node,
            "position": position,
            "details": operation_details,
            "original_prompt": prompt
        }

        logger.info(f"Edit intent analysis: {intent.value} (confidence: {confidence:.2f})")
        return result

    def _detect_intent(self, prompt: str) -> Tuple[EditIntent, float]:
        """
        Detect the primary edit intent from the prompt.

        Returns tuple of (intent, confidence).
        """
        scores = {
            EditIntent.ADD: 0.0,
            EditIntent.REMOVE: 0.0,
            EditIntent.MODIFY: 0.0,
            EditIntent.REORDER: 0.0,
            EditIntent.REPLACE: 0.0
        }

        # Score each intent based on keyword matches
        for keyword in self.ADD_KEYWORDS:
            if keyword in prompt:
                scores[EditIntent.ADD] += 1.0

        for keyword in self.REMOVE_KEYWORDS:
            if keyword in prompt:
                scores[EditIntent.REMOVE] += 1.0

        for keyword in self.MODIFY_KEYWORDS:
            if keyword in prompt:
                scores[EditIntent.MODIFY] += 1.0

        for keyword in self.REORDER_KEYWORDS:
            if keyword in prompt:
                scores[EditIntent.REORDER] += 1.0

        for keyword in self.REPLACE_KEYWORDS:
            if keyword in prompt:
                scores[EditIntent.REPLACE] += 1.0

        # Find the highest scoring intent
        max_score = max(scores.values())

        if max_score == 0:
            return EditIntent.UNKNOWN, 0.3

        best_intent = max(scores, key=scores.get)

        # Calculate confidence (normalize by number of possible keywords)
        keyword_counts = {
            EditIntent.ADD: len(self.ADD_KEYWORDS),
            EditIntent.REMOVE: len(self.REMOVE_KEYWORDS),
            EditIntent.MODIFY: len(self.MODIFY_KEYWORDS),
            EditIntent.REORDER: len(self.REORDER_KEYWORDS),
            EditIntent.REPLACE: len(self.REPLACE_KEYWORDS)
        }

        # Confidence based on score relative to total possible
        # Also consider if the intent is much stronger than alternatives
        confidence = min(0.95, 0.5 + (max_score / 3))  # Cap at 0.95

        return best_intent, confidence

    def _extract_target_node(
        self,
        prompt: str,
        current_workflow: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Try to identify which node(s) the user is referring to.

        Returns node information if identifiable.
        """
        if not current_workflow:
            return None

        nodes = current_workflow.get('nodes', [])
        if not nodes:
            return None

        # Try to match node names or types mentioned in the prompt
        best_match = None
        best_score = 0

        for node in nodes:
            node_name = node.get('data', {}).get('label', '') or node.get('name', '')
            node_type = node.get('type', '')

            # Check if node name is mentioned
            if node_name and node_name.lower() in prompt:
                score = len(node_name)  # Longer matches are more specific
                if score > best_score:
                    best_score = score
                    best_match = {
                        'id': node.get('id'),
                        'name': node_name,
                        'type': node_type,
                        'match_type': 'name'
                    }

            # Check if node type is mentioned
            if node_type and node_type.lower() in prompt:
                score = len(node_type) * 0.8  # Slightly lower priority than name
                if score > best_score:
                    best_score = score
                    best_match = {
                        'id': node.get('id'),
                        'name': node_name,
                        'type': node_type,
                        'match_type': 'type'
                    }

        return best_match

    def _extract_position(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Extract position information for where to place new nodes.

        Returns position dict with type and reference node(s).
        """
        # Check each position pattern
        for pos_type, pattern in self._compiled_patterns.items():
            match = pattern.search(prompt)
            if match:
                groups = match.groups()

                if pos_type == 'after' and groups:
                    return {
                        'type': 'after',
                        'reference': groups[0].strip()
                    }
                elif pos_type == 'before' and groups:
                    return {
                        'type': 'before',
                        'reference': groups[0].strip()
                    }
                elif pos_type == 'between' and len(groups) >= 2:
                    return {
                        'type': 'between',
                        'reference_start': groups[0].strip(),
                        'reference_end': groups[1].strip()
                    }
                elif pos_type == 'first':
                    return {'type': 'first'}
                elif pos_type == 'last':
                    return {'type': 'last'}

        return None

    def _extract_operation_details(
        self,
        prompt: str,
        intent: EditIntent
    ) -> Dict[str, Any]:
        """
        Extract details specific to the operation type.
        """
        details = {}

        if intent == EditIntent.ADD:
            # Try to extract what's being added
            add_patterns = [
                r'add\s+(?:a\s+)?(?:new\s+)?(.+?)(?:\s+step|\s+node|\s+after|\s+before|\s*$)',
                r'insert\s+(?:a\s+)?(?:new\s+)?(.+?)(?:\s+step|\s+node|\s*$)',
                r'include\s+(?:a\s+)?(.+?)(?:\s+in|\s+to|\s*$)',
            ]
            for pattern in add_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    details['new_element'] = match.group(1).strip()
                    break

        elif intent == EditIntent.REMOVE:
            # Try to extract what's being removed
            remove_patterns = [
                r'remove\s+(?:the\s+)?(.+?)(?:\s+step|\s+node|\s*$)',
                r'delete\s+(?:the\s+)?(.+?)(?:\s+step|\s+node|\s*$)',
            ]
            for pattern in remove_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    details['element_to_remove'] = match.group(1).strip()
                    break

        elif intent == EditIntent.MODIFY:
            # Try to extract the property being changed and new value
            change_patterns = [
                r'change\s+(?:the\s+)?(.+?)\s+to\s+(.+?)(?:\s*$)',
                r'set\s+(?:the\s+)?(.+?)\s+to\s+(.+?)(?:\s*$)',
                r'update\s+(?:the\s+)?(.+?)\s+to\s+(.+?)(?:\s*$)',
            ]
            for pattern in change_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    details['property'] = match.group(1).strip()
                    details['new_value'] = match.group(2).strip()
                    break

        elif intent == EditIntent.REPLACE:
            # Try to extract what's being replaced and with what
            replace_patterns = [
                r'replace\s+(?:the\s+)?(.+?)\s+with\s+(.+?)(?:\s*$)',
                r'swap\s+(?:the\s+)?(.+?)\s+(?:for|with)\s+(.+?)(?:\s*$)',
            ]
            for pattern in replace_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    details['old_element'] = match.group(1).strip()
                    details['new_element'] = match.group(2).strip()
                    break

        return details


# Singleton instance for easy import
workflow_edit_analyzer = WorkflowEditIntentAnalyzer()
