"""Abstract Request Processing Services for AICtrlNet Intelligent Assistant.

This module provides services for handling high-level, abstract requests
by decomposing them, asking clarifying questions, and synthesizing complete solutions.
"""

from .request_decomposer import (
    RequestDecomposer,
    DecomposedRequest,
    DecomposedComponent,
    ComponentType
)

from .clarification_engine import (
    ClarificationEngine,
    ClarifyingQuestion,
    ClarificationContext,
    QuestionType,
    QuestionPriority,
    QuestionOption
)

from .solution_synthesizer import (
    SolutionSynthesizer,
    CompleteSolution,
    ImplementationPlan,
    ResourceRequirement,
    RiskMitigation
)

__all__ = [
    # Request Decomposer
    "RequestDecomposer",
    "DecomposedRequest",
    "DecomposedComponent",
    "ComponentType",

    # Clarification Engine
    "ClarificationEngine",
    "ClarifyingQuestion",
    "ClarificationContext",
    "QuestionType",
    "QuestionPriority",
    "QuestionOption",

    # Solution Synthesizer
    "SolutionSynthesizer",
    "CompleteSolution",
    "ImplementationPlan",
    "ResourceRequirement",
    "RiskMitigation"
]