"""Unified Analysis Parser for LLM Call Optimization.

This module provides unified parsing of intent, domain, and company context
from a single LLM response, reducing workflow generation latency by combining
three sequential FAST-tier LLM calls into one.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAnalysisResult:
    """Structured result from unified initial analysis.

    This dataclass encapsulates all three components that were previously
    extracted from separate LLM calls:
    - Intent classification
    - Domain/industry identification
    - Company automation context
    """

    # Intent fields
    primary_intent: str
    intent_confidence: float
    intent_reasoning: str

    # Domain fields
    industry: str
    domain_type: str
    domain_confidence: float
    domain_keywords: List[str] = field(default_factory=list)

    # Company context fields
    company_size: str = "unknown"
    business_type: str = "unknown"
    automation_needs: List[str] = field(default_factory=list)
    specific_processes: List[str] = field(default_factory=list)


class UnifiedAnalysisParser:
    """Parses unified LLM analysis response.

    This parser handles the structured JSON response from the unified
    LLM call that combines intent classification, domain identification,
    and company automation analysis.
    """

    async def parse_response(
        self,
        llm_response: str
    ) -> UnifiedAnalysisResult:
        """Parse LLM JSON response into structured result.

        Args:
            llm_response: JSON string from LLM containing unified analysis

        Returns:
            UnifiedAnalysisResult with all extracted fields

        Raises:
            ValueError: If response cannot be parsed or is incomplete
        """
        try:
            # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith('```'):
                # Remove opening fence (```json or just ```)
                lines = cleaned_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # Remove first line
                # Remove closing fence
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]  # Remove last line
                cleaned_response = '\n'.join(lines).strip()

            # Extract ONLY the JSON object, ignoring any extra text after it
            # This fixes the "Extra data: line X column Y" error where LLM adds explanatory text
            json_start = cleaned_response.find('{')
            if json_start == -1:
                raise ValueError("No JSON object found in response")

            # Find matching closing brace using brace counting
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(cleaned_response)):
                if cleaned_response[i] == '{':
                    brace_count += 1
                elif cleaned_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end == -1:
                raise ValueError("No matching closing brace found for JSON object")

            json_only = cleaned_response[json_start:json_end]
            logger.debug(f"Extracted JSON object ({len(json_only)} chars) from response ({len(cleaned_response)} chars)")

            # Parse JSON response
            data = json.loads(json_only)

            # Validate required top-level keys
            if not all(key in data for key in ['intent', 'domain', 'company_context']):
                raise ValueError(
                    "Missing required sections in unified analysis response. "
                    f"Expected 'intent', 'domain', 'company_context'. Got: {list(data.keys())}"
                )

            # Extract intent fields
            intent_data = data['intent']
            if not all(key in intent_data for key in ['primary_intent', 'confidence']):
                raise ValueError(
                    "Missing required fields in intent section. "
                    f"Expected 'primary_intent', 'confidence'. Got: {list(intent_data.keys())}"
                )

            primary_intent = str(intent_data['primary_intent'])
            intent_confidence = float(intent_data['confidence'])
            intent_reasoning = str(intent_data.get('reasoning', ''))

            # Validate intent confidence
            if not 0.0 <= intent_confidence <= 1.0:
                logger.warning(
                    f"Intent confidence {intent_confidence} out of range [0.0, 1.0]. "
                    "Clamping to valid range."
                )
                intent_confidence = max(0.0, min(1.0, intent_confidence))

            # Extract domain fields
            domain_data = data['domain']
            if not all(key in domain_data for key in ['industry', 'confidence']):
                raise ValueError(
                    "Missing required fields in domain section. "
                    f"Expected 'industry', 'confidence'. Got: {list(domain_data.keys())}"
                )

            industry = str(domain_data['industry'])
            domain_type = str(domain_data.get('domain_type', ''))
            domain_confidence = float(domain_data['confidence'])
            domain_keywords = [str(kw) for kw in domain_data.get('keywords', [])]

            # Validate domain confidence
            if not 0.0 <= domain_confidence <= 1.0:
                logger.warning(
                    f"Domain confidence {domain_confidence} out of range [0.0, 1.0]. "
                    "Clamping to valid range."
                )
                domain_confidence = max(0.0, min(1.0, domain_confidence))

            # Extract company context fields (optional fields with defaults)
            company_data = data['company_context']
            company_size = str(company_data.get('company_size', 'unknown'))
            business_type = str(company_data.get('business_type', 'unknown'))
            automation_needs = [str(need) for need in company_data.get('automation_needs', [])]
            specific_processes = [str(proc) for proc in company_data.get('specific_processes', [])]

            # Create and return structured result
            result = UnifiedAnalysisResult(
                primary_intent=primary_intent,
                intent_confidence=intent_confidence,
                intent_reasoning=intent_reasoning,
                industry=industry,
                domain_type=domain_type,
                domain_confidence=domain_confidence,
                domain_keywords=domain_keywords,
                company_size=company_size,
                business_type=business_type,
                automation_needs=automation_needs,
                specific_processes=specific_processes
            )

            logger.info(
                f"Unified analysis parsed successfully: "
                f"intent={primary_intent} (conf={intent_confidence:.2f}), "
                f"industry={industry} (conf={domain_confidence:.2f}), "
                f"company_size={company_size}"
            )

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
        except KeyError as e:
            raise ValueError(f"Missing required field in LLM response: {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid data type in LLM response: {e}") from e
