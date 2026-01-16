"""NLP Service for workflow generation from natural language."""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from models.community import WorkflowDefinition
from schemas.workflow import (
    WorkflowCreate, NLPWorkflowResponse, TemplateMatch,
    ExtractedParameter, EditionRequirement, UpgradeSuggestion,
    TemplatePreview, WorkflowResponse
)
from core.config import get_settings
from core.tenant_context import get_current_tenant_id
from .workflow import WorkflowService
from .workflow_template_service import create_workflow_template_service
from .llm_helpers import get_user_llm_settings, get_system_llm_settings
# Import from LLM module instead of performance_optimizer
from llm import llm_service, UserLLMSettings, WorkflowStep
from llm.model_selection import estimate_complexity_hybrid
from .adapter import AdapterService
from .workflow_security import WorkflowSecurityService, SecurityLevel

settings = get_settings()

logger = logging.getLogger(__name__)


class NLPService:
    """Service for NLP-driven workflow generation."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.workflow_service = WorkflowService(db)
        self.template_service = create_workflow_template_service()
        # Using template_service (WorkflowTemplateService) directly instead of workflow_templates
        self.adapter_service = AdapterService(db)
        self.security_service = WorkflowSecurityService()
        # Connect to native Ollama via host network
        self.ollama_url = "http://host.docker.internal:11434"
        self._available_models = None
        self._cached_adapters = None
        
        # Initialize edition-specific enhancements
        self._init_edition_enhancements()
        
        # Log template count (will be loaded dynamically from database)
        logger.info(f"NLP Service initialized with WorkflowTemplateService - Edition: {self.edition}")
        
    def _init_edition_enhancements(self):
        """Initialize edition-specific enhancement features."""
        import os
        
        # Detect edition from environment
        self.edition = os.getenv('AICTRLNET_EDITION', 'community').lower()
        
        # Initialize enhancement components based on edition
        self.enhancement_service = None
        
        if self.edition in ['business', 'enterprise']:
            try:
                # Import Business edition enhancement service (which includes RAG and Spec Generator)
                from aictrlnet_business.services.workflow_enhancement_service import WorkflowEnhancementService
                
                self.enhancement_service = WorkflowEnhancementService()
                logger.info(f"Initialized {self.edition.title()} edition with WorkflowEnhancementService")
            except ImportError as e:
                logger.warning(f"Could not import Business edition enhancements: {e}")
                # Fall back to basic functionality
                
        if self.edition == 'enterprise':
            # Enterprise gets all Business features plus more
            # The enhancement service already handles enterprise features
            pass
            
    async def process_natural_language(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        return_transparency: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process natural language to generate a workflow with full transparency."""
        processing_steps = []
        templates_used = []
        extracted_parameters = []
        generation_method = "fallback"
        confidence_score = 0.5
        ai_model_used = None
        
        # Security validation
        if user_id is None:
            user_id = context.get('user_id', 'anonymous') if context else 'anonymous'
            
        security_validation = self.security_service.validate_prompt(prompt, user_id)
        
        processing_steps.append({
            "step": "security_validation",
            "result": {
                "allowed": security_validation.allowed,
                "level": security_validation.level,
                "reason": security_validation.reason
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if not security_validation.allowed:
            # Return error response with transparency data
            return {
                "error": f"Security validation failed: {security_validation.reason}",
                "workflow": None,
                "transparency": {
                    "processing_steps": processing_steps,
                    "security_blocked": True,
                    "security_level": security_validation.level
                } if return_transparency else None
            }
        
        try:
            # Track intent analysis
            intent_analysis = await self._analyze_intent(prompt)
            processing_steps.append({
                "step": "intent_analysis",
                "result": intent_analysis,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 1: Try AI-driven generation
            ai_result = await self._generate_with_ai_enhanced(prompt, context, user_id)
            if ai_result:
                ai_workflow = ai_result['workflow']
                ai_model_used = ai_result.get('model', settings.DEFAULT_LLM_MODEL)
                
                processing_steps.append({
                    "step": "ai_generation",
                    "success": bool(ai_workflow),
                    "model": ai_model_used
                })
                
                if ai_workflow and self._is_complete_workflow(ai_workflow):
                    # AI generated a complete workflow
                    generation_method = "ai_generated"
                    confidence_score = 0.85
                    
                    # Apply edition-based enhancement if available
                    # Check if user requested Business/Enterprise edition via context
                    requested_edition = context.get('edition', self.edition) if context else self.edition
                    
                    # If Community edition needs to escalate to Business/Enterprise, use API call
                    if self.edition == 'community' and requested_edition in ['business', 'enterprise']:
                        try:
                            import httpx
                            # Determine target service URL
                            target_port = 8001 if requested_edition == 'business' else 8002
                            target_url = f"http://{requested_edition}:{target_port}/api/v1/workflows/enhanced/generate"
                            
                            async with httpx.AsyncClient(timeout=60.0) as client:
                                response = await client.post(
                                    target_url,
                                    json={
                                        "prompt": prompt,
                                        "context": context or {},
                                        "base_workflow": ai_workflow
                                    },
                                    # DEV_ONLY_START
                                    headers={"Authorization": "Bearer dev-token-for-testing"}
                                    # DEV_ONLY_END
                                    # In production, use: headers={"Authorization": f"Bearer {user_token}"}
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    if result.get("success") and result.get("workflow"):
                                        ai_workflow = result["workflow"]
                                        generation_method = f"ai_generated_{requested_edition}"
                                        confidence_score = 0.95
                                        
                                        processing_steps.append({
                                            "step": f"{requested_edition}_enhancement_api",
                                            "enhancement_level": requested_edition,
                                            "success": True,
                                            "timestamp": datetime.utcnow().isoformat()
                                        })
                                        logger.info(f"Successfully enhanced via {requested_edition} API")
                                else:
                                    logger.warning(f"Enhancement API returned {response.status_code}")
                                    
                        except Exception as e:
                            logger.warning(f"API-based enhancement failed: {e}")
                            processing_steps.append({
                                "step": f"{requested_edition}_enhancement_api",
                                "success": False,
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    
                    # Direct enhancement for Business/Enterprise editions
                    elif self.enhancement_service and requested_edition in ['business', 'enterprise']:
                        try:
                            enhancement_level = context.get('enhancement_level', 'enhanced') if context else 'enhanced'
                            if requested_edition == 'enterprise':
                                enhancement_level = 'premium'
                            
                            # Use the enhancement service to generate enhanced workflow
                            from aictrlnet_business.services.workflow_enhancement_service import EnhancementLevel
                            
                            # Convert string to enum
                            level_map = {
                                'basic': EnhancementLevel.BASIC,
                                'enhanced': EnhancementLevel.ENHANCED,
                                'premium': EnhancementLevel.PREMIUM
                            }
                            level_enum = level_map.get(enhancement_level, EnhancementLevel.ENHANCED)
                            
                            enhanced_workflow = await self.enhancement_service.generate_workflow(
                                prompt=prompt,
                                context=context or {},
                                enhancement_level=level_enum,
                                user_tier=requested_edition,
                                base_workflow=ai_workflow
                            )
                            
                            if enhanced_workflow:
                                # The enhancement service returns the workflow directly
                                ai_workflow = enhanced_workflow
                                generation_method = f"ai_generated_{enhancement_level}"
                                confidence_score = 0.95
                                
                                processing_steps.append({
                                    "step": f"{requested_edition}_enhancement",
                                    "enhancement_level": enhancement_level,
                                    "success": True,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                        except Exception as e:
                            logger.warning(f"Enhancement failed, using basic workflow: {e}")
                            processing_steps.append({
                                "step": f"{requested_edition}_enhancement",
                                "success": False,
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    
                    workflow = await self._create_workflow_from_nlp(
                        prompt, ai_workflow, generation_method,
                        tenant_id=context.get('tenant_id') or get_current_tenant_id() if context else get_current_tenant_id(),
                        context=context
                    )
                    
                    if return_transparency:
                        return await self._build_transparency_response(
                            workflow, generation_method, templates_used,
                            extracted_parameters, intent_analysis, confidence_score,
                            processing_steps, ai_model_used, prompt
                        )
                    return workflow
                
                # Step 2: Enhance with templates if AI workflow is incomplete
                if ai_workflow:
                    enhancement_result = await self._enhance_with_templates_tracked(
                        prompt, ai_workflow, context
                    )
                    if enhancement_result:
                        enhanced_workflow = enhancement_result['workflow']
                        templates_used.extend(enhancement_result['templates_used'])
                        generation_method = "ai_enhanced"
                        confidence_score = 0.75
                        
                        workflow = await self._create_workflow_from_nlp(
                            prompt, enhanced_workflow, generation_method,
                            tenant_id=context.get('tenant_id') or get_current_tenant_id() if context else get_current_tenant_id(),
                            context=context
                        )
                        
                        # Validate generated workflow
                        workflow_validation = self.security_service.validate_generated_workflow(
                            workflow, user_id
                        )
                        
                        processing_steps.append({
                            "step": "workflow_security_validation",
                            "result": {
                                "allowed": workflow_validation.allowed,
                                "level": workflow_validation.level,
                                "reason": workflow_validation.reason
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        if not workflow_validation.allowed:
                            # Generate safe fallback workflow
                            workflow = await self._create_fallback_workflow(prompt, context)
                            generation_method = "security_fallback"
                            confidence_score = 0.4
                        
                        if return_transparency:
                            return await self._build_transparency_response(
                                workflow, generation_method, templates_used,
                                extracted_parameters, intent_analysis, confidence_score,
                                processing_steps, ai_model_used, prompt
                            )
                        return workflow
            
            # Step 3: Try template matching if AI fails
            template_result = await self._match_templates_tracked(prompt, context)
            if template_result:
                template_workflow = template_result['workflow']
                templates_used = template_result['templates_used']
                extracted_parameters = template_result['parameters']
                generation_method = "template_based"
                confidence_score = 0.7
                
                workflow = await self._create_workflow_from_nlp(
                    prompt, template_workflow, generation_method,
                    tenant_id=context.get('tenant_id') or get_current_tenant_id() if context else get_current_tenant_id(),
                    context=context
                )
                
                if return_transparency:
                    return await self._build_transparency_response(
                        workflow, generation_method, templates_used,
                        extracted_parameters, intent_analysis, confidence_score,
                        processing_steps, ai_model_used, prompt
                    )
                return workflow
            
            # Step 4: Fallback to basic workflow
            generation_method = "intelligent_fallback"
            confidence_score = 0.5
            workflow = await self._create_fallback_workflow(prompt, context)
            
            if return_transparency:
                logger.info(f"Building transparency response for fallback workflow, return_transparency={return_transparency}")
                transparency_response = await self._build_transparency_response(
                    workflow, generation_method, templates_used,
                    extracted_parameters, intent_analysis, confidence_score,
                    processing_steps, ai_model_used, prompt
                )
                logger.info(f"Transparency response type: {type(transparency_response)}, is_dict: {isinstance(transparency_response, dict)}")
                if isinstance(transparency_response, dict):
                    logger.info(f"Transparency response keys: {list(transparency_response.keys())[:10]}")
                return transparency_response
            return workflow
            
        except Exception as e:
            logger.error(f"Error processing natural language: {e}", exc_info=True)
            # Always return something usable
            workflow = await self._create_fallback_workflow(prompt, context)
            if return_transparency:
                return await self._build_transparency_response(
                    workflow, "error_fallback", [], [], {},
                    0.3, processing_steps, None, prompt
                )
            return workflow
    
    async def _discover_available_tools(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Discover available adapters and tools from aictrlnet."""
        try:
            # Get user's edition from context
            user_edition = context.get("user_edition", "community") if context else "community"
            
            # Discover all available adapters
            adapters, total = await self.adapter_service.discover_adapters(
                edition=user_edition,
                include_unavailable=True,  # Include all to show what's possible
                limit=1000  # Get all adapters
            )
            
            # Group adapters by category
            adapters_by_category = {}
            for adapter in adapters:
                category = adapter.category
                if category not in adapters_by_category:
                    adapters_by_category[category] = []
                adapters_by_category[category].append({
                    "id": adapter.id,
                    "name": adapter.name,
                    "description": adapter.description,
                    "type": adapter.type,
                    "available": adapter.min_edition <= user_edition,
                    "min_edition": adapter.min_edition
                })
            
            self._cached_adapters = {
                "total": total,
                "by_category": adapters_by_category,
                "all": [{"id": a.id, "name": a.name, "type": a.type, "category": a.category} for a in adapters]
            }
            
            logger.info(f"Discovered {total} adapters across {len(adapters_by_category)} categories")
            
            return self._cached_adapters
            
        except Exception as e:
            logger.warning(f"Failed to discover adapters: {e}")
            return {"total": 0, "by_category": {}, "all": []}
    
    async def _generate_with_ai(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate workflow using AI (Ollama) with sophisticated step extraction."""
        try:
            # Discover available tools first
            available_tools = await self._discover_available_tools(context)

            # Always use AI/LLM for workflow generation (removed pattern matching)
            # This ensures user preferences are always respected
            # Estimate complexity using hybrid approach (non-async version)
            complexity = estimate_complexity_hybrid(prompt)
            logger.info(f"Prompt word count: {len(prompt.split())}, Hybrid complexity: {complexity:.2f}")

            # Use LLM service for workflow generation with user preference resolution
            if user_id and user_id != "system":
                # User-initiated request - use their preferences
                user_settings = await get_user_llm_settings(
                    db=self.db,
                    user_id=user_id
                )
            else:
                # System task - use system defaults
                user_settings = get_system_llm_settings()
            
            # Enhance prompt with available tools
            enhanced_prompt = self._enhance_prompt_with_tools(prompt, available_tools)
            
            # Use LLM service to generate workflow steps
            workflow_steps = await llm_service.generate_workflow_steps(
                prompt=enhanced_prompt,
                user_settings=user_settings,
                context={"complexity": complexity}
            )
            
            if workflow_steps and len(workflow_steps) > 0:
                # Convert WorkflowStep objects to the format expected by _steps_to_workflow_config
                steps_data = [step.to_dict() for step in workflow_steps]

                # Convert to workflow config
                workflow_config = self._steps_to_workflow_config(steps_data)
                # Record the actual model that was used (from user_settings)
                workflow_config['_model_used'] = user_settings.selected_model
                workflow_config['_model_tier'] = "balanced"
                logger.info(f"Generated workflow with {len(workflow_steps)} steps using model: {user_settings.selected_model}")
                return workflow_config
            else:
                logger.warning(f"LLM service failed to generate steps for prompt: {prompt[:50]}...")
                    
        except Exception as e:
            logger.warning(f"AI generation failed: {e}")
            return None
    
    def _is_complete_workflow(self, workflow: Dict[str, Any]) -> bool:
        """Check if a workflow is complete and valid."""
        if not workflow:
            return False
            
        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])
        
        # Must have at least start and end nodes
        has_start = any(n.get("type") == "start" for n in nodes)
        has_end = any(n.get("type") == "end" for n in nodes)
        
        # Must have at least one edge
        has_edges = len(edges) > 0
        
        # Must have at least 3 nodes (start, process, end)
        has_process = len(nodes) >= 3
        
        return has_start and has_end and has_edges and has_process
    
    async def _enhance_with_templates(
        self, 
        prompt: str, 
        ai_workflow: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Enhance an incomplete AI workflow with templates."""
        # Extract intent from prompt
        intent = await self._analyze_intent(prompt)
        
        # Find matching templates
        templates = await self.template_service.get_templates_by_category_async(
            self.db, intent.get("category", "general")
        )
        
        if not templates:
            return ai_workflow
            
        # Merge AI workflow with best matching template
        best_template = templates[0]  # Simple selection for now
        template_workflow = await self.template_service.generate_workflow_for_template(
            best_template
        )
        
        # Merge nodes and edges
        merged_nodes = ai_workflow.get("nodes", []) + template_workflow.get("nodes", [])
        merged_edges = ai_workflow.get("edges", []) + template_workflow.get("edges", [])
        
        # Remove duplicates and fix IDs
        return self._deduplicate_workflow({
            "nodes": merged_nodes,
            "edges": merged_edges
        })
    
    async def _match_templates(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Match prompt to sophisticated templates using AI for semantic understanding."""
        logger.info(f"_match_templates called with prompt: {prompt}")
        intent = await self._analyze_intent(prompt)
        logger.info(f"Intent analysis result: {intent}")
        
        # Use AI to find the best matching template
        template = await self._match_template_with_ai(prompt, intent)
        logger.info(f"AI template matching result: {template.get('name') if template else 'None'}")
        
        if template:
            # Generate workflow from template using the template service
            workflow_config = await self.template_service.generate_workflow_for_template(template)
            return workflow_config
        
        # Fallback to basic templates
        templates = await self.template_service.get_templates_by_category_async(
            self.db, intent.get("category", "general")
        )
        
        if not templates:
            return None
            
        # Use the best matching template
        template = templates[0]
        
        # Extract parameters from prompt
        parameters = self._extract_parameters(prompt, template)
        
        # Generate workflow from template
        workflow_config = await self.template_service.generate_workflow_for_template(template)
        
        # Apply parameters
        if parameters:
            workflow_config = self._apply_parameters_to_workflow(
                workflow_config, parameters
            )
            
        return workflow_config
    
    async def _analyze_intent(self, prompt: str) -> Dict[str, Any]:
        """Analyze user intent from prompt with enhanced understanding."""
        prompt_lower = prompt.lower()
        
        # Enhanced intent analysis with more categories
        intent_patterns = {
            "approval": ["approve", "approval", "review", "authorization", "sign off"],
            "notification": ["notify", "alert", "email", "sms", "message", "inform"],
            "validation": ["validate", "check", "verify", "ensure", "confirm"],
            "analysis": ["analyze", "sentiment", "classify", "predict", "assess"],
            "processing": ["process", "transform", "convert", "pipeline", "etl"],
            "monitoring": ["monitor", "track", "observe", "watch", "metric"],
            "support": ["support", "help", "assist", "customer", "inquiry"],
            "sales": ["lead", "sales", "opportunity", "crm", "prospect"],
            "hr": ["employee", "onboard", "hire", "hr", "human resources"],
            "finance": ["expense", "invoice", "payment", "reimburse", "financial"],
            "devops": ["deploy", "build", "cicd", "pipeline", "release"]
        }
        
        # Find primary intent
        primary_action = "process"
        category = "general"
        confidence = 0.5
        
        for intent_type, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in prompt_lower)
            if matches > 0:
                match_confidence = min(matches * 0.3, 0.9)
                if match_confidence > confidence:
                    primary_action = intent_type
                    category = self._map_intent_to_category(intent_type)
                    confidence = match_confidence
        
        # Extract entities
        entities = self._extract_entities(prompt)
        
        return {
            "primary_action": primary_action,
            "category": category,
            "intent": primary_action,
            "entities": entities,
            "confidence": confidence
        }
    
    def _map_intent_to_category(self, intent: str) -> str:
        """Map intent to workflow category."""
        category_map = {
            "approval": "business",
            "notification": "communication",
            "validation": "data",
            "analysis": "ai",
            "processing": "data",
            "monitoring": "operations",
            "support": "support",
            "sales": "sales",
            "hr": "hr",
            "finance": "finance",
            "devops": "it"
        }
        return category_map.get(intent, "general")
    
    def _extract_entities(self, prompt: str) -> Dict[str, Any]:
        """Extract entities from prompt."""
        entities = {}
        
        # Extract common entities
        if "email" in prompt.lower():
            entities["channel"] = "email"
        elif "sms" in prompt.lower():
            entities["channel"] = "sms"
        elif "slack" in prompt.lower():
            entities["channel"] = "slack"
        
        # Extract risk levels
        if "high risk" in prompt.lower():
            entities["risk_level"] = "high"
        elif "low risk" in prompt.lower():
            entities["risk_level"] = "low"
        
        # Extract urgency
        if "urgent" in prompt.lower() or "immediately" in prompt.lower():
            entities["urgency"] = "high"
        
        return entities
    
    def _extract_parameters(
        self, 
        prompt: str, 
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract parameters from prompt for template."""
        # Simple parameter extraction
        # In production, this would use NLP/AI
        parameters = {}
        
        # Extract common parameters
        if "email" in prompt.lower():
            parameters["channel"] = "email"
        elif "sms" in prompt.lower():
            parameters["channel"] = "sms"
            
        return parameters
    
    def _apply_parameters_to_workflow(
        self, 
        workflow: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply parameters to workflow configuration."""
        # Apply parameters to node data
        for node in workflow.get("nodes", []):
            if node.get("data"):
                node["data"].update(parameters)
                
        return workflow
    
    def _deduplicate_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate nodes and fix edge references."""
        seen_nodes = {}
        unique_nodes = []
        
        # Deduplicate nodes
        for node in workflow.get("nodes", []):
            node_key = f"{node.get('type')}_{node.get('name', '')}"
            if node_key not in seen_nodes:
                seen_nodes[node_key] = node["id"]
                unique_nodes.append(node)
                
        # Fix edge references
        unique_edges = []
        for edge in workflow.get("edges", []):
            # Skip edges with invalid references
            if edge.get("from") in seen_nodes.values() and \
               edge.get("to") in seen_nodes.values():
                unique_edges.append(edge)
                
        return {
            "nodes": unique_nodes,
            "edges": unique_edges
        }
    
    async def _match_template_with_ai(self, prompt: str, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use AI to find the best matching template based on semantic understanding."""
        logger.info(f"Starting AI template matching for prompt: {prompt}")
        try:
            # Get all available templates from database
            from schemas.workflow_templates import TemplateListRequest
            request = TemplateListRequest(
                include_system=True,
                include_public=True,
                include_private=False,
                limit=1000  # Get all templates
            )
            all_templates, total = await self.template_service.list_templates(self.db, "system", request)
            # Convert to dict format for compatibility
            all_templates = [template.model_dump() for template in all_templates]
            logger.info(f"Found {len(all_templates)} templates for AI matching")
            
            if not all_templates:
                logger.warning("No templates available for AI matching")
                return None
            
            # Prepare template summaries for AI
            template_summaries = []
            for i, template in enumerate(all_templates):
                template_summaries.append({
                    "index": i,
                    "id": str(template.get("id", "")),  # Convert UUID to string for JSON serialization
                    "name": template.get("name", ""),
                    "description": template.get("description", ""),
                    "category": template.get("category", ""),
                    "tags": template.get("tags", [])
                })
            
            # Pre-sort templates to help AI by putting likely matches first
            # This helps when we can only show a subset of templates to the AI
            prompt_words = set(prompt.lower().split())
            
            def relevance_score(template_summary):
                """Calculate basic relevance score for pre-sorting."""
                score = 0
                template_text = (
                    template_summary.get("name", "").lower() + " " +
                    template_summary.get("description", "").lower() + " " +
                    " ".join(template_summary.get("tags", []))
                ).lower()
                
                # Check for word matches
                for word in prompt_words:
                    if len(word) > 3:  # Skip very short words
                        if word in template_text:
                            score += 1
                            if word in template_summary.get("name", "").lower():
                                score += 2  # Extra weight for name matches
                
                # Add popularity if available
                idx = template_summary["index"]
                if idx < len(all_templates) and "popularity" in all_templates[idx]:
                    score += all_templates[idx]["popularity"] / 100.0
                
                return score
            
            # Sort by relevance score
            sorted_summaries = sorted(template_summaries, 
                                    key=relevance_score, 
                                    reverse=True)
            
            # Create AI prompt for template matching
            ai_prompt = f"""Given this user request: "{prompt}"

Select the SINGLE BEST matching workflow template from the list below.

Available templates:
{json.dumps(sorted_summaries[:25], indent=2)}

Instructions:
1. Understand the user's intent and desired workflow
2. Match it to the most appropriate template based on:
   - Semantic similarity of the request to template name/description
   - Alignment of business domain (category)
   - Relevant functionality described in tags
3. Return ONLY the index number

Return ONLY a single number (the index). Example: 5
"""
            
            # Use LLM service for template selection
            # Using system settings since template selection is an internal operation
            user_settings = get_system_llm_settings(
                temperature=0.3,  # Lower temperature for more deterministic selection
                max_tokens=10
            )
            
            # Call LLM service for template selection
            llm_response = await llm_service.generate(
                prompt=ai_prompt,
                user_settings=user_settings,
                task_type="template_matching",
                temperature=0.3,  # Lower temperature for more deterministic selection
                max_tokens=10
            )
            
            if llm_response and llm_response.text:
                ai_response = llm_response.text.strip()
                
                # Extract index from response
                try:
                    # Try to extract just the number
                    import re
                    numbers = re.findall(r'\d+', ai_response)
                    if numbers:
                        index = int(numbers[0])
                        if 0 <= index < len(all_templates):
                            selected_template = all_templates[index]
                            logger.info(f"AI selected template: {selected_template.get('name')} for prompt: {prompt}")
                            return selected_template
                except Exception as e:
                    logger.warning(f"Failed to parse AI template selection: {e}")
                
        except Exception as e:
            logger.warning(f"AI template matching failed: {e}")
        
        # Fallback to keyword matching if AI fails
        logger.info("Falling back to keyword-based template matching")
        # Simple keyword matching fallback
        return await self._simple_template_match(prompt, intent.get("entities", {}))
    
    async def _create_workflow_from_nlp(
        self, 
        prompt: str, 
        workflow_config: Dict[str, Any],
        generation_method: str,
        tenant_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """Create a workflow from NLP-generated configuration."""
        # Generate a name from the prompt
        name = self._generate_workflow_name(prompt)
        
        # Use tenant context if not provided
        if tenant_id is None:
            tenant_id = get_current_tenant_id()
        
        logger.info(f"Creating workflow with tenant_id: {tenant_id}")
        
        # Create the workflow directly in the database
        # Preserve all enhancement data in the metadata field
        base_metadata = {
            "generated_from": "nlp",
            "generation_method": generation_method,
            "original_prompt": prompt,
            "generated_at": datetime.utcnow().isoformat() + 'Z'
        }
        
        # Add enhancement data to metadata if present
        if "qgm_configuration" in workflow_config:
            base_metadata["qgm_configuration"] = workflow_config["qgm_configuration"]
        if "best_practices" in workflow_config:
            base_metadata["best_practices"] = workflow_config["best_practices"]
        if "tools_recommended" in workflow_config:
            base_metadata["tools_recommended"] = workflow_config["tools_recommended"]
        if "industry_insights" in workflow_config:
            base_metadata["industry_insights"] = workflow_config["industry_insights"]
        if "success_metrics" in workflow_config:
            base_metadata["success_metrics"] = workflow_config["success_metrics"]
        if "risk_assessment" in workflow_config:
            base_metadata["risk_assessment"] = workflow_config["risk_assessment"]
        if "implementation_guidance" in workflow_config:
            base_metadata["implementation_guidance"] = workflow_config["implementation_guidance"]
        if "enhancement_metadata" in workflow_config:
            base_metadata["enhancement_metadata"] = workflow_config["enhancement_metadata"]
        if "multi_tenant_config" in workflow_config:
            base_metadata["multi_tenant_config"] = workflow_config["multi_tenant_config"]
        if "advanced_governance" in workflow_config:
            base_metadata["advanced_governance"] = workflow_config["advanced_governance"]
        if "federation_config" in workflow_config:
            base_metadata["federation_config"] = workflow_config["federation_config"]
        if "ml_optimizations" in workflow_config:
            base_metadata["ml_optimizations"] = workflow_config["ml_optimizations"]
        if "execution_config" in workflow_config:
            base_metadata["execution_config"] = workflow_config["execution_config"]
        if "hub_connections" in workflow_config:
            base_metadata["hub_connections"] = workflow_config["hub_connections"]
        
        # Merge metadata from workflow_config if it exists (for expansion patterns, etc.)
        if "metadata" in workflow_config and isinstance(workflow_config["metadata"], dict):
            # Merge workflow metadata with base metadata
            base_metadata.update(workflow_config["metadata"])
        
        # Create workflow definition structure preserving ALL node data including agent and metadata
        # Use the workflow_config directly which already has the rich node structure
        definition = {
            "nodes": workflow_config.get("nodes", []),
            "edges": workflow_config.get("edges", []),
            "metadata": base_metadata
        }

        # Post-process branching: use LLM-based pattern detection for intelligent workflow structure
        definition = await self._post_process_branching_async(definition)

        # Log to verify rich metadata is preserved
        for node in definition["nodes"]:
            if node.get("type") == "ai_agent" or node.get("agent"):
                logger.info(f"Node {node['id']} has agent: {node.get('agent')}, has metadata: {bool(node.get('metadata'))}")
        
        logger.info(f"Workflow config keys: {list(workflow_config.keys())}")
        logger.info(f"Metadata contains enhancement: {'qgm_configuration' in base_metadata}")
        logger.info(f"Final definition keys before save: {list(definition.keys())}")
        logger.info(f"Metadata keys: {list(base_metadata.keys())}")
        
        # Apply unified enhancement pipeline (Business/Enterprise only)
        if self.edition in ['business', 'enterprise']:
            try:
                from aictrlnet_business.services.unified_enhancement_pipeline import unified_enhancement_pipeline
                
                # Create workflow dict for pipeline
                workflow_dict = {
                    "name": name,
                    "definition": definition
                }
                
                # Build enhancement config for NLP path
                enhancement_config = {
                    'source': 'nlp',
                    'automatic': True,  # NLP always applies all enhancements automatically
                    'context': context.copy() if context else {}
                }
                
                # Ensure essential context fields
                if "edition" not in enhancement_config['context']:
                    enhancement_config['context']["edition"] = self.edition
                if "tenant_id" not in enhancement_config['context']:
                    enhancement_config['context']["tenant_id"] = tenant_id
                if "generation_method" not in enhancement_config['context']:
                    enhancement_config['context']["generation_method"] = generation_method
                if "original_prompt" not in enhancement_config['context']:
                    enhancement_config['context']["original_prompt"] = prompt
                
                # Apply unified enhancement pipeline (includes defaults, AI, context, etc.)
                enhanced_workflow = await unified_enhancement_pipeline.enhance(
                    workflow_dict,
                    enhancement_config,
                    self.db
                )
                
                # Update definition with enhanced metadata
                definition = enhanced_workflow.get("definition", definition)
                
                logger.info(f"Applied unified enhancement pipeline. Metadata keys: {list(definition.get('metadata', {}).keys())}")
                
            except ImportError as e:
                logger.warning(f"Unified enhancement pipeline not available, falling back to defaults engine: {e}")
                # Fall back to just defaults engine
                try:
                    from aictrlnet_business.services.workflow_defaults_engine import workflow_defaults_engine
                    
                    workflow_dict = {
                        "name": name,
                        "definition": definition
                    }
                    
                    defaults_context = context.copy() if context else {}
                    if "edition" not in defaults_context:
                        defaults_context["edition"] = self.edition
                    if "tenant_id" not in defaults_context:
                        defaults_context["tenant_id"] = tenant_id
                    
                    workflow_dict = await workflow_defaults_engine.apply_defaults(workflow_dict, defaults_context)
                    definition = workflow_dict["definition"]
                    
                    logger.info(f"Applied defaults engine fallback. Metadata keys: {list(definition.get('metadata', {}).keys())}")
                    
                except Exception as fallback_e:
                    logger.warning(f"Could not apply defaults (non-critical): {fallback_e}")
                    
            except Exception as e:
                logger.warning(f"Could not apply enhancements (non-critical): {e}")
                # Continue without enhancements
        
        workflow = WorkflowDefinition(
            name=name,
            description=f"Generated from: {prompt[:200]}...",
            definition=definition,
            active=True,  # NLP workflows are active by default
            tags=["nlp_generated", generation_method],
            tenant_id=tenant_id or get_current_tenant_id()  # Ensure tenant_id is never None
        )
        
        self.db.add(workflow)
        await self.db.commit()
        await self.db.refresh(workflow)
        
        logger.info(f"Saved workflow definition keys: {list(workflow.definition.keys())}")
        
        # Track in learning loop for continuous improvement (Business/Enterprise only)
        if self.edition in ['business', 'enterprise']:
            try:
                from aictrlnet_business.services.learning_loop_service import LearningLoopService
                learning_service = LearningLoopService()
                
                # Track this workflow creation for pattern learning
                await learning_service.track_workflow_execution(
                    db=self.db,
                    workflow_id=workflow.id,
                    workflow_config=workflow.definition,
                    initiated_by='ai_agent',  # NLP is AI-initiated
                    initiator_id=None,  # System-generated
                    modifications_made={
                        'source': 'nlp',
                        'generation_method': generation_method,
                        'prompt': prompt[:500],  # Track first 500 chars of prompt
                        'enhancements_applied': workflow.definition.get('metadata', {}).get('enhancements_applied', [])
                    }
                )
                logger.info(f"✅ Tracked workflow creation in learning loop")
            except Exception as e:
                logger.warning(f"Could not track in learning loop (non-critical): {e}")
        
        # Phase 6: Connect metadata to hub features (Business/Enterprise only)
        if self.edition in ['business', 'enterprise'] and workflow.id:
            try:
                from aictrlnet_business.services.workflow_metadata_connector import workflow_metadata_connector
                
                # Convert workflow to dict format for connector
                workflow_dict = {
                    "id": str(workflow.id),
                    "name": workflow.name,
                    "definition": workflow.definition
                }
                
                connections = await workflow_metadata_connector.connect_metadata_to_hub_features(
                    workflow_dict,
                    self.db
                )
                
                logger.info(f"Connections returned: {connections}")
                logger.info(f"Features enabled: {connections.get('features_enabled') if connections else None}")
                
                # Add hub_connections to the workflow metadata
                if connections and connections.get("features_enabled"):
                    # Need to create a new dict to trigger SQLAlchemy change detection for JSON fields
                    updated_definition = dict(workflow.definition)
                    updated_definition["metadata"]["hub_connections"] = connections
                    workflow.definition = updated_definition
                    
                    # Mark as modified to ensure SQLAlchemy detects the change
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(workflow, "definition")
                    
                    self.db.add(workflow)
                    await self.db.commit()
                    await self.db.refresh(workflow)
                    logger.info(f"✅ Connected to {len(connections.get('hubs_connected', []))} hubs")
                    logger.info(f"Hub connections added to metadata: {list(workflow.definition['metadata'].get('hub_connections', {}).keys())}")
                    
            except Exception as e:
                logger.warning(f"Could not connect metadata to hubs: {e}")
                # Non-critical, continue without connections
        
        return workflow
    
    async def _create_fallback_workflow(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> WorkflowDefinition:
        """Create an intelligent fallback workflow based on intent analysis."""
        # Analyze intent even for fallback
        intent = await self._analyze_intent(prompt)
        
        # Create a more intelligent fallback based on intent
        nodes = [
            {
                "id": "start",
                "type": "start",
                "name": "Start",
                "position": {"x": 100, "y": 100}
            }
        ]
        
        edges = []
        x_pos = 300
        prev_node = "start"
        
        # Add intent-specific nodes
        if intent["category"] == "ai":
            # Add AI processing node
            node_id = "ai_process"
            nodes.append({
                "id": node_id,
                "type": "aiProcess",
                "name": "AI Processing",
                "position": {"x": x_pos, "y": 100},
                "data": {
                    "description": f"AI {intent['primary_action']} based on: {prompt[:50]}...",
                    "capability": "ai_processing"
                }
            })
            edges.append({"from": prev_node, "to": node_id})
            prev_node = node_id
            x_pos += 200
        
        elif intent["category"] == "business":
            # Add approval node
            node_id = "approval"
            nodes.append({
                "id": node_id,
                "type": "approval",
                "name": "Approval Step",
                "position": {"x": x_pos, "y": 100},
                "data": {
                    "description": f"Approval for: {prompt[:50]}...",
                    "capability": "approval_workflow"
                }
            })
            edges.append({"from": prev_node, "to": node_id})
            prev_node = node_id
            x_pos += 200
        
        elif intent["category"] == "communication":
            # Add notification node
            node_id = "notify"
            nodes.append({
                "id": node_id,
                "type": "adapter",
                "name": "Send Notification",
                "position": {"x": x_pos, "y": 100},
                "data": {
                    "description": f"Notification: {prompt[:50]}...",
                    "capability": "notification",
                    "channel": intent["entities"].get("channel", "email")
                }
            })
            edges.append({"from": prev_node, "to": node_id})
            prev_node = node_id
            x_pos += 200
        
        else:
            # Default process node
            node_id = "process"
            nodes.append({
                "id": node_id,
                "type": "process",
                "name": "Process Request",
                "position": {"x": x_pos, "y": 100},
                "data": {
                    "description": prompt[:100],
                    "requiresManualSetup": True,
                    "intent": intent
                }
            })
            edges.append({"from": prev_node, "to": node_id})
            prev_node = node_id
            x_pos += 200
        
        # Add end node
        nodes.append({
            "id": "end",
            "type": "end",
            "name": "End",
            "position": {"x": x_pos, "y": 100}
        })
        edges.append({"from": prev_node, "to": "end"})
        
        workflow_config = {
            "nodes": nodes,
            "edges": edges
        }
        
        return await self._create_workflow_from_nlp(
            prompt, workflow_config, "intelligent_fallback",
            tenant_id=context.get('tenant_id') or get_current_tenant_id() if context else get_current_tenant_id(),
            context=context
        )
    
    def _convert_steps_to_workflow(self, steps: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """Convert workflow steps to a complete workflow configuration."""
        nodes = [{
            "id": "start",
            "type": "start",
            "name": "Start",
            "position": {"x": 100, "y": 100}
        }]
        
        edges = []
        x_pos = 300
        y_pos = 100
        prev_node = "start"
        
        # Add nodes for each step
        for i, step in enumerate(steps):
            node_id = f"step_{i+1}"
            node_name = step.get("intent", {}).get("label", step["action"])
            
            nodes.append({
                "id": node_id,
                "type": step.get("node_type", "process"),
                "name": node_name,
                "position": {"x": x_pos, "y": y_pos},
                "data": {
                    "capability": step.get("capability"),
                    "parameters": step.get("parameters", {}),
                    "intent": step.get("intent"),
                    "description": step.get("intent", {}).get("description", "")
                }
            })
            
            # Handle dependencies
            if step.get("dependencies"):
                for dep in step["dependencies"]:
                    # Find the step with this action
                    dep_idx = next((j for j, s in enumerate(steps) if s["action"] == dep), None)
                    if dep_idx is not None:
                        edges.append({"from": f"step_{dep_idx+1}", "to": node_id})
            else:
                edges.append({"from": prev_node, "to": node_id})
            
            prev_node = node_id
            x_pos += 200
            
            # Stagger y position for parallel branches
            if i % 3 == 0:
                y_pos = 100
            elif i % 3 == 1:
                y_pos = 200
            else:
                y_pos = 300
        
        # Add end node
        nodes.append({
            "id": "end",
            "type": "end",
            "name": "End",
            "position": {"x": x_pos, "y": 100}
        })
        edges.append({"from": prev_node, "to": "end"})
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "generated_from": "template",
                "original_prompt": prompt
            }
        }
    
    def _generate_workflow_name(self, prompt: str) -> str:
        """Generate a workflow name from prompt.

        Extracts quoted names if present, otherwise uses first few words.
        Handles patterns like:
        - "Create a workflow named 'My Workflow' that..."
        - "Create workflow 'Legal Matter' for..."
        """
        import re

        # First, try to extract a quoted name from the prompt
        # Pattern: named 'X' or named "X" or called 'X' or called "X"
        quoted_pattern = r"(?:named|called)\s+['\"]([^'\"]+)['\"]"
        match = re.search(quoted_pattern, prompt, re.IGNORECASE)

        if match:
            name = match.group(1).strip()
        else:
            # Fallback: take first few words, but skip common prefixes
            words = prompt.split()

            # Skip common instructional prefixes
            skip_words = {'create', 'a', 'an', 'the', 'workflow', 'named', 'called', 'that', 'for', 'to'}
            meaningful_words = []
            started = False

            for word in words[:10]:
                word_lower = word.lower().strip("'\"")
                if word_lower in skip_words and not started:
                    continue
                started = True
                meaningful_words.append(word)
                if len(meaningful_words) >= 5:
                    break

            if meaningful_words:
                name = " ".join(meaningful_words)
            else:
                # Ultimate fallback
                name = " ".join(words[:5])

        # Clean up any remaining quotes
        name = name.strip("'\" \t")

        # Ensure reasonable length
        if len(name) > 50:
            name = name[:47] + "..."

        return name.title()
    
    def _extract_workflow_steps(self, prompt: str) -> List[Dict[str, Any]]:
        """Extract workflow steps from natural language using pattern matching."""
        import re
        
        steps = []
        prompt_lower = prompt.lower()
        
        # Approval workflow pattern
        if 'approval' in prompt_lower or 'approve' in prompt_lower:
            if 'ai' in prompt_lower and 'compliance' in prompt_lower:
                # AI compliance check
                steps.append({
                    "action": "check_compliance",
                    "intent": {
                        "label": "AI agent checks for compliance issues",
                        "originalText": "AI agent checks for compliance issues",
                        "description": "AI reviews documents for compliance violations"
                    },
                    "node_type": "aiProcess",
                    "capability": "ai_processing"
                })
                
                # Conditional routing
                steps.append({
                    "action": "route_decision",
                    "intent": {
                        "label": "Route based on compliance check",
                        "originalText": "routes to a human approver if issues are found",
                        "description": "Conditional routing based on compliance results"
                    },
                    "node_type": "decision",
                    "capability": "conditional_logic",
                    "dependencies": ["check_compliance"]
                })
                
                # Human approval path
                steps.append({
                    "action": "human_approval",
                    "intent": {
                        "label": "Human approver reviews",
                        "originalText": "human approver",
                        "description": "Human reviews and approves when issues found"
                    },
                    "node_type": "approval",
                    "capability": "human_approval",
                    "dependencies": ["route_decision"]
                })
                
                # Auto approval path
                steps.append({
                    "action": "auto_approve",
                    "intent": {
                        "label": "Auto-approve if compliant",
                        "originalText": "auto-approves if compliant",
                        "description": "Automatic approval when no issues detected"
                    },
                    "node_type": "approval",
                    "capability": "automated_approval",
                    "dependencies": ["route_decision"]
                })
                
                return steps
        
        # Data processing pipeline pattern
        if any(word in prompt_lower for word in ['etl', 'pipeline', 'data processing', 'transform']):
            steps.extend([
                {
                    "action": "extract_data",
                    "intent": {"label": "Extract data from source"},
                    "node_type": "dataSource",
                    "capability": "data_extraction"
                },
                {
                    "action": "validate_data",
                    "intent": {"label": "Validate data quality"},
                    "node_type": "process",
                    "capability": "data_validation",
                    "dependencies": ["extract_data"]
                },
                {
                    "action": "transform_data",
                    "intent": {"label": "Transform data"},
                    "node_type": "transformer",
                    "capability": "data_transformation",
                    "dependencies": ["validate_data"]
                },
                {
                    "action": "load_data",
                    "intent": {"label": "Load to destination"},
                    "node_type": "output",
                    "capability": "data_loading",
                    "dependencies": ["transform_data"]
                }
            ])
            return steps
        
        # Customer support pattern
        if 'customer' in prompt_lower and 'support' in prompt_lower:
            steps.extend([
                {
                    "action": "receive_inquiry",
                    "intent": {"label": "Receive customer inquiry"},
                    "node_type": "input",
                    "capability": "data_input"
                },
                {
                    "action": "classify_issue",
                    "intent": {"label": "AI classifies the issue"},
                    "node_type": "aiProcess",
                    "capability": "classification",
                    "dependencies": ["receive_inquiry"]
                },
                {
                    "action": "search_knowledge",
                    "intent": {"label": "Search knowledge base"},
                    "node_type": "dataSource",
                    "capability": "knowledge_search",
                    "dependencies": ["classify_issue"]
                },
                {
                    "action": "generate_response",
                    "intent": {"label": "AI generates response"},
                    "node_type": "aiProcess",
                    "capability": "text_generation",
                    "dependencies": ["search_knowledge"]
                },
                {
                    "action": "send_response",
                    "intent": {"label": "Send response to customer"},
                    "node_type": "output",
                    "capability": "notification",
                    "dependencies": ["generate_response"]
                }
            ])
            return steps
        
        # If no specific pattern matches, extract general action phrases
        action_patterns = [
            r'(?:first|initially|begin by|start with)\s+([^,\.]+?)(?=\s*(?:,|\.|then|$))',
            r'(?:then|next|after that)\s+([^,\.]+?)(?=\s*(?:,|\.|then|finally|$))',
            r'(?:finally|lastly|at the end)\s+([^,\.]+?)(?=\s*(?:,|\.|$))',
            r'((?:analyze|process|validate|check|send|notify|transform|extract|load)\s+[^,\.]+?)(?=\s*(?:,|and\s+|then\s+|$))'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                action_text = match.strip()
                if action_text and len(action_text) > 5:
                    steps.append({
                        "action": action_text.lower().replace(' ', '_'),
                        "intent": {
                            "label": action_text,
                            "originalText": action_text
                        },
                        "node_type": self._infer_node_type(action_text),
                        "capability": self._infer_capability(action_text)
                    })
        
        return steps
    
    def _steps_to_workflow_config(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert workflow steps to a complete workflow configuration with rich AI agent nodes."""
        nodes = [{
            "id": "start",
            "type": "start",
            "name": "Start",
            "position": {"x": 100, "y": 100}
        }]

        edges = []
        x_pos = 300
        y_pos = 100

        # Track which nodes have outgoing edges to find terminal nodes
        nodes_with_outgoing = set()
        all_node_ids = []

        # Add nodes for each step
        for i, step in enumerate(steps):
            node_id = f"node_{i+1}"
            all_node_ids.append(node_id)

            # Extract AI agent information from step
            agent_name = self._extract_agent_from_step(step)

            # Determine node type with proper precedence:
            # 1. Explicit node_type from step (LLM-generated)
            # 2. Inferred from action text (for decision, approval, etc.)
            # 3. AI agent if agent assigned
            # 4. Default to process
            is_ai_agent = agent_name is not None
            explicit_type = step.get("node_type") or step.get("type")

            # Special node types that shouldn't be overridden
            special_types = {"decision", "conditional", "parallel", "approval", "humanAgent", "merge"}

            if explicit_type and explicit_type.lower() in special_types:
                node_type = explicit_type.lower()
            else:
                # Try to infer from action text
                inferred_type = self._infer_node_type(step.get("action", ""))
                if inferred_type in special_types:
                    node_type = inferred_type
                elif is_ai_agent:
                    node_type = "ai_agent"
                else:
                    node_type = explicit_type if explicit_type else "process"

            # Generate rich metadata for the node
            metadata = self._generate_node_metadata(step, agent_name)

            # Get sensible node name (Community edition - rule-based improvement)
            node_name = self._generate_sensible_node_name(step, i, len(steps))

            node_data = {
                "id": node_id,
                "type": node_type,
                "name": node_name,
                "position": {"x": x_pos, "y": y_pos},
                "data": {
                    "action": step["action"],
                    "capability": step.get("capability"),
                    "intent": step.get("intent"),
                    "parameters": step.get("parameters", {})
                }
            }
            
            # Add AI agent information if this is an AI agent node
            if is_ai_agent:
                node_data["agent"] = agent_name  # This will be resolved by UnifiedAgentResolver
                node_data["data"]["agent_type"] = "intelligence"
                node_data["data"]["ai_required"] = True

            # Preserve branches for decision/parallel nodes (for _post_process_branching)
            if node_type in ("decision", "conditional", "parallel"):
                branches = step.get("branches", [])
                if branches:
                    node_data["branches"] = branches
                    node_data["data"]["branches"] = branches
                # Also preserve condition if present
                condition = step.get("condition")
                if condition:
                    node_data["data"]["condition"] = condition

            # Add rich metadata to the node
            node_data["metadata"] = metadata
            
            nodes.append(node_data)
            
            # Handle dependencies
            if step.get("dependencies"):
                for dep in step["dependencies"]:
                    # Find the node with this action
                    dep_idx = next((j for j, s in enumerate(steps) if s["action"] == dep), None)
                    if dep_idx is not None:
                        from_node = f"node_{dep_idx+1}"
                        edges.append({"from": from_node, "to": node_id})
                        nodes_with_outgoing.add(from_node)
            else:
                # Connect to previous node if no explicit dependencies
                if i == 0:
                    edges.append({"from": "start", "to": node_id})
                    nodes_with_outgoing.add("start")
                else:
                    from_node = f"node_{i}"
                    edges.append({"from": from_node, "to": node_id})
                    nodes_with_outgoing.add(from_node)
            
            # Update position for next node
            x_pos += 200
            # Stagger y position to avoid overlapping for long workflows
            if i % 4 == 3:  # Every 4th node, drop down
                y_pos += 150
                x_pos = 300  # Reset x position
        
        # Add end node
        nodes.append({
            "id": "end",
            "type": "end",
            "name": "End",
            "position": {"x": x_pos, "y": y_pos}
        })
        
        # Connect all terminal nodes (nodes without outgoing edges) to end
        if steps:
            terminal_nodes = [node_id for node_id in all_node_ids if node_id not in nodes_with_outgoing]
            if terminal_nodes:
                for terminal_node in terminal_nodes:
                    edges.append({"from": terminal_node, "to": "end"})
            else:
                # Fallback: if no terminal nodes found, connect the last node
                edges.append({"from": f"node_{len(steps)}", "to": "end"})
        else:
            edges.append({"from": "start", "to": "end"})
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _generate_sensible_node_name(self, step: Dict[str, Any], index: int, total_steps: int) -> str:
        """
        Generate sensible node names for Community edition.

        Priority order (Domain-First approach per WORKFLOW_CREATION_IMPLEMENTATION_SPEC):
        1. Trust LLM-generated labels if they are domain-specific (not generic)
        2. Fall back to rule-based naming only for truly generic labels

        This preserves domain-specific workflow structure from RAG-enhanced generation.
        """
        import re

        # Get the label from the step - handle both nested and flat formats
        # Steps from _extract_workflow_steps use nested format: step["intent"]["label"]
        # LLM-generated steps may use flat format: step["label"]
        intent = step.get("intent", {})
        if isinstance(intent, dict):
            label = intent.get("label", "") or step.get("label", "")
            description = intent.get("description", "") or step.get("description", "")
        else:
            label = step.get("label", "")
            description = step.get("description", "")

        action = step.get("action", "")
        capability = step.get("capability", "")

        # Define patterns that indicate generic/placeholder labels
        # These should be replaced with rule-based names
        generic_patterns = [
            r'^step[_\s]?\d+$',           # step_1, step 1, Step_1, etc.
            r'^\d+\.?\s*$',               # Just numbers: 1, 2., etc.
            r'^task[_\s]?\d+$',           # task_1, task 1, etc.
            r'^node[_\s]?\d+$',           # node_1, node 1, etc.
            r'^action[_\s]?\d+$',         # action_1, etc.
            r'^process[_\s]?\d+$',        # process_1, etc.
            r'^step$',                    # Just "step"
            r'^task$',                    # Just "task"
            r'^untitled',                 # Untitled, untitled_1, etc.
            r'^unnamed',                  # Unnamed, etc.
        ]

        # Check if label is meaningful (not generic)
        if label:
            label_lower = label.lower().strip()
            is_generic = any(re.match(pattern, label_lower, re.IGNORECASE) for pattern in generic_patterns)

            # Also check for very short labels (likely generic)
            is_too_short = len(label_lower) < 3

            # Trust the LLM label if it's not generic and has reasonable length
            if not is_generic and not is_too_short:
                # Clean up and title-case if needed
                if label == label.lower():
                    return label.title()
                return label

        # Check if description can be used as a better name
        if description and len(description) > 5 and len(description) < 50:
            desc_lower = description.lower()
            is_generic_desc = any(re.match(pattern, desc_lower, re.IGNORECASE) for pattern in generic_patterns)
            if not is_generic_desc:
                # Use first sentence/clause of description
                first_part = description.split('.')[0].split(',')[0].strip()
                if len(first_part) > 3 and len(first_part) < 40:
                    return first_part.title() if first_part == first_part.lower() else first_part

        # Fall back to rule-based naming based on action/capability
        # Analyze action/capability to determine semantic role
        action_lower = (action + " " + capability).lower()

        # Pattern-based naming (generic but sensible)
        if any(word in action_lower for word in ['start', 'begin', 'initialize', 'input', 'receive']):
            return "Initialize Workflow"
        elif any(word in action_lower for word in ['validate', 'check', 'verify']):
            return "Validate Input"
        elif any(word in action_lower for word in ['extract', 'parse', 'read']):
            return "Extract Data"
        elif any(word in action_lower for word in ['transform', 'convert', 'map', 'format']):
            return "Transform Data"
        elif any(word in action_lower for word in ['analyze', 'assess', 'evaluate', 'calculate']):
            return "Analyze Information"
        elif any(word in action_lower for word in ['classify', 'categorize', 'sort']):
            return "Classify Data"
        elif any(word in action_lower for word in ['decide', 'route', 'branch']):
            return "Decision Point"
        elif any(word in action_lower for word in ['approve', 'approval', 'review']):
            return "Approval Step"
        elif any(word in action_lower for word in ['notify', 'send', 'email', 'message', 'alert']):
            return "Send Notification"
        elif any(word in action_lower for word in ['store', 'save', 'persist', 'write']):
            return "Store Results"
        elif any(word in action_lower for word in ['generate', 'create', 'produce']):
            return "Generate Output"
        elif any(word in action_lower for word in ['monitor', 'track', 'log']):
            return "Monitor Process"
        elif any(word in action_lower for word in ['complete', 'finish', 'end', 'finalize']):
            return "Complete Workflow"

        # Positional naming as fallback (but better than step_1, step_2)
        if index < total_steps * 0.2:
            return f"Process Input {index + 1}"
        elif index < total_steps * 0.4:
            return f"Transform Data {index + 1}"
        elif index < total_steps * 0.6:
            return f"Execute Action {index + 1}"
        elif index < total_steps * 0.8:
            return f"Verify Results {index + 1}"
        else:
            return f"Finalize Process {index + 1}"

    def _infer_node_type(self, action_text: str) -> str:
        """Infer node type from action text.

        Enhanced to detect a comprehensive set of workflow patterns including:
        - Control flow: decision, loop, parallel, join
        - Error handling: error_handler, retry, compensate
        - Human interaction: human_task, approval, manual_review
        - Integration: api_call, webhook, database, mcp
        - Processing: transform, aggregate, quality_check
        """
        action_lower = action_text.lower()

        # === CONTROL FLOW NODES ===

        # Decision/branching detection - check first as these are structural
        if any(word in action_lower for word in [
            'decide', 'decision', 'check if', 'if ', 'when', 'branch', 'route',
            'conditional', 'evaluate', 'determine', 'based on', 'depending on',
            'either', 'or else', 'otherwise', 'switch', 'case'
        ]):
            return "decision"

        # Loop/iteration detection (Business Edition+)
        if any(word in action_lower for word in [
            'loop', 'iterate', 'for each', 'foreach', 'for all', 'process each',
            'process all', 'repeat', 'batch', 'bulk', 'multiple', 'every item',
            'each record', 'all items', 'one by one', 'sequentially process'
        ]):
            return "loop"

        # Parallel execution detection (Business Edition+)
        if any(word in action_lower for word in [
            'parallel', 'concurrent', 'simultaneously', 'at the same time',
            'in parallel', 'fan out', 'fork', 'split into', 'multiple branches'
        ]):
            return "parallel"

        # Join/synchronization detection
        if any(word in action_lower for word in [
            'join', 'synchronize', 'wait for all', 'merge results', 'fan in',
            'converge', 'collect results', 'combine branches'
        ]):
            return "join"

        # === ERROR HANDLING NODES ===

        # Error handler detection
        if any(word in action_lower for word in [
            'error handler', 'handle error', 'on error', 'catch error',
            'error handling', 'exception', 'failure handler', 'on failure'
        ]):
            return "error_handler"

        # Retry detection
        if any(word in action_lower for word in [
            'retry', 'try again', 'reattempt', 'retry on failure',
            'exponential backoff', 'retry with delay'
        ]):
            return "retry"

        # Compensation/rollback detection
        if any(word in action_lower for word in [
            'compensate', 'rollback', 'undo', 'revert', 'cancel',
            'compensation', 'cleanup on failure'
        ]):
            return "compensate"

        # === HUMAN INTERACTION NODES ===

        # Human task detection (Business Edition+)
        if any(word in action_lower for word in [
            'human task', 'assign to', 'assign task', 'manual task',
            'escalate to', 'create task for', 'await human', 'human input'
        ]):
            return "human_task"

        # Approval detection
        if any(word in action_lower for word in [
            'approve', 'approval', 'get approval', 'request approval',
            'sign off', 'authorize', 'authorization required'
        ]):
            return "approval"

        # Manual review detection
        if any(word in action_lower for word in [
            'manual review', 'human review', 'review by', 'person review',
            'quality review', 'manual check', 'human verification'
        ]):
            return "manual_review"

        # Wait for input
        if any(word in action_lower for word in [
            'wait for', 'await', 'pause until', 'hold until',
            'wait for response', 'pending input'
        ]):
            return "wait_message"

        # === AI/ML NODES ===

        # AI processing detection
        if any(word in action_lower for word in [
            'ai', 'classify', 'analyze with ai', 'predict', 'generate',
            'machine learning', 'ml', 'nlp', 'llm', 'summarize',
            'sentiment', 'extract entities', 'ai agent', 'intelligent'
        ]):
            return "ai_process"

        # === QUALITY & GOVERNANCE NODES ===

        # Quality check detection
        if any(word in action_lower for word in [
            'quality check', 'validate', 'validation', 'quality assurance',
            'qa check', 'data quality', 'verify data', 'check quality'
        ]):
            return "quality_check"

        # Policy/compliance check (Enterprise Edition+)
        if any(word in action_lower for word in [
            'policy check', 'compliance', 'compliance check', 'audit',
            'governance', 'regulatory', 'policy validation'
        ]):
            return "policy_check"

        # Risk assessment (Enterprise Edition+)
        if any(word in action_lower for word in [
            'risk assessment', 'assess risk', 'risk analysis', 'risk score',
            'evaluate risk', 'risk evaluation'
        ]):
            return "risk_assessment"

        # === INTEGRATION NODES ===

        # API call detection
        if any(word in action_lower for word in [
            'api call', 'call api', 'http request', 'rest api', 'api request',
            'post to', 'get from api', 'call endpoint', 'invoke api'
        ]):
            return "api_call"

        # Webhook detection
        if any(word in action_lower for word in [
            'webhook', 'trigger webhook', 'send webhook', 'webhook notification'
        ]):
            return "webhook"

        # Database detection
        if any(word in action_lower for word in [
            'database', 'db', 'query database', 'insert into', 'update database',
            'sql', 'store in database', 'save to database', 'fetch from database'
        ]):
            return "database"

        # MCP tool detection
        if any(word in action_lower for word in [
            'mcp', 'mcp tool', 'external tool', 'tool server',
            'use tool', 'execute tool'
        ]):
            return "mcp_tool_execute"

        # === DATA PROCESSING NODES ===

        # Transform detection
        if any(word in action_lower for word in [
            'transform', 'convert', 'map', 'format', 'restructure',
            'normalize', 'denormalize', 'reshape'
        ]):
            return "transform"

        # Aggregate detection
        if any(word in action_lower for word in [
            'aggregate', 'combine', 'merge', 'collect', 'sum up',
            'consolidate', 'gather', 'combine data'
        ]):
            return "aggregate"

        # Extract/data source detection
        if any(word in action_lower for word in [
            'extract', 'fetch', 'retrieve', 'get data', 'read from',
            'load', 'pull', 'source data'
        ]):
            return "dataSource"

        # === MESSAGING NODES ===

        # Notification/send message detection
        if any(word in action_lower for word in [
            'send', 'notify', 'email', 'sms', 'slack', 'notification',
            'alert', 'message', 'push notification'
        ]):
            return "send_message"

        # Default to process
        return "process"
    
    def _infer_capability(self, action_text: str) -> str:
        """Infer capability from action text.

        Enhanced to match the expanded node type detection.
        """
        action_lower = action_text.lower()

        # Control flow capabilities
        if any(word in action_lower for word in ['decide', 'branch', 'route', 'conditional']):
            return "conditional_routing"
        if any(word in action_lower for word in ['loop', 'iterate', 'for each', 'batch']):
            return "iteration"
        if any(word in action_lower for word in ['parallel', 'concurrent', 'simultaneously']):
            return "parallel_execution"

        # Error handling capabilities
        if any(word in action_lower for word in ['error', 'handle', 'catch', 'exception']):
            return "error_handling"
        if any(word in action_lower for word in ['retry', 'reattempt']):
            return "retry_logic"
        if any(word in action_lower for word in ['rollback', 'compensate', 'undo']):
            return "compensation"

        # Human interaction capabilities
        if any(word in action_lower for word in ['approve', 'approval']):
            return "approval_workflow"
        if any(word in action_lower for word in ['human', 'manual', 'escalate']):
            return "human_task"
        if any(word in action_lower for word in ['review', 'check']):
            return "manual_review"

        # AI/ML capabilities
        if any(word in action_lower for word in ['ai', 'analyze', 'predict', 'classify', 'ml']):
            return "ai_processing"

        # Quality/Governance capabilities
        if any(word in action_lower for word in ['validate', 'quality', 'qa']):
            return "data_validation"
        if any(word in action_lower for word in ['compliance', 'policy', 'audit']):
            return "compliance_check"
        if any(word in action_lower for word in ['risk']):
            return "risk_assessment"

        # Integration capabilities
        if any(word in action_lower for word in ['api', 'http', 'rest']):
            return "api_integration"
        if any(word in action_lower for word in ['webhook']):
            return "webhook_trigger"
        if any(word in action_lower for word in ['database', 'db', 'sql']):
            return "database_operation"
        if any(word in action_lower for word in ['mcp', 'tool']):
            return "tool_execution"

        # Data processing capabilities
        if any(word in action_lower for word in ['transform', 'convert', 'map']):
            return "data_transformation"
        if any(word in action_lower for word in ['aggregate', 'combine', 'merge']):
            return "data_aggregation"
        if any(word in action_lower for word in ['extract', 'fetch', 'retrieve']):
            return "data_extraction"

        # Messaging capabilities
        if any(word in action_lower for word in ['notify', 'send', 'email', 'sms', 'alert']):
            return "notification"

        return "data_processing"
    
    async def _generate_with_ai_enhanced(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Enhanced AI generation that returns additional metadata."""
        workflow = await self._generate_with_ai(prompt, context, user_id)
        if workflow:
            # Extract model metadata
            # _model_used should always be set now (no more pattern matching)
            model_used = workflow.pop('_model_used', settings.DEFAULT_LLM_MODEL)
            model_tier = workflow.pop('_model_tier', 'balanced')

            return {
                'workflow': workflow,
                'model': model_used,
                'model_tier': model_tier,
                'provider': 'ollama'
            }
        return None
    
    async def _enhance_with_templates_tracked(
        self, 
        prompt: str, 
        ai_workflow: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Enhance workflow with templates and track which were used."""
        enhanced = await self._enhance_with_templates(prompt, ai_workflow, context)
        if not enhanced:
            return None
            
        # Track templates used
        intent = await self._analyze_intent(prompt)
        templates = await self.template_service.get_templates_by_category_async(
            self.db, intent.get("category", "general")
        )
        
        templates_used = []
        if templates:
            template = templates[0]
            templates_used.append(TemplateMatch(
                template_id=template.get('id', 'unknown'),
                template_name=template.get('name', 'Unknown'),
                confidence=0.8,
                category=template.get('category', 'general'),
                tags=template.get('tags', [])
            ))
        
        return {
            'workflow': enhanced,
            'templates_used': templates_used
        }
    
    async def _match_templates_tracked(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Match templates and track detailed information."""
        intent = await self._analyze_intent(prompt)
        
        # Use AI to find the best matching template
        template = await self._match_template_with_ai(prompt, intent)
        
        templates_used = []
        extracted_parameters = []
        
        if template:
            # Track template match (safely handle missing 'id' field)
            template_id = template.get('id') or template.get('name', 'unknown')
            templates_used.append(TemplateMatch(
                template_id=template_id,
                template_name=template.get('name', 'Unknown Template'),
                confidence=intent.get('confidence', 0.7),
                category=template.get('category', 'general'),
                tags=template.get('tags', [])
            ))
            
            # Extract real parameters from prompt
            extracted_parameters = self.extract_real_parameters(prompt, template)
            
            # Generate workflow from template using the template service
            workflow_config = await self.template_service.generate_workflow_for_template(template)
            
            return {
                'workflow': workflow_config,
                'templates_used': templates_used,
                'parameters': extracted_parameters
            }
        
        # Fallback to basic templates
        workflow = await self._match_templates(prompt, context)
        if workflow:
            return {
                'workflow': workflow,
                'templates_used': templates_used,
                'parameters': extracted_parameters
            }
        
        return None
    
    async def _build_transparency_response(
        self,
        workflow: WorkflowDefinition,
        generation_method: str,
        templates_used: List[TemplateMatch],
        extracted_parameters: List[ExtractedParameter],
        intent_analysis: Dict[str, Any],
        confidence_score: float,
        processing_steps: List[Dict[str, Any]],
        ai_model_used: Optional[str],
        original_prompt: str
    ) -> Dict[str, Any]:
        """Build the complete transparency response."""
        # Analyze edition requirements
        edition_requirements = await self._analyze_edition_requirements(workflow)
        
        # Get current edition from instance (set during initialization from environment)
        current_edition = self.edition
        
        # Generate upgrade suggestions
        upgrade_suggestions = self._generate_upgrade_suggestions(
            edition_requirements, current_edition, intent_analysis
        )
        
        # Get related templates
        related_templates = await self._get_related_templates(
            intent_analysis.get('category', 'general'),
            templates_used
        )
        
        # Get alternative templates
        alternative_templates = await self._get_alternative_templates(
            intent_analysis.get('category', 'general'),
            templates_used
        )
        
        # Build response
        workflow_response = WorkflowResponse.model_validate(workflow)
        
        return NLPWorkflowResponse(
            workflow=workflow_response,
            generation_method=generation_method,
            templates_used=templates_used,
            extracted_parameters=extracted_parameters,
            intent_analysis=intent_analysis,
            confidence_score=confidence_score,
            edition_requirements=edition_requirements,
            current_edition=current_edition,
            upgrade_suggestions=upgrade_suggestions,
            related_templates=related_templates,
            alternative_templates=alternative_templates,
            processing_steps=processing_steps,
            ai_model_used=ai_model_used
        ).model_dump()
    
    async def _analyze_edition_requirements(
        self, 
        workflow: WorkflowDefinition
    ) -> List[EditionRequirement]:
        """Analyze which edition features are required."""
        requirements = []
        
        # Get the workflow definition
        definition = workflow.definition
        nodes = definition.get('nodes', [])
        
        # Analyze each node
        for node in nodes:
            node_type = node.get('type')
            node_name = node.get('name', 'Unknown')
            
            # Community features
            if node_type in ['start', 'end', 'process', 'input', 'output']:
                requirements.append(EditionRequirement(
                    feature=f"{node_name} ({node_type})",
                    required_edition="community",
                    reason="Basic workflow functionality",
                    current_availability=True
                ))
            
            # Business features
            elif node_type in ['approval', 'decision', 'humanAgent', 'adapter']:
                requirements.append(EditionRequirement(
                    feature=f"{node_name} ({node_type})",
                    required_edition="business",
                    reason="Advanced business process features",
                    current_availability=False  # Would check actual edition
                ))
            
            # Enterprise features
            elif node_type in ['aiProcess', 'mcpNode', 'transformer', 'dataSource']:
                requirements.append(EditionRequirement(
                    feature=f"{node_name} ({node_type})",
                    required_edition="enterprise",
                    reason="AI and advanced data processing capabilities",
                    current_availability=False  # Would check actual edition
                ))
        
        return requirements
    
    def _generate_upgrade_suggestions(
        self,
        edition_requirements: List[EditionRequirement],
        current_edition: str,
        intent_analysis: Dict[str, Any]
    ) -> List[UpgradeSuggestion]:
        """Generate upgrade suggestions based on requirements."""
        suggestions = []
        
        # Count features by edition
        business_features = [r for r in edition_requirements if r.required_edition == "business"]
        enterprise_features = [r for r in edition_requirements if r.required_edition == "enterprise"]
        
        # Suggest Business upgrade if on Community
        if current_edition == "community" and business_features:
            suggestions.append(UpgradeSuggestion(
                target_edition="business",
                features_unlocked=[f.feature for f in business_features],
                reason=f"Unlock {len(business_features)} business process features for your workflow",
                priority="high" if len(business_features) > 2 else "medium"
            ))
        
        # Suggest Enterprise upgrade if features needed
        if current_edition in ["community", "business"] and enterprise_features:
            suggestions.append(UpgradeSuggestion(
                target_edition="enterprise",
                features_unlocked=[f.feature for f in enterprise_features],
                reason=f"Enable {len(enterprise_features)} AI and advanced features",
                priority="high" if intent_analysis.get('category') == 'ai' else "medium"
            ))
        
        return suggestions
    
    async def _get_related_templates(
        self,
        category: str,
        used_templates: List[TemplateMatch]
    ) -> List[TemplatePreview]:
        """Get related templates for customization."""
        # Get templates from the same category
        category_templates = await self.template_service.get_templates_by_category_async(self.db, category)
        
        # Filter out already used templates
        used_ids = {t.template_id for t in used_templates}
        related = []
        
        for template in category_templates[:3]:  # Limit to 3 related
            if template['id'] not in used_ids:
                # Convert UUID to string if needed
                template_id = str(template['id']) if hasattr(template['id'], 'hex') else template['id']
                related.append(TemplatePreview(
                    id=template_id,
                    name=template['name'],
                    description=template['description'],
                    category=template['category'],
                    tags=template['tags'],
                    node_count=template['node_count'],
                    complexity=template['complexity'],
                    edition_required=template.get('edition_required', 'community'),
                    preview_available=template.get('preview_available', False),
                    example_use_cases=template.get('example_use_cases', [])
                ))
        
        return related
    
    async def _get_alternative_templates(
        self,
        category: str,
        used_templates: List[TemplateMatch]
    ) -> List[TemplatePreview]:
        """Get alternative templates from different categories."""
        # Get templates from different categories
        from schemas.workflow_templates import TemplateListRequest
        request = TemplateListRequest(
            include_system=True,
            include_public=True,
            include_private=False,
            limit=1000
        )
        all_templates, _ = await self.template_service.list_templates(self.db, "system", request)
        all_templates = [template.model_dump() for template in all_templates]
        
        # Filter out same category and used templates
        used_ids = {t.template_id for t in used_templates}
        alternatives = []
        
        for template in all_templates:
            if (template['category'] != category and 
                template['id'] not in used_ids and
                len(alternatives) < 3):
                
                # Create basic preview from template data
                # Convert UUID to string if needed
                template_id = str(template['id']) if hasattr(template['id'], 'hex') else template['id']
                preview = {
                    'id': template_id,
                    'name': template['name'],
                    'description': template['description'],
                    'category': template['category'],
                    'tags': template.get('tags', []),
                    'node_count': len(template.get('nodes', [])),
                    'complexity': template.get('complexity', 'moderate'),
                    'edition_required': template.get('edition', 'community'),
                    'preview_available': True,
                    'example_use_cases': template.get('example_use_cases', [])
                }
                if preview:
                    alternatives.append(TemplatePreview(
                        id=preview['id'],
                        name=preview['name'],
                        description=preview['description'],
                        category=preview['category'],
                        tags=preview['tags'],
                        node_count=preview['node_count'],
                        complexity=preview['complexity'],
                        edition_required=preview['edition_required'],
                        preview_available=preview['preview_available'],
                        example_use_cases=preview.get('example_use_cases', [])
                    ))
        
        return alternatives
    
    def extract_real_parameters(self, prompt: str, template: Dict[str, Any]) -> List[ExtractedParameter]:
        """Extract real parameters from user prompt based on template requirements."""
        parameters = []
        prompt_lower = prompt.lower()
        
        # Common parameter patterns
        param_patterns = {
            'threshold': r'threshold[:\s]+(\d+(?:\.\d+)?%?)',
            'limit': r'limit[:\s]+(\d+)',
            'timeout': r'timeout[:\s]+(\d+)\s*(seconds?|minutes?|hours?)?',
            'channel': r'(email|sms|slack|teams|push)',
            'priority': r'(high|medium|low)\s*priority',
            'risk_level': r'(high|medium|low)\s*risk',
            'approval_levels': r'(\d+)\s*levels?\s*of\s*approval',
            'schedule': r'(daily|weekly|monthly|hourly)',
            'format': r'format[:\s]+(json|xml|csv|pdf)',
        }
        
        import re
        
        for param_name, pattern in param_patterns.items():
            match = re.search(pattern, prompt_lower, re.IGNORECASE)
            if match:
                value = match.group(1)
                # Clean up value
                if param_name == 'threshold' and '%' in value:
                    value = float(value.rstrip('%')) / 100
                elif param_name in ['limit', 'approval_levels']:
                    value = int(value)
                elif param_name == 'timeout':
                    # Convert to seconds
                    unit = match.group(2) if match.lastindex > 1 else 'seconds'
                    multiplier = {'hours': 3600, 'minutes': 60, 'seconds': 1}.get(unit.rstrip('s'), 1)
                    value = int(value) * multiplier
                
                parameters.append(ExtractedParameter(
                    name=param_name,
                    value=value,
                    confidence=0.9,
                    source='explicit'
                ))
        
        # Extract from entities in intent
        entities = self._extract_entities(prompt)
        for key, value in entities.items():
            if not any(p.name == key for p in parameters):
                parameters.append(ExtractedParameter(
                    name=key,
                    value=value,
                    confidence=0.7,
                    source='inferred'
                ))
        
        # Add default parameters from template if needed
        if template and 'default_parameters' in template:
            for key, value in template['default_parameters'].items():
                if not any(p.name == key for p in parameters):
                    parameters.append(ExtractedParameter(
                        name=key,
                        value=value,
                        confidence=0.5,
                        source='default'
                    ))
        
        return parameters
    
    async def _simple_template_match(self, prompt: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simple keyword-based template matching fallback."""
        try:
            # Get templates matching the prompt keywords
            from schemas.workflow_templates import TemplateListRequest
            
            # Extract key words from prompt
            prompt_words = set(word.lower() for word in prompt.split() if len(word) > 3)
            
            # Try to find templates with matching keywords
            request = TemplateListRequest(
                include_system=True,
                include_public=True,
                include_private=False,
                limit=50
            )
            
            templates, _ = await self.template_service.list_templates(self.db, "system", request)
            
            best_match = None
            best_score = 0
            
            for template in templates:
                template_dict = template.model_dump()
                template_text = (
                    template_dict.get('name', '').lower() + ' ' +
                    template_dict.get('description', '').lower() + ' ' +
                    ' '.join(template_dict.get('tags', []))
                ).lower()
                
                # Score based on word matches
                score = sum(1 for word in prompt_words if word in template_text)
                
                # Boost score for category match
                if template_dict.get('category') in prompt.lower():
                    score += 2
                
                # Boost score for entity matches
                for entity_value in entities.values():
                    if str(entity_value).lower() in template_text:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = template_dict
            
            if best_match and best_score > 0:
                logger.info(f"Found template match: {best_match['name']} (score: {best_score})")
                return await self.template_service.generate_workflow_for_template(best_match)
            
            return None
            
        except Exception as e:
            logger.warning(f"Simple template matching failed: {e}")
            return None
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models from LLM service."""
        if self._available_models is not None:
            return self._available_models
            
        try:
            # Use LLM service to get available models
            model_infos = await llm_service.get_available_models()
            self._available_models = [model.name for model in model_infos]
            logger.info(f"Available models from LLM service: {self._available_models}")
            return self._available_models
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
        
        # Return default models if failed
        self._available_models = ["llama3.2:3b", "llama3.2:1b"]
        return self._available_models
    
    def _enhance_prompt_with_tools(self, prompt: str, available_tools: Dict[str, Any]) -> str:
        """Enhance the prompt with domain-aware guidance for AI-powered workflow generation.

        DOMAIN-FIRST APPROACH (December 2025):
        - Generate step names that match the DOMAIN of the request
        - Still leverage AI agents, governance, and quality features
        - Add Q/G/M as capabilities ON nodes, not generic standalone nodes
        """
        # Build tool information
        tool_info = []
        if available_tools and available_tools.get("total", 0) > 0:
            for category, adapters in available_tools.get("by_category", {}).items():
                available_adapters = [a for a in adapters if a.get("available", True)]
                if available_adapters:
                    tool_info.append(f"\n{category.title()} tools:")
                    for adapter in available_adapters[:3]:  # Limit to top 3 per category
                        tool_info.append(f"- {adapter['name']} (ID: {adapter['id']})")

        # Determine target complexity
        complexity = self.context.get('complexity', 'standard') if hasattr(self, 'context') else 'standard'
        min_nodes = 8 if complexity == 'standard' else (15 if complexity == 'enterprise' else 5)
        max_nodes = 15 if complexity == 'standard' else (25 if complexity == 'enterprise' else 8)

        enhanced_prompt = f"""Generate an AI-powered workflow for: {prompt}

CRITICAL - DOMAIN-SPECIFIC NODE NAMES WITH AI CAPABILITIES:
Create step names that are SPECIFIC to the domain while leveraging AI/ML capabilities.

GOOD examples (domain name + AI capability):
- For email marketing:
  * "Audience Segmentation" (AI Agent: ML clustering for customer segments)
  * "Content Personalization" (AI Agent: NLP for personalized messaging)
  * "Campaign Strategy Review" (Human Agent: Marketing manager approval - can be automated later)
  * "Campaign Performance Analysis" (AI Agent: Predictive analytics)

- For invoice processing:
  * "Invoice Data Extraction" (AI Agent: OCR + NLP extraction)
  * "Validation & Anomaly Detection" (AI Agent: ML-based validation)
  * "Finance Manager Approval" (Human Agent: High-value approvals - can be automated for low-risk)
  * "Payment Processing" (Integration: ERP/payment system)

- For customer onboarding:
  * "Application Screening" (AI Agent: Risk scoring, fraud detection)
  * "Document Verification" (AI Agent: Document analysis)
  * "Account Manager Review" (Human Agent: Complex cases - AI handles simple ones)
  * "Welcome & Setup" (Integration: CRM, email systems)

BAD examples (DO NOT USE):
- "Document Processing Intelligence AI Agent" - generic, not domain-specific
- "Risk Assessment Intelligence AI Agent" - doesn't say what's being assessed
- "AI Agent 1", "Process Node", "node_1" - meaningless

REQUIREMENTS:
1. Generate {min_nodes}-{max_nodes} workflow steps
2. Step names MUST reflect the specific domain action
3. Node type distribution:
   - 40-50% AI Agent nodes (automation, analysis, predictions)
   - 15-25% Human Agent nodes (approvals, reviews, complex decisions)
   - 15-20% Integration nodes (external systems, APIs)
   - 10-15% Decision/routing nodes

HUMAN+AI COLLABORATION (Critical Value Proposition):
AICtrlNet enables seamless collaboration between humans and AI agents:
- Human Agent nodes represent points where human judgment is valuable
- These can be REPLACED by AI Agents later as confidence increases
- Include escalation paths: AI handles routine cases, humans handle exceptions
- Pattern: AI does analysis → Human validates → AI learns from human decisions

Examples of Human+AI collaboration patterns:
- "AI Pre-Screening" → "Human Final Review" (AI filters, human decides)
- "AI Risk Scoring" → "Manager Override" (AI recommends, human can override)
- "Automated Processing" → "Exception Handler (Human)" (AI handles 80%, human handles 20%)

AI CAPABILITIES TO LEVERAGE:
- ML Classification & Clustering
- NLP & Text Analysis
- Predictive Analytics & Forecasting
- Anomaly Detection & Fraud Prevention
- Risk Scoring & Assessment
- Document Intelligence (OCR, extraction)
- Smart Routing & Decision Trees
- Sentiment Analysis
- Recommendation Engines

GOVERNANCE & QUALITY (as properties on nodes):
- Data quality validation on data-handling steps
- Compliance checks where regulatory requirements apply
- Audit logging on decision points
- SLA tracking on human review steps
- Bias detection on AI decision nodes

Available tools and adapters:
{' '.join(tool_info) if tool_info else 'Standard AI/ML capabilities'}

AVAILABLE NODE TYPES (use appropriate types for richer workflows):

Basic Control Flow:
- "start" - Workflow entry point (auto-added)
- "end" - Workflow exit point (auto-added)
- "decision" - Conditional branching (CREATES MULTIPLE PATHS - see below)
- "parallel" - Split into parallel execution paths
- "merge" - Join parallel paths back together
- "loop" - Iterate over items or until condition

AI & Processing:
- "ai_agent" - AI-powered processing (ML, NLP, predictions)
- "human_agent" - Human review/approval required
- "process" - Generic processing step
- "transform" - Data transformation
- "dataSource" - Fetch data from source

Integration:
- "integration" - Connect to external systems
- "apiCall" - Make API calls
- "adapter" - Use specific adapter (Slack, Email, etc.)
- "notification" - Send notifications

Approval & Human-in-Loop:
- "approval" - Formal approval workflow
- "humanAgent" - Human task assignment

BRANCHING WORKFLOWS (Critical for real-world processes):
When using "decision" nodes, ALWAYS specify the conditional branches:

Example - Decision with branches:
{{
  "name": "Approval Routing",
  "type": "decision",
  "condition": "amount > 10000",
  "branches": [
    {{"condition": "amount > 10000", "label": "High Value", "target": "manager_review"}},
    {{"condition": "amount <= 10000", "label": "Standard", "target": "auto_approve"}}
  ]
}}

Example - Parallel execution:
{{
  "name": "Parallel Processing",
  "type": "parallel",
  "branches": ["validate_data", "check_compliance", "notify_stakeholders"]
}}

WORKFLOW STRUCTURE REQUIREMENTS:
1. Generate {min_nodes}-{max_nodes} workflow steps
2. Include at least ONE decision node with branches for non-trivial workflows
3. Use parallel nodes when multiple independent tasks can run simultaneously
4. Include human checkpoints at critical decision points

Node type distribution:
- 35-45% AI Agent nodes (automation, analysis, predictions)
- 15-20% Human Agent nodes (approvals, reviews)
- 15-20% Integration nodes (external systems, APIs)
- 10-15% Decision/routing nodes (with branches!)
- 5-10% Data nodes (dataSource, transform)
- 0-5% Parallel/Loop nodes (for complex workflows)

OUTPUT FORMAT for each step:
- name: Domain-specific action name (e.g., "Audience Segmentation")
- type: One of the types listed above
- description: What this step does
- ai_capability: What AI/ML powers this (if ai_agent)
- can_automate: true/false - if human_agent, can it be replaced by AI?
- branches: Array of branch definitions (if decision/parallel type)
- condition: Condition expression (if decision type)
- quality: Data quality requirements (if applicable)
- governance: Compliance/audit requirements (if applicable)
- dependencies: Array of step names this depends on (for non-linear flows)
"""

        logger.info(f"Enhanced prompt with {available_tools.get('total', 0)} available tools (domain-first with AI capabilities)")
        return enhanced_prompt
    
    def _extract_agent_from_step(self, step: Dict[str, Any]) -> Optional[str]:
        """Extract AI agent name from a workflow step."""
        action = step.get("action", "")
        
        # Check if action contains AI agent specification
        if "AI Agent:" in action:
            # Parse format: "AI Agent: [Agent Name] | Action: ..."
            parts = action.split("|")
            for part in parts:
                if "AI Agent:" in part:
                    agent_name = part.replace("AI Agent:", "").strip()
                    return agent_name
        
        # Check for agent in step fields
        if step.get("agent"):
            return step["agent"]
        
        # Check if this looks like an AI/ML task based on keywords
        ai_keywords = ["intelligence", "ai", "ml", "predict", "analyze", "extract", "classify", "detect"]
        if any(keyword in action.lower() for keyword in ai_keywords):
            # Generate a descriptive agent name based on the action
            if "document" in action.lower() or "extract" in action.lower():
                return "Document Processing Intelligence AI Agent"
            elif "decision" in action.lower() or "approve" in action.lower():
                return "Decision Intelligence AI Agent"
            elif "risk" in action.lower() or "fraud" in action.lower():
                return "Risk Assessment Intelligence AI Agent"
            elif "predict" in action.lower() or "forecast" in action.lower():
                return "Predictive Analytics Intelligence AI Agent"
            else:
                return "Intelligent Processing AI Agent"
        
        return None
    
    def _generate_node_metadata(self, step: Dict[str, Any], agent_name: Optional[str]) -> Dict[str, Any]:
        """Generate rich metadata for a workflow node."""
        action = step.get("action", "")
        
        # Extract intelligence reason if provided
        intelligence_reason = ""
        if "Intelligence:" in action:
            parts = action.split("|")
            for part in parts:
                if "Intelligence:" in part:
                    intelligence_reason = part.replace("Intelligence:", "").strip()
        
        # Generate metadata structure (similar to Rich Workflow Architecture)
        metadata = {
            "business_value": self._estimate_business_value(step),
            "implementation_details": self._get_implementation_details(step, agent_name),
            "quality_gates": {
                "accuracy_threshold": 90 if agent_name else 80,
                "validation_rules": ["schema", "format"] if agent_name else ["basic"],
                "ai_confidence_required": 0.85 if agent_name else None
            },
            "governance": {
                "risk_level": self._assess_risk_level(step),
                "compliance": self._get_compliance_requirements(step),
                "audit_trail": True if agent_name else False
            },
            "memory_config": {
                "persist_context": True if agent_name else False,
                "learning_enabled": True if agent_name else False,
                "scope": "workflow"
            }
        }
        
        # Add AI-specific metadata if this is an AI agent node
        if agent_name:
            metadata["ai_capabilities"] = {
                "agent_name": agent_name,
                "intelligence_reason": intelligence_reason or "AI/ML processing required",
                "ml_models": self._suggest_ml_models(step),
                "confidence_threshold": 0.85
            }
        
        return metadata
    
    def _estimate_business_value(self, step: Dict[str, Any]) -> str:
        """Estimate business value for a workflow step."""
        action = step.get("action", "").lower()
        
        if "automate" in action or "ai" in action:
            return "75-90% reduction in manual processing time"
        elif "extract" in action or "process" in action:
            return "60-80% faster document processing"
        elif "decision" in action or "approve" in action:
            return "50% reduction in decision time"
        elif "predict" in action or "forecast" in action:
            return "30-40% improvement in accuracy"
        else:
            return "Improved efficiency and accuracy"
    
    def _get_implementation_details(self, step: Dict[str, Any], agent_name: Optional[str]) -> str:
        """Get implementation details for a workflow step."""
        if agent_name:
            if "Document" in agent_name:
                return "Uses NLP for text extraction, computer vision for layout analysis"
            elif "Decision" in agent_name:
                return "ML-based decision tree with confidence scoring"
            elif "Risk" in agent_name:
                return "Ensemble ML models for risk scoring and anomaly detection"
            elif "Predictive" in agent_name:
                return "Time series analysis and deep learning for predictions"
            else:
                return "AI/ML processing with automated intelligence"
        else:
            return "Standard workflow processing"
    
    def _assess_risk_level(self, step: Dict[str, Any]) -> str:
        """Assess risk level for a workflow step."""
        action = step.get("action", "").lower()
        
        if any(word in action for word in ["payment", "financial", "compliance", "security"]):
            return "high"
        elif any(word in action for word in ["approval", "decision", "customer", "data"]):
            return "medium"
        else:
            return "low"
    
    def _get_compliance_requirements(self, step: Dict[str, Any]) -> List[str]:
        """Get compliance requirements for a workflow step."""
        action = step.get("action", "").lower()
        requirements = []
        
        if "customer" in action or "personal" in action or "data" in action:
            requirements.append("GDPR")
        if "financial" in action or "payment" in action:
            requirements.append("SOC2")
        if "health" in action or "medical" in action:
            requirements.append("HIPAA")
        
        return requirements if requirements else ["General"]

    async def _detect_patterns_with_llm(self, prompt: str, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to detect workflow patterns from natural language prompt.

        This replaces regex-based pattern detection with intelligent LLM analysis
        that can understand any natural language expression of branching, parallel,
        loops, and human tasks.

        Args:
            prompt: The original user prompt
            nodes: The workflow nodes generated so far (for context)

        Returns:
            Dict with detected patterns:
            {
                "branching": {...} or None,
                "parallel": {...} or None,
                "loop": {...} or None,
                "human_tasks": [...] or None
            }
        """
        # Build node labels for context
        node_labels = [n.get("data", {}).get("label", n.get("name", "")) for n in nodes if n.get("type") not in ("start", "end")]

        detection_prompt = f"""Analyze this workflow request and identify any non-linear patterns:

USER REQUEST: "{prompt}"

EXISTING WORKFLOW NODES: {node_labels[:10]}

Identify these patterns if present:

1. BRANCHING/DECISION: Conditional logic like "if X then Y otherwise Z", "when payment fails route to review", "based on amount", "depending on", etc.
   - Return: condition (what triggers the decision), true_path (action when condition is true), false_path (action when condition is false)

2. PARALLEL EXECUTION: Tasks that should run simultaneously like "check inventory AND verify payment at the same time", "simultaneously", "in parallel", "concurrently"
   - Return: list of task descriptions that should run in parallel

3. LOOPS/ITERATION: Repeated processing like "for each order", "process all items", "iterate through", "batch process"
   - Return: what is being iterated, whether it's parallel or sequential

4. HUMAN TASKS: Tasks requiring human intervention like "manual review", "approval needed", "escalate to manager", "human verification"
   - Return: list of tasks that need human involvement

Return ONLY valid JSON. If a pattern is not present, use null for that field.
Do NOT return patterns that aren't clearly stated in the request."""

        schema = {
            "type": "object",
            "properties": {
                "branching": {
                    "type": ["object", "null"],
                    "properties": {
                        "condition": {"type": "string"},
                        "true_path": {"type": "string"},
                        "false_path": {"type": "string"}
                    }
                },
                "parallel": {
                    "type": ["object", "null"],
                    "properties": {
                        "tasks": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "loop": {
                    "type": ["object", "null"],
                    "properties": {
                        "item_type": {"type": "string"},
                        "parallel": {"type": "boolean"}
                    }
                },
                "human_tasks": {
                    "type": ["array", "null"],
                    "items": {"type": "string"}
                }
            }
        }

        try:
            # Use fast model for pattern detection (low latency)
            user_settings = get_system_llm_settings()
            if hasattr(user_settings, 'preferredFastModel') and user_settings.preferredFastModel:
                # Override to use fast model
                user_settings.selected_model = user_settings.preferredFastModel

            result = await llm_service.generate_structured(
                prompt=detection_prompt,
                schema=schema,
                user_settings=user_settings
            )

            logger.info(f"LLM PATTERN DETECTION result: {result}")
            return result

        except Exception as e:
            logger.warning(f"LLM pattern detection failed, falling back to regex: {e}")
            # Fall back to regex-based detection
            return {
                "branching": self._detect_branching_from_prompt(prompt),
                "parallel": self._detect_parallel_from_prompt(prompt),
                "loop": self._detect_loop_from_prompt(prompt),
                "human_tasks": None
            }

    async def _post_process_branching_async(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of post-process branching that uses LLM for pattern detection.

        This handles LLM-generated workflows and transforms linear workflows into
        non-linear structures based on patterns detected in the original prompt.
        """
        logger.info(f"DEBUG _post_process_branching_async called with workflow keys: {list(workflow.keys())}")

        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])
        original_prompt = workflow.get("metadata", {}).get("original_prompt", "")

        logger.info(f"DEBUG nodes count: {len(nodes)}, edges count: {len(edges)}")
        logger.info(f"DEBUG original_prompt found: '{original_prompt[:100] if original_prompt else 'EMPTY'}...'")

        if original_prompt and len(nodes) >= 3:
            # Use LLM for intelligent pattern detection
            patterns = await self._detect_patterns_with_llm(original_prompt, nodes)

            # 1. BRANCHING: Apply branching pattern if detected
            if patterns.get("branching"):
                branching = patterns["branching"]
                branching_info = {
                    "type": "conditional",
                    "condition_field": branching.get("condition", "condition"),
                    "true_branch": {"label": branching.get("true_path", "Yes"), "target": branching.get("true_path", "")},
                    "false_branch": {"label": branching.get("false_path", "No"), "target": branching.get("false_path", "")}
                }
                logger.info(f"LLM BRANCHING: Detected pattern: {branching_info}")
                workflow = self._inject_decision_node(workflow, branching_info)
                nodes = workflow.get("nodes", [])
                edges = workflow.get("edges", [])

            # 2. PARALLEL: Apply parallel pattern if detected
            parallel_data = patterns.get("parallel")
            if parallel_data:
                # Handle both list and dict formats from LLM
                if isinstance(parallel_data, list):
                    tasks = parallel_data  # LLM returned list directly
                elif isinstance(parallel_data, dict):
                    tasks = parallel_data.get("tasks", [])
                else:
                    tasks = []

                if tasks and len(tasks) >= 2:
                    parallel_info = {
                        "type": "parallel",
                        "branches": tasks
                    }
                    logger.info(f"LLM PARALLEL: Detected pattern: {parallel_info}")
                    workflow = self._inject_parallel_nodes(workflow, parallel_info)
                    nodes = workflow.get("nodes", [])
                    edges = workflow.get("edges", [])

            # 3. LOOP: Apply loop pattern if detected
            if patterns.get("loop"):
                loop_info = {
                    "type": "foreach" if not patterns["loop"].get("parallel") else "batch",
                    "item_type": patterns["loop"].get("item_type", "item"),
                    "parallel": patterns["loop"].get("parallel", False)
                }
                logger.info(f"LLM LOOP: Detected pattern: {loop_info}")
                workflow = self._inject_loop_node(workflow, loop_info)
                nodes = workflow.get("nodes", [])
                edges = workflow.get("edges", [])

        # Continue with existing branch processing for LLM-generated branches
        return self._process_existing_branches(workflow)

    def _post_process_branching(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process workflow to convert decision/parallel nodes into actual branches.

        This handles LLM-generated workflows that include:
        - Decision nodes with 'branches' array
        - Parallel nodes with multiple branches
        - Dependencies between nodes
        - SMART: Detects branching from original prompt when LLM fails to generate decision nodes

        Returns updated workflow with proper branching edges.

        NOTE: This is the synchronous version. Use _post_process_branching_async for
        intelligent LLM-based pattern detection.
        """
        logger.info(f"DEBUG _post_process_branching called with workflow keys: {list(workflow.keys())}")
        logger.info(f"DEBUG workflow.metadata keys: {list(workflow.get('metadata', {}).keys())}")

        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])

        logger.info(f"DEBUG nodes count: {len(nodes)}, edges count: {len(edges)}")

        # SMART PATTERN DETECTION: Detect workflow patterns from original prompt
        # This fixes the issue where LLM generates linear workflows for complex prompts
        original_prompt = workflow.get("metadata", {}).get("original_prompt", "")
        logger.info(f"DEBUG original_prompt found: '{original_prompt[:100] if original_prompt else 'EMPTY'}...'")

        if original_prompt:
            # 1. BRANCHING DETECTION: if/then/else, route based on, etc.
            branching_info = self._detect_branching_from_prompt(original_prompt)
            if branching_info:
                logger.info(f"SMART BRANCHING: Detected branching pattern in prompt: {branching_info}")
                workflow = self._inject_decision_node(workflow, branching_info)
                # Refresh nodes and edges after injection
                nodes = workflow.get("nodes", [])
                edges = workflow.get("edges", [])

            # 2. LOOP DETECTION: for each, process all, iterate, etc.
            loop_info = self._detect_loop_from_prompt(original_prompt)
            if loop_info:
                logger.info(f"SMART LOOP: Detected loop pattern in prompt: {loop_info}")
                workflow = self._inject_loop_node(workflow, loop_info)
                # Refresh nodes and edges after injection
                nodes = workflow.get("nodes", [])
                edges = workflow.get("edges", [])

            # 3. PARALLEL DETECTION: simultaneously, in parallel, etc.
            parallel_info = self._detect_parallel_from_prompt(original_prompt)
            if parallel_info:
                logger.info(f"SMART PARALLEL: Detected parallel pattern in prompt: {parallel_info}")
                workflow = self._inject_parallel_nodes(workflow, parallel_info)
                # Refresh nodes and edges after injection
                nodes = workflow.get("nodes", [])
                edges = workflow.get("edges", [])

        return self._process_existing_branches(workflow)

    def _process_existing_branches(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Process branches that already exist in workflow nodes (from LLM generation).

        This is extracted from the original _post_process_branching to allow
        both sync and async versions to share this logic.
        """
        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])

        if not nodes:
            return workflow

        # Build node lookup by id and name
        node_by_id = {n.get("id"): n for n in nodes}
        node_by_name = {n.get("name", "").lower().replace(" ", "_").replace("-", "_"): n for n in nodes}

        # Track edges to add and remove, and new nodes to create
        edges_to_add = []
        edges_to_remove = set()
        nodes_to_add = []
        created_node_counter = len(nodes) + 1

        # Find all node IDs for reference
        all_node_ids = set(n.get("id") for n in nodes)

        # Helper function to find or create a target node
        def find_or_create_target_node(target_name: str, source_node: Dict, branch_index: int) -> str:
            nonlocal created_node_counter

            if not target_name:
                return None

            target_lower = target_name.lower().replace(" ", "_").replace("-", "_")

            # Try to find by ID
            if target_name in all_node_ids:
                return target_name

            # Try to find by name match (fuzzy)
            for n in nodes + nodes_to_add:
                n_name = n.get("name", "").lower().replace(" ", "_").replace("-", "_")
                n_id = n.get("id", "")
                # Match by partial name or action
                if (target_lower in n_name or n_name in target_lower or
                    target_lower in n_id.lower() or
                    target_lower in n.get("data", {}).get("action", "").lower()):
                    return n.get("id")

            # If not found, CREATE a new node for this branch target
            new_node_id = f"branch_node_{created_node_counter}"
            created_node_counter += 1

            # Calculate position based on source node
            source_pos = source_node.get("position", {"x": 300, "y": 100})
            new_x = source_pos.get("x", 300) + 250
            new_y = source_pos.get("y", 100) + (branch_index * 150)  # Stagger vertically

            new_node = {
                "id": new_node_id,
                "type": "process",
                "name": target_name.title(),  # Capitalize properly
                "position": {"x": new_x, "y": new_y},
                "data": {
                    "action": target_name,
                    "capability": "process",
                    "intent": {
                        "label": target_name.title(),
                        "description": f"Process for {target_name}"
                    }
                }
            }

            nodes_to_add.append(new_node)
            all_node_ids.add(new_node_id)
            logger.info(f"Created branch target node: {new_node_id} for target '{target_name}'")

            return new_node_id

        for node in nodes:
            node_id = node.get("id")
            node_type = node.get("type", "").lower()
            node_data = node.get("data", {})

            # Handle decision nodes with branches
            if node_type in ("decision", "conditional"):
                branches = node_data.get("branches") or node.get("branches", [])

                if branches:
                    logger.info(f"Processing decision node {node_id} with {len(branches)} branches")

                    # Remove the default sequential edge from this node
                    for edge in edges:
                        if edge.get("from") == node_id or edge.get("source") == node_id:
                            edges_to_remove.add((edge.get("from") or edge.get("source"),
                                                 edge.get("to") or edge.get("target")))

                    # Create edges for each branch
                    for i, branch in enumerate(branches):
                        target = branch.get("target") or branch.get("to") or branch.get("name")
                        # Use simple Yes/No labels for decision branches (cleaner UI)
                        label = "Yes" if i == 0 else "No"

                        # Find or create target node
                        target_node = find_or_create_target_node(target, node, i)

                        if target_node:
                            edge_id = f"edge-{node_id}-{target_node}-{i}"
                            edges_to_add.append({
                                "id": edge_id,
                                "from": node_id,
                                "to": target_node,
                                "source": node_id,
                                "target": target_node,
                                "label": label,
                                "animated": True,
                                "type": "smoothstep"
                            })
                            logger.info(f"Created branch edge: {node_id} -> {target_node} ({label})")

            # Handle parallel nodes
            elif node_type == "parallel":
                branches = node_data.get("branches") or node.get("branches", [])

                if branches:
                    logger.info(f"Processing parallel node {node_id} with {len(branches)} branches")

                    # Remove default edge
                    for edge in edges:
                        if edge.get("from") == node_id or edge.get("source") == node_id:
                            edges_to_remove.add((edge.get("from") or edge.get("source"),
                                                 edge.get("to") or edge.get("target")))

                    # Create edges to each parallel branch
                    for i, branch_item in enumerate(branches):
                        # Handle both string branch names and dict branch objects
                        if isinstance(branch_item, str):
                            branch_name = branch_item
                        else:
                            branch_name = branch_item.get("target") or branch_item.get("name") or f"Parallel {i+1}"

                        # Find or create target node
                        target_node = find_or_create_target_node(branch_name, node, i)

                        if target_node:
                            edge_id = f"edge-{node_id}-{target_node}-parallel-{i}"
                            edges_to_add.append({
                                "id": edge_id,
                                "from": node_id,
                                "to": target_node,
                                "source": node_id,
                                "target": target_node,
                                # No label for parallel edges - the visual connection speaks for itself
                                "animated": True,
                                "type": "smoothstep"
                            })
                            logger.info(f"Created parallel edge: {node_id} -> {target_node}")

            # Handle dependencies (for non-linear flows)
            dependencies = node_data.get("dependencies") or node.get("dependencies", [])
            if dependencies:
                # Remove any auto-generated sequential edges TO this node
                for edge in edges:
                    if edge.get("to") == node_id:
                        edges_to_remove.add((edge.get("from"), edge.get("to")))

                # Create edges from each dependency
                for dep in dependencies:
                    dep_lower = dep.lower().replace(" ", "_")

                    for n in nodes:
                        n_name = n.get("name", "").lower().replace(" ", "_")
                        n_action = n.get("data", {}).get("action", "").lower()

                        if dep_lower in n_name or dep_lower in n_action or n_name in dep_lower:
                            edges_to_add.append({
                                "from": n.get("id"),
                                "to": node_id
                            })
                            break

        # Apply edge modifications
        new_edges = [e for e in edges if (e.get("from"), e.get("to")) not in edges_to_remove]

        # Add new edges (avoid duplicates)
        existing_edge_keys = set((e.get("from"), e.get("to")) for e in new_edges)
        for edge in edges_to_add:
            key = (edge.get("from"), edge.get("to"))
            if key not in existing_edge_keys:
                new_edges.append(edge)
                existing_edge_keys.add(key)

        workflow["edges"] = new_edges

        # CRITICAL: Add any newly created branch target nodes to the workflow
        if nodes_to_add:
            workflow["nodes"] = nodes + nodes_to_add
            logger.info(f"Added {len(nodes_to_add)} new branch target nodes: {[n.get('id') for n in nodes_to_add]}")

        logger.info(f"Post-processed branching: removed {len(edges_to_remove)} edges, added {len(edges_to_add)} edges, created {len(nodes_to_add)} new nodes")

        return workflow

    def _detect_branching_from_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Detect branching patterns from the original user prompt.

        Handles patterns like:
        - "route based on X to Y otherwise Z"
        - "if X then Y else Z"
        - "when X do Y otherwise do Z"
        - "above/below/over/under threshold"
        """
        import re

        prompt_lower = prompt.lower()

        # Pattern 1: "route based on X to Y otherwise Z"
        route_pattern = r'route\s+(?:based\s+on\s+)?(\w+(?:\s+\w+)*)\s+(?:over|above|greater\s+than|>)\s*(\d+)\s+to\s+(\w+(?:\s+\w+)*)\s+(?:otherwise|else|or)\s+(\w+(?:[\-_]\w+)*)'
        match = re.search(route_pattern, prompt_lower)
        if match:
            condition_field = match.group(1).strip()
            threshold = match.group(2)
            true_target = match.group(3).strip()
            false_target = match.group(4).strip()
            return {
                "type": "threshold",
                "condition_field": condition_field,
                "threshold": int(threshold),
                "operator": ">",
                "true_branch": {"label": f"{condition_field} > {threshold}", "target": true_target},
                "false_branch": {"label": f"{condition_field} <= {threshold}", "target": false_target}
            }

        # Pattern 2: "if X over/above Y then A else B"
        if_pattern = r'if\s+(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:over|above|greater\s+than|>)\s*(\d+)\s+(?:then\s+)?(\w+(?:[\-_\s]\w+)*)\s+(?:otherwise|else|or\s+else)\s+(\w+(?:[\-_\s]\w+)*)'
        match = re.search(if_pattern, prompt_lower)
        if match:
            condition_field = match.group(1).strip()
            threshold = match.group(2)
            true_target = match.group(3).strip()
            false_target = match.group(4).strip()
            return {
                "type": "threshold",
                "condition_field": condition_field,
                "threshold": int(threshold),
                "operator": ">",
                "true_branch": {"label": f"if {condition_field} > {threshold}", "target": true_target},
                "false_branch": {"label": "else", "target": false_target}
            }

        # Pattern 3: Generic "X otherwise Y" with keywords
        generic_pattern = r'(?:route|send|forward|direct)\s+(?:to\s+)?(\w+(?:[\-_\s]\w+)*)\s+(?:otherwise|else|or)\s+(\w+(?:[\-_\s]\w+)*)'
        match = re.search(generic_pattern, prompt_lower)
        if match:
            true_target = match.group(1).strip()
            false_target = match.group(2).strip()
            return {
                "type": "conditional",
                "true_branch": {"label": "Condition met", "target": true_target},
                "false_branch": {"label": "Otherwise", "target": false_target}
            }

        # Pattern 4: "based on" indicates conditional logic
        if "based on" in prompt_lower and ("otherwise" in prompt_lower or "else" in prompt_lower):
            # Try to extract branch targets from keyword matching
            targets = []
            for keyword in ["manager", "approve", "reject", "escalate", "auto"]:
                if keyword in prompt_lower:
                    targets.append(keyword)
            if len(targets) >= 2:
                return {
                    "type": "generic_conditional",
                    "true_branch": {"label": "Primary path", "target": targets[0]},
                    "false_branch": {"label": "Alternative path", "target": targets[1]}
                }

        return None

    def _fuzzy_match_text(self, pattern: str, text: str, threshold: float = 0.4) -> bool:
        """Fuzzy match pattern against text using word overlap.

        This handles cases where LLM pattern descriptions don't exactly match node names:
        - Pattern: "Check Inventory Availability"
        - Text: "Verify Available Inventory"
        - Result: True (overlap of "inventory", "available/availability")

        Args:
            pattern: The pattern to search for (from LLM pattern detection)
            text: The text to match against (node name or action)
            threshold: Minimum overlap ratio (0.0 to 1.0, default 0.4 = 40%)

        Returns:
            True if word overlap exceeds threshold
        """
        if not pattern or not text:
            return False

        # Normalize and tokenize
        stop_words = {'a', 'an', 'the', 'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'and', 'or', 'is', 'are', 'be'}

        def tokenize(s: str) -> set:
            # Normalize: lowercase, replace punctuation with spaces
            s = s.lower()
            for char in '-_.,;:!?()[]{}':
                s = s.replace(char, ' ')
            # Split and remove stop words and short words
            words = set(w for w in s.split() if w not in stop_words and len(w) > 2)
            # Also add word stems (simple suffix removal)
            stemmed = set()
            for w in words:
                stemmed.add(w)
                # Common suffixes
                for suffix in ['ing', 'tion', 'ation', 'ment', 'ness', 'ity', 'able', 'ible', 'ed', 'es', 's']:
                    if w.endswith(suffix) and len(w) > len(suffix) + 2:
                        stemmed.add(w[:-len(suffix)])
            return stemmed

        pattern_words = tokenize(pattern)
        text_words = tokenize(text)

        if not pattern_words or not text_words:
            return False

        # Calculate overlap
        overlap = pattern_words.intersection(text_words)

        # Ratio of overlap to smaller set (more lenient matching)
        smaller_set = min(len(pattern_words), len(text_words))
        if smaller_set == 0:
            return False

        overlap_ratio = len(overlap) / smaller_set

        # Log for debugging
        if overlap_ratio >= threshold:
            logger.debug(f"FUZZY MATCH: '{pattern}' ~ '{text}' (overlap={overlap}, ratio={overlap_ratio:.2f})")

        return overlap_ratio >= threshold

    def _find_matching_node(self, nodes: List[Dict[str, Any]], pattern: str, exclude_types: List[str] = None) -> Optional[Dict[str, Any]]:
        """Find a node matching the pattern using fuzzy matching.

        Args:
            nodes: List of workflow nodes
            pattern: Pattern to search for
            exclude_types: Node types to exclude (e.g., ['start', 'end'])

        Returns:
            Matching node or None
        """
        if not pattern:
            return None

        exclude_types = exclude_types or ['start', 'end']

        for node in nodes:
            if node.get("type") in exclude_types:
                continue

            node_name = node.get("name", "")
            node_action = node.get("data", {}).get("action", "") if node.get("data") else ""
            node_label = node.get("data", {}).get("label", "") if node.get("data") else ""

            # Try fuzzy matching against name, action, and label
            if self._fuzzy_match_text(pattern, node_name):
                logger.info(f"FUZZY MATCH: Pattern '{pattern}' matched node '{node_name}' (id={node.get('id')})")
                return node
            if node_action and self._fuzzy_match_text(pattern, node_action):
                logger.info(f"FUZZY MATCH: Pattern '{pattern}' matched action '{node_action}' (id={node.get('id')})")
                return node
            if node_label and self._fuzzy_match_text(pattern, node_label):
                logger.info(f"FUZZY MATCH: Pattern '{pattern}' matched label '{node_label}' (id={node.get('id')})")
                return node

        logger.warning(f"FUZZY MATCH: No match found for pattern '{pattern}'")
        return None

    def _inject_decision_node(self, workflow: Dict[str, Any], branching_info: Dict[str, Any]) -> Dict[str, Any]:
        """Inject a decision node into a linear workflow based on detected branching.

        This transforms a linear workflow into a branching one by:
        1. Finding nodes that match branch targets
        2. Creating a decision node
        3. Rewiring edges to create branching structure
        """
        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])

        if len(nodes) < 3:  # Need at least start, some process, and end
            return workflow

        # Find node indices (skip start and end)
        process_nodes = [n for n in nodes if n.get("type") not in ("start", "end")]
        if len(process_nodes) < 2:
            return workflow

        # Find nodes that might match branch targets using fuzzy matching
        true_target = branching_info.get("true_branch", {}).get("target", "")
        false_target = branching_info.get("false_branch", {}).get("target", "")

        logger.info(f"DECISION INJECTION: Looking for true_target='{true_target}', false_target='{false_target}'")
        logger.info(f"DECISION INJECTION: Available nodes: {[n.get('name') for n in process_nodes]}")

        # Use fuzzy matching to find nodes
        true_node = self._find_matching_node(nodes, true_target)
        false_node = self._find_matching_node(nodes, false_target)

        logger.info(f"DECISION INJECTION: Found true_node={true_node.get('id') if true_node else None}, false_node={false_node.get('id') if false_node else None}")

        # If we found both branch targets, or at least have a clear decision point
        if true_node or false_node:
            # Find a good insertion point for the decision node
            # Insert after the first couple of process nodes (setup/analysis)
            insert_idx = min(2, len(process_nodes) - 1)
            insert_after_node = process_nodes[insert_idx] if insert_idx < len(process_nodes) else process_nodes[0]

            # Create decision node
            decision_node_id = "decision_node_1"
            decision_pos = insert_after_node.get("position", {"x": 500, "y": 100})

            decision_node = {
                "id": decision_node_id,
                "type": "decision",
                "name": branching_info.get("condition_field", "Check Condition").title(),
                "position": {"x": decision_pos.get("x", 500) + 100, "y": decision_pos.get("y", 100)},
                "data": {
                    "action": "evaluate_condition",
                    "condition": f"{branching_info.get('condition_field', 'value')} {branching_info.get('operator', '>')} {branching_info.get('threshold', 0)}",
                    "branches": [
                        {
                            "condition": branching_info.get("true_branch", {}).get("label", "Yes"),
                            "target": true_node.get("id") if true_node else "end"
                        },
                        {
                            "condition": branching_info.get("false_branch", {}).get("label", "No"),
                            "target": false_node.get("id") if false_node else "end"
                        }
                    ]
                },
                "branches": [
                    {
                        "condition": branching_info.get("true_branch", {}).get("label", "Yes"),
                        "target": true_node.get("id") if true_node else "end"
                    },
                    {
                        "condition": branching_info.get("false_branch", {}).get("label", "No"),
                        "target": false_node.get("id") if false_node else "end"
                    }
                ],
                "metadata": {
                    "smart_branching": True,
                    "detected_from_prompt": True,
                    "branching_info": branching_info
                }
            }

            # Insert decision node after insert_after_node
            node_list = list(nodes)
            insert_position = next((i for i, n in enumerate(node_list) if n.get("id") == insert_after_node.get("id")), -1)
            if insert_position >= 0:
                node_list.insert(insert_position + 1, decision_node)

            # Rewire edges
            new_edges = []
            for edge in edges:
                source = edge.get("source") or edge.get("from")
                target = edge.get("target") or edge.get("to")

                # If edge goes from insert_after_node, redirect to decision node
                if source == insert_after_node.get("id"):
                    new_edges.append({
                        "id": f"edge-{source}-{decision_node_id}",
                        "source": source,
                        "target": decision_node_id,
                        "from": source,
                        "to": decision_node_id
                    })
                # Skip edges that would be replaced by branching
                elif source == decision_node_id:
                    continue
                else:
                    new_edges.append(edge)

            # Add branch edges from decision node
            # Use clean short labels - "Yes"/"No" - not full action descriptions
            if true_node:
                new_edges.append({
                    "id": f"edge-{decision_node_id}-{true_node.get('id')}-true",
                    "source": decision_node_id,
                    "target": true_node.get("id"),
                    "from": decision_node_id,
                    "to": true_node.get("id"),
                    "label": "Yes",
                    "animated": True,
                    "type": "smoothstep"
                })

            if false_node:
                new_edges.append({
                    "id": f"edge-{decision_node_id}-{false_node.get('id')}-false",
                    "source": decision_node_id,
                    "target": false_node.get("id"),
                    "from": decision_node_id,
                    "to": false_node.get("id"),
                    "label": "No",
                    "animated": True,
                    "type": "smoothstep"
                })

            # If no targets found, at least create edges to next sequential nodes
            if not true_node and not false_node and len(process_nodes) >= 2:
                next_nodes = process_nodes[insert_idx + 1:insert_idx + 3] if insert_idx + 1 < len(process_nodes) else []
                for i, next_node in enumerate(next_nodes[:2]):
                    # Use clean "Yes"/"No" labels, not full action descriptions
                    label = "Yes" if i == 0 else "No"
                    new_edges.append({
                        "id": f"edge-{decision_node_id}-{next_node.get('id')}-branch{i}",
                        "source": decision_node_id,
                        "target": next_node.get("id"),
                        "from": decision_node_id,
                        "to": next_node.get("id"),
                        "label": label,
                        "animated": True,
                        "type": "smoothstep"
                    })

            workflow["nodes"] = node_list
            workflow["edges"] = new_edges
            logger.info(f"SMART BRANCHING: Injected decision node {decision_node_id} with branches to {true_node.get('id') if true_node else 'end'} and {false_node.get('id') if false_node else 'end'}")

        return workflow

    def _detect_loop_from_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Detect loop/iteration patterns from the original user prompt.

        Handles patterns like:
        - "for each invoice", "process all items", "iterate through records"
        - "batch process", "handle each", "one by one"
        """
        import re

        prompt_lower = prompt.lower()

        # Pattern 1: "for each X" or "for all X"
        foreach_pattern = r'(?:for\s+each|for\s+all|process\s+each|process\s+all|iterate\s+(?:through|over))\s+(\w+(?:\s+\w+)*)'
        match = re.search(foreach_pattern, prompt_lower)
        if match:
            item_type = match.group(1).strip()
            return {
                "type": "foreach",
                "item_type": item_type,
                "parallel": False  # Default to sequential
            }

        # Pattern 2: "batch X" or "bulk X"
        batch_pattern = r'(?:batch|bulk)\s+(?:process|handle|send|update)\s+(\w+(?:\s+\w+)*)'
        match = re.search(batch_pattern, prompt_lower)
        if match:
            item_type = match.group(1).strip()
            return {
                "type": "batch",
                "item_type": item_type,
                "parallel": True  # Batch typically implies parallel
            }

        # Pattern 3: "every X" or "all the X"
        every_pattern = r'(?:every|all\s+the|each\s+of\s+the)\s+(\w+(?:\s+\w+)*)'
        match = re.search(every_pattern, prompt_lower)
        if match:
            item_type = match.group(1).strip()
            return {
                "type": "foreach",
                "item_type": item_type,
                "parallel": False
            }

        # Pattern 4: Generic iteration keywords
        if any(word in prompt_lower for word in ['iterate', 'loop through', 'one by one', 'sequentially']):
            return {
                "type": "generic_iteration",
                "parallel": False
            }

        return None

    def _detect_parallel_from_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Detect parallel execution patterns from the original user prompt.

        Handles patterns like:
        - "simultaneously", "in parallel", "at the same time"
        - "concurrently", "fan out"
        """
        import re

        prompt_lower = prompt.lower()

        # Pattern 1: "X and Y simultaneously/in parallel/at the same time"
        parallel_pattern = r'(\w+(?:\s+\w+)*)\s+(?:and|,)\s+(\w+(?:\s+\w+)*)\s+(?:simultaneously|in\s+parallel|at\s+the\s+same\s+time|concurrently)'
        match = re.search(parallel_pattern, prompt_lower)
        if match:
            branch1 = match.group(1).strip()
            branch2 = match.group(2).strip()
            return {
                "type": "parallel",
                "branches": [branch1, branch2]
            }

        # Pattern 2: "run/execute X in parallel"
        run_parallel_pattern = r'(?:run|execute|process|perform)\s+(\w+(?:\s+\w+)*(?:\s*,\s*\w+(?:\s+\w+)*)*)\s+in\s+parallel'
        match = re.search(run_parallel_pattern, prompt_lower)
        if match:
            tasks_str = match.group(1).strip()
            # Split by comma
            branches = [t.strip() for t in tasks_str.split(',') if t.strip()]
            if branches:
                return {
                    "type": "parallel",
                    "branches": branches
                }

        # Pattern 3: "fan out to X"
        fanout_pattern = r'fan\s+out\s+to\s+(\w+(?:\s+\w+)*(?:\s*(?:,|and)\s*\w+(?:\s+\w+)*)*)'
        match = re.search(fanout_pattern, prompt_lower)
        if match:
            targets_str = match.group(1).strip()
            branches = [t.strip() for t in re.split(r'[,\s]+and\s+|,', targets_str) if t.strip()]
            if branches:
                return {
                    "type": "fanout",
                    "branches": branches
                }

        # Pattern 4: Generic parallel keywords
        if any(word in prompt_lower for word in ['simultaneously', 'in parallel', 'concurrently', 'at the same time']):
            return {
                "type": "generic_parallel"
            }

        return None

    def _inject_loop_node(self, workflow: Dict[str, Any], loop_info: Dict[str, Any]) -> Dict[str, Any]:
        """Inject a loop node into a workflow based on detected iteration pattern.

        This transforms process nodes that should iterate into loop nodes.
        """
        nodes = workflow.get("nodes", [])

        if not nodes:
            return workflow

        # Find process nodes that might need to become loops
        for node in nodes:
            node_type = node.get("type", "")
            node_name = node.get("name", "").lower() if node.get("name") else ""
            node_label = node.get("data", {}).get("label", "").lower() if node.get("data") else ""

            # Check if this node matches the loop pattern
            item_type = loop_info.get("item_type", "").lower()

            if item_type and (item_type in node_name or item_type in node_label):
                # Transform this node into a loop node
                node["type"] = "loop"
                if "data" not in node:
                    node["data"] = {}
                node["data"]["node_type"] = "loop"
                node["data"]["parallel"] = loop_info.get("parallel", False)
                node["data"]["items_source"] = f"${{{item_type}}}"

                # Add loop metadata
                if "metadata" not in node:
                    node["metadata"] = {}
                node["metadata"]["loop_config"] = {
                    "item_type": item_type,
                    "parallel": loop_info.get("parallel", False),
                    "smart_detection": True
                }

                logger.info(f"SMART LOOP: Converted node {node.get('id')} to loop node for {item_type}")
                break  # Only convert one node

        return workflow

    def _inject_parallel_nodes(self, workflow: Dict[str, Any], parallel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Inject parallel and join nodes into a workflow based on detected parallel pattern.

        This creates parallel execution structure with a parallel node and join node.
        """
        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])

        if not nodes or len(nodes) < 3:
            return workflow

        branches = parallel_info.get("branches", [])
        if not branches:
            return workflow

        logger.info(f"PARALLEL INJECTION: Looking for branches: {branches}")
        logger.info(f"PARALLEL INJECTION: Available nodes: {[n.get('name') for n in nodes if n.get('type') not in ('start', 'end')]}")

        # Find nodes matching the branch descriptions using fuzzy matching
        branch_nodes = []
        matched_node_ids = set()  # Avoid matching same node twice
        for branch_desc in branches:
            matching_node = self._find_matching_node(nodes, branch_desc)
            if matching_node and matching_node.get("id") not in matched_node_ids:
                branch_nodes.append(matching_node)
                matched_node_ids.add(matching_node.get("id"))
                logger.info(f"PARALLEL INJECTION: Matched branch '{branch_desc}' to node '{matching_node.get('name')}' (id={matching_node.get('id')})")

        logger.info(f"PARALLEL INJECTION: Found {len(branch_nodes)} matching nodes for {len(branches)} branches")

        if len(branch_nodes) < 2:
            logger.warning(f"PARALLEL INJECTION: Need at least 2 branches but only found {len(branch_nodes)}, skipping")
            return workflow  # Need at least 2 branches for parallel

        # Create parallel node - minimal labels, let visual design communicate node type
        parallel_node_id = f"parallel-{uuid.uuid4().hex[:8]}"
        parallel_node = {
            "id": parallel_node_id,
            "type": "parallel",
            "name": "Parallel",
            "position": {"x": 300, "y": 200},
            "data": {
                "node_type": "parallel",
                "branches": [n.get("id") for n in branch_nodes]
            },
            "metadata": {
                "smart_parallel": True,
                "branch_count": len(branch_nodes)
            }
        }

        # Create join node - minimal labels, let visual design communicate node type
        join_node_id = f"join-{uuid.uuid4().hex[:8]}"
        join_node = {
            "id": join_node_id,
            "type": "join",
            "name": "Sync",
            "position": {"x": 600, "y": 200},
            "data": {
                "node_type": "join"
            },
            "metadata": {
                "smart_join": True
            }
        }

        # Find insertion point - before the first branch node
        first_branch_idx = next((i for i, n in enumerate(nodes) if n.get("id") == branch_nodes[0].get("id")), -1)
        if first_branch_idx > 0:
            # Insert parallel node before branches
            node_list = list(nodes)
            node_list.insert(first_branch_idx, parallel_node)

            # Find last branch node and insert join after it
            last_branch_idx = max(i for i, n in enumerate(node_list) if n in branch_nodes)
            node_list.insert(last_branch_idx + 1, join_node)

            # Rewire edges
            new_edges = []
            # Find the node before the first branch
            pre_parallel_edge = None
            for edge in edges:
                target = edge.get("target") or edge.get("to")
                if target == branch_nodes[0].get("id"):
                    pre_parallel_edge = edge
                    break

            if pre_parallel_edge:
                source = pre_parallel_edge.get("source") or pre_parallel_edge.get("from")
                # Edge from previous node to parallel
                new_edges.append({
                    "id": f"edge-{source}-{parallel_node_id}",
                    "source": source,
                    "target": parallel_node_id
                })

            # Edges from parallel to each branch
            for bn in branch_nodes:
                new_edges.append({
                    "id": f"edge-{parallel_node_id}-{bn.get('id')}",
                    "source": parallel_node_id,
                    "target": bn.get("id"),
                    "animated": True
                })

            # Edges from each branch to join
            for bn in branch_nodes:
                new_edges.append({
                    "id": f"edge-{bn.get('id')}-{join_node_id}",
                    "source": bn.get("id"),
                    "target": join_node_id,
                    "animated": True
                })

            # Find the edge that continues from the last branch to the next node
            # and rewire it to go from join → next node
            branch_ids = [n.get("id") for n in branch_nodes]
            continuation_target = None
            for edge in edges:
                source = edge.get("source") or edge.get("from")
                target = edge.get("target") or edge.get("to")
                # If this edge goes from a branch node to a non-branch node, that's the continuation
                if source in branch_ids and target not in branch_ids and target not in ("start", "end"):
                    continuation_target = target
                    break

            # Add edge from join to continuation target
            if continuation_target:
                new_edges.append({
                    "id": f"edge-{join_node_id}-{continuation_target}",
                    "source": join_node_id,
                    "target": continuation_target
                })
                logger.info(f"SMART PARALLEL: Added continuation edge from join to {continuation_target}")

            # Keep other edges that don't involve branch nodes
            for edge in edges:
                source = edge.get("source") or edge.get("from")
                target = edge.get("target") or edge.get("to")
                if source not in branch_ids and target not in branch_ids:
                    new_edges.append(edge)

            workflow["nodes"] = node_list
            workflow["edges"] = new_edges
            logger.info(f"SMART PARALLEL: Injected parallel node {parallel_node_id} with {len(branch_nodes)} branches and join node {join_node_id}")

        return workflow

    def _suggest_ml_models(self, step: Dict[str, Any]) -> List[str]:
        """Suggest ML models for a workflow step."""
        action = step.get("action", "").lower()
        models = []
        
        if "classify" in action:
            models.append("RandomForest Classifier")
        if "predict" in action:
            models.append("LSTM Time Series")
        if "extract" in action or "nlp" in action:
            models.append("BERT/Transformer")
        if "vision" in action or "image" in action:
            models.append("CNN/YOLO")
        if "decision" in action:
            models.append("XGBoost")
        
        return models if models else ["General ML Model"]
    
    def _map_to_adapter(self, step: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Map a workflow step to an actual adapter if available."""
        if not self._cached_adapters:
            return step.get("node_type", "process"), None
        
        capability = step.get("capability", "")
        action = step.get("action", "")
        node_type = step.get("node_type", "process")
        
        # Mapping of capabilities/actions to adapter categories
        capability_map = {
            "notification": "communication",
            "messaging": "communication",
            "email": "communication",
            "slack": "communication",
            "sms": "communication",
            "ai_processing": "ai",
            "text_generation": "ai", 
            "classification": "ai",
            "analysis": "ai",
            "payment": "payment",
            "billing": "payment",
            "human_approval": "human",
            "manual_review": "human",
            "data_extraction": "data",
            "data_transformation": "data",
            "database": "data"
        }
        
        # Try to find matching adapter
        for key, category in capability_map.items():
            if key in capability.lower() or key in action.lower():
                adapters = self._cached_adapters.get("by_category", {}).get(category, [])
                if adapters:
                    # Get first available adapter
                    for adapter in adapters:
                        if adapter.get("available", True):
                            # Return specific adapter node type
                            return f"{adapter['id']}Adapter", adapter['id']
        
        # No specific adapter found, return original node type
        return node_type, None