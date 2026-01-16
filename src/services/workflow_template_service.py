"""Workflow template service for managing template operations."""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from models.workflow_templates import (
    WorkflowTemplate,
    WorkflowTemplatePermission,
    WorkflowTemplateUsage,
    WorkflowTemplateReview
)
from models.user import User
from models.community_complete import WorkflowInstance
from schemas.workflow_templates import (
    WorkflowTemplateCreate,
    WorkflowTemplateUpdate,
    WorkflowTemplateResponse,
    WorkflowTemplateDetail,
    TemplatePermission,
    TemplatePermissionCreate,
    TemplateListRequest,
    TemplateUsageCreate,
    TemplateReviewCreate,
    TemplateReviewUpdate,
    InstantiateTemplateRequest,
    ForkTemplateRequest,
    CustomizationLevel
)
from core.exceptions import NotFoundError, ForbiddenError, ValidationError

logger = logging.getLogger(__name__)


class WorkflowTemplateService:
    """Service for managing workflow templates."""
    
    def __init__(self):
        """Initialize the workflow template service."""
        # Set up base template directory (Community only)
        # Try multiple locations in order of preference, but be explicit about Community templates
        community_template_paths = [
            Path("/app/workflow-templates"),  # Community container
            Path("/workspace/aictrlnet-fastapi/workflow-templates"),  # Business/Enterprise containers
            Path("/workflow-templates"),  # Volume mount
            Path("workflow-templates")  # Development
        ]
        
        # Choose the FIRST Community template location that exists and has Community templates
        self.base_template_dir = None
        for template_path in community_template_paths:
            if template_path.exists():
                system_path = template_path / "system"
                if system_path.exists():
                    # Check if this contains Community-specific templates
                    community_categories = {'communication', 'data', 'reliability'}
                    found_categories = {d.name for d in system_path.iterdir() if d.is_dir()}
                    
                    # If this path contains ANY Community categories, use it
                    if community_categories.intersection(found_categories):
                        self.base_template_dir = template_path
                        logger.info(f"Community service using template directory: {template_path.absolute()}")
                        logger.info(f"Found Community categories: {found_categories.intersection(community_categories)}")
                        break
        
        if self.base_template_dir is None:
            # Fallback: prefer user home directory for pip installs, /app for Docker
            import os
            home_templates = Path.home() / ".aictrlnet" / "workflow-templates"
            if os.environ.get("AICTRLNET_TEMPLATE_DIR"):
                self.base_template_dir = Path(os.environ["AICTRLNET_TEMPLATE_DIR"])
            elif Path("/app").exists() and os.access("/app", os.W_OK):
                self.base_template_dir = Path("/app/workflow-templates")
            else:
                self.base_template_dir = home_templates
            logger.warning(f"No Community templates found, using fallback: {self.base_template_dir.absolute()}")
        
        self.system_dir = self.base_template_dir / "system"
        self.org_dir = self.base_template_dir / "organization"
        self.personal_dir = self.base_template_dir / "personal"
        
        # Create directories if they don't exist
        for directory in [self.system_dir, self.org_dir, self.personal_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_template_preview(self, definition_path: str) -> Optional[Dict[str, Any]]:
        """Load preview data from a template definition file."""
        try:
            # Try multiple potential paths
            paths_to_try = [
                Path(definition_path),
                self.base_template_dir.parent / definition_path,
                Path("/workspace/aictrlnet-fastapi") / definition_path,
                Path("/workspace/aictrlnet-fastapi-business") / definition_path,
                Path("/workspace/aictrlnet-fastapi-enterprise") / definition_path,
            ]
            
            template_file = None
            for path in paths_to_try:
                if path.exists():
                    template_file = path
                    break
            
            if not template_file:
                logger.debug(f"Template file not found: {definition_path}")
                return None
            
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            
            # Extract preview data (first 5 nodes max)
            # Check if nodes are in workflow field (new format) or root (old format)
            if 'workflow' in template_data and isinstance(template_data['workflow'], dict):
                nodes = template_data['workflow'].get('nodes', [])
                edges = template_data['workflow'].get('edges', [])
            else:
                nodes = template_data.get('nodes', [])
                edges = template_data.get('edges', [])
            
            preview_nodes = nodes[:5] if nodes else []
            preview_edges = [e for e in edges if 
                            e.get('source') in [n.get('id') for n in preview_nodes] and 
                            e.get('target') in [n.get('id') for n in preview_nodes]]
            
            return {
                'preview': {
                    'nodes': preview_nodes,
                    'edges': preview_edges
                },
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
        except Exception as e:
            logger.debug(f"Error loading template preview from {definition_path}: {e}")
            return None
    
    async def _load_template_definition(self, definition_path: str) -> Dict[str, Any]:
        """Load the full template definition from a file."""
        try:
            # Try multiple potential paths
            paths_to_try = [
                Path(definition_path),
                self.base_template_dir.parent / definition_path,
                Path("/workspace/aictrlnet-fastapi") / definition_path,
                Path("/workspace/aictrlnet-fastapi-business") / definition_path,
                Path("/workspace/aictrlnet-fastapi-enterprise") / definition_path,
            ]
            
            template_file = None
            for path in paths_to_try:
                if path.exists():
                    template_file = path
                    break
            
            if not template_file:
                raise FileNotFoundError(f"Template file not found: {definition_path}")
            
            with open(template_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading template definition from {definition_path}: {e}")
            raise
    
    async def list_templates(
        self,
        db: AsyncSession,
        user_id: str,
        request: TemplateListRequest
    ) -> tuple[List[WorkflowTemplateResponse], int]:
        """List workflow templates accessible to the user."""
        query = select(WorkflowTemplate)
        
        # Build access filter
        access_conditions = []
        
        if request.include_public:
            access_conditions.append(WorkflowTemplate.is_public == True)
        
        if request.include_system:
            access_conditions.append(WorkflowTemplate.is_system == True)
        
        if request.include_private:
            # User's own templates
            access_conditions.append(WorkflowTemplate.owner_id == user_id)
            
            # Templates with explicit permissions
            permission_subquery = select(WorkflowTemplatePermission.template_id).where(
                WorkflowTemplatePermission.user_id == user_id,
                WorkflowTemplatePermission.permission.in_([TemplatePermission.VIEW, TemplatePermission.USE])
            )
            access_conditions.append(WorkflowTemplate.id.in_(permission_subquery))
        
        if access_conditions:
            query = query.where(or_(*access_conditions))
        
        # Apply filters
        if request.category:
            query = query.where(WorkflowTemplate.category == request.category)
        
        if request.tags:
            # Filter by any matching tag
            query = query.where(WorkflowTemplate.tags.overlap(request.tags))
        
        if request.edition:
            query = query.where(WorkflowTemplate.edition == request.edition)
        else:
            # Community edition can only see Community templates by default
            query = query.where(WorkflowTemplate.edition == 'community')
        
        if request.complexity:
            query = query.where(WorkflowTemplate.complexity == request.complexity)
        
        if request.search:
            search_term = f"%{request.search}%"
            query = query.where(
                or_(
                    WorkflowTemplate.name.ilike(search_term),
                    WorkflowTemplate.description.ilike(search_term)
                )
            )
        
        # Count total before pagination
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply sorting
        if request.sort_by == "name":
            order_by = WorkflowTemplate.name
        elif request.sort_by == "updated_at":
            order_by = WorkflowTemplate.updated_at
        elif request.sort_by == "usage_count":
            order_by = WorkflowTemplate.usage_count
        elif request.sort_by == "rating":
            order_by = WorkflowTemplate.rating
        else:
            order_by = WorkflowTemplate.created_at
        
        if request.sort_desc:
            order_by = order_by.desc()
        
        query = query.order_by(order_by)
        
        # Apply pagination
        query = query.offset(request.skip).limit(request.limit)
        
        # Execute query
        result = await db.execute(query)
        templates = result.scalars().all()
        
        # Add preview data to each template
        responses = []
        for template in templates:
            response = WorkflowTemplateResponse.model_validate(template)
            
            # Try to load preview data from the template definition file
            if template.definition_path:
                preview_data = self._load_template_preview(template.definition_path)
                if preview_data:
                    response.preview = preview_data.get('preview')
                    response.node_count = preview_data.get('node_count')
                    response.edge_count = preview_data.get('edge_count')
            
            responses.append(response)
        
        return responses, total
    
    async def get_template(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str,
        load_definition: bool = False
    ) -> WorkflowTemplateDetail:
        """Get a specific workflow template."""
        query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
        result = await db.execute(query)
        template = result.scalar_one_or_none()
        
        if not template:
            raise NotFoundError(f"Template {template_id} not found")
        
        # Check access permissions
        if not await self._can_access_template(db, template, user_id):
            raise ForbiddenError("Access denied to this template")
        
        # Create response
        response = WorkflowTemplateDetail.model_validate(template)
        
        # Load owner name if available
        if template.owner_id:
            owner_query = select(User).where(User.id == template.owner_id)
            owner_result = await db.execute(owner_query)
            owner = owner_result.scalar_one_or_none()
            if owner:
                response.owner_name = owner.email
        
        # Check edit/delete permissions
        response.can_edit = await self._can_edit_template(db, template, user_id)
        response.can_delete = await self._can_delete_template(db, template, user_id)
        
        # Load workflow definition if requested
        if load_definition:
            try:
                # Construct full path - keep relative paths relative
                if template.definition_path.startswith('/'):
                    full_path = template.definition_path
                else:
                    # Keep relative paths relative so they work from current directory
                    full_path = template.definition_path
                
                definition = await self._load_template_definition(full_path)
                # Check if workflow is in a nested 'workflow' field (new format) or at root (old format)
                if 'workflow' in definition and isinstance(definition['workflow'], dict):
                    response.workflow_definition = definition['workflow']
                else:
                    response.workflow_definition = definition
                response.parameters = definition.get("parameters", [])
            except Exception as e:
                logger.error(f"Failed to load template definition: {e}")
                response.workflow_definition = None
        
        return response
    
    async def create_template(
        self,
        db: AsyncSession,
        user_id: str,
        create_data: WorkflowTemplateCreate
    ) -> WorkflowTemplateResponse:
        """Create a new workflow template."""
        # Generate file path
        file_path = self._generate_file_path(user_id, create_data.name, is_personal=True)
        
        # Create template record
        template = WorkflowTemplate(
            **create_data.model_dump(exclude={"definition_path"}),
            owner_id=user_id,
            definition_path=str(file_path)
        )
        
        db.add(template)
        await db.commit()
        await db.refresh(template)
        
        # Create initial template file
        await self._create_template_file(file_path, {
            "version": "1.0",
            "id": str(template.id),
            "name": template.name,
            "description": template.description,
            "metadata": {
                "created": datetime.utcnow().isoformat(),
                "author": user_id
            },
            "workflow": {
                "nodes": [],
                "edges": []
            }
        })
        
        return WorkflowTemplateResponse.model_validate(template)
    
    async def get_template_by_id(
        self,
        db: AsyncSession,
        template_id: UUID
    ) -> Optional[WorkflowTemplateDetail]:
        """Get a template by ID without user authorization checks (for internal use)."""
        query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
        result = await db.execute(query)
        template = result.scalar_one_or_none()
        
        if not template:
            return None
        
        # Create response
        response = WorkflowTemplateDetail.model_validate(template)
        
        # Load owner name if available
        if template.owner_id:
            owner_query = select(User).where(User.id == template.owner_id)
            owner_result = await db.execute(owner_query)
            owner = owner_result.scalar_one_or_none()
            if owner:
                response.owner_name = owner.email
        
        # Load definition if available
        try:
            if template.definition_path:
                definition = await self._load_template_definition(Path(template.definition_path))
                response.workflow_definition = definition
                response.parameters = definition.get("parameters", [])
        except Exception as e:
            logger.error(f"Failed to load template definition: {e}")
            response.workflow_definition = None
        
        return response
    
    async def update_template(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str,
        update_data: WorkflowTemplateUpdate
    ) -> WorkflowTemplateResponse:
        """Update a workflow template."""
        # Get template
        query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
        result = await db.execute(query)
        template = result.scalar_one_or_none()
        
        if not template:
            raise NotFoundError(f"Template {template_id} not found")
        
        # Check permissions
        if not await self._can_edit_template(db, template, user_id):
            raise ForbiddenError("You don't have permission to edit this template")
        
        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(template, field, value)
        
        template.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(template)
        
        return WorkflowTemplateResponse.model_validate(template)
    
    async def delete_template(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str
    ) -> None:
        """Delete a workflow template."""
        # Get template
        query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
        result = await db.execute(query)
        template = result.scalar_one_or_none()
        
        if not template:
            raise NotFoundError(f"Template {template_id} not found")
        
        # Check permissions
        if not await self._can_delete_template(db, template, user_id):
            raise ForbiddenError("You don't have permission to delete this template")
        
        # Delete file
        try:
            if os.path.exists(template.definition_path):
                os.remove(template.definition_path)
        except Exception as e:
            logger.error(f"Failed to delete template file: {e}")
        
        # Delete record
        await db.delete(template)
        await db.commit()
    
    async def instantiate_template(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str,
        request: InstantiateTemplateRequest
    ) -> Dict[str, Any]:
        """Create a workflow instance from a template."""
        # Get template
        template = await self.get_template(db, template_id, user_id, load_definition=True)
        
        if not template.workflow_definition:
            raise ValidationError("Template has no workflow definition")
        
        # Apply parameters to workflow
        workflow_config = self._apply_parameters(
            template.workflow_definition,
            request.parameters
        )
        
        # Check if we should use the unified enhancement pipeline
        use_unified_pipeline = hasattr(request, 'enhancements')
        
        if use_unified_pipeline:
            # Use the new unified enhancement pipeline
            try:
                from aictrlnet_business.services.unified_enhancement_pipeline import unified_enhancement_pipeline
                
                # Prepare enhancement config with both main options and sub-options
                user_options = {}
                if hasattr(request, 'enhancements') and request.enhancements:
                    user_options.update(request.enhancements)
                if hasattr(request, 'enhancement_sub_options') and request.enhancement_sub_options:
                    user_options['sub_options'] = request.enhancement_sub_options
                
                enhancement_config = {
                    'source': 'template',
                    'automatic': False,
                    'user_options': user_options,
                    'context': {
                        'template_id': str(template_id),
                        'template_name': template.name,
                        'template_category': template.category,
                        'user_id': user_id
                    }
                }
                
                # Enhance the workflow
                workflow_config = await unified_enhancement_pipeline.enhance(
                    {'definition': workflow_config, 'name': request.name},
                    enhancement_config,
                    db
                )
                
                # Extract the definition back out
                if 'definition' in workflow_config:
                    enhanced_definition = workflow_config['definition']
                    enhanced_metadata = workflow_config.get('metadata', {})
                else:
                    enhanced_definition = workflow_config
                    enhanced_metadata = {}
                    
                workflow_config = enhanced_definition
                
            except ImportError:
                # Pipeline not available (Community edition), proceed without enhancement
                logger.info("Unified enhancement pipeline not available, using basic instantiation")
                enhanced_metadata = {}
        
        else:
            # No enhancement requested
            enhanced_metadata = {}
        
        # First, we need to create a WorkflowDefinition
        from models.community_complete import WorkflowDefinition, WorkflowInstance, WorkflowStatus
        from datetime import datetime
        
        # Merge template metadata with enhancement metadata
        full_metadata = {
            **(template.metadata if hasattr(template, 'metadata') and template.metadata else {}),
            **enhanced_metadata
        }
        
        # Create workflow definition with enhanced metadata
        workflow_def = WorkflowDefinition(
            name=request.name,
            description=request.description or template.description,
            definition=workflow_config,
            active=True,
            tags=template.tags,
            metadata=full_metadata if full_metadata else None,
            tenant_id=user_id  # Using user_id as tenant_id for now
        )
        db.add(workflow_def)
        await db.flush()  # Get the ID without committing
        
        # Create workflow instance
        workflow = WorkflowInstance(
            definition_id=workflow_def.id,
            name=request.name,
            status=WorkflowStatus.PENDING,
            input_data=request.parameters,
            tenant_id=user_id  # Using user_id as tenant_id for now
        )
        
        db.add(workflow)
        await db.flush()  # Get the ID without committing
        
        # Record usage
        usage = WorkflowTemplateUsage(
            template_id=template_id,
            user_id=user_id,
            workflow_id=workflow.id,
            customization_level=CustomizationLevel.NONE
        )
        db.add(usage)
        
        # Update usage count
        update_query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
        update_result = await db.execute(update_query)
        template_record = update_result.scalar_one()
        template_record.usage_count += 1
        
        await db.commit()
        await db.refresh(workflow)
        await db.refresh(workflow_def)
        
        # Note: WorkflowMetadataConnector is in Business edition
        # Template instantiation runs in Community edition
        # Service connection will happen when accessed through Business edition API
        
        return {
            "workflow_id": workflow.id,  # Return the instance ID for tracking
            "workflow_definition_id": workflow_def.id,  # Also include definition ID
            "workflow_name": workflow.name,
            "template_id": str(template_id),
            "template_name": template.name,
            "created_at": workflow.created_at.isoformat() if hasattr(workflow.created_at, 'isoformat') else str(workflow.created_at)
        }
    
    async def fork_template(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str,
        request: ForkTemplateRequest
    ) -> WorkflowTemplateResponse:
        """Fork a template for customization."""
        # Get original template
        original = await self.get_template(db, template_id, user_id, load_definition=True)
        
        # Create new template
        create_data = WorkflowTemplateCreate(
            name=request.new_name,
            description=request.description or f"Forked from {original.name}",
            category=original.category,
            tags=original.tags + ["forked"],
            edition=original.edition,
            is_public=request.make_public,
            parent_template_id=template_id,
            complexity=original.complexity,
            estimated_duration=original.estimated_duration,
            required_adapters=original.required_adapters,
            required_capabilities=original.required_capabilities,
            definition_path="temp"  # Will be updated
        )
        
        new_template = await self.create_template(db, user_id, create_data)
        
        # Copy workflow definition
        if original.workflow_definition:
            definition_path = Path(new_template.definition_path)
            await self._create_template_file(definition_path, {
                "version": "1.0",
                "id": str(new_template.id),
                "name": new_template.name,
                "description": new_template.description,
                "metadata": {
                    "created": datetime.utcnow().isoformat(),
                    "author": user_id,
                    "forked_from": str(template_id)
                },
                "parameters": original.parameters,
                "workflow": original.workflow_definition
            })
        
        return new_template
    
    async def add_review(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str,
        review_data: TemplateReviewCreate
    ) -> None:
        """Add a review to a template."""
        # Check if template exists and user has access
        template = await self.get_template(db, template_id, user_id)
        
        # Check if user already reviewed
        existing_query = select(WorkflowTemplateReview).where(
            and_(
                WorkflowTemplateReview.template_id == template_id,
                WorkflowTemplateReview.user_id == user_id
            )
        )
        existing_result = await db.execute(existing_query)
        existing = existing_result.scalar_one_or_none()
        
        if existing:
            # Update existing review
            existing.rating = review_data.rating
            existing.review = review_data.review
        else:
            # Create new review
            review = WorkflowTemplateReview(
                template_id=template_id,
                user_id=user_id,
                **review_data.model_dump()
            )
            db.add(review)
        
        # Update template rating
        await self._update_template_rating(db, template_id)
        
        await db.commit()
    
    # Helper methods
    
    async def _can_access_template(
        self,
        db: AsyncSession,
        template: WorkflowTemplate,
        user_id: str
    ) -> bool:
        """Check if user can access a template."""
        # Public or system templates
        if template.is_public or template.is_system:
            return True
        
        # Owner
        if template.owner_id == user_id:
            return True
        
        # Check explicit permissions
        perm_query = select(WorkflowTemplatePermission).where(
            and_(
                WorkflowTemplatePermission.template_id == template.id,
                WorkflowTemplatePermission.user_id == user_id,
                WorkflowTemplatePermission.permission.in_([
                    TemplatePermission.VIEW,
                    TemplatePermission.USE,
                    TemplatePermission.EDIT,
                    TemplatePermission.DELETE
                ])
            )
        )
        perm_result = await db.execute(perm_query)
        return perm_result.scalar_one_or_none() is not None
    
    async def _can_edit_template(
        self,
        db: AsyncSession,
        template: WorkflowTemplate,
        user_id: str
    ) -> bool:
        """Check if user can edit a template."""
        # System templates cannot be edited
        if template.is_system:
            return False
        
        # Owner can edit
        if template.owner_id == user_id:
            return True
        
        # Check explicit edit permission
        perm_query = select(WorkflowTemplatePermission).where(
            and_(
                WorkflowTemplatePermission.template_id == template.id,
                WorkflowTemplatePermission.user_id == user_id,
                WorkflowTemplatePermission.permission == TemplatePermission.EDIT
            )
        )
        perm_result = await db.execute(perm_query)
        return perm_result.scalar_one_or_none() is not None
    
    async def _can_delete_template(
        self,
        db: AsyncSession,
        template: WorkflowTemplate,
        user_id: str
    ) -> bool:
        """Check if user can delete a template."""
        # System templates cannot be deleted
        if template.is_system:
            return False
        
        # Only owner can delete
        return template.owner_id == user_id
    
    # NOTE: _load_template_definition is defined at line ~138 with path resolution logic
    # DO NOT add another definition here - it would shadow the correct implementation
    
    async def _create_template_file(self, file_path: Path, content: Dict[str, Any]) -> None:
        """Create a template file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create template file: {e}")
            raise ValidationError(f"Failed to create template file: {str(e)}")
    
    def _generate_file_path(
        self,
        user_id: str,
        template_name: str,
        is_personal: bool = True
    ) -> Path:
        """Generate a file path for a template."""
        # Sanitize name for filesystem
        safe_name = "".join(c for c in template_name if c.isalnum() or c in (' ', '-', '_'))
        safe_name = safe_name.replace(' ', '-').lower()
        
        if is_personal:
            return self.personal_dir / user_id / f"{safe_name}.json"
        else:
            # Organization templates would go here
            return self.org_dir / f"{safe_name}.json"
    
    def _apply_parameters(
        self,
        workflow_definition: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply parameters to a workflow definition."""
        # This is a simplified version - in reality, you'd need to:
        # 1. Validate parameters against template parameter definitions
        # 2. Apply conditionals based on parameters
        # 3. Filter nodes/edges based on conditions
        # 4. Replace parameter placeholders in prompts/configs
        
        # Fix position format if needed (convert arrays to objects)
        workflow_def = self._fix_workflow_positions(workflow_definition)
        
        return workflow_def
    
    def _fix_workflow_positions(self, workflow_def: Dict[str, Any]) -> Dict[str, Any]:
        """Convert position arrays [x, y] to objects {x: x, y: y} for compatibility."""
        import copy
        fixed_def = copy.deepcopy(workflow_def)
        
        # Fix nodes positions
        if 'nodes' in fixed_def and isinstance(fixed_def['nodes'], list):
            for node in fixed_def['nodes']:
                if 'position' in node:
                    pos = node['position']
                    # Convert array to object if needed
                    if isinstance(pos, list) and len(pos) >= 2:
                        node['position'] = {'x': float(pos[0]), 'y': float(pos[1])}
                    elif isinstance(pos, dict):
                        # Ensure x and y are floats
                        if 'x' in pos:
                            pos['x'] = float(pos['x'])
                        if 'y' in pos:
                            pos['y'] = float(pos['y'])
        
        return fixed_def
    
    async def _update_template_rating(
        self,
        db: AsyncSession,
        template_id: UUID
    ) -> None:
        """Update the average rating for a template."""
        # Calculate average rating
        rating_query = select(func.avg(WorkflowTemplateReview.rating)).where(
            WorkflowTemplateReview.template_id == template_id
        )
        rating_result = await db.execute(rating_query)
        avg_rating = rating_result.scalar()
        
        if avg_rating is not None:
            # Update template
            template_query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
            template_result = await db.execute(template_query)
            template = template_result.scalar_one()
            template.rating = round(avg_rating, 2)
    
    async def get_template_usage_count(
        self,
        db: AsyncSession,
        template_id: UUID
    ) -> int:
        """Get the number of workflows using this template."""
        query = select(func.count()).where(
            WorkflowTemplateUsage.template_id == template_id
        )
        result = await db.execute(query)
        return result.scalar() or 0
    
    async def delete_template(
        self,
        db: AsyncSession,
        template_id: UUID,
        user_id: str,
        soft_delete: bool = True
    ) -> None:
        """Delete a template (soft or hard delete)."""
        # Get template
        template = await self.get_template(db, template_id, user_id)
        
        # Check permissions
        if not self._can_delete(template, user_id):
            raise ForbiddenError("You don't have permission to delete this template")
        
        if soft_delete:
            # Mark as deleted
            template.is_deleted = True
            template.deleted_at = datetime.utcnow()
        else:
            # Hard delete
            await db.delete(template)
        
        await db.commit()
    
    async def copy_template_definition(
        self,
        source_path: str,
        dest_path: str
    ) -> None:
        """Copy a template definition file."""
        try:
            source_file = Path(source_path)
            dest_file = Path(dest_path)
            
            # Create destination directory
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            with open(source_file, 'r') as src:
                content = json.load(src)
            
            with open(dest_file, 'w') as dst:
                json.dump(content, dst, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to copy template: {e}")
            raise ValidationError(f"Failed to copy template: {str(e)}")
    
    async def initialize_system_templates(self, db: AsyncSession) -> int:
        """Initialize system templates from the filesystem."""
        count = 0
        
        if not self.system_dir.exists():
            logger.warning(f"System template directory does not exist: {self.system_dir}")
            return 0
        
        for category_dir in self.system_dir.iterdir():
            if category_dir.is_dir():
                for template_file in category_dir.glob("*.json"):
                    # Skip metadata files
                    if template_file.name.endswith('.metadata.json'):
                        continue
                    
                    try:
                        # Load template definition
                        with open(template_file, 'r') as f:
                            template_data = json.load(f)
                        
                        # Check if this is a Community template
                        template_edition = template_data.get('edition', 'community')
                        if template_edition != 'community':
                            # Skip non-community templates in Community edition
                            continue
                        
                        # Check if template already exists
                        existing = await db.execute(
                            select(WorkflowTemplate).where(
                                and_(
                                    WorkflowTemplate.name == template_data.get('name'),
                                    WorkflowTemplate.is_system == True
                                )
                            )
                        )
                        if existing.scalar_one_or_none():
                            logger.debug(f"Template already exists: {template_data.get('name')}")
                            continue
                        
                        # Create template record
                        template = WorkflowTemplate(
                            name=template_data.get('name'),
                            description=template_data.get('description'),
                            category=category_dir.name,
                            tags=template_data.get('tags', []),
                            edition='community',
                            is_public=True,
                            is_system=True,
                            version=int(template_data.get('version', '1.0').replace('.', '')),
                            definition_path=str(template_file.relative_to(self.base_template_dir.parent)),
                            complexity=template_data.get('metadata', {}).get('complexity', 'moderate'),
                            estimated_duration=template_data.get('metadata', {}).get('estimatedDuration'),
                            required_adapters=template_data.get('metadata', {}).get('requiredAdapters', []),
                            required_capabilities=template_data.get('metadata', {}).get('requiredCapabilities', []),
                            usage_count=0
                        )
                        
                        db.add(template)
                        count += 1
                        logger.info(f"Added Community template: {template.name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading template {template_file}: {e}")
        
        await db.commit()
        logger.info(f"Initialized {count} Community system templates")
        return count
    
    async def get_templates_by_ids(
        self,
        db: AsyncSession,
        template_ids: List[str],
        user_id: str
    ) -> List[WorkflowTemplate]:
        """Get multiple templates by their IDs."""
        if not template_ids:
            return []
        
        # Convert string IDs to UUIDs
        uuid_ids = [UUID(tid) for tid in template_ids]
        
        query = select(WorkflowTemplate).where(
            WorkflowTemplate.id.in_(uuid_ids)
        )
        
        result = await db.execute(query)
        templates = result.scalars().all()
        
        # Filter by access permissions
        accessible_templates = []
        for template in templates:
            if await self._can_view(db, template, user_id):
                accessible_templates.append(template)
        
        return accessible_templates
    
    async def import_template(
        self,
        source_url: str,
        format: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Import a template from external source."""
        # This is a simplified implementation
        # In production, you'd need to:
        # 1. Fetch content from URL
        # 2. Parse based on format (JSON, YAML, BPMN)
        # 3. Convert to internal format
        # 4. Validate structure
        
        import httpx
        import yaml
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(source_url)
                response.raise_for_status()
                content = response.text
            except Exception as e:
                raise ValidationError(f"Failed to fetch template: {str(e)}")
        
        # Parse based on format
        try:
            if format == "json":
                data = json.loads(content)
            elif format == "yaml":
                data = yaml.safe_load(content)
            elif format == "bpmn":
                # Would need BPMN parser
                raise ValidationError("BPMN import not yet implemented")
            else:
                raise ValidationError(f"Unsupported format: {format}")
        except Exception as e:
            raise ValidationError(f"Failed to parse template: {str(e)}")
        
        # Convert to internal format
        template_data = {
            "name": data.get("name", "Imported Template"),
            "description": data.get("description", ""),
            "category": data.get("category", "imported"),
            "tags": data.get("tags", ["imported"]),
            "edition": "business",
            "is_public": False,
            "complexity": data.get("complexity", "moderate"),
            "estimated_duration": data.get("estimatedDuration", "varies"),
            "required_adapters": data.get("requiredAdapters", []),
            "required_capabilities": data.get("requiredCapabilities", []),
            "definition_path": f"imported/{user_id}/{data.get('name', 'template').lower().replace(' ', '-')}.json"
        }
        
        return template_data
    
    async def export_template(
        self,
        export_data: Dict[str, Any],
        format: str
    ) -> str:
        """Export a template to specified format."""
        if format == "json":
            return json.dumps(export_data, indent=2)
        elif format == "yaml":
            import yaml
            return yaml.dump(export_data, default_flow_style=False)
        elif format == "pdf":
            # Would need PDF generation library
            raise ValidationError("PDF export not yet implemented")
        elif format == "bpmn":
            # Would need BPMN converter
            raise ValidationError("BPMN export not yet implemented")
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def get_content_type(self, format: str) -> str:
        """Get the content type for a format."""
        content_types = {
            "json": "application/json",
            "yaml": "application/x-yaml",
            "pdf": "application/pdf",
            "bpmn": "application/xml"
        }
        return content_types.get(format, "application/octet-stream")
    
    def _validate_template_schema(self, template_data: Dict[str, Any], file_path: Path) -> List[str]:
        """Validate template data against schema requirements."""
        errors = []
        
        # Required fields
        required_fields = ['name', 'description', 'category', 'edition', 'version']
        for field in required_fields:
            if field not in template_data or not template_data[field]:
                errors.append(f"Missing required field: '{field}'")
        
        # Validate edition
        valid_editions = ['community', 'business', 'enterprise']
        if 'edition' in template_data and template_data['edition'] not in valid_editions:
            errors.append(f"Invalid edition '{template_data['edition']}'. Must be one of: {valid_editions}")
        
        # Validate complexity if present
        if 'complexity' in template_data:
            valid_complexities = ['simple', 'moderate', 'complex', 'expert', 'advanced']
            if template_data['complexity'] not in valid_complexities:
                errors.append(f"Invalid complexity '{template_data['complexity']}'. Must be one of: {valid_complexities}")
        
        # Validate steps structure if present
        if 'steps' in template_data:
            if not isinstance(template_data['steps'], list):
                errors.append("Field 'steps' must be a list")
            else:
                for i, step in enumerate(template_data['steps']):
                    if not isinstance(step, dict):
                        errors.append(f"Step {i} must be a dictionary")
                    elif 'name' not in step:
                        errors.append(f"Step {i} missing required field 'name'")
        
        # Validate tags if present
        if 'tags' in template_data and not isinstance(template_data['tags'], list):
            errors.append("Field 'tags' must be a list")
        
        return errors
    
    async def initialize_system_templates(self, db: AsyncSession) -> int:
        """Initialize system templates by scanning ONLY Community template directories.
        
        This method ONLY loads templates from Community-specific directories to prevent
        cross-contamination when called by Business/Enterprise services through inheritance.
        
        Returns the number of templates initialized.
        """
        count = 0
        errors = []
        skipped = 0
        
        # Define Community-specific template categories ONLY
        # This prevents loading Business/Enterprise templates when this method is called via super()
        community_categories = {'communication', 'data', 'reliability'}
        
        logger.info(f"Community service initializing templates for categories: {community_categories}")
        
        # Scan system template directory, but ONLY Community categories
        if self.system_dir.exists():
            for category_dir in self.system_dir.iterdir():
                if category_dir.is_dir() and category_dir.name in community_categories:
                    logger.info(f"Processing Community category: {category_dir.name}")
                    for template_file in category_dir.glob("*.json"):
                        # Skip metadata files
                        if template_file.name.endswith('.metadata.json'):
                            continue
                        
                        try:
                            # Load template definition
                            with open(template_file, 'r') as f:
                                template_data = json.load(f)
                            
                            # Validate template schema
                            schema_errors = self._validate_template_schema(template_data, template_file)
                            if schema_errors:
                                error_msg = f"Schema validation failed for {template_file}: {'; '.join(schema_errors)}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                                continue
                                
                            # Check if template already exists
                            existing = await db.execute(
                                select(WorkflowTemplate).where(
                                    and_(
                                        WorkflowTemplate.name == template_data.get('name'),
                                        WorkflowTemplate.is_system == True
                                    )
                                )
                            )
                            if existing.scalar_one_or_none():
                                logger.debug(f"Template already exists: {template_data.get('name')}")
                                skipped += 1
                                continue
                            
                            # Force edition to 'community' for all templates loaded by this method
                            template = WorkflowTemplate(
                                name=template_data.get('name'),
                                description=template_data.get('description', ''),
                                category=template_data.get('category', category_dir.name),
                                tags=template_data.get('tags', []),
                                edition='community',  # Force Community edition
                                is_public=True,
                                is_system=True,
                                version=int(template_data.get('version', '1.0.0').split('.')[0]),
                                definition_path=str(template_file.relative_to(self.base_template_dir.parent)),
                                complexity=template_data.get('complexity', 'moderate'),
                                estimated_duration=template_data.get('estimatedDuration'),
                                required_adapters=template_data.get('requiredAdapters', []),
                                required_capabilities=template_data.get('requiredCapabilities', []),
                                usage_count=0
                            )
                            
                            db.add(template)
                            count += 1
                            logger.info(f"Initialized Community template: {template.name}")
                            
                        except json.JSONDecodeError as e:
                            error_msg = f"JSON syntax error in {template_file}: Line {e.lineno}, Col {e.colno}: {e.msg}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                        except Exception as e:
                            error_msg = f"Error loading template {template_file}: {type(e).__name__}: {str(e)}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                else:
                    # Log skipped categories to help debugging
                    if category_dir.is_dir():
                        logger.debug(f"Skipping non-Community category: {category_dir.name}")
            
            await db.commit()
        
        # Summary logging
        logger.info(f"Community template initialization complete:")
        logger.info(f"  - Loaded: {count} templates")
        logger.info(f"  - Skipped (already exist): {skipped} templates")
        if errors:
            logger.warning(f"  - Errors: {len(errors)} templates failed to load")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"    • {error}")
            if len(errors) > 5:
                logger.warning(f"    ... and {len(errors) - 5} more errors")
        
        return count
    
    async def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get templates by category for legacy compatibility."""
        from schemas.workflow_templates import TemplateListRequest
        
        # Create a request to list templates by category
        request = TemplateListRequest(
            category=category,
            include_system=True,
            include_public=True,
            include_private=False,
            limit=100
        )
        
        # Use existing list_templates method but need a db session
        # For compatibility, we'll create a simple sync wrapper
        import asyncio
        from core.database import get_session_maker

        async def _get_templates():
            async with get_session_maker()() as db:
                templates, _ = await self.list_templates(db, "system", request)
                return [template.model_dump() for template in templates]
        
        # Run async method
        if asyncio.get_event_loop().is_running():
            # We're already in an async context
            raise RuntimeError("get_templates_by_category should be called from async context. Use async version.")
        else:
            return asyncio.run(_get_templates())
    
    async def get_templates_by_category_async(self, db: AsyncSession, category: str) -> List[Dict[str, Any]]:
        """Get templates by category (async version)."""
        from schemas.workflow_templates import TemplateListRequest
        
        request = TemplateListRequest(
            category=category,
            include_system=True,
            include_public=True,
            include_private=False,
            limit=100
        )
        
        templates, _ = await self.list_templates(db, "system", request)
        return [template.model_dump() for template in templates]
    
    async def generate_workflow_for_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow configuration from template for legacy compatibility."""
        import traceback
        template_keys = template.keys() if isinstance(template, dict) else type(template)
        logger.info(f"[TEMPLATE_DEBUG] generate_workflow_for_template called with template keys: {template_keys}")
        # Log stack trace to find caller when called with nodes/edges only
        if isinstance(template, dict) and 'nodes' in template and 'edges' in template and 'id' not in template:
            stack = ''.join(traceback.format_stack()[-6:-1])
            logger.warning(f"[TEMPLATE_DEBUG] Called with workflow (nodes/edges) instead of template! Stack:\n{stack}")

        # If template is already a dict, extract workflow definition from it
        if isinstance(template, dict):
            template_id = template.get('id')
            logger.info(f"[TEMPLATE_DEBUG] template_id={template_id}, is truthy={bool(template_id)}")
            if template_id:
                try:
                    import uuid
                    from core.database import get_session_maker

                    template_uuid = uuid.UUID(template_id) if isinstance(template_id, str) else template_id
                    logger.info(f"[TEMPLATE_DEBUG] Looking up template UUID: {template_uuid}")
                    async with get_session_maker()() as db:
                        template_detail = await self.get_template(db, template_uuid, "system", load_definition=True)
                        logger.info(f"[TEMPLATE_DEBUG] template_detail found: {template_detail is not None}, workflow_definition: {template_detail.workflow_definition is not None if template_detail else 'N/A'}")
                        if template_detail and template_detail.workflow_definition:
                            logger.info(f"[TEMPLATE_DEBUG] Returning workflow with {len(template_detail.workflow_definition.get('nodes', []))} nodes")
                            return template_detail.workflow_definition
                        else:
                            logger.warning(f"[TEMPLATE_DEBUG] No workflow_definition found for template {template_id}")
                except Exception as e:
                    logger.error(f"Failed to load template {template_id} from DB: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: if template has workflow_definition or workflow directly
            # Templates can use either key - check both for compatibility
            if 'workflow_definition' in template:
                return template['workflow_definition']
            if 'workflow' in template:
                return template['workflow']

            # Generate basic workflow structure from template metadata
            return self._generate_basic_workflow_from_template(template)
        
        # If template is a Pydantic model, get its workflow definition
        # Check both workflow_definition and workflow attributes for compatibility
        if hasattr(template, 'workflow_definition') and template.workflow_definition:
            return template.workflow_definition
        if hasattr(template, 'workflow') and template.workflow:
            return template.workflow

        # Generate basic workflow from template data
        template_dict = template.model_dump() if hasattr(template, 'model_dump') else dict(template)

        # Before falling back to basic, check if the dict conversion has workflow
        if 'workflow' in template_dict:
            return template_dict['workflow']
        if 'workflow_definition' in template_dict:
            return template_dict['workflow_definition']

        return self._generate_basic_workflow_from_template(template_dict)
    
    def _generate_basic_workflow_from_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a basic workflow structure from template metadata."""
        category = template.get('category', 'general')
        name = template.get('name', 'Workflow')
        
        # Create basic workflow structure based on category
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
        
        # Add category-specific nodes
        if category == "ai":
            node_id = "ai_process"
            nodes.append({
                "id": node_id,
                "type": "aiProcess",
                "name": "AI Processing",
                "position": {"x": x_pos, "y": 100},  
                "data": {"capability": "ai_processing", "description": f"AI processing for {name}"}
            })
            edges.append({"from": prev_node, "to": node_id})
            prev_node = node_id
            x_pos += 200
            
        elif category == "business":
            node_id = "approval"
            nodes.append({
                "id": node_id,
                "type": "approval",
                "name": "Approval",
                "position": {"x": x_pos, "y": 100},
                "data": {"capability": "approval_workflow", "description": f"Approval step for {name}"}
            })
            edges.append({"from": prev_node, "to": node_id})
            prev_node = node_id
            x_pos += 200
            
        elif category == "communication":
            node_id = "notify"
            nodes.append({
                "id": node_id,
                "type": "adapter",
                "name": "Send Notification", 
                "position": {"x": x_pos, "y": 100},
                "data": {"capability": "notification", "description": f"Notification for {name}"}
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
                "name": "Process",
                "position": {"x": x_pos, "y": 100},
                "data": {"description": f"Process step for {name}"}
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
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "template_id": template.get('id'),
                "template_name": template.get('name'),
                "generated_from": "template"
            }
        }


def create_workflow_template_service() -> 'WorkflowTemplateService':
    """Factory function to create the appropriate WorkflowTemplateService based on edition.
    
    This enables the accretive model where Business and Enterprise editions extend Community.
    """
    edition = os.getenv('AICTRLNET_EDITION', 'community').lower()
    
    if edition == 'business':
        try:
            # Import Business service that extends Community
            from aictrlnet_business.services.workflow_template_service import WorkflowTemplateService as BusinessService
            logger.info("Using Business WorkflowTemplateService")
            return BusinessService()
        except ImportError:
            logger.warning("Business WorkflowTemplateService not found, falling back to Community")
            return WorkflowTemplateService()
    
    elif edition == 'enterprise':
        try:
            # Import Enterprise service that extends Business (which extends Community)
            from aictrlnet_enterprise.services.workflow_template_service import WorkflowTemplateService as EnterpriseService
            logger.info("Using Enterprise WorkflowTemplateService")
            return EnterpriseService()
        except ImportError:
            logger.warning("Enterprise WorkflowTemplateService not found, falling back to Community")
            return WorkflowTemplateService()
    
    else:
        logger.info("Using Community WorkflowTemplateService")
        return WorkflowTemplateService()