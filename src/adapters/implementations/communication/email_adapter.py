"""Email adapter implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import base64
from datetime import datetime

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class EmailAdapter(BaseAdapter):
    """Adapter for email integration via SMTP."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)
        
        # Discovery mode support
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        # SMTP configuration
        self.smtp_host = config.credentials.get("smtp_host", "smtp.gmail.com") if config.credentials else "smtp.gmail.com"
        self.smtp_port = config.credentials.get("smtp_port", 587) if config.credentials else 587
        self.smtp_username = (config.credentials.get("smtp_username") or config.api_key) if config.credentials else None
        self.smtp_password = (config.credentials.get("smtp_password") or config.api_secret) if config.credentials else None
        self.smtp_use_tls = config.credentials.get("smtp_use_tls", True) if config.credentials else True
        
        # Sender configuration
        self.default_from_email = config.credentials.get("from_email", self.smtp_username) if config.credentials else None
        self.default_from_name = config.credentials.get("from_name", "AICtrlNet") if config.credentials else "AICtrlNet"
        
        # Skip validation in discovery mode
        if not self.discovery_only and (not self.smtp_username or not self.smtp_password):
            raise ValueError("SMTP username and password are required")
    
    async def initialize(self) -> None:
        """Initialize the email adapter."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Email adapter initialized in discovery mode")
            return
        
        # Test SMTP connection
        try:
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.smtp_use_tls
            ) as smtp:
                await smtp.login(self.smtp_username, self.smtp_password)
                
            logger.info(f"Email adapter initialized with {self.smtp_host}:{self.smtp_port}")
        except Exception as e:
            logger.error(f"Failed to initialize email adapter: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        logger.info("Email adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return email adapter capabilities."""
        return [
            AdapterCapability(
                name="send_email",
                description="Send an email message",
                category="messaging",
                parameters={
                    "to": {"type": "string|array", "description": "Recipient email address(es)"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body (plain text or HTML)"},
                    "body_html": {"type": "string", "description": "HTML body (optional, overrides body)"},
                    "cc": {"type": "string|array", "description": "CC recipients"},
                    "bcc": {"type": "string|array", "description": "BCC recipients"},
                    "from_email": {"type": "string", "description": "Override sender email"},
                    "from_name": {"type": "string", "description": "Override sender name"},
                    "reply_to": {"type": "string", "description": "Reply-to address"},
                    "attachments": {"type": "array", "description": "File attachments"}
                },
                required_parameters=["to", "subject", "body"],
                async_supported=True,
                estimated_duration_seconds=1.0
            ),
            AdapterCapability(
                name="send_template_email",
                description="Send an email using a template",
                category="messaging",
                parameters={
                    "to": {"type": "string|array", "description": "Recipient email address(es)"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "template_name": {"type": "string", "description": "Template name"},
                    "template_data": {"type": "object", "description": "Template variables"},
                    "cc": {"type": "string|array", "description": "CC recipients"},
                    "bcc": {"type": "string|array", "description": "BCC recipients"}
                },
                required_parameters=["to", "subject", "template_name"],
                async_supported=True,
                estimated_duration_seconds=1.0
            ),
            AdapterCapability(
                name="send_bulk_email",
                description="Send emails to multiple recipients",
                category="messaging",
                parameters={
                    "recipients": {"type": "array", "description": "Array of recipient objects"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body template"},
                    "personalization": {"type": "boolean", "description": "Enable personalization", "default": True},
                    "batch_size": {"type": "integer", "description": "Emails per batch", "default": 50}
                },
                required_parameters=["recipients", "subject", "body"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                rate_limit=100  # per minute
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to send email."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        if request.capability == "send_email":
            return await self._handle_send_email(request)
        elif request.capability == "send_template_email":
            return await self._handle_send_template_email(request)
        elif request.capability == "send_bulk_email":
            return await self._handle_send_bulk_email(request)
        else:
            raise ValueError(f"Unknown capability: {request.capability}")
    
    async def _handle_send_email(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a single email."""
        start_time = datetime.utcnow()
        
        try:
            # Prepare recipients
            to_addresses = self._normalize_addresses(request.parameters["to"])
            cc_addresses = self._normalize_addresses(request.parameters.get("cc", []))
            bcc_addresses = self._normalize_addresses(request.parameters.get("bcc", []))
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = request.parameters["subject"]
            msg["From"] = self._format_address(
                request.parameters.get("from_email", self.default_from_email),
                request.parameters.get("from_name", self.default_from_name)
            )
            msg["To"] = ", ".join(to_addresses)
            
            if cc_addresses:
                msg["Cc"] = ", ".join(cc_addresses)
            
            if reply_to := request.parameters.get("reply_to"):
                msg["Reply-To"] = reply_to
            
            # Add body
            body_text = request.parameters["body"]
            body_html = request.parameters.get("body_html", body_text)
            
            # Add text part
            msg.attach(MIMEText(body_text, "plain"))
            
            # Add HTML part if different from text
            if body_html != body_text:
                msg.attach(MIMEText(body_html, "html"))
            
            # Add attachments
            if attachments := request.parameters.get("attachments", []):
                for attachment in attachments:
                    await self._add_attachment(msg, attachment)
            
            # Send email
            all_recipients = to_addresses + cc_addresses + bcc_addresses
            
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.smtp_use_tls
            ) as smtp:
                await smtp.login(self.smtp_username, self.smtp_password)
                await smtp.send_message(msg, recipients=all_recipients)
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish email sent event
            await event_bus.publish(
                "adapter.email.sent",
                {
                    "to": to_addresses,
                    "subject": request.parameters["subject"],
                    "recipients_count": len(all_recipients)
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "sent": True,
                    "recipients": all_recipients,
                    "message_id": msg.get("Message-ID")
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_send_template_email(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a template email."""
        # Get template
        template_name = request.parameters["template_name"]
        template_data = request.parameters.get("template_data", {})
        
        # Load template (in a real implementation, this would load from a template engine)
        templates = {
            "welcome": {
                "body": "Welcome {{name}}! Thank you for joining {{company}}.",
                "body_html": "<h1>Welcome {{name}}!</h1><p>Thank you for joining {{company}}.</p>"
            },
            "notification": {
                "body": "Hi {{name}}, {{message}}",
                "body_html": "<p>Hi {{name}},</p><p>{{message}}</p>"
            }
        }
        
        template = templates.get(template_name)
        if not template:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=f"Template '{template_name}' not found"
            )
        
        # Render template
        body = self._render_template(template["body"], template_data)
        body_html = self._render_template(template.get("body_html", body), template_data)
        
        # Send email with rendered content
        email_request = AdapterRequest(
            capability="send_email",
            parameters={
                **request.parameters,
                "body": body,
                "body_html": body_html
            }
        )
        
        return await self._handle_send_email(email_request)
    
    async def _handle_send_bulk_email(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending bulk emails."""
        start_time = datetime.utcnow()
        
        try:
            recipients = request.parameters["recipients"]
            subject = request.parameters["subject"]
            body_template = request.parameters["body"]
            personalization = request.parameters.get("personalization", True)
            batch_size = request.parameters.get("batch_size", 50)
            
            sent_count = 0
            failed_count = 0
            errors = []
            
            # Process in batches
            for i in range(0, len(recipients), batch_size):
                batch = recipients[i:i + batch_size]
                
                # Send emails in parallel within batch
                tasks = []
                for recipient in batch:
                    if personalization:
                        # Personalize content
                        personalized_body = self._render_template(body_template, recipient)
                        personalized_subject = self._render_template(subject, recipient)
                    else:
                        personalized_body = body_template
                        personalized_subject = subject
                    
                    email_request = AdapterRequest(
                        capability="send_email",
                        parameters={
                            "to": recipient.get("email", recipient),
                            "subject": personalized_subject,
                            "body": personalized_body
                        }
                    )
                    
                    task = asyncio.create_task(self._handle_send_email(email_request))
                    tasks.append(task)
                
                # Wait for batch to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, AdapterResponse) and result.status == "success":
                        sent_count += 1
                    else:
                        failed_count += 1
                        if isinstance(result, AdapterResponse):
                            errors.append(result.error)
                
                # Rate limiting between batches
                if i + batch_size < len(recipients):
                    await asyncio.sleep(1)  # 1 second between batches
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success" if failed_count == 0 else "partial",
                data={
                    "sent_count": sent_count,
                    "failed_count": failed_count,
                    "total_recipients": len(recipients),
                    "errors": errors[:10]  # First 10 errors
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def _normalize_addresses(self, addresses: Any) -> List[str]:
        """Normalize email addresses to a list."""
        if not addresses:
            return []
        if isinstance(addresses, str):
            return [addr.strip() for addr in addresses.split(",")]
        if isinstance(addresses, list):
            return addresses
        return [str(addresses)]
    
    def _format_address(self, email: str, name: Optional[str] = None) -> str:
        """Format email address with optional name."""
        if name:
            return f'"{name}" <{email}>'
        return email
    
    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """Simple template rendering (replace with Jinja2 in production)."""
        result = template
        for key, value in data.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result
    
    async def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]):
        """Add an attachment to the email."""
        filename = attachment.get("filename", "attachment")
        content = attachment.get("content")
        content_type = attachment.get("content_type", "application/octet-stream")
        
        if not content:
            return
        
        # Create attachment
        part = MIMEBase(*content_type.split("/", 1))
        
        # Handle base64 encoded content
        if attachment.get("is_base64", False):
            part.set_payload(base64.b64decode(content))
        else:
            part.set_payload(content.encode() if isinstance(content, str) else content)
        
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"'
        )
        
        msg.attach(part)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform email adapter health check."""
        try:
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.smtp_use_tls,
                timeout=5
            ) as smtp:
                await smtp.login(self.smtp_username, self.smtp_password)
            
            return {
                "status": "healthy",
                "smtp_host": self.smtp_host,
                "smtp_port": self.smtp_port
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }