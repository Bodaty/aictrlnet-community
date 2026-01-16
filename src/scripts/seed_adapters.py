#!/usr/bin/env python3
"""
Adapter Database Seeding Script

This script populates the adapters table with all known adapters from the 
AdapterFactory mapping. It reads adapter metadata from the implementations
and creates database records for proper discovery.

Usage:
    python seed_adapters.py [--force] [--dry-run]
    
Options:
    --force     Force re-seed even if adapters already exist
    --dry-run   Show what would be done without making changes

Note: Handles SQLAlchemy 'metadata' reserved word by using adapter_metadata field.
"""

import sys
import os
import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete
from models.community import Adapter


class AdapterSeeder:
    """Adapter database seeder."""
    
    def __init__(self):
        # Use environment variables directly
        db_url = (
            f"postgresql+asyncpg://{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}"
            f"@{os.getenv('POSTGRES_SERVER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}"
            f"/{os.getenv('POSTGRES_DB', 'aictrlnet')}"
        )
        self.engine = create_async_engine(db_url)
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
    async def get_adapter_metadata(self) -> List[Dict[str, Any]]:
        """Extract adapter metadata using proper SQLAlchemy format."""
        
        # Adapter definitions (using Python dicts, not JSON strings)
        adapters = [
            # COMMUNITY EDITION ADAPTERS (17 total - includes limited AI agents + missing adapters)
            {
                "id": "openai",
                "name": "OpenAI",
                "category": "ai",
                "description": "OpenAI GPT models for text generation, completion, and analysis",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {  # This maps to the 'metadata' column
                    "factory_path": "adapters.implementations.ai.openai_adapter.OpenAIAdapter",
                    "capabilities": ["text_generation", "text_completion", "embeddings", "chat"],
                    "tags": ["llm", "gpt", "text", "ai"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI API key"},
                        "model": {"type": "string", "default": "gpt-3.5-turbo"},
                        "max_tokens": {"type": "integer", "default": 1000}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["text_generation", "text_completion", "embeddings", "chat"],
                "tags": ["llm", "gpt", "text", "ai"]
            },
            {
                "id": "claude",
                "name": "Claude",
                "category": "ai",
                "description": "Anthropic Claude for conversational AI and text analysis",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.claude_adapter.ClaudeAdapter",
                    "capabilities": ["conversation", "text_analysis", "reasoning", "code_generation"],
                    "tags": ["llm", "anthropic", "conversation", "ai"]
                },
                "config_schema": {
                    "type": "object", 
                    "properties": {
                        "api_key": {"type": "string", "description": "Anthropic API key"},
                        "model": {"type": "string", "default": "claude-3-sonnet-20240229"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["conversation", "text_analysis", "reasoning", "code_generation"],
                "tags": ["llm", "anthropic", "conversation", "ai"]
            },
            {
                "id": "slack",
                "name": "Slack",
                "category": "communication",
                "description": "Team messaging, notifications, and collaboration via Slack",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.communication.slack_adapter.SlackAdapter",
                    "capabilities": ["messaging", "notifications", "file_sharing", "channels"],
                    "tags": ["messaging", "team", "collaboration", "notifications"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "bot_token": {"type": "string", "description": "Slack bot token"},
                        "channel": {"type": "string", "description": "Default channel"}
                    },
                    "required": ["bot_token"]
                },
                "capabilities": ["messaging", "notifications", "file_sharing", "channels"],
                "tags": ["messaging", "team", "collaboration", "notifications"]
            },
            {
                "id": "upwork",
                "name": "Upwork",
                "category": "human",
                "description": "Freelance marketplace integration for hiring skilled professionals",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.human.upwork_adapter.UpworkAdapter",
                    "capabilities": ["freelancer_search", "job_posting", "contract_management", "payments"],
                    "tags": ["freelance", "marketplace", "professionals", "hiring"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string", "description": "Upwork OAuth client ID"},
                        "client_secret": {"type": "string", "description": "Upwork OAuth client secret"},
                        "access_token": {"type": "string", "description": "Upwork access token"}
                    },
                    "required": ["client_id", "client_secret"]
                },
                "capabilities": ["freelancer_search", "job_posting", "contract_management", "payments"],
                "tags": ["freelance", "marketplace", "professionals", "hiring"]
            },
            {
                "id": "fiverr",
                "name": "Fiverr",
                "category": "human", 
                "description": "Gig economy platform for quick professional services",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.human.fiverr_adapter.FiverrAdapter",
                    "capabilities": ["gig_search", "order_management", "service_delivery", "reviews"],
                    "tags": ["gigs", "services", "marketplace", "quick-delivery"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "Fiverr API key"},
                        "api_secret": {"type": "string", "description": "Fiverr API secret"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["gig_search", "order_management", "service_delivery", "reviews"],
                "tags": ["gigs", "services", "marketplace", "quick-delivery"]
            },
            {
                "id": "taskrabbit",
                "name": "TaskRabbit",
                "category": "human",
                "description": "Physical world task marketplace for local services and tasks",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.human.taskrabbit_adapter.TaskRabbitAdapter",
                    "capabilities": ["task_search", "tasker_booking", "physical_services", "local_tasks"],
                    "tags": ["physical", "local", "tasks", "services", "marketplace"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string", "description": "TaskRabbit OAuth client ID"},
                        "client_secret": {"type": "string", "description": "TaskRabbit OAuth client secret"},
                        "access_token": {"type": "string", "description": "TaskRabbit access token"}
                    },
                    "required": ["client_id", "client_secret"]
                },
                "capabilities": ["task_search", "tasker_booking", "physical_services", "local_tasks"],
                "tags": ["physical", "local", "tasks", "services", "marketplace"]
            },
            {
                "id": "stripe",
                "name": "Stripe",
                "category": "payment",
                "description": "Payment processing, subscriptions, and financial transactions",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.payment.stripe_adapter.StripeAdapter",
                    "capabilities": ["payment_processing", "subscriptions", "invoicing", "webhooks"],
                    "tags": ["payments", "subscriptions", "finance", "transactions"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "Stripe API key"},
                        "webhook_secret": {"type": "string", "description": "Stripe webhook secret"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["payment_processing", "subscriptions", "invoicing", "webhooks"],
                "tags": ["payments", "subscriptions", "finance", "transactions"]
            },
            
            # Additional Community adapters (missing from original seed)
            {
                "id": "discord",
                "name": "Discord",
                "category": "communication",
                "description": "Discord bot integration for community management and notifications",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.communication.discord_adapter.DiscordAdapter",
                    "capabilities": ["messaging", "voice_channels", "community_management", "webhooks"],
                    "tags": ["discord", "gaming", "community", "voice", "chat"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "bot_token": {"type": "string", "description": "Discord bot token"},
                        "guild_id": {"type": "string", "description": "Discord server/guild ID"}
                    },
                    "required": ["bot_token"]
                },
                "capabilities": ["messaging", "voice_channels", "community_management", "webhooks"],
                "tags": ["discord", "gaming", "community", "voice", "chat"]
            },
            {
                "id": "email",
                "name": "Email",
                "category": "communication",
                "description": "Email integration for sending notifications and reports",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.communication.email_adapter.EmailAdapter",
                    "capabilities": ["send_email", "attachments", "html_content", "templates"],
                    "tags": ["email", "smtp", "notifications", "reports"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "smtp_host": {"type": "string", "description": "SMTP server host"},
                        "smtp_port": {"type": "integer", "default": 587},
                        "username": {"type": "string", "description": "SMTP username"},
                        "password": {"type": "string", "description": "SMTP password"},
                        "use_tls": {"type": "boolean", "default": True}
                    },
                    "required": ["smtp_host", "username", "password"]
                },
                "capabilities": ["send_email", "attachments", "html_content", "templates"],
                "tags": ["email", "smtp", "notifications", "reports"]
            },
            {
                "id": "webhook",
                "name": "Webhook",
                "category": "communication",
                "description": "Generic webhook integration for HTTP notifications",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.communication.webhook_adapter.WebhookAdapter",
                    "capabilities": ["http_post", "http_get", "custom_headers", "retry_logic"],
                    "tags": ["webhook", "http", "api", "notifications", "integration"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Webhook URL"},
                        "method": {"type": "string", "default": "POST"},
                        "headers": {"type": "object", "description": "Custom headers"},
                        "retry_count": {"type": "integer", "default": 3}
                    },
                    "required": ["url"]
                },
                "capabilities": ["http_post", "http_get", "custom_headers", "retry_logic"],
                "tags": ["webhook", "http", "api", "notifications", "integration"]
            },
            {
                "id": "huggingface",
                "name": "Hugging Face",
                "category": "ai",
                "description": "Hugging Face Hub integration for open-source models",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.huggingface_adapter.HuggingFaceAdapter",
                    "capabilities": ["inference", "model_download", "dataset_access", "spaces"],
                    "tags": ["huggingface", "transformers", "open_source", "models", "ai"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_token": {"type": "string", "description": "Hugging Face API token"},
                        "model_id": {"type": "string", "description": "Model identifier"}
                    },
                    "required": ["api_token"]
                },
                "capabilities": ["inference", "model_download", "dataset_access", "spaces"],
                "tags": ["huggingface", "transformers", "open_source", "models", "ai"]
            },
            {
                "id": "ollama",
                "name": "Ollama",
                "category": "ai",
                "description": "Ollama local LLM integration for on-premise AI",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.ollama_adapter.OllamaAdapter",
                    "capabilities": ["local_inference", "model_management", "streaming", "embeddings"],
                    "tags": ["ollama", "local", "llm", "on_premise", "ai"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "default": "http://localhost:11434"},
                        "model": {"type": "string", "default": "llama2"}
                    },
                    "required": []
                },
                "capabilities": ["local_inference", "model_management", "streaming", "embeddings"],
                "tags": ["ollama", "local", "llm", "on_premise", "ai"]
            },
            
            # Community AI Agent adapters (limited versions)
            {
                "id": "langchain",
                "name": "LangChain (Community)",
                "category": "ai_agent",
                "description": "LangChain agent framework adapter - Community Edition with 100/day limit",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.langchain_adapter.LangChainAdapter",
                    "capabilities": ["basic_chains", "simple_agents", "memory", "tools"],
                    "tags": ["langchain", "ai_agent", "chains", "agents", "limited"],
                    "daily_limit": 100,
                    "limitations": ["rate_limited", "basic_features_only"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI API key"},
                        "model": {"type": "string", "default": "gpt-3.5-turbo"},
                        "max_retries": {"type": "integer", "default": 3}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["basic_chains", "simple_agents", "memory", "tools"],
                "tags": ["langchain", "ai_agent", "chains", "agents", "limited"]
            },
            {
                "id": "autogpt-limited",
                "name": "AutoGPT (Community)",
                "category": "ai_agent",
                "description": "AutoGPT autonomous AI agent - Community Edition with cascading adapters",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.autogpt_adapter.AutoGPTAdapter",
                    "capabilities": ["autonomous_basic", "memory_simple", "tool_usage", "cascading_llm"],
                    "tags": ["autogpt", "ai_agent", "autonomous", "cascading", "limited"],
                    "max_iterations": 10,
                    "limitations": ["reduced_iterations", "cascading_fallback"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "workspace_folder": {"type": "string", "default": "/tmp/autogpt"},
                        "max_iterations": {"type": "integer", "default": 10}
                    },
                    "required": []
                },
                "capabilities": ["autonomous_basic", "memory_simple", "tool_usage", "cascading_llm"],
                "tags": ["autogpt", "ai_agent", "autonomous", "cascading", "limited"]
            },
            {
                "id": "autogen-limited",
                "name": "AutoGen (Community)",
                "category": "ai_agent",
                "description": "AutoGen multi-agent conversation - Community Edition with cascading adapters",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.autogen_adapter.AutoGenAdapter",
                    "capabilities": ["multi_agent_basic", "conversations", "cascading_llm"],
                    "tags": ["autogen", "ai_agent", "multi_agent", "cascading", "limited"],
                    "max_agents": 3,
                    "max_rounds": 10,
                    "limitations": ["reduced_agents", "reduced_rounds", "cascading_fallback"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "max_agents": {"type": "integer", "default": 3},
                        "max_rounds": {"type": "integer", "default": 10}
                    },
                    "required": []
                },
                "capabilities": ["multi_agent_basic", "conversations", "cascading_llm"],
                "tags": ["autogen", "ai_agent", "multi_agent", "cascading", "limited"]
            },
            {
                "id": "crewai-limited",
                "name": "CrewAI (Community)",
                "category": "ai_agent",
                "description": "CrewAI crew-based task execution - Community Edition with cascading adapters",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.crewai_adapter.CrewAIAdapter",
                    "capabilities": ["crew_basic", "task_simple", "cascading_llm"],
                    "tags": ["crewai", "ai_agent", "crew", "cascading", "limited"],
                    "max_crew_size": 3,
                    "limitations": ["reduced_crew_size", "cascading_fallback"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "process": {"type": "string", "default": "sequential"},
                        "max_crew_size": {"type": "integer", "default": 3}
                    },
                    "required": []
                },
                "capabilities": ["crew_basic", "task_simple", "cascading_llm"],
                "tags": ["crewai", "ai_agent", "crew", "cascading", "limited"]
            },
            {
                "id": "semantic-kernel-limited",
                "name": "Semantic Kernel (Community)",
                "category": "ai_agent",
                "description": "Microsoft Semantic Kernel - Community Edition with cascading adapters",
                "version": "1.0.0",
                "min_edition": "community",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.semantic_kernel_adapter.SemanticKernelAdapter",
                    "capabilities": ["skills_basic", "plugins_simple", "cascading_llm"],
                    "tags": ["semantic_kernel", "ai_agent", "microsoft", "cascading", "limited"],
                    "limitations": ["basic_skills_only", "cascading_fallback"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "skill_directory": {"type": "string", "default": "/tmp/skills"}
                    },
                    "required": []
                },
                "capabilities": ["skills_basic", "plugins_simple", "cascading_llm"],
                "tags": ["semantic_kernel", "ai_agent", "microsoft", "cascading", "limited"]
            },
            
            # BUSINESS EDITION ADAPTERS (14 total)
            {
                "id": "azure-openai",
                "name": "Azure OpenAI",
                "category": "ai",
                "description": "Microsoft Azure OpenAI Service for enterprise AI capabilities",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.azure_openai_adapter.AzureOpenAIAdapter",
                    "capabilities": ["text_generation", "chat", "embeddings", "enterprise_ai"],
                    "tags": ["azure", "microsoft", "enterprise", "ai", "llm"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "Azure OpenAI API key"},
                        "endpoint": {"type": "string", "description": "Azure OpenAI endpoint"},
                        "deployment_name": {"type": "string", "description": "Deployment name"}
                    },
                    "required": ["api_key", "endpoint"]
                },
                "capabilities": ["text_generation", "chat", "embeddings", "enterprise_ai"],
                "tags": ["azure", "microsoft", "enterprise", "ai", "llm"]
            },
            {
                "id": "bedrock",
                "name": "AWS Bedrock",
                "category": "ai",
                "description": "Amazon Bedrock managed AI service with foundation models",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.aws_bedrock_adapter.AWSBedrockAdapter",
                    "capabilities": ["text_generation", "embeddings", "foundation_models"],
                    "tags": ["aws", "bedrock", "foundation", "models", "ai"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "aws_access_key": {"type": "string", "description": "AWS access key"},
                        "aws_secret_key": {"type": "string", "description": "AWS secret key"},
                        "region": {"type": "string", "default": "us-east-1"}
                    },
                    "required": ["aws_access_key", "aws_secret_key"]
                },
                "capabilities": ["text_generation", "embeddings", "foundation_models"],
                "tags": ["aws", "bedrock", "foundation", "models", "ai"]
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "category": "ai",
                "description": "Google Gemini multimodal AI for advanced reasoning",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.google_gemini_adapter.GoogleGeminiAdapter",
                    "capabilities": ["multimodal", "reasoning", "text_generation", "vision"],
                    "tags": ["google", "gemini", "multimodal", "vision", "ai"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "Google AI API key"},
                        "model": {"type": "string", "default": "gemini-pro"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["multimodal", "reasoning", "text_generation", "vision"],
                "tags": ["google", "gemini", "multimodal", "vision", "ai"]
            },
            {
                "id": "vertex-ai",
                "name": "Google Vertex AI",
                "category": "ai",
                "description": "Google Cloud Vertex AI platform for ML workflows",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.vertex_ai_adapter.VertexAIAdapter",
                    "capabilities": ["ml_workflows", "training", "inference", "pipelines"],
                    "tags": ["google", "vertex", "ml", "pipelines", "training"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "GCP Project ID"},
                        "location": {"type": "string", "default": "us-central1"},
                        "credentials": {"type": "string", "description": "Service account JSON"}
                    },
                    "required": ["project_id", "credentials"]
                },
                "capabilities": ["ml_workflows", "training", "inference", "pipelines"],
                "tags": ["google", "vertex", "ml", "pipelines", "training"]
            },
            {
                "id": "teams",
                "name": "Microsoft Teams",
                "category": "communication",
                "description": "Microsoft Teams integration for enterprise collaboration",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.communication.teams_adapter.TeamsAdapter",
                    "capabilities": ["messaging", "meetings", "channels", "enterprise_collaboration"],
                    "tags": ["microsoft", "teams", "enterprise", "collaboration", "messaging"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "description": "Azure AD tenant ID"},
                        "client_id": {"type": "string", "description": "App registration client ID"},
                        "client_secret": {"type": "string", "description": "App registration secret"}
                    },
                    "required": ["tenant_id", "client_id", "client_secret"]
                },
                "capabilities": ["messaging", "meetings", "channels", "enterprise_collaboration"],
                "tags": ["microsoft", "teams", "enterprise", "collaboration", "messaging"]
            },
            {
                "id": "postgresql",
                "name": "PostgreSQL",
                "category": "database",
                "description": "PostgreSQL database integration for data operations",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.database.postgresql_adapter.PostgreSQLAdapter",
                    "capabilities": ["sql_queries", "data_operations", "transactions", "indexing"],
                    "tags": ["postgresql", "database", "sql", "relational", "data"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "Database host"},
                        "port": {"type": "integer", "default": 5432},
                        "database": {"type": "string", "description": "Database name"},
                        "username": {"type": "string", "description": "Database username"},
                        "password": {"type": "string", "description": "Database password"}
                    },
                    "required": ["host", "database", "username", "password"]
                },
                "capabilities": ["sql_queries", "data_operations", "transactions", "indexing"],
                "tags": ["postgresql", "database", "sql", "relational", "data"]
            },
            {
                "id": "mysql",
                "name": "MySQL",
                "category": "database",
                "description": "MySQL database integration for data operations",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.database.mysql_adapter.MySQLAdapter",
                    "capabilities": ["sql_queries", "data_operations", "transactions", "replication"],
                    "tags": ["mysql", "database", "sql", "relational", "data"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "Database host"},
                        "port": {"type": "integer", "default": 3306},
                        "database": {"type": "string", "description": "Database name"},
                        "username": {"type": "string", "description": "Database username"},
                        "password": {"type": "string", "description": "Database password"}
                    },
                    "required": ["host", "database", "username", "password"]
                },
                "capabilities": ["sql_queries", "data_operations", "transactions", "replication"],
                "tags": ["mysql", "database", "sql", "relational", "data"]
            },
            {
                "id": "salesforce",
                "name": "Salesforce",
                "category": "crm",
                "description": "Salesforce CRM integration for customer relationship management",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.crm.salesforce_adapter.SalesforceAdapter",
                    "capabilities": ["lead_management", "opportunity_tracking", "account_management", "reporting"],
                    "tags": ["salesforce", "crm", "leads", "opportunities", "sales"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "Salesforce username"},
                        "password": {"type": "string", "description": "Salesforce password"},
                        "security_token": {"type": "string", "description": "Security token"},
                        "domain": {"type": "string", "default": "login"}
                    },
                    "required": ["username", "password", "security_token"]
                },
                "capabilities": ["lead_management", "opportunity_tracking", "account_management", "reporting"],
                "tags": ["salesforce", "crm", "leads", "opportunities", "sales"]
            },
            {
                "id": "hubspot",
                "name": "HubSpot",
                "category": "crm",
                "description": "HubSpot CRM and marketing automation platform",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.crm.hubspot_adapter.HubSpotAdapter",
                    "capabilities": ["contact_management", "marketing_automation", "pipeline_management", "analytics"],
                    "tags": ["hubspot", "crm", "marketing", "automation", "analytics"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "HubSpot API key"},
                        "portal_id": {"type": "string", "description": "HubSpot portal ID"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["contact_management", "marketing_automation", "pipeline_management", "analytics"],
                "tags": ["hubspot", "crm", "marketing", "automation", "analytics"]
            },
            {
                "id": "jira",
                "name": "Atlassian Jira",
                "category": "project_management",
                "description": "Jira project management and issue tracking",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.project_management.jira_adapter.JiraAdapter",
                    "capabilities": ["issue_tracking", "project_management", "workflow_automation", "reporting"],
                    "tags": ["jira", "atlassian", "project", "issues", "tracking"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "server": {"type": "string", "description": "Jira server URL"},
                        "username": {"type": "string", "description": "Username or email"},
                        "api_token": {"type": "string", "description": "API token"}
                    },
                    "required": ["server", "username", "api_token"]
                },
                "capabilities": ["issue_tracking", "project_management", "workflow_automation", "reporting"],
                "tags": ["jira", "atlassian", "project", "issues", "tracking"]
            },
            
            # ENTERPRISE EDITION ADAPTERS (11 total)
            {
                "id": "cohere",
                "name": "Cohere",
                "category": "ai",
                "description": "Cohere AI platform for enterprise NLP capabilities",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai.cohere_adapter.CohereAdapter",
                    "capabilities": ["text_generation", "embeddings", "classification", "summarization"],
                    "tags": ["cohere", "nlp", "enterprise", "embeddings", "ai"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "Cohere API key"},
                        "model": {"type": "string", "default": "command"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["text_generation", "embeddings", "classification", "summarization"],
                "tags": ["cohere", "nlp", "enterprise", "embeddings", "ai"]
            },
            {
                "id": "redis",
                "name": "Redis",
                "category": "database",
                "description": "Redis in-memory data store for high-performance caching",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.database.redis_adapter.RedisAdapter",
                    "capabilities": ["caching", "pub_sub", "data_structures", "high_performance"],
                    "tags": ["redis", "cache", "memory", "performance", "data"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "default": "localhost"},
                        "port": {"type": "integer", "default": 6379},
                        "password": {"type": "string", "description": "Redis password"},
                        "db": {"type": "integer", "default": 0}
                    },
                    "required": ["host"]
                },
                "capabilities": ["caching", "pub_sub", "data_structures", "high_performance"],
                "tags": ["redis", "cache", "memory", "performance", "data"]
            },
            {
                "id": "s3",
                "name": "AWS S3",
                "category": "storage",
                "description": "Amazon S3 cloud storage for scalable object storage",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.cloud.aws_s3_adapter.AWSS3Adapter",
                    "capabilities": ["object_storage", "file_management", "versioning", "lifecycle"],
                    "tags": ["aws", "s3", "storage", "cloud", "objects"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "aws_access_key": {"type": "string", "description": "AWS access key"},
                        "aws_secret_key": {"type": "string", "description": "AWS secret key"},
                        "bucket": {"type": "string", "description": "S3 bucket name"},
                        "region": {"type": "string", "default": "us-east-1"}
                    },
                    "required": ["aws_access_key", "aws_secret_key", "bucket"]
                },
                "capabilities": ["object_storage", "file_management", "versioning", "lifecycle"],
                "tags": ["aws", "s3", "storage", "cloud", "objects"]
            },
            {
                "id": "hipaa",
                "name": "HIPAA Compliance",
                "category": "compliance",
                "description": "HIPAA compliance monitoring and enforcement",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.compliance.hipaa_adapter.HIPAAAdapter",
                    "capabilities": ["privacy_monitoring", "audit_trails", "data_encryption", "access_control"],
                    "tags": ["hipaa", "healthcare", "compliance", "privacy", "audit"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "covered_entity_id": {"type": "string", "description": "HIPAA covered entity ID"},
                        "compliance_officer": {"type": "string", "description": "Compliance officer contact"},
                        "audit_level": {"type": "string", "default": "standard"}
                    },
                    "required": ["covered_entity_id"]
                },
                "capabilities": ["privacy_monitoring", "audit_trails", "data_encryption", "access_control"],
                "tags": ["hipaa", "healthcare", "compliance", "privacy", "audit"]
            },
            {
                "id": "gdpr",
                "name": "GDPR Compliance",
                "category": "compliance",
                "description": "GDPR compliance monitoring and data protection",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.compliance.gdpr_adapter.GDPRAdapter",
                    "capabilities": ["data_protection", "consent_management", "right_to_erasure", "breach_notification"],
                    "tags": ["gdpr", "privacy", "compliance", "eu", "data_protection"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "data_controller": {"type": "string", "description": "Data controller information"},
                        "dpo_contact": {"type": "string", "description": "Data Protection Officer contact"},
                        "jurisdiction": {"type": "string", "default": "EU"}
                    },
                    "required": ["data_controller"]
                },
                "capabilities": ["data_protection", "consent_management", "right_to_erasure", "breach_notification"],
                "tags": ["gdpr", "privacy", "compliance", "eu", "data_protection"]
            },
            {
                "id": "soc2",
                "name": "SOC 2 Compliance",
                "category": "compliance",
                "description": "SOC 2 compliance monitoring and controls",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.compliance.soc2_adapter.SOC2Adapter",
                    "capabilities": ["security_controls", "availability_monitoring", "confidentiality", "integrity"],
                    "tags": ["soc2", "security", "compliance", "controls", "audit"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "audit_firm": {"type": "string", "description": "SOC 2 audit firm"},
                        "report_type": {"type": "string", "default": "Type II"},
                        "trust_criteria": {"type": "array", "default": ["security", "availability"]}
                    },
                    "required": ["audit_firm"]
                },
                "capabilities": ["security_controls", "availability_monitoring", "confidentiality", "integrity"],
                "tags": ["soc2", "security", "compliance", "controls", "audit"]
            },
            {
                "id": "healthcare",
                "name": "Healthcare Industry",
                "category": "industry",
                "description": "Healthcare industry-specific integrations and workflows",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.industry.healthcare_adapter.HealthcareAdapter",
                    "capabilities": ["hl7_integration", "fhir_support", "patient_data", "clinical_workflows"],
                    "tags": ["healthcare", "hl7", "fhir", "medical", "clinical"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "hl7_endpoint": {"type": "string", "description": "HL7 integration endpoint"},
                        "fhir_version": {"type": "string", "default": "R4"},
                        "facility_id": {"type": "string", "description": "Healthcare facility ID"}
                    },
                    "required": ["facility_id"]
                },
                "capabilities": ["hl7_integration", "fhir_support", "patient_data", "clinical_workflows"],
                "tags": ["healthcare", "hl7", "fhir", "medical", "clinical"]
            },
            {
                "id": "finance",
                "name": "Finance Industry",
                "category": "industry",
                "description": "Financial services industry integrations and compliance",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.industry.finance_adapter.FinanceAdapter",
                    "capabilities": ["regulatory_reporting", "risk_management", "trading_integration", "kyc_aml"],
                    "tags": ["finance", "banking", "regulatory", "risk", "trading"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "institution_id": {"type": "string", "description": "Financial institution ID"},
                        "regulatory_framework": {"type": "string", "default": "US"},
                        "risk_tolerance": {"type": "string", "default": "medium"}
                    },
                    "required": ["institution_id"]
                },
                "capabilities": ["regulatory_reporting", "risk_management", "trading_integration", "kyc_aml"],
                "tags": ["finance", "banking", "regulatory", "risk", "trading"]
            },
            {
                "id": "retail",
                "name": "Retail Industry",
                "category": "industry",
                "description": "Retail industry integrations for commerce and inventory",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.industry.retail_adapter.RetailAdapter",
                    "capabilities": ["pos_integration", "inventory_management", "customer_analytics", "omnichannel"],
                    "tags": ["retail", "pos", "inventory", "commerce", "customer"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string", "description": "Retail store identifier"},
                        "pos_system": {"type": "string", "description": "POS system type"},
                        "inventory_sync": {"type": "boolean", "default": True}
                    },
                    "required": ["store_id"]
                },
                "capabilities": ["pos_integration", "inventory_management", "customer_analytics", "omnichannel"],
                "tags": ["retail", "pos", "inventory", "commerce", "customer"]
            },
            {
                "id": "manufacturing",
                "name": "Manufacturing Industry",
                "category": "industry",
                "description": "Manufacturing industry integrations for production and supply chain",
                "version": "1.0.0",
                "min_edition": "enterprise",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.industry.manufacturing_adapter.ManufacturingAdapter",
                    "capabilities": ["erp_integration", "supply_chain", "quality_control", "production_planning"],
                    "tags": ["manufacturing", "erp", "supply_chain", "production", "quality"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "facility_id": {"type": "string", "description": "Manufacturing facility ID"},
                        "erp_system": {"type": "string", "description": "ERP system type"},
                        "production_line": {"type": "string", "description": "Production line identifier"}
                    },
                    "required": ["facility_id"]
                },
                "capabilities": ["erp_integration", "supply_chain", "quality_control", "production_planning"],
                "tags": ["manufacturing", "erp", "supply_chain", "production", "quality"]
            },
            
            # AI AGENT ADAPTERS - BUSINESS EDITION (5 total - full featured versions)
            {
                "id": "langchain-full",
                "name": "LangChain (Business)",
                "category": "ai_agent",
                "description": "LangChain agent framework adapter - Business Edition with full features",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.langchain_adapter_full.LangChainFullAdapter",
                    "capabilities": ["all_chains", "advanced_agents", "memory", "tools", "callbacks", "streaming"],
                    "tags": ["langchain", "ai_agent", "chains", "agents", "full"],
                    "unlimited": True
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI API key"},
                        "model": {"type": "string", "default": "gpt-4"},
                        "max_retries": {"type": "integer", "default": 3},
                        "streaming": {"type": "boolean", "default": True}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["all_chains", "advanced_agents", "memory", "tools", "callbacks", "streaming"],
                "tags": ["langchain", "ai_agent", "chains", "agents", "full"]
            },
            {
                "id": "autogpt",
                "name": "AutoGPT",
                "category": "ai_agent",
                "description": "AutoGPT autonomous AI agent adapter - Business Edition",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.autogpt_adapter.AutoGPTAdapter",
                    "capabilities": ["autonomous_operation", "memory_management", "tool_usage", "goal_oriented"],
                    "tags": ["autogpt", "ai_agent", "autonomous", "goals", "memory"],
                    "max_iterations": 25
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI API key"},
                        "workspace_folder": {"type": "string", "default": "/tmp/autogpt"},
                        "max_iterations": {"type": "integer", "default": 25},
                        "smart_llm_model": {"type": "string", "default": "gpt-4"},
                        "fast_llm_model": {"type": "string", "default": "gpt-3.5-turbo"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["autonomous_operation", "memory_management", "tool_usage", "goal_oriented"],
                "tags": ["autogpt", "ai_agent", "autonomous", "goals", "memory"]
            },
            {
                "id": "autogen",
                "name": "AutoGen",
                "category": "ai_agent",
                "description": "AutoGen multi-agent conversation framework - Business Edition",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.autogen_adapter.AutoGenAdapter",
                    "capabilities": ["multi_agent", "conversations", "role_based", "code_execution"],
                    "tags": ["autogen", "ai_agent", "multi_agent", "conversation", "microsoft"],
                    "max_agents": 10,
                    "max_rounds": 20
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI API key"},
                        "config_list": {"type": "array", "description": "Model configurations"},
                        "max_agents": {"type": "integer", "default": 10},
                        "max_rounds": {"type": "integer", "default": 20},
                        "human_input_mode": {"type": "string", "default": "NEVER"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["multi_agent", "conversations", "role_based", "code_execution"],
                "tags": ["autogen", "ai_agent", "multi_agent", "conversation", "microsoft"]
            },
            {
                "id": "crewai",
                "name": "CrewAI",
                "category": "ai_agent",
                "description": "CrewAI crew-based task execution framework - Business Edition",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.crewai_adapter.CrewAIAdapter",
                    "capabilities": ["crew_management", "task_delegation", "agent_collaboration", "memory_sharing"],
                    "tags": ["crewai", "ai_agent", "crew", "tasks", "collaboration"],
                    "process_types": ["sequential", "hierarchical", "consensual"],
                    "max_crew_size": 8
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI API key"},
                        "process": {"type": "string", "default": "sequential"},
                        "verbose": {"type": "boolean", "default": True},
                        "max_crew_size": {"type": "integer", "default": 8}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["crew_management", "task_delegation", "agent_collaboration", "memory_sharing"],
                "tags": ["crewai", "ai_agent", "crew", "tasks", "collaboration"]
            },
            {
                "id": "semantic-kernel",
                "name": "Semantic Kernel",
                "category": "ai_agent",
                "description": "Microsoft Semantic Kernel AI orchestration - Business Edition",
                "version": "1.0.0",
                "min_edition": "business",
                "enabled": True,
                "available": True,
                "installed": True,
                "install_count": 1,
                "adapter_metadata": {
                    "factory_path": "adapters.implementations.ai_agents.semantic_kernel_adapter.SemanticKernelAdapter",
                    "capabilities": ["skills", "plugins", "planners", "memory", "connectors"],
                    "tags": ["semantic_kernel", "ai_agent", "microsoft", "skills", "plugins"],
                    "skill_types": ["semantic", "native", "planner"],
                    "supported_planners": ["sequential", "action", "stepwise"]
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "OpenAI or Azure OpenAI API key"},
                        "deployment_name": {"type": "string", "description": "Azure deployment name"},
                        "endpoint": {"type": "string", "description": "Azure endpoint"},
                        "org_id": {"type": "string", "description": "OpenAI organization ID"}
                    },
                    "required": ["api_key"]
                },
                "capabilities": ["skills", "plugins", "planners", "memory", "connectors"],
                "tags": ["semantic_kernel", "ai_agent", "microsoft", "skills", "plugins"]
            }
        ]
            
        return adapters
    
    async def clear_existing_adapters(self, session: AsyncSession):
        """Clear existing adapter records."""
        print("Clearing existing adapter records...")
        result = await session.execute(delete(Adapter))
        print(f"Deleted {result.rowcount} existing adapter records")
        
    async def seed_adapters(self, adapters: List[Dict[str, Any]], session: AsyncSession):
        """Seed adapters into database using SQLAlchemy model."""
        print(f"Seeding {len(adapters)} adapters...")
        
        for adapter_data in adapters:
            adapter = Adapter(**adapter_data)
            session.add(adapter)
            print(f"  + {adapter_data['id']}: {adapter_data['name']} ({adapter_data['category']})")
            
        await session.commit()
        print(f"Successfully seeded {len(adapters)} adapters!")
        
    async def verify_seeding(self, session: AsyncSession):
        """Verify adapters were seeded correctly."""
        print("\nVerifying seeded adapters...")
        
        # Get total count
        result = await session.execute(select(Adapter))
        adapters = result.scalars().all()
        
        print(f"Total adapters in database: {len(adapters)}")
        
        # Group by category
        by_category = {}
        by_edition = {}
        
        for adapter in adapters:
            # By category
            if adapter.category not in by_category:
                by_category[adapter.category] = []
            by_category[adapter.category].append(adapter.name)
            
            # By edition 
            if adapter.min_edition not in by_edition:
                by_edition[adapter.min_edition] = []
            by_edition[adapter.min_edition].append(adapter.name)
        
        print("\nAdapters by category:")
        for category, adapter_names in sorted(by_category.items()):
            print(f"  {category}: {len(adapter_names)} adapters")
            for name in sorted(adapter_names):
                print(f"    - {name}")
                
        print("\nAdapters by edition:")
        for edition, adapter_names in sorted(by_edition.items()):
            print(f"  {edition}: {len(adapter_names)} adapters")
            for name in sorted(adapter_names):
                print(f"    - {name}")
    
    async def run_seeding(self, force: bool = False, dry_run: bool = False):
        """Run the seeding process."""
        print("🌱 AICtrlNet Adapter Database Seeding")
        print("=" * 50)
        
        # Get adapter metadata
        adapters = await self.get_adapter_metadata()
        
        if dry_run:
            print(f"\n🔍 DRY RUN: Would seed {len(adapters)} adapters:")
            for adapter in adapters:
                print(f"  + {adapter['id']}: {adapter['name']} ({adapter['category']}) - {adapter['min_edition']}")
            return
        
        async with self.SessionLocal() as session:
            # Check if adapters already exist
            result = await session.execute(select(Adapter))
            existing_adapters = result.scalars().all()
            
            if existing_adapters and not force:
                print(f"❌ Database already contains {len(existing_adapters)} adapters.")
                print("Use --force to re-seed or --dry-run to preview changes.")
                return
                
            # Clear existing if force mode
            if force and existing_adapters:
                await self.clear_existing_adapters(session)
                
            # Seed new adapters
            await self.seed_adapters(adapters, session)
            
            # Verify seeding
            await self.verify_seeding(session)
            
        print("\n✅ Adapter seeding completed successfully!")
        
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()


async def main():
    """Main seeding function."""
    parser = argparse.ArgumentParser(description="Seed adapters database")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-seed even if adapters exist")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # Run seeding
    seeder = AdapterSeeder()
    try:
        await seeder.run_seeding(force=args.force, dry_run=args.dry_run)
    except Exception as e:
        print(f"❌ Seeding failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await seeder.close()


if __name__ == "__main__":
    asyncio.run(main())