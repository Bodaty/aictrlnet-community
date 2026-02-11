"""Factory for creating adapter instances."""

import logging
from typing import Dict, Any, Optional, Type
import importlib
import inspect

from .base_adapter import BaseAdapter
from .models import AdapterConfig, AdapterCategory
from .registry import adapter_registry


logger = logging.getLogger(__name__)


class AdapterFactory:
    """Factory for creating adapter instances dynamically."""
    
    # Known adapter implementations - Community Edition only
    ADAPTER_MAPPINGS = {
        # AI Adapters (Open Source)
        "openai": "adapters.implementations.ai.openai_adapter.OpenAIAdapter",
        "claude": "adapters.implementations.ai.claude_adapter.ClaudeAdapter",
        "anthropic": "adapters.implementations.ai.claude_adapter.ClaudeAdapter",  # Alias
        "ollama": "adapters.implementations.ai.ollama_adapter.OllamaAdapter",
        "huggingface": "adapters.implementations.ai.huggingface_adapter.HuggingFaceAdapter",
        "hf": "adapters.implementations.ai.huggingface_adapter.HuggingFaceAdapter",  # Alias
        "deepseek": "adapters.implementations.ai.deepseek_adapter.DeepSeekAdapter",
        
        # Internal Service Adapters (for unified model access)
        "llm-service": "adapters.implementations.ai.llm_service_adapter.LLMServiceAdapter",
        "ml-service": "adapters.implementations.ai.ml_service_adapter.MLServiceAdapter",
        "mcp-client": "adapters.implementations.ai.mcp_client_adapter.MCPClientAdapter",
        
        # Communication Adapters (Community)
        "slack": "adapters.implementations.communication.slack_adapter.SlackAdapter",
        "email": "adapters.implementations.communication.email_adapter.EmailAdapter",
        "webhook": "adapters.implementations.communication.webhook_adapter.WebhookAdapter",
        "discord": "adapters.implementations.communication.discord_adapter.DiscordAdapter",
        "telegram": "adapters.implementations.communication.telegram_adapter.TelegramAdapter",
        "whatsapp": "adapters.implementations.communication.whatsapp_adapter.WhatsAppAdapter",
        "twilio": "adapters.implementations.communication.twilio_adapter.TwilioAdapter",
        "sms": "adapters.implementations.communication.twilio_adapter.TwilioAdapter",  # Alias
        
        # Human Service Adapters
        "upwork": "adapters.implementations.human.upwork_adapter.UpworkAdapter",
        "fiverr": "adapters.implementations.human.fiverr_adapter.FiverrAdapter",
        "taskrabbit": "adapters.implementations.human.taskrabbit_adapter.TaskRabbitAdapter",
        
        # Payment Adapters
        "stripe": "adapters.implementations.payment.stripe_adapter.StripeAdapter",
        
        # AI Agent Adapters (Community Edition)
        "langchain": "adapters.implementations.ai_agents.langchain_adapter.LangChainAdapter",
        "autogpt": "adapters.implementations.ai_agents.autogpt_adapter.AutoGPTAdapter",
        "crewai": "adapters.implementations.ai_agents.crewai_adapter.CrewAIAdapter",
        "autogen": "adapters.implementations.ai_agents.autogen_adapter.AutoGenAdapter",
        "semantic-kernel": "adapters.implementations.ai_agents.semantic_kernel_adapter.SemanticKernelAdapter",
        "openclaw": "adapters.implementations.ai_agents.openclaw_adapter.OpenClawAdapter",
    }
    
    @classmethod
    def create_adapter(
        cls,
        adapter_type: str,
        config: Dict[str, Any],
        auto_start: bool = True
    ) -> BaseAdapter:
        """Create an adapter instance from type and config."""
        # Normalize adapter type
        adapter_type = adapter_type.lower()
        
        # Get adapter class
        adapter_class = cls._get_adapter_class(adapter_type)
        
        # Create config object
        adapter_config = cls._create_config(adapter_type, config)
        
        # Create adapter instance
        adapter = adapter_class(adapter_config)
        
        # Register and start if requested
        if auto_start:
            # This would be async in real usage
            logger.info(f"Created adapter {adapter_type} (auto_start={auto_start})")
        
        return adapter
    
    @classmethod
    async def create_and_register_adapter(
        cls,
        adapter_type: str,
        config: Dict[str, Any]
    ) -> BaseAdapter:
        """Create an adapter and register it with the registry."""
        # Get adapter class
        adapter_class = cls._get_adapter_class(adapter_type)
        
        # Register class with registry if not already registered
        if adapter_type not in adapter_registry.get_available_adapter_classes():
            # Determine category from adapter type or config
            category = cls._determine_category(adapter_type, config)
            adapter_registry.register_adapter_class(
                adapter_type,
                adapter_class,
                category
            )
        
        # Create config
        adapter_config = cls._create_config(adapter_type, config)
        
        # Create and start adapter through registry
        adapter = await adapter_registry.create_adapter(adapter_type, adapter_config)
        
        return adapter
    
    @classmethod
    def _get_adapter_class(cls, adapter_type: str) -> Type[BaseAdapter]:
        """Get adapter class by type."""
        # Check if it's a known adapter
        if adapter_type in cls.ADAPTER_MAPPINGS:
            class_path = cls.ADAPTER_MAPPINGS[adapter_type]
            return cls._load_adapter_class(class_path)
        
        # Check registry for already loaded classes
        available_classes = adapter_registry.get_available_adapter_classes()
        if adapter_type in available_classes:
            return available_classes[adapter_type]
        
        # Try to load dynamically
        # Assume pattern: adapters.implementations.{category}.{type}_adapter.{Type}Adapter
        for category in ["ai", "communication", "human", "payment", "data"]:
            try:
                module_name = f"adapters.implementations.{category}.{adapter_type}_adapter"
                class_name = f"{adapter_type.title()}Adapter"
                class_path = f"{module_name}.{class_name}"
                return cls._load_adapter_class(class_path)
            except Exception:
                continue
        
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    @classmethod
    def _load_adapter_class(cls, class_path: str) -> Type[BaseAdapter]:
        """Load an adapter class from module path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            
            # Verify it's a BaseAdapter subclass
            if not inspect.isclass(adapter_class) or not issubclass(adapter_class, BaseAdapter):
                raise ValueError(f"{class_path} is not a valid BaseAdapter subclass")
            
            return adapter_class
            
        except Exception as e:
            raise ValueError(f"Failed to load adapter class {class_path}: {str(e)}")
    
    @classmethod
    def _create_config(cls, adapter_type: str, config_dict: Dict[str, Any]) -> AdapterConfig:
        """Create adapter config from dictionary."""
        # Set defaults
        config_dict.setdefault("name", adapter_type)
        config_dict.setdefault("version", "1.0.0")
        
        # Determine category if not provided
        if "category" not in config_dict:
            config_dict["category"] = cls._determine_category(adapter_type, config_dict)
        
        # Create config object
        return AdapterConfig(**config_dict)
    
    @classmethod
    def _determine_category(cls, adapter_type: str, config: Dict[str, Any]) -> AdapterCategory:
        """Determine adapter category from type or config."""
        # Check config first
        if "category" in config:
            return AdapterCategory(config["category"])
        
        # Determine from adapter type
        ai_adapters = ["openai", "claude", "anthropic", "ollama", "huggingface", "hf", "deepseek"]
        communication_adapters = ["slack", "email", "webhook", "discord", "telegram", "whatsapp"]
        human_adapters = ["upwork", "fiverr", "taskrabbit", "mechanical_turk", "scale"]
        payment_adapters = ["stripe", "paypal", "square"]
        database_adapters = ["sqlite"]  # Community only has basic database support
        cloud_storage_adapters = []  # No cloud storage in Community Edition
        
        if adapter_type in ai_adapters:
            return AdapterCategory.AI
        elif adapter_type in communication_adapters:
            return AdapterCategory.COMMUNICATION
        elif adapter_type in human_adapters:
            return AdapterCategory.HUMAN
        elif adapter_type in payment_adapters:
            return AdapterCategory.PAYMENT
        elif adapter_type in database_adapters:
            return AdapterCategory.DATA
        elif adapter_type in cloud_storage_adapters:
            return AdapterCategory.INTEGRATION  # Cloud storage uses INTEGRATION category
        else:
            return AdapterCategory.INTEGRATION
    
    @classmethod
    def get_available_adapters(cls) -> Dict[str, str]:
        """Get list of available adapter types and their class paths."""
        # Return the actual mappings, not descriptions
        return cls.ADAPTER_MAPPINGS.copy()
    
    @classmethod
    def validate_config(cls, adapter_type: str, config: Dict[str, Any]) -> bool:
        """Validate adapter configuration."""
        try:
            # Try to create config object
            adapter_config = cls._create_config(adapter_type, config)
            
            # Get adapter class and check required config
            adapter_class = cls._get_adapter_class(adapter_type)
            
            # Could add more validation here based on adapter requirements
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed for {adapter_type}: {str(e)}")
            return False