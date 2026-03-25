"""
Swarm Factory: Adaptive Subagent Customization Engine

This component allows the IntrospectionAgent to dynamically instantiate 
subagents from the integrated reference core (nanobot, microgravity) 
while overriding key behavioral parameters for experimentation.
"""

import importlib
import logging
from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)

class SwarmFactory:
    """
    Factory for creating and customizing subagents from integrated reference repositories.
    """

    @staticmethod
    def create_custom_agent(
        module_path: str, 
        class_name: str, 
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Dynamically imports a class and instantiates it with customized overrides.
        
        Args:
            module_path: Dot-separated path to the module (e.g. 'coding_agent.integrated_core.microgravity.agents.architect')
            class_name: The name of the class to instantiate.
            custom_params: A dictionary of key-value pairs to set/override on the instance.
        """
        try:
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            
            # Instantiate the agent
            instance = agent_class()
            
            # Apply customizations
            if custom_params:
                for attr, value in custom_params.items():
                    if hasattr(instance, attr):
                        setattr(instance, attr, value)
                        logger.info(f"Overrode {attr} on {class_name} with custom value.")
                    else:
                        # If the attribute doesn't exist, we can still try to set it for metadata
                        setattr(instance, attr, value)
                        logger.warning(f"Set NEW attribute {attr} on {class_name} (not found in base class).")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create custom agent {class_name} from {module_path}: {e}")
            raise e

    @staticmethod
    def get_ui_agent_config(precision_mode: bool = False) -> Dict[str, Any]:
        """Returns standard or high-precision configs for UI tasks."""
        if precision_mode:
            return {
                "scan_delay": 0.5,
                "ocr_confidence_threshold": 0.8,
                "use_advanced_edge_detection": True
            }
        return {
            "scan_delay": 0.1,
            "ocr_confidence_threshold": 0.5,
            "use_advanced_edge_detection": False
        }
