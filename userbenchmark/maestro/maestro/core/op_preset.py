from abc import ABC, abstractmethod
from abc import ABCMeta

from core.utils.logging import get_logger

logger = get_logger(__name__)

class OpPresetRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, op_preset: type):
        if op_preset.name in cls._registry:
            raise ValueError(f"Op preset {op_preset.name} already registered")

        cls._registry[op_preset.name] = op_preset
    
    @classmethod
    def get_op_preset_cls(cls, name: str) -> type:
        """Return the op preset class with this name"""
        if name not in cls._registry:
            raise NameError(f"Could not find op preset with name {name}, registered: {list(cls._registry.keys())}")
        op_preset_cls = cls._registry[name]
        return op_preset_cls
    
    @classmethod
    def get_all_op_presets(cls) -> list[type]:
        return list(cls._registry.values())


class _OpPresetRegistrer(ABCMeta):
    registry = {}
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        # Don't register abstract classes
        if getattr(cls, "__abstractmethods__", False):
            return
        
        logger.debug(f"[_OpPresetRegistry] Registering op preset {cls}")
        OpPresetRegistry.register(cls)


class OpPreset(ABC, metaclass=_OpPresetRegistrer):
    params_schema = {}

    @classmethod
    def validate_params(cls, params: dict) -> dict:
        """Validate the parameters for the MOE operation"""
        missing = []
        for param, schema in cls.params_schema.items():
            if param in params:
                continue

            if "default" in schema:
                params[param] = schema["default"]
            elif schema.get("required", False):
                missing.append(param)
            
        
        if missing:
            msg = f"Missing required parameters for operation preset {cls.__name__}:\n"
            for param in missing:
                msg += f"  - {param} ({cls.params_schema[param].get('description', 'No description')})\n"
            raise ValueError(msg)

    @classmethod
    @abstractmethod
    def create_op(cls, params: dict, axes_config: dict) -> tuple[dict[str, dict], list]:
        """
        Params:
            world_size - World size
            params - Parameters for the operation
            axes_config - The axes config parsed by config_parser

        Returns:
            The axes dictionary, a list of ops 
            The axes dictionary is a dictionary of structure {axis_name: axis_config} where axis_config should match the schema of the axis config in the config_parser (same as input YAML).

        The axis can also be predefined in the configuration and passed to 'params', in this case the returned axis dictionary will be empty.
        It is the responsability of the user to check if the axis is defined in the configuration and pass it to 'params'.
        """
        pass

import ops_presets