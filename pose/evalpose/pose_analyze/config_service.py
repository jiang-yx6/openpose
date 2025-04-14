from evalpose.models import VideoConfig
from evalpose.models import VideoConfig
from .config import Config as DefaultConfig
import types

def get_config_class(numeric_id):
    """
    Dynamically create a Config class based on database configuration for the given numeric_id.
    
    Args:
        numeric_id (str): The numeric ID of the standard video (e.g., "01_01")
        
    Returns:
        type: A dynamically created Config class with attributes from database
    """
    try:
        # Get configuration from database
        db_config = VideoConfig.objects.get(numeric_id=numeric_id)
        
        # Create a new Config class
        class DynamicConfig(DefaultConfig):
            KEY_ANGLES = db_config.key_angles
            NORMALIZATION_JOINTS = db_config.normalization_joints
            DESCRIPTION = db_config.description
            NUMERIC_ID = numeric_id
            
            @classmethod
            def __str__(cls):
                return f"Config({cls.NUMERIC_ID}: {cls.DESCRIPTION})"
            
        return DynamicConfig
        
    except VideoConfig.DoesNotExist:
        # Return the default Config class if no configuration found
        return DefaultConfig

def get_config_instance(numeric_id):
    """
    Get a Config instance based on database configuration for the given numeric_id.
    
    Args:
        numeric_id (str): The numeric ID of the standard video (e.g., "01_01")
        
    Returns:
        object: An instance of the dynamically created Config class
    """
    ConfigClass = get_config_class(numeric_id)
    return ConfigClass()

def get_video_config(numeric_id):
    """
    Get the configuration for a given numeric_id
    
    Args:
        numeric_id (str): The numeric ID of the standard video (e.g., "01_01")
        
    Returns:
        dict: Configuration dictionary with KEY_ANGLES and NORMALIZATION_JOINTS
    """
    try:
        config = VideoConfig.objects.get(numeric_id=numeric_id)
        return {
            'KEY_ANGLES': config.key_angles,
            'NORMALIZATION_JOINTS': config.normalization_joints,
            'Describe': config.description
        }
    except VideoConfig.DoesNotExist:
        # Fall back to default configuration if not found
        from .config import Config
        return {
            'KEY_ANGLES': Config.KEY_ANGLES,
            'NORMALIZATION_JOINTS': Config.NORMALIZATION_JOINTS,
            'Describe': 'Default Configuration'
        }

def create_dynamic_config_from_dict(config_dict, numeric_id=None):
    """
    Create a dynamic configuration object from a dictionary.
    
    Args:
        config_dict (dict): The configuration dictionary with KEY_ANGLES, etc.
        numeric_id (str, optional): The numeric ID to associate with the config
        
    Returns:
        object: A dynamic configuration object with attributes
    """
    config_class = type('DynamicConfig', (), {
        'KEY_ANGLES': config_dict.get('KEY_ANGLES', {}),
        'NORMALIZATION_JOINTS': config_dict.get('NORMALIZATION_JOINTS', []),
        'DESCRIPTION': config_dict.get('Describe', ''),
        'NUMERIC_ID': numeric_id
    })
    return config_class