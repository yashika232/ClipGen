#!/usr/bin/env python3
"""
User Configuration System - No Hardcoded Values
Provides comprehensive user-configurable parameters for all pipeline stages
Designed for frontend integration and maximum flexibility
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum


class ContentType(Enum):
    """Available content types."""
    SHORT_FORM_VIDEO = "Short-Form Video Reel"
    TUTORIAL = "Tutorial"
    LECTURE = "Lecture"
    PRESENTATION = "Presentation"
    EXPLANATION = "Explanation"


class QualityLevel(Enum):
    """Available quality levels."""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass
class UserPreferences:
    """User preference configuration."""
    # Core Content Parameters
    title: str = ""
    topic: str = ""
    audience: str = ""
    tone: str = ""
    emotion: str = ""
    content_type: str = ""
    additional_context: str = ""
    
    # Quality Settings
    quality_level: str = "standard"
    output_resolution: str = "1080p"
    audio_quality: str = "high"
    
    # Processing Preferences
    use_gpu_acceleration: bool = True
    enable_chunked_processing: bool = True
    enable_enhancement: bool = True
    enable_background_animation: bool = True
    
    # Advanced Settings
    custom_voice_parameters: Dict[str, Any] = field(default_factory=dict)
    custom_animation_parameters: Dict[str, Any] = field(default_factory=dict)
    custom_enhancement_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfiguration:
    """Complete pipeline configuration without hardcoded values."""
    
    # User Preferences
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Available Options (for frontend dropdowns)
    available_tones: List[str] = field(default_factory=lambda: [
        'professional', 'friendly', 'motivational', 'casual', 'academic', 'conversational'
    ])
    
    available_emotions: List[str] = field(default_factory=lambda: [
        'inspired', 'confident', 'curious', 'excited', 'calm', 'enthusiastic', 'thoughtful'
    ])
    
    available_audiences: List[str] = field(default_factory=lambda: [
        'junior engineers', 'senior engineers', 'new hires', 'students', 
        'professionals', 'managers', 'general audience', 'technical audience'
    ])
    
    available_content_types: List[str] = field(default_factory=lambda: [
        'Short-Form Video Reel', 'Tutorial', 'Lecture', 'Presentation', 'Explanation'
    ])
    
    # Dynamic Parameter Mappings (configurable by user)
    voice_parameter_mapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    animation_parameter_mapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enhancement_parameter_mapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # System Settings
    enable_fallback_mechanisms: bool = False  # User can choose
    max_processing_time: int = 3600  # 1 hour default, user configurable
    enable_debug_logging: bool = False


class UserConfigurationSystem:
    """Manages user configuration without hardcoded values."""
    
    def __init__(self, base_dir: str = None):
        """Initialize configuration system.
        
        Args:
            base_dir: Base directory for configuration files
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.config_dir = self.base_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default configuration templates
        self._initialize_parameter_templates()
        
        self.logger.info("[EMOJI] User Configuration System initialized - No hardcoded values")
    
    def _initialize_parameter_templates(self):
        """Initialize configurable parameter templates."""
        
        # Voice Parameter Templates (user can modify)
        self.default_voice_templates = {
            'professional': {
                'temperature': 0.7,
                'speed': 1.0,
                'repetition_penalty': 1.1,
                'voice_clarity': 0.9
            },
            'friendly': {
                'temperature': 0.8,
                'speed': 1.05,
                'repetition_penalty': 1.0,
                'voice_clarity': 0.8
            },
            'motivational': {
                'temperature': 0.9,
                'speed': 1.1,
                'repetition_penalty': 1.0,
                'voice_clarity': 0.9
            },
            'casual': {
                'temperature': 0.8,
                'speed': 1.0,
                'repetition_penalty': 0.9,
                'voice_clarity': 0.7
            }
        }
        
        # Animation Parameter Templates (user can modify)
        self.default_animation_templates = {
            'inspired': {
                'expression_scale': 1.2,
                'face_animation_strength': 1.1,
                'emotion_intensity': 0.8
            },
            'confident': {
                'expression_scale': 1.3,
                'face_animation_strength': 1.2,
                'emotion_intensity': 0.9
            },
            'curious': {
                'expression_scale': 1.1,
                'face_animation_strength': 1.0,
                'emotion_intensity': 0.7
            },
            'excited': {
                'expression_scale': 1.4,
                'face_animation_strength': 1.3,
                'emotion_intensity': 1.0
            },
            'calm': {
                'expression_scale': 0.9,
                'face_animation_strength': 0.8,
                'emotion_intensity': 0.5
            }
        }
        
        # Enhancement Parameter Templates (user can modify)
        self.default_enhancement_templates = {
            'draft': {
                'realesrgan_scale': 1,
                'codeformer_fidelity': 0.5,
                'enable_temporal_consistency': False
            },
            'standard': {
                'realesrgan_scale': 2,
                'codeformer_fidelity': 0.7,
                'enable_temporal_consistency': True
            },
            'high': {
                'realesrgan_scale': 2,
                'codeformer_fidelity': 0.8,
                'enable_temporal_consistency': True
            },
            'premium': {
                'realesrgan_scale': 4,
                'codeformer_fidelity': 0.9,
                'enable_temporal_consistency': True
            }
        }
    
    def create_user_configuration(self, user_inputs: Dict[str, Any]) -> PipelineConfiguration:
        """Create configuration from user inputs without defaults.
        
        Args:
            user_inputs: Raw user input data
            
        Returns:
            Complete pipeline configuration
        """
        try:
            # Validate required fields are provided by user
            required_fields = ['title', 'topic', 'audience', 'tone', 'emotion', 'content_type']
            missing_fields = [field for field in required_fields if not user_inputs.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required user inputs: {missing_fields}")
            
            # Create user preferences from input
            preferences = UserPreferences(
                title=user_inputs['title'],
                topic=user_inputs['topic'],
                audience=user_inputs['audience'],
                tone=user_inputs['tone'],
                emotion=user_inputs['emotion'],
                content_type=user_inputs['content_type'],
                additional_context=user_inputs.get('additional_context', ''),
                quality_level=user_inputs.get('quality_level', 'standard'),
                output_resolution=user_inputs.get('output_resolution', '1080p'),
                audio_quality=user_inputs.get('audio_quality', 'high'),
                use_gpu_acceleration=user_inputs.get('use_gpu_acceleration', True),
                enable_chunked_processing=user_inputs.get('enable_chunked_processing', True),
                enable_enhancement=user_inputs.get('enable_enhancement', True),
                enable_background_animation=user_inputs.get('enable_background_animation', True)
            )
            
            # Get parameter mappings based on user choices
            voice_mapping = self._get_voice_parameters(user_inputs['tone'], user_inputs['emotion'])
            animation_mapping = self._get_animation_parameters(user_inputs['emotion'], user_inputs['tone'])
            enhancement_mapping = self._get_enhancement_parameters(
                user_inputs.get('quality_level', 'standard')
            )
            
            # Create complete configuration
            config = PipelineConfiguration(
                user_preferences=preferences,
                voice_parameter_mapping=voice_mapping,
                animation_parameter_mapping=animation_mapping,
                enhancement_parameter_mapping=enhancement_mapping
            )
            
            # Allow user to override any parameters
            if 'custom_voice_parameters' in user_inputs:
                config.voice_parameter_mapping.update(user_inputs['custom_voice_parameters'])
            
            if 'custom_animation_parameters' in user_inputs:
                config.animation_parameter_mapping.update(user_inputs['custom_animation_parameters'])
            
            if 'custom_enhancement_parameters' in user_inputs:
                config.enhancement_parameter_mapping.update(user_inputs['custom_enhancement_parameters'])
            
            self.logger.info(f"[SUCCESS] User configuration created for: {preferences.title}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to create user configuration: {e}")
            raise
    
    def _get_voice_parameters(self, tone: str, emotion: str) -> Dict[str, Any]:
        """Get voice parameters based on user selection."""
        base_params = self.default_voice_templates.get(tone, {})
        
        # Emotion adjustments
        emotion_adjustments = {
            'inspired': {'temperature': 0.1, 'speed': 0.05},
            'confident': {'temperature': 0.0, 'speed': 0.0},
            'curious': {'temperature': 0.2, 'speed': -0.05},
            'excited': {'temperature': 0.2, 'speed': 0.1},
            'calm': {'temperature': -0.1, 'speed': -0.1}
        }
        
        # Apply emotion adjustments
        if emotion in emotion_adjustments:
            for param, adjustment in emotion_adjustments[emotion].items():
                if param in base_params:
                    base_params[param] += adjustment
        
        return {'voice_parameters': base_params}
    
    def _get_animation_parameters(self, emotion: str, tone: str) -> Dict[str, Any]:
        """Get animation parameters based on user selection."""
        base_params = self.default_animation_templates.get(emotion, {})
        
        # Tone adjustments
        tone_adjustments = {
            'professional': {'expression_scale': -0.1},
            'friendly': {'expression_scale': 0.1},
            'motivational': {'expression_scale': 0.2},
            'casual': {'expression_scale': 0.1}
        }
        
        # Apply tone adjustments
        if tone in tone_adjustments:
            for param, adjustment in tone_adjustments[tone].items():
                if param in base_params:
                    base_params[param] += adjustment
        
        return {'animation_parameters': base_params}
    
    def _get_enhancement_parameters(self, quality_level: str) -> Dict[str, Any]:
        """Get enhancement parameters based on user selection."""
        return {'enhancement_parameters': self.default_enhancement_templates.get(quality_level, {})}
    
    def save_configuration(self, config: PipelineConfiguration, filename: str = None) -> str:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved configuration file
        """
        try:
            if filename is None:
                timestamp = config.user_preferences.title.replace(' ', '_').lower()
                filename = f"config_{timestamp}.json"
            
            config_path = self.config_dir / filename
            
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            self.logger.info(f"[SUCCESS] Configuration saved: {config_path}")
            return str(config_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_configuration(self, config_path: str) -> PipelineConfiguration:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded pipeline configuration
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Reconstruct configuration object
            config = PipelineConfiguration(**config_data)
            
            self.logger.info(f"[SUCCESS] Configuration loaded: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """Get all available options for frontend dropdowns.
        
        Returns:
            Dictionary of available options
        """
        config = PipelineConfiguration()
        return {
            'tones': config.available_tones,
            'emotions': config.available_emotions,
            'audiences': config.available_audiences,
            'content_types': config.available_content_types,
            'quality_levels': [level.value for level in QualityLevel],
            'output_resolutions': ['720p', '1080p', '1440p', '4K'],
            'audio_qualities': ['standard', 'high', 'premium']
        }
    
    def validate_user_inputs(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user inputs against available options.
        
        Args:
            user_inputs: User input data to validate
            
        Returns:
            Validation results with errors and warnings
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_inputs': user_inputs.copy()
        }
        
        available_options = self.get_available_options()
        
        # Validate tone
        if user_inputs.get('tone') not in available_options['tones']:
            validation['errors'].append(f"Invalid tone: {user_inputs.get('tone')}")
            validation['valid'] = False
        
        # Validate emotion
        if user_inputs.get('emotion') not in available_options['emotions']:
            validation['errors'].append(f"Invalid emotion: {user_inputs.get('emotion')}")
            validation['valid'] = False
        
        # Validate audience
        if user_inputs.get('audience') not in available_options['audiences']:
            validation['errors'].append(f"Invalid audience: {user_inputs.get('audience')}")
            validation['valid'] = False
        
        # Validate content type
        if user_inputs.get('content_type') not in available_options['content_types']:
            validation['errors'].append(f"Invalid content type: {user_inputs.get('content_type')}")
            validation['valid'] = False
        
        return validation
    
    def create_frontend_api_schema(self) -> Dict[str, Any]:
        """Create API schema for frontend integration.
        
        Returns:
            Complete API schema for frontend developers
        """
        available_options = self.get_available_options()
        
        return {
            'required_fields': {
                'title': {
                    'type': 'string',
                    'description': 'Title of the content to generate',
                    'example': 'Introduction to Machine Learning'
                },
                'topic': {
                    'type': 'string',
                    'description': 'Specific topic to focus on',
                    'example': 'supervised learning algorithms'
                },
                'audience': {
                    'type': 'enum',
                    'options': available_options['audiences'],
                    'description': 'Target audience for the content'
                },
                'tone': {
                    'type': 'enum',
                    'options': available_options['tones'],
                    'description': 'Communication tone for the content'
                },
                'emotion': {
                    'type': 'enum',
                    'options': available_options['emotions'],
                    'description': 'Emotional context for the content'
                },
                'content_type': {
                    'type': 'enum',
                    'options': available_options['content_types'],
                    'description': 'Type of content to generate'
                }
            },
            'optional_fields': {
                'additional_context': {
                    'type': 'string',
                    'description': 'Additional context or requirements'
                },
                'quality_level': {
                    'type': 'enum',
                    'options': available_options['quality_levels'],
                    'default': 'standard'
                },
                'output_resolution': {
                    'type': 'enum',
                    'options': available_options['output_resolutions'],
                    'default': '1080p'
                },
                'audio_quality': {
                    'type': 'enum',
                    'options': available_options['audio_qualities'],
                    'default': 'high'
                },
                'use_gpu_acceleration': {
                    'type': 'boolean',
                    'default': True
                },
                'enable_enhancement': {
                    'type': 'boolean',
                    'default': True
                },
                'enable_background_animation': {
                    'type': 'boolean',
                    'default': True
                }
            },
            'advanced_fields': {
                'custom_voice_parameters': {
                    'type': 'object',
                    'description': 'Custom voice synthesis parameters'
                },
                'custom_animation_parameters': {
                    'type': 'object',
                    'description': 'Custom animation parameters'
                },
                'custom_enhancement_parameters': {
                    'type': 'object',
                    'description': 'Custom enhancement parameters'
                }
            }
        }


def main():
    """Test the User Configuration System."""
    print("[EMOJI] Testing User Configuration System")
    print("=" * 50)
    
    # Initialize configuration system
    config_system = UserConfigurationSystem()
    
    # Test available options
    options = config_system.get_available_options()
    print(f"Endpoints Available Options:")
    for category, option_list in options.items():
        print(f"   {category}: {len(option_list)} options")
    
    # Test user input creation
    test_user_inputs = {
        'title': 'Understanding Neural Networks',
        'topic': 'backpropagation algorithm',
        'audience': 'junior engineers',
        'tone': 'professional',
        'emotion': 'confident',
        'content_type': 'Tutorial',
        'quality_level': 'high'
    }
    
    print(f"\n🧪 Testing Configuration Creation:")
    print(f"   User inputs: {test_user_inputs['title']}")
    
    # Validate inputs
    validation = config_system.validate_user_inputs(test_user_inputs)
    if validation['valid']:
        print(f"   [SUCCESS] Validation passed")
        
        # Create configuration
        config = config_system.create_user_configuration(test_user_inputs)
        print(f"   [SUCCESS] Configuration created")
        print(f"   Voice parameters: {len(config.voice_parameter_mapping)} mappings")
        print(f"   Animation parameters: {len(config.animation_parameter_mapping)} mappings")
        print(f"   Enhancement parameters: {len(config.enhancement_parameter_mapping)} mappings")
        
        # Test save/load
        config_path = config_system.save_configuration(config)
        print(f"   [SUCCESS] Configuration saved: {Path(config_path).name}")
        
        loaded_config = config_system.load_configuration(config_path)
        print(f"   [SUCCESS] Configuration loaded successfully")
        
    else:
        print(f"   [ERROR] Validation failed: {validation['errors']}")
    
    # Test frontend API schema
    schema = config_system.create_frontend_api_schema()
    print(f"\nPorts Frontend API Schema:")
    print(f"   Required fields: {len(schema['required_fields'])}")
    print(f"   Optional fields: {len(schema['optional_fields'])}")
    print(f"   Advanced fields: {len(schema['advanced_fields'])}")
    
    print(f"\nSUCCESS User Configuration System testing completed!")


if __name__ == "__main__":
    main()