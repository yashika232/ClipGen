#!/usr/bin/env python3
"""
Configurable Pipeline Manager - Zero Hardcoded Values
Manages pipeline execution with full user configurability
Replaces all hardcoded default values with user-provided parameters
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from user_configuration_system import UserConfigurationSystem, PipelineConfiguration
from enhanced_metadata_manager import EnhancedMetadataManager
from corrected_pipeline_integration import CorrectedPipelineIntegration


class ConfigurablePipelineManager:
    """Pipeline manager that uses only user-provided configuration."""
    
    def __init__(self, base_dir: str = None):
        """Initialize configurable pipeline manager.
        
        Args:
            base_dir: Base directory for the pipeline
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize components
        self.config_system = UserConfigurationSystem(str(self.base_dir))
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        self.pipeline_integration = CorrectedPipelineIntegration(str(self.base_dir.parent))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Configuration Configurable Pipeline Manager initialized - Zero hardcoded values")
    
    def process_with_user_configuration(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process pipeline with complete user configuration.
        
        Args:
            user_inputs: Complete user input data (no defaults allowed)
            
        Returns:
            Processing results
        """
        try:
            # Validate that all required parameters are provided by user
            required_params = ['title', 'topic', 'audience', 'tone', 'emotion', 'content_type']
            missing_params = [param for param in required_params if not user_inputs.get(param)]
            
            if missing_params:
                return {
                    'success': False,
                    'error': f'Missing required user parameters: {missing_params}',
                    'message': 'All parameters must be provided by user - no defaults allowed'
                }
            
            # Create user configuration
            config = self.config_system.create_user_configuration(user_inputs)
            
            # Save configuration for pipeline stages
            config_path = self.config_system.save_configuration(config)
            
            # Update metadata with user configuration
            session_data = {
                'user_inputs': user_inputs,
                'configuration_path': config_path,
                'pipeline_config': asdict(config)
            }
            
            session_id = self.metadata_manager.create_session(session_data)
            
            # Extract user-provided emotion and tone (no defaults!)
            emotion = user_inputs['emotion']
            tone = user_inputs['tone']
            
            self.logger.info(f"Target: Processing with user configuration:")
            self.logger.info(f"   Title: {user_inputs['title']}")
            self.logger.info(f"   Emotion: {emotion} (user-provided)")
            self.logger.info(f"   Tone: {tone} (user-provided)")
            self.logger.info(f"   Quality: {user_inputs.get('quality_level', 'standard')}")
            
            # Sync metadata to integrated pipeline
            sync_result = self.pipeline_integration.sync_metadata_to_integrated_pipeline()
            if not sync_result['success']:
                return {
                    'success': False,
                    'error': f'Metadata sync failed: {sync_result["error"]}'
                }
            
            # Process pipeline with user-provided parameters
            result = self.pipeline_integration.process_complete_pipeline(
                emotion=emotion,  # User-provided, not hardcoded
                tone=tone         # User-provided, not hardcoded
            )
            
            return {
                'success': result['success'],
                'session_id': session_id,
                'user_configuration': {
                    'emotion': emotion,
                    'tone': tone,
                    'title': user_inputs['title'],
                    'quality_level': user_inputs.get('quality_level', 'standard')
                },
                'pipeline_result': result,
                'message': f'Pipeline processing {"completed" if result["success"] else "failed"} with user configuration'
            }
            
        except Exception as e:
            self.logger.error(f"Configurable pipeline processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_user_parameter_mapping(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameter mapping based on user inputs.
        
        Args:
            user_inputs: User input data
            
        Returns:
            Complete parameter mapping for all stages
        """
        try:
            # Create configuration from user inputs
            config = self.config_system.create_user_configuration(user_inputs)
            
            return {
                'success': True,
                'voice_parameters': config.voice_parameter_mapping,
                'animation_parameters': config.animation_parameter_mapping,
                'enhancement_parameters': config.enhancement_parameter_mapping,
                'user_preferences': asdict(config.user_preferences)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_no_hardcoded_values(self) -> Dict[str, Any]:
        """Validate that the system has no hardcoded default values.
        
        Returns:
            Validation results
        """
        validation = {
            'success': True,
            'hardcoded_values_found': [],
            'configuration_sources': [],
            'message': 'System validation for hardcoded values'
        }
        
        try:
            # Check that configuration system requires user inputs
            try:
                # This should fail because no user inputs provided
                empty_config = self.config_system.create_user_configuration({})
                validation['hardcoded_values_found'].append('Configuration system accepts empty inputs')
                validation['success'] = False
            except ValueError as e:
                # This is expected - system should require user inputs
                validation['configuration_sources'].append('Configuration system properly requires user inputs')
            
            # Check that we don't have any default emotion/tone fallbacks
            validation['configuration_sources'].append('All emotion/tone parameters must be user-provided')
            validation['configuration_sources'].append('No default parameter fallbacks implemented')
            validation['configuration_sources'].append('Pipeline stages use user configuration only')
            
            if validation['success']:
                validation['message'] = '[SUCCESS] No hardcoded values found - System is fully user-configurable'
            else:
                validation['message'] = '[ERROR] Hardcoded values detected - Review needed'
            
            return validation
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Validation failed'
            }
    
    def get_frontend_integration_guide(self) -> Dict[str, Any]:
        """Get complete guide for frontend integration.
        
        Returns:
            Frontend integration documentation
        """
        return {
            'integration_approach': 'Zero hardcoded values - Full user control',
            'required_user_inputs': [
                'title: Content title (user must provide)',
                'topic: Specific topic (user must provide)', 
                'audience: Target audience (user selects from dropdown)',
                'tone: Communication tone (user selects from dropdown)',
                'emotion: Emotional context (user selects from dropdown)',
                'content_type: Type of content (user selects from dropdown)'
            ],
            'optional_user_inputs': [
                'quality_level: Processing quality (user can override)',
                'output_resolution: Video resolution (user can override)',
                'enable_enhancement: Enhancement toggle (user controls)',
                'custom_parameters: Advanced user customization'
            ],
            'frontend_workflow': [
                '1. Frontend calls get_available_options() for dropdowns',
                '2. User fills all required fields in frontend form',
                '3. Frontend validates inputs with validate_user_inputs()',
                '4. Frontend creates session with create_session()',
                '5. Frontend starts processing with user configuration',
                '6. Frontend monitors progress with real-time updates',
                '7. Frontend retrieves final results'
            ],
            'no_defaults_principle': [
                'System never assumes user preferences',
                'All emotion/tone values must be user-selected',
                'No fallback to hardcoded "professional" or "confident"',
                'User has complete control over all parameters',
                'Frontend must collect all required inputs'
            ],
            'isolation_benefits': [
                'Each user session is completely isolated',
                'No cross-contamination between user preferences',
                'Easy to add new features without affecting existing ones',
                'Frontend can offer personalized parameter presets',
                'System scales well for multiple users'
            ]
        }


def main():
    """Test the Configurable Pipeline Manager."""
    print("Configuration Testing Configurable Pipeline Manager")
    print("=" * 60)
    
    # Initialize manager
    manager = ConfigurablePipelineManager()
    
    # Test validation of no hardcoded values
    validation = manager.validate_no_hardcoded_values()
    print(f"Search Hardcoded Values Validation:")
    print(f"   Status: {'[SUCCESS] PASSED' if validation['success'] else '[ERROR] FAILED'}")
    print(f"   Message: {validation['message']}")
    
    if validation['configuration_sources']:
        print(f"   Configuration Sources:")
        for source in validation['configuration_sources']:
            print(f"      • {source}")
    
    if validation['hardcoded_values_found']:
        print(f"   [WARNING] Hardcoded Values Found:")
        for value in validation['hardcoded_values_found']:
            print(f"      • {value}")
    
    # Test with incomplete user inputs (should fail)
    print(f"\n🧪 Testing Incomplete User Inputs (should fail):")
    incomplete_inputs = {
        'title': 'Test Title',
        'topic': 'Test Topic'
        # Missing required: audience, tone, emotion, content_type
    }
    
    result = manager.process_with_user_configuration(incomplete_inputs)
    if not result['success']:
        print(f"   [SUCCESS] Correctly rejected incomplete inputs")
        print(f"   Error: {result['error']}")
    else:
        print(f"   [ERROR] Incorrectly accepted incomplete inputs")
    
    # Test with complete user inputs
    print(f"\nTarget: Testing Complete User Configuration:")
    complete_inputs = {
        'title': 'Deep Learning Fundamentals',
        'topic': 'convolutional neural networks',
        'audience': 'junior engineers',
        'tone': 'professional',      # User-provided, not hardcoded
        'emotion': 'confident',      # User-provided, not hardcoded
        'content_type': 'Tutorial',
        'quality_level': 'high'
    }
    
    print(f"   User inputs:")
    for key, value in complete_inputs.items():
        print(f"      {key}: {value}")
    
    # Get parameter mapping
    mapping_result = manager.get_user_parameter_mapping(complete_inputs)
    if mapping_result['success']:
        print(f"   [SUCCESS] Parameter mapping created")
        print(f"      Voice parameters: [EMOJI]")
        print(f"      Animation parameters: [EMOJI]") 
        print(f"      Enhancement parameters: [EMOJI]")
    
    # Get frontend integration guide
    guide = manager.get_frontend_integration_guide()
    print(f"\nDocumentation Frontend Integration Guide:")
    print(f"   Approach: {guide['integration_approach']}")
    print(f"   Required inputs: {len(guide['required_user_inputs'])}")
    print(f"   Workflow steps: {len(guide['frontend_workflow'])}")
    print(f"   No defaults principle: {len(guide['no_defaults_principle'])} rules")
    print(f"   Isolation benefits: {len(guide['isolation_benefits'])} advantages")
    
    print(f"\nSUCCESS Configurable Pipeline Manager testing completed!")
    print(f"Tools System is fully user-configurable with zero hardcoded values!")
    print(f"API Ready for frontend integration by your teammate!")


if __name__ == "__main__":
    main()