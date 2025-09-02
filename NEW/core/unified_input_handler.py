#!/usr/bin/env python3
"""
Unified Input Handler for Video Synthesis Pipeline
Handles all user inputs with validation, processing, and metadata integration
"""

import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
import json
import hashlib
import mimetypes
from PIL import Image
import wave
import librosa

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager


class UnifiedInputHandler:
    """Unified handler for all user inputs with validation and processing."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the unified input handler.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize metadata manager
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Supported file formats
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_audio_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        self.supported_document_formats = {'.pdf', '.txt', '.md', '.docx'}
        
        # Validation constraints
        self.image_constraints = {
            'min_width': 256,
            'max_width': 4096,
            'min_height': 256,
            'max_height': 4096,
            'max_file_size_mb': 50
        }
        
        self.audio_constraints = {
            'min_duration_seconds': 5,
            'max_duration_seconds': 300,  # 5 minutes
            'min_sample_rate': 16000,
            'max_file_size_mb': 100
        }
        
        # Valid input values
        self.valid_tones = {'professional', 'friendly', 'motivational', 'casual'}
        self.valid_emotions = {'inspired', 'confident', 'curious', 'excited', 'calm'}
        self.valid_audiences = {'junior engineers', 'senior engineers', 'new hires', 'students', 'professionals', 'managers'}
        self.valid_content_types = {'Short-Form Video Reel', 'Full Training Module', 'Quick Tutorial', 'Presentation'}
        
        self.logger.info("STARTING Unified Input Handler initialized")
    
    def create_session_with_inputs(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session with validated user inputs.
        
        Args:
            user_inputs: Dictionary containing all user inputs
            
        Returns:
            Dictionary with session creation results
        """
        try:
            # Validate user inputs
            validation_result = self.validate_user_inputs(user_inputs)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'Input validation failed',
                    'validation_errors': validation_result['errors'],
                    'warnings': validation_result.get('warnings', [])
                }
            
            # Create session with validated inputs
            session_id = self.metadata_manager.create_session(validation_result['validated_inputs'])
            
            if session_id:
                self.logger.info(f"[SUCCESS] Session created successfully: {session_id}")
                return {
                    'success': True,
                    'session_id': session_id,
                    'validated_inputs': validation_result['validated_inputs'],
                    'warnings': validation_result.get('warnings', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create session',
                    'session_id': None
                }
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error creating session: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': None
            }
    
    def validate_user_inputs(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all user inputs.
        
        Args:
            user_inputs: Dictionary containing user inputs
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_inputs': {}
        }
        
        # Required fields
        required_fields = ['title', 'topic', 'audience', 'tone', 'emotion', 'content_type']
        
        for field in required_fields:
            if field not in user_inputs or not user_inputs[field]:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required field: {field}")
            else:
                validation_result['validated_inputs'][field] = str(user_inputs[field]).strip()
        
        # Validate specific fields
        if validation_result['valid']:
            # Validate tone
            tone = validation_result['validated_inputs'].get('tone', '').lower()
            if tone not in self.valid_tones:
                validation_result['warnings'].append(f"Invalid tone '{tone}', using 'professional'")
                validation_result['validated_inputs']['tone'] = 'professional'
            else:
                validation_result['validated_inputs']['tone'] = tone
            
            # Validate emotion
            emotion = validation_result['validated_inputs'].get('emotion', '').lower()
            if emotion not in self.valid_emotions:
                validation_result['warnings'].append(f"Invalid emotion '{emotion}', using 'inspired'")
                validation_result['validated_inputs']['emotion'] = 'inspired'
            else:
                validation_result['validated_inputs']['emotion'] = emotion
            
            # Validate audience
            audience = validation_result['validated_inputs'].get('audience', '').lower()
            if audience not in self.valid_audiences:
                validation_result['warnings'].append(f"Invalid audience '{audience}', using 'junior engineers'")
                validation_result['validated_inputs']['audience'] = 'junior engineers'
            else:
                validation_result['validated_inputs']['audience'] = audience
            
            # Validate content type
            content_type = validation_result['validated_inputs'].get('content_type', '')
            if content_type not in self.valid_content_types:
                validation_result['warnings'].append(f"Invalid content type '{content_type}', using 'Short-Form Video Reel'")
                validation_result['validated_inputs']['content_type'] = 'Short-Form Video Reel'
            else:
                validation_result['validated_inputs']['content_type'] = content_type
            
            # Validate text lengths
            title = validation_result['validated_inputs'].get('title', '')
            if len(title) > 100:
                validation_result['warnings'].append("Title truncated to 100 characters")
                validation_result['validated_inputs']['title'] = title[:100].strip()
            
            topic = validation_result['validated_inputs'].get('topic', '')
            if len(topic) > 200:
                validation_result['warnings'].append("Topic truncated to 200 characters")
                validation_result['validated_inputs']['topic'] = topic[:200].strip()
            
            # Add optional context
            additional_context = user_inputs.get('additional_context', '')
            if additional_context:
                if len(additional_context) > 1000:
                    validation_result['warnings'].append("Additional context truncated to 1000 characters")
                    additional_context = additional_context[:1000].strip()
                validation_result['validated_inputs']['additional_context'] = additional_context
            else:
                validation_result['validated_inputs']['additional_context'] = ''
        
        return validation_result
    
    def process_user_assets(self, face_image_path: str = None, 
                           voice_sample_path: str = None,
                           document_path: str = None) -> Dict[str, Any]:
        """Process and validate user assets (image, voice, document).
        
        Args:
            face_image_path: Path to face image file
            voice_sample_path: Path to voice sample file
            document_path: Path to document file
            
        Returns:
            Dictionary with processing results
        """
        processing_result = {
            'success': True,
            'processed_assets': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Process face image
            if face_image_path:
                image_result = self._process_face_image(face_image_path)
                if image_result['success']:
                    processing_result['processed_assets']['face_image'] = image_result['processed_path']
                    if image_result.get('warnings'):
                        processing_result['warnings'].extend(image_result['warnings'])
                else:
                    processing_result['success'] = False
                    processing_result['errors'].extend(image_result['errors'])
            
            # Process voice sample
            if voice_sample_path:
                voice_result = self._process_voice_sample(voice_sample_path)
                if voice_result['success']:
                    processing_result['processed_assets']['voice_sample'] = voice_result['processed_path']
                    if voice_result.get('warnings'):
                        processing_result['warnings'].extend(voice_result['warnings'])
                else:
                    processing_result['success'] = False
                    processing_result['errors'].extend(voice_result['errors'])
            
            # Process document
            if document_path:
                document_result = self._process_document(document_path)
                if document_result['success']:
                    processing_result['processed_assets']['document'] = document_result['processed_path']
                    if document_result.get('warnings'):
                        processing_result['warnings'].extend(document_result['warnings'])
                else:
                    processing_result['success'] = False
                    processing_result['errors'].extend(document_result['errors'])
            
            # Update metadata with processed assets
            if processing_result['success'] and processing_result['processed_assets']:
                success = self.metadata_manager.update_user_assets(
                    face_image=processing_result['processed_assets'].get('face_image'),
                    voice_sample=processing_result['processed_assets'].get('voice_sample'),
                    document=processing_result['processed_assets'].get('document')
                )
                
                if success:
                    self.logger.info("[SUCCESS] User assets updated in metadata")
                else:
                    processing_result['warnings'].append("Failed to update metadata with processed assets")
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error processing user assets: {e}")
            return {
                'success': False,
                'processed_assets': {},
                'errors': [str(e)],
                'warnings': []
            }
    
    def _process_face_image(self, image_path: str) -> Dict[str, Any]:
        """Process and validate face image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with processing results
        """
        try:
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                return {
                    'success': False,
                    'errors': [f"Image file not found: {image_path}"],
                    'processed_path': None
                }
            
            # Check file format
            file_extension = image_path.suffix.lower()
            if file_extension not in self.supported_image_formats:
                return {
                    'success': False,
                    'errors': [f"Unsupported image format: {file_extension}. Supported: {self.supported_image_formats}"],
                    'processed_path': None
                }
            
            # Check file size
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.image_constraints['max_file_size_mb']:
                return {
                    'success': False,
                    'errors': [f"Image file too large: {file_size_mb:.1f}MB (max: {self.image_constraints['max_file_size_mb']}MB)"],
                    'processed_path': None
                }
            
            # Load and validate image
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check dimensions
                if width < self.image_constraints['min_width'] or height < self.image_constraints['min_height']:
                    return {
                        'success': False,
                        'errors': [f"Image too small: {width}x{height} (min: {self.image_constraints['min_width']}x{self.image_constraints['min_height']})"],
                        'processed_path': None
                    }
                
                if width > self.image_constraints['max_width'] or height > self.image_constraints['max_height']:
                    return {
                        'success': False,
                        'errors': [f"Image too large: {width}x{height} (max: {self.image_constraints['max_width']}x{self.image_constraints['max_height']})"],
                        'processed_path': None
                    }
                
                # Generate unique filename
                timestamp = int(datetime.now().timestamp())
                file_hash = self._get_file_hash(image_path)
                new_filename = f"face_{timestamp}_{file_hash[:8]}.jpg"
                destination_path = self.base_dir / "user_assets" / "faces" / new_filename
                
                # Convert to RGB if necessary and save as JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(destination_path, 'JPEG', quality=95)
                
                # Get relative path
                relative_path = destination_path.relative_to(self.base_dir)
                
                self.logger.info(f"[SUCCESS] Face image processed: {relative_path}")
                
                return {
                    'success': True,
                    'processed_path': str(relative_path),
                    'warnings': [],
                    'image_info': {
                        'width': width,
                        'height': height,
                        'format': 'JPEG',
                        'size_mb': destination_path.stat().st_size / (1024 * 1024)
                    }
                }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error processing image: {str(e)}"],
                'processed_path': None
            }
    
    def _process_voice_sample(self, voice_path: str) -> Dict[str, Any]:
        """Process and validate voice sample.
        
        Args:
            voice_path: Path to the voice file
            
        Returns:
            Dictionary with processing results
        """
        try:
            voice_path = Path(voice_path)
            
            # Check if file exists
            if not voice_path.exists():
                return {
                    'success': False,
                    'errors': [f"Voice file not found: {voice_path}"],
                    'processed_path': None
                }
            
            # Check file format
            file_extension = voice_path.suffix.lower()
            if file_extension not in self.supported_audio_formats:
                return {
                    'success': False,
                    'errors': [f"Unsupported audio format: {file_extension}. Supported: {self.supported_audio_formats}"],
                    'processed_path': None
                }
            
            # Check file size
            file_size_mb = voice_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.audio_constraints['max_file_size_mb']:
                return {
                    'success': False,
                    'errors': [f"Audio file too large: {file_size_mb:.1f}MB (max: {self.audio_constraints['max_file_size_mb']}MB)"],
                    'processed_path': None
                }
            
            # Load and analyze audio
            try:
                audio_data, sample_rate = librosa.load(voice_path, sr=None)
                duration = len(audio_data) / sample_rate
            except Exception as e:
                return {
                    'success': False,
                    'errors': [f"Error reading audio file: {str(e)}"],
                    'processed_path': None
                }
            
            # Validate duration
            if duration < self.audio_constraints['min_duration_seconds']:
                return {
                    'success': False,
                    'errors': [f"Audio too short: {duration:.1f}s (min: {self.audio_constraints['min_duration_seconds']}s)"],
                    'processed_path': None
                }
            
            if duration > self.audio_constraints['max_duration_seconds']:
                return {
                    'success': False,
                    'errors': [f"Audio too long: {duration:.1f}s (max: {self.audio_constraints['max_duration_seconds']}s)"],
                    'processed_path': None
                }
            
            # Validate sample rate
            if sample_rate < self.audio_constraints['min_sample_rate']:
                return {
                    'success': False,
                    'errors': [f"Sample rate too low: {sample_rate}Hz (min: {self.audio_constraints['min_sample_rate']}Hz)"],
                    'processed_path': None
                }
            
            # Generate unique filename
            timestamp = int(datetime.now().timestamp())
            file_hash = self._get_file_hash(voice_path)
            new_filename = f"voice_{timestamp}_{file_hash[:8]}.wav"
            destination_path = self.base_dir / "user_assets" / "voices" / new_filename
            
            # Convert to WAV format if necessary
            if file_extension != '.wav':
                # Save as WAV using wave module
                import wave
                with wave.open(str(destination_path), 'w') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
            else:
                # Copy WAV file
                shutil.copy2(voice_path, destination_path)
            
            # Get relative path
            relative_path = destination_path.relative_to(self.base_dir)
            
            self.logger.info(f"[SUCCESS] Voice sample processed: {relative_path}")
            
            return {
                'success': True,
                'processed_path': str(relative_path),
                'warnings': [],
                'audio_info': {
                    'duration_seconds': duration,
                    'sample_rate': sample_rate,
                    'format': 'WAV',
                    'size_mb': destination_path.stat().st_size / (1024 * 1024)
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error processing audio: {str(e)}"],
                'processed_path': None
            }
    
    def _process_document(self, document_path: str) -> Dict[str, Any]:
        """Process and validate document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Dictionary with processing results
        """
        try:
            document_path = Path(document_path)
            
            # Check if file exists
            if not document_path.exists():
                return {
                    'success': False,
                    'errors': [f"Document file not found: {document_path}"],
                    'processed_path': None
                }
            
            # Check file format
            file_extension = document_path.suffix.lower()
            if file_extension not in self.supported_document_formats:
                return {
                    'success': False,
                    'errors': [f"Unsupported document format: {file_extension}. Supported: {self.supported_document_formats}"],
                    'processed_path': None
                }
            
            # Check file size (max 10MB for documents)
            file_size_mb = document_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:
                return {
                    'success': False,
                    'errors': [f"Document file too large: {file_size_mb:.1f}MB (max: 10MB)"],
                    'processed_path': None
                }
            
            # Generate unique filename
            timestamp = int(datetime.now().timestamp())
            file_hash = self._get_file_hash(document_path)
            new_filename = f"document_{timestamp}_{file_hash[:8]}{file_extension}"
            destination_path = self.base_dir / "user_assets" / "documents" / new_filename
            
            # Copy document
            shutil.copy2(document_path, destination_path)
            
            # Get relative path
            relative_path = destination_path.relative_to(self.base_dir)
            
            self.logger.info(f"[SUCCESS] Document processed: {relative_path}")
            
            return {
                'success': True,
                'processed_path': str(relative_path),
                'warnings': [],
                'document_info': {
                    'format': file_extension.upper(),
                    'size_mb': file_size_mb
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error processing document: {str(e)}"],
                'processed_path': None
            }
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_input_validation_info(self) -> Dict[str, Any]:
        """Get information about input validation constraints.
        
        Returns:
            Dictionary with validation information
        """
        return {
            'supported_formats': {
                'images': list(self.supported_image_formats),
                'audio': list(self.supported_audio_formats),
                'documents': list(self.supported_document_formats)
            },
            'constraints': {
                'image': self.image_constraints,
                'audio': self.audio_constraints
            },
            'valid_values': {
                'tones': list(self.valid_tones),
                'emotions': list(self.valid_emotions),
                'audiences': list(self.valid_audiences),
                'content_types': list(self.valid_content_types)
            },
            'text_limits': {
                'title_max_length': 100,
                'topic_max_length': 200,
                'context_max_length': 1000
            }
        }
    
    def validate_session_prerequisites(self) -> Dict[str, Any]:
        """Validate that current session has all required inputs and assets.
        
        Returns:
            Dictionary with validation results
        """
        return self.metadata_manager.validate_prerequisites()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session inputs and assets.
        
        Returns:
            Dictionary with session summary
        """
        try:
            # Get session info
            session_info = self.metadata_manager.get_session_info()
            user_inputs = self.metadata_manager.get_user_inputs()
            user_assets = self.metadata_manager.get_user_assets()
            
            # Calculate asset sizes and info
            asset_info = {}
            if user_assets:
                for asset_type, asset_path in user_assets.items():
                    if asset_path and asset_type != 'added_at':
                        full_path = self.base_dir / asset_path
                        if full_path.exists():
                            file_size_mb = full_path.stat().st_size / (1024 * 1024)
                            asset_info[asset_type] = {
                                'path': asset_path,
                                'size_mb': round(file_size_mb, 2),
                                'exists': True
                            }
                        else:
                            asset_info[asset_type] = {
                                'path': asset_path,
                                'size_mb': 0,
                                'exists': False
                            }
            
            return {
                'session_id': session_info.get('session_id'),
                'created_at': session_info.get('created_at'),
                'last_modified': session_info.get('last_modified'),
                'user_inputs': user_inputs,
                'user_assets': asset_info,
                'prerequisites_met': self.validate_session_prerequisites()['valid']
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting session summary: {e}")
            return {
                'error': str(e),
                'session_id': None
            }


def main():
    """Test the unified input handler."""
    handler = UnifiedInputHandler()
    
    # Test user inputs
    test_user_inputs = {
        'title': 'Advanced Machine Learning Concepts',
        'topic': 'Deep Learning and Neural Networks',
        'audience': 'senior engineers',
        'tone': 'professional',
        'emotion': 'confident',
        'content_type': 'Full Training Module',
        'additional_context': 'Focus on practical applications and real-world examples'
    }
    
    print("🧪 Testing Unified Input Handler")
    print("=" * 50)
    
    # Test input validation
    validation_result = handler.validate_user_inputs(test_user_inputs)
    print(f"[SUCCESS] Input validation: {'Valid' if validation_result['valid'] else 'Invalid'}")
    if validation_result['warnings']:
        print(f"   Warnings: {validation_result['warnings']}")
    
    # Test session creation
    session_result = handler.create_session_with_inputs(test_user_inputs)
    if session_result['success']:
        print(f"[SUCCESS] Session created: {session_result['session_id']}")
    else:
        print(f"[ERROR] Session creation failed: {session_result['error']}")
        return
    
    # Test validation info
    validation_info = handler.get_input_validation_info()
    print(f"[SUCCESS] Validation info: {len(validation_info['supported_formats'])} format categories")
    
    # Test session summary
    summary = handler.get_session_summary()
    if 'session_id' in summary:
        print(f"[SUCCESS] Session summary: {summary['user_inputs']['title']}")
    else:
        print(f"[ERROR] Session summary failed: {summary.get('error')}")
    
    # Test prerequisites validation
    prerequisites = handler.validate_session_prerequisites()
    print(f"[SUCCESS] Prerequisites: {'Valid' if prerequisites['valid'] else 'Invalid'}")
    if prerequisites['errors']:
        print(f"   Missing: {prerequisites['errors']}")
    
    print("\nSUCCESS All unified input handler tests completed!")


if __name__ == "__main__":
    main()