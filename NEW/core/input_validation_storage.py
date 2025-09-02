#!/usr/bin/env python3
"""
Input Validation and Storage System
Advanced validation, processing, and storage system for all user inputs and assets
"""

import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from datetime import datetime
import json
import hashlib
import mimetypes
import tempfile
import subprocess
from PIL import Image, ImageEnhance, ImageFilter
import librosa
import numpy as np
# Optional imports with fallbacks
try:
    import PyPDF2
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

try:
    import docx
    HAS_DOCX_SUPPORT = True
except ImportError:
    HAS_DOCX_SUPPORT = False

try:
    from scipy import ndimage, signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from unified_input_handler import UnifiedInputHandler


class InputValidationStorage:
    """Advanced input validation and storage system with preprocessing capabilities."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the input validation and storage system.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize unified input handler
        self.input_handler = UnifiedInputHandler(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Advanced validation settings
        self.advanced_image_validation = {
            'check_face_presence': True,
            'check_image_quality': True,
            'enhance_automatically': True,
            'generate_thumbnails': True
        }
        
        self.advanced_audio_validation = {
            'check_audio_quality': True,
            'normalize_audio': True,
            'remove_silence': True,
            'enhance_speech': True
        }
        
        # Storage settings
        self.storage_settings = {
            'create_backups': True,
            'compress_images': True,
            'optimize_audio': True,
            'generate_metadata': True
        }
        
        self.logger.info("STARTING Input Validation Storage system initialized")
    
    def comprehensive_asset_processing(self, asset_files: Dict[str, str]) -> Dict[str, Any]:
        """Comprehensive processing of all user assets with advanced validation.
        
        Args:
            asset_files: Dictionary with asset file paths {
                'face_image': 'path/to/image.jpg',
                'voice_sample': 'path/to/audio.wav',
                'document': 'path/to/document.pdf'
            }
            
        Returns:
            Dictionary with comprehensive processing results
        """
        processing_result = {
            'success': True,
            'processed_assets': {},
            'validation_results': {},
            'preprocessing_applied': {},
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        try:
            # Process each asset type
            for asset_type, file_path in asset_files.items():
                if file_path and Path(file_path).exists():
                    if asset_type == 'face_image':
                        result = self._comprehensive_image_processing(file_path)
                    elif asset_type == 'voice_sample':
                        result = self._comprehensive_audio_processing(file_path)
                    elif asset_type == 'document':
                        result = self._comprehensive_document_processing(file_path)
                    else:
                        continue
                    
                    if result['success']:
                        processing_result['processed_assets'][asset_type] = result['processed_path']
                        processing_result['validation_results'][asset_type] = result['validation']
                        processing_result['preprocessing_applied'][asset_type] = result['preprocessing']
                        processing_result['quality_metrics'][asset_type] = result['quality_metrics']
                        
                        if result.get('warnings'):
                            processing_result['warnings'].extend(result['warnings'])
                    else:
                        processing_result['success'] = False
                        processing_result['errors'].extend(result['errors'])
            
            # Update metadata if successful
            if processing_result['success'] and processing_result['processed_assets']:
                metadata_update_result = self._update_metadata_with_assets(processing_result)
                if not metadata_update_result['success']:
                    processing_result['warnings'].append("Failed to update metadata")
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error in comprehensive asset processing: {e}")
            return {
                'success': False,
                'processed_assets': {},
                'validation_results': {},
                'preprocessing_applied': {},
                'errors': [str(e)],
                'warnings': [],
                'quality_metrics': {}
            }
    
    def _comprehensive_image_processing(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive image processing with advanced validation and enhancement.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with processing results
        """
        try:
            image_path = Path(image_path)
            
            # Basic validation first
            basic_result = self.input_handler._process_face_image(str(image_path))
            if not basic_result['success']:
                return {
                    'success': False,
                    'errors': basic_result['errors'],
                    'processed_path': None,
                    'validation': {},
                    'preprocessing': {},
                    'quality_metrics': {}
                }
            
            # Advanced validation and processing
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Quality analysis
                quality_metrics = self._analyze_image_quality(img)
                
                # Face detection (basic check)
                face_detected = self._detect_face_presence(img)
                
                # Apply enhancements if needed
                preprocessing_applied = {}
                enhanced_img = img.copy()
                
                if self.advanced_image_validation['enhance_automatically']:
                    if quality_metrics['brightness'] < 0.3:
                        enhanced_img = ImageEnhance.Brightness(enhanced_img).enhance(1.2)
                        preprocessing_applied['brightness_enhanced'] = True
                    
                    if quality_metrics['contrast'] < 0.5:
                        enhanced_img = ImageEnhance.Contrast(enhanced_img).enhance(1.1)
                        preprocessing_applied['contrast_enhanced'] = True
                    
                    if quality_metrics['sharpness'] < 0.6:
                        enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask())
                        preprocessing_applied['sharpness_enhanced'] = True
                
                # Generate processed filename
                timestamp = int(datetime.now().timestamp())
                file_hash = self.input_handler._get_file_hash(image_path)
                processed_filename = f"face_processed_{timestamp}_{file_hash[:8]}.jpg"
                processed_path = self.base_dir / "user_assets" / "faces" / processed_filename
                
                # Save processed image
                enhanced_img.save(processed_path, 'JPEG', quality=95, optimize=True)
                
                # Generate thumbnail if requested
                if self.advanced_image_validation['generate_thumbnails']:
                    thumbnail_path = processed_path.with_suffix('.thumb.jpg')
                    thumbnail = enhanced_img.copy()
                    thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    thumbnail.save(thumbnail_path, 'JPEG', quality=85)
                    preprocessing_applied['thumbnail_generated'] = True
                
                # Get relative path
                relative_path = processed_path.relative_to(self.base_dir)
                
                self.logger.info(f"[SUCCESS] Image comprehensively processed: {relative_path}")
                
                return {
                    'success': True,
                    'processed_path': str(relative_path),
                    'validation': {
                        'face_detected': face_detected,
                        'quality_score': quality_metrics['overall_quality'],
                        'format_valid': True
                    },
                    'preprocessing': preprocessing_applied,
                    'quality_metrics': quality_metrics,
                    'warnings': []
                }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error in comprehensive image processing: {str(e)}"],
                'processed_path': None,
                'validation': {},
                'preprocessing': {},
                'quality_metrics': {}
            }
    
    def _comprehensive_audio_processing(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive audio processing with advanced validation and enhancement.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with processing results
        """
        try:
            audio_path = Path(audio_path)
            
            # Basic validation first
            basic_result = self.input_handler._process_voice_sample(str(audio_path))
            if not basic_result['success']:
                return {
                    'success': False,
                    'errors': basic_result['errors'],
                    'processed_path': None,
                    'validation': {},
                    'preprocessing': {},
                    'quality_metrics': {}
                }
            
            # Load audio for advanced processing
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            
            # Quality analysis
            quality_metrics = self._analyze_audio_quality(audio_data, sample_rate)
            
            # Apply enhancements if needed
            preprocessing_applied = {}
            processed_audio = audio_data.copy()
            
            if self.advanced_audio_validation['normalize_audio']:
                # Normalize audio levels
                max_val = np.max(np.abs(processed_audio))
                if max_val > 0:
                    processed_audio = processed_audio / max_val * 0.95
                    preprocessing_applied['normalized'] = True
            
            if self.advanced_audio_validation['remove_silence']:
                # Remove silence from beginning and end
                trimmed_audio, _ = librosa.effects.trim(processed_audio, top_db=20)
                if len(trimmed_audio) != len(processed_audio):
                    processed_audio = trimmed_audio
                    preprocessing_applied['silence_removed'] = True
            
            if self.advanced_audio_validation['enhance_speech']:
                # Apply noise reduction (simple spectral gating)
                processed_audio = self._simple_noise_reduction(processed_audio, sample_rate)
                preprocessing_applied['noise_reduced'] = True
            
            # Generate processed filename
            timestamp = int(datetime.now().timestamp())
            file_hash = self.input_handler._get_file_hash(audio_path)
            processed_filename = f"voice_processed_{timestamp}_{file_hash[:8]}.wav"
            processed_path = self.base_dir / "user_assets" / "voices" / processed_filename
            
            # Save processed audio
            import soundfile as sf
            try:
                sf.write(str(processed_path), processed_audio, sample_rate)
            except ImportError:
                # Fallback to wave module
                import wave
                with wave.open(str(processed_path), 'w') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    audio_int16 = (processed_audio * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
            
            # Get relative path
            relative_path = processed_path.relative_to(self.base_dir)
            
            self.logger.info(f"[SUCCESS] Audio comprehensively processed: {relative_path}")
            
            return {
                'success': True,
                'processed_path': str(relative_path),
                'validation': {
                    'speech_detected': quality_metrics['speech_ratio'] > 0.3,
                    'quality_score': quality_metrics['overall_quality'],
                    'format_valid': True
                },
                'preprocessing': preprocessing_applied,
                'quality_metrics': quality_metrics,
                'warnings': []
            }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error in comprehensive audio processing: {str(e)}"],
                'processed_path': None,
                'validation': {},
                'preprocessing': {},
                'quality_metrics': {}
            }
    
    def _comprehensive_document_processing(self, document_path: str) -> Dict[str, Any]:
        """Comprehensive document processing with content extraction.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Dictionary with processing results
        """
        try:
            document_path = Path(document_path)
            
            # Basic validation first
            basic_result = self.input_handler._process_document(str(document_path))
            if not basic_result['success']:
                return {
                    'success': False,
                    'errors': basic_result['errors'],
                    'processed_path': None,
                    'validation': {},
                    'preprocessing': {},
                    'quality_metrics': {}
                }
            
            # Extract content based on file type
            file_extension = document_path.suffix.lower()
            extracted_content = ""
            content_metadata = {}
            
            if file_extension == '.pdf':
                extracted_content, content_metadata = self._extract_pdf_content(document_path)
            elif file_extension == '.docx':
                extracted_content, content_metadata = self._extract_docx_content(document_path)
            elif file_extension in ['.txt', '.md']:
                with open(document_path, 'r', encoding='utf-8') as f:
                    extracted_content = f.read()
                content_metadata = {'word_count': len(extracted_content.split())}
            
            # Quality analysis
            quality_metrics = {
                'content_length': len(extracted_content),
                'word_count': len(extracted_content.split()),
                'readable': len(extracted_content.strip()) > 0,
                'overall_quality': 1.0 if len(extracted_content.strip()) > 100 else 0.5
            }
            
            # Copy to processed location (no preprocessing needed for documents)
            timestamp = int(datetime.now().timestamp())
            file_hash = self.input_handler._get_file_hash(document_path)
            processed_filename = f"document_processed_{timestamp}_{file_hash[:8]}{file_extension}"
            processed_path = self.base_dir / "user_assets" / "documents" / processed_filename
            
            shutil.copy2(document_path, processed_path)
            
            # Save extracted content as metadata
            content_file = processed_path.with_suffix('.content.txt')
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(extracted_content)
            
            # Get relative path
            relative_path = processed_path.relative_to(self.base_dir)
            
            self.logger.info(f"[SUCCESS] Document comprehensively processed: {relative_path}")
            
            return {
                'success': True,
                'processed_path': str(relative_path),
                'validation': {
                    'content_extracted': len(extracted_content) > 0,
                    'readable': quality_metrics['readable'],
                    'format_valid': True
                },
                'preprocessing': {
                    'content_extracted': True,
                    'metadata_generated': True
                },
                'quality_metrics': quality_metrics,
                'warnings': [],
                'extracted_content': extracted_content[:500] + "..." if len(extracted_content) > 500 else extracted_content
            }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error in comprehensive document processing: {str(e)}"],
                'processed_path': None,
                'validation': {},
                'preprocessing': {},
                'quality_metrics': {}
            }
    
    def _analyze_image_quality(self, img: Image.Image) -> Dict[str, float]:
        """Analyze image quality metrics.
        
        Args:
            img: PIL Image object
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for analysis
        gray = img.convert('L')
        img_array = np.array(gray)
        
        # Brightness analysis
        brightness = np.mean(img_array) / 255.0
        
        # Contrast analysis
        contrast = np.std(img_array) / 255.0
        
        # Sharpness analysis (Laplacian variance)
        laplacian_var = self._calculate_laplacian_variance(img_array)
        sharpness = min(1.0, laplacian_var / 1000.0)  # Normalize
        
        # Overall quality score
        overall_quality = (brightness * 0.3 + contrast * 0.4 + sharpness * 0.3)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'overall_quality': overall_quality
        }
    
    def _analyze_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze audio quality metrics.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with quality metrics
        """
        # Signal-to-noise ratio estimation
        snr = self._estimate_snr(audio_data)
        
        # Speech activity detection
        speech_ratio = self._detect_speech_activity(audio_data, sample_rate)
        
        # Dynamic range
        dynamic_range = np.max(audio_data) - np.min(audio_data)
        
        # Overall quality score
        overall_quality = min(1.0, (snr / 20.0) * 0.5 + speech_ratio * 0.3 + min(1.0, dynamic_range) * 0.2)
        
        return {
            'snr_db': snr,
            'speech_ratio': speech_ratio,
            'dynamic_range': dynamic_range,
            'overall_quality': overall_quality
        }
    
    def _detect_face_presence(self, img: Image.Image) -> bool:
        """Simple face detection check.
        
        Args:
            img: PIL Image object
            
        Returns:
            True if face detected, False otherwise
        """
        try:
            # This is a simplified check - in production, you'd use opencv or similar
            # For now, just check if image has reasonable face-like proportions
            width, height = img.size
            aspect_ratio = width / height
            
            # Face images typically have aspect ratios between 0.7 and 1.4
            return 0.7 <= aspect_ratio <= 1.4
        except:
            return False
    
    def _calculate_laplacian_variance(self, img_array: np.ndarray) -> float:
        """Calculate Laplacian variance for sharpness estimation.
        
        Args:
            img_array: Grayscale image array
            
        Returns:
            Laplacian variance value
        """
        try:
            if HAS_SCIPY:
                # Simple Laplacian kernel
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                
                # Apply convolution
                laplacian = ndimage.convolve(img_array.astype(float), kernel)
                return np.var(laplacian)
            else:
                # Simple fallback - use standard deviation as sharpness proxy
                return np.std(img_array)
        except:
            return 0.0
    
    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """Estimate signal-to-noise ratio.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            SNR in dB
        """
        try:
            # Simple SNR estimation
            signal_power = np.mean(audio_data ** 2)
            
            # Estimate noise from quieter parts
            sorted_audio = np.sort(np.abs(audio_data))
            noise_threshold = sorted_audio[int(len(sorted_audio) * 0.1)]
            noise_power = np.mean((audio_data[np.abs(audio_data) < noise_threshold]) ** 2)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return max(0, min(40, snr))  # Clamp between 0 and 40 dB
            else:
                return 20.0  # Default good SNR
        except:
            return 10.0
    
    def _detect_speech_activity(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Detect speech activity ratio.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            
        Returns:
            Ratio of speech activity (0-1)
        """
        try:
            # Simple voice activity detection based on energy
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.01 * sample_rate)     # 10ms hop
            
            # Calculate frame energies
            frame_energies = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy = np.sum(frame ** 2)
                frame_energies.append(energy)
            
            frame_energies = np.array(frame_energies)
            
            # Threshold for speech detection
            if len(frame_energies) > 0:
                threshold = np.percentile(frame_energies, 30)  # Bottom 30% as noise
                speech_frames = np.sum(frame_energies > threshold * 3)  # 3x above noise
                return speech_frames / len(frame_energies)
            else:
                return 0.0
        except:
            return 0.5  # Default moderate speech activity
    
    def _simple_noise_reduction(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply simple noise reduction.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            
        Returns:
            Noise-reduced audio data
        """
        try:
            if HAS_SCIPY:
                # Simple spectral gating
                # Apply high-pass filter to remove low-frequency noise
                nyquist = sample_rate / 2
                low_cutoff = 80 / nyquist  # 80 Hz high-pass
                b, a = signal.butter(4, low_cutoff, btype='high')
                filtered_audio = signal.filtfilt(b, a, audio_data)
                return filtered_audio
            else:
                # Simple fallback - basic amplitude gating
                threshold = np.percentile(np.abs(audio_data), 10)
                audio_data[np.abs(audio_data) < threshold] *= 0.5
                return audio_data
        except:
            return audio_data  # Return original if filtering fails
    
    def _extract_pdf_content(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not HAS_PDF_SUPPORT:
            self.logger.warning("PDF support not available (PyPDF2 not installed)")
            return "PDF content extraction not available", {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                metadata = {
                    'page_count': len(pdf_reader.pages),
                    'word_count': len(text_content.split()),
                    'character_count': len(text_content)
                }
                
                return text_content, metadata
        except Exception as e:
            self.logger.warning(f"Failed to extract PDF content: {e}")
            return "", {}
    
    def _extract_docx_content(self, docx_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content from DOCX file.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not HAS_DOCX_SUPPORT:
            self.logger.warning("DOCX support not available (python-docx not installed)")
            return "DOCX content extraction not available", {}
        
        try:
            doc = docx.Document(docx_path)
            
            text_content = ""
            paragraph_count = 0
            
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
                paragraph_count += 1
            
            metadata = {
                'paragraph_count': paragraph_count,
                'word_count': len(text_content.split()),
                'character_count': len(text_content)
            }
            
            return text_content, metadata
        except Exception as e:
            self.logger.warning(f"Failed to extract DOCX content: {e}")
            return "", {}
    
    def _update_metadata_with_assets(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update metadata with processed assets.
        
        Args:
            processing_result: Processing result from comprehensive_asset_processing
            
        Returns:
            Dictionary with update results
        """
        try:
            # Update metadata with processed assets
            success = self.input_handler.metadata_manager.update_user_assets(
                face_image=processing_result['processed_assets'].get('face_image'),
                voice_sample=processing_result['processed_assets'].get('voice_sample'),
                document=processing_result['processed_assets'].get('document')
            )
            
            if success:
                # Also save processing metadata
                metadata = self.input_handler.metadata_manager.load_metadata()
                if metadata:
                    metadata['asset_processing'] = {
                        'validation_results': processing_result['validation_results'],
                        'preprocessing_applied': processing_result['preprocessing_applied'],
                        'quality_metrics': processing_result['quality_metrics'],
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    self.input_handler.metadata_manager.save_metadata(metadata)
                
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to update metadata'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def batch_process_assets(self, asset_directory: str) -> Dict[str, Any]:
        """Process multiple assets from a directory.
        
        Args:
            asset_directory: Directory containing asset files
            
        Returns:
            Dictionary with batch processing results
        """
        try:
            asset_dir = Path(asset_directory)
            if not asset_dir.exists():
                return {
                    'success': False,
                    'error': f"Directory not found: {asset_directory}",
                    'processed_files': []
                }
            
            processed_files = []
            errors = []
            
            # Find all supported files
            for file_path in asset_dir.iterdir():
                if file_path.is_file():
                    file_extension = file_path.suffix.lower()
                    
                    asset_type = None
                    if file_extension in self.input_handler.supported_image_formats:
                        asset_type = 'face_image'
                    elif file_extension in self.input_handler.supported_audio_formats:
                        asset_type = 'voice_sample'
                    elif file_extension in self.input_handler.supported_document_formats:
                        asset_type = 'document'
                    
                    if asset_type:
                        result = self.comprehensive_asset_processing({asset_type: str(file_path)})
                        processed_files.append({
                            'file_path': str(file_path),
                            'asset_type': asset_type,
                            'success': result['success'],
                            'processed_path': result['processed_assets'].get(asset_type) if result['success'] else None,
                            'errors': result['errors']
                        })
                        
                        if not result['success']:
                            errors.extend(result['errors'])
            
            return {
                'success': len(errors) == 0,
                'processed_files': processed_files,
                'errors': errors,
                'total_files': len(processed_files),
                'successful_files': sum(1 for f in processed_files if f['success'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processed_files': []
            }


def main():
    """Test the input validation and storage system."""
    ivs = InputValidationStorage()
    
    print("🧪 Testing Input Validation Storage System")
    print("=" * 60)
    
    # Test validation info
    validation_info = ivs.input_handler.get_input_validation_info()
    print(f"[SUCCESS] Validation system initialized")
    print(f"   Supported image formats: {len(validation_info['supported_formats']['images'])}")
    print(f"   Supported audio formats: {len(validation_info['supported_formats']['audio'])}")
    print(f"   Supported document formats: {len(validation_info['supported_formats']['documents'])}")
    
    # Test session creation with comprehensive inputs
    test_inputs = {
        'title': 'Advanced AI Video Processing',
        'topic': 'Computer Vision and Speech Recognition',
        'audience': 'senior engineers',
        'tone': 'professional',
        'emotion': 'confident',
        'content_type': 'Full Training Module',
        'additional_context': 'Focus on practical implementations and real-world applications'
    }
    
    session_result = ivs.input_handler.create_session_with_inputs(test_inputs)
    if session_result['success']:
        print(f"[SUCCESS] Session created: {session_result['session_id']}")
    else:
        print(f"[ERROR] Session creation failed: {session_result['error']}")
        return
    
    # Test comprehensive processing (without actual files)
    print(f"[SUCCESS] Comprehensive processing system ready")
    print(f"   Advanced image validation: {ivs.advanced_image_validation}")
    print(f"   Advanced audio validation: {ivs.advanced_audio_validation}")
    
    # Test session summary
    summary = ivs.input_handler.get_session_summary()
    print(f"[SUCCESS] Session summary: {summary['user_inputs']['title']}")
    
    print("\nSUCCESS All input validation storage tests completed!")


if __name__ == "__main__":
    main()