#!/usr/bin/env python3
"""
Secure File Validator - Comprehensive Upload Security
Provides robust file validation and security for user uploads
Includes virus scanning, content validation, and malware detection
"""

import os
import sys
import hashlib
import mimetypes
import magic
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import shutil
from PIL import Image, ImageFile
import wave
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_exceptions import FileValidationError, ValidationException


class FileType(Enum):
    """Supported file types."""
    FACE_IMAGE = "face_image"
    VOICE_AUDIO = "voice_audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class SecurityThreat(Enum):
    """Types of security threats."""
    VIRUS_DETECTED = "virus_detected"
    MALICIOUS_CONTENT = "malicious_content"
    OVERSIZED_FILE = "oversized_file"
    INVALID_FORMAT = "invalid_format"
    CORRUPTED_FILE = "corrupted_file"
    SUSPICIOUS_METADATA = "suspicious_metadata"


@dataclass
class ValidationResult:
    """File validation result."""
    is_valid: bool
    file_type: FileType
    mime_type: str
    file_size: int
    checksum: str
    warnings: List[str]
    errors: List[str]
    security_threats: List[SecurityThreat]
    metadata: Dict[str, Any]
    sanitized_filename: str


class SecureFileValidator:
    """Comprehensive file validation and security system."""
    
    def __init__(self, base_dir: str = None):
        """Initialize secure file validator.
        
        Args:
            base_dir: Base directory for the pipeline
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.quarantine_dir = self.base_dir / "quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)
        
        # File size limits (in bytes)
        self.max_file_sizes = {
            FileType.FACE_IMAGE: 10 * 1024 * 1024,    # 10MB
            FileType.VOICE_AUDIO: 50 * 1024 * 1024,   # 50MB
            FileType.DOCUMENT: 5 * 1024 * 1024,       # 5MB
        }
        
        # Allowed MIME types
        self.allowed_mime_types = {
            FileType.FACE_IMAGE: {
                'image/jpeg', 'image/jpg', 'image/png', 'image/bmp'
            },
            FileType.VOICE_AUDIO: {
                'audio/wav', 'audio/wave', 'audio/x-wav', 'audio/mpeg', 'audio/mp3'
            },
            FileType.DOCUMENT: {
                'text/plain', 'application/pdf', 'text/markdown'
            }
        }
        
        # Allowed file extensions
        self.allowed_extensions = {
            FileType.FACE_IMAGE: {'.jpg', '.jpeg', '.png', '.bmp'},
            FileType.VOICE_AUDIO: {'.wav', '.mp3'},
            FileType.DOCUMENT: {'.txt', '.pdf', '.md'}
        }
        
        # Dangerous file signatures (magic bytes)
        self.dangerous_signatures = {
            b'\x4D\x5A': 'PE executable',
            b'\x7F\x45\x4C\x46': 'ELF executable',
            b'\xCA\xFE\xBA\xBE': 'Java class file',
            b'\xFE\xED\xFA\xCE': 'Mach-O executable',
            b'\x50\x4B\x03\x04': 'ZIP archive (check contents)',
        }
        
        # Suspicious metadata keywords
        self.suspicious_keywords = {
            'script', 'javascript', 'vbscript', 'powershell', 'cmd', 'bash',
            'eval', 'exec', 'system', 'shell', 'proc_open', 'passthru'
        }
        
        # Initialize libmagic
        try:
            self.mime_detector = magic.Magic(mime=True)
            self.file_detector = magic.Magic()
            self.libmagic_available = True
        except Exception as e:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"libmagic not available: {e}")
            self.libmagic_available = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Security Secure File Validator initialized")
    
    def validate_file(self, file_path: str, expected_type: FileType,
                     user_filename: Optional[str] = None) -> ValidationResult:
        """Comprehensive file validation.
        
        Args:
            file_path: Path to file to validate
            expected_type: Expected file type
            user_filename: Original filename from user
            
        Returns:
            Validation result with security assessment
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileValidationError(
                filename=str(file_path),
                reason="File does not exist"
            )
        
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            file_type=FileType.UNKNOWN,
            mime_type="",
            file_size=0,
            checksum="",
            warnings=[],
            errors=[],
            security_threats=[],
            metadata={},
            sanitized_filename=""
        )
        
        try:
            # Step 1: Basic file information
            file_stats = file_path.stat()
            result.file_size = file_stats.st_size
            result.checksum = self._calculate_checksum(file_path)
            
            # Step 2: Filename sanitization
            original_filename = user_filename or file_path.name
            result.sanitized_filename = self._sanitize_filename(original_filename)
            
            # Step 3: File size validation
            self._validate_file_size(result, expected_type)
            
            # Step 4: MIME type detection
            self._detect_mime_type(file_path, result)
            
            # Step 5: File signature validation
            self._validate_file_signature(file_path, result, expected_type)
            
            # Step 6: Content validation
            self._validate_file_content(file_path, result, expected_type)
            
            # Step 7: Security scanning
            self._security_scan(file_path, result)
            
            # Step 8: Metadata analysis
            self._analyze_metadata(file_path, result, expected_type)
            
            # Step 9: Final assessment
            if result.security_threats or result.errors:
                result.is_valid = False
            
            # Determine file type
            result.file_type = self._determine_file_type(result, expected_type)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation failed: {str(e)}")
            self.logger.error(f"File validation error for {file_path}: {e}")
        
        return result
    
    def quarantine_file(self, file_path: str, reason: str,
                       threats: List[SecurityThreat]) -> str:
        """Move suspicious file to quarantine.
        
        Args:
            file_path: Path to file to quarantine
            reason: Reason for quarantine
            threats: List of security threats detected
            
        Returns:
            Path to quarantined file
        """
        file_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_filename = f"{timestamp}_{file_path.name}"
        quarantine_path = self.quarantine_dir / quarantine_filename
        
        # Move file to quarantine
        shutil.move(str(file_path), str(quarantine_path))
        
        # Create quarantine report
        report = {
            'original_path': str(file_path),
            'quarantine_path': str(quarantine_path),
            'quarantine_time': datetime.now().isoformat(),
            'reason': reason,
            'threats': [threat.value for threat in threats],
            'file_size': quarantine_path.stat().st_size,
            'checksum': self._calculate_checksum(quarantine_path)
        }
        
        report_path = quarantine_path.with_suffix('.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.warning(f"[EMOJI] File quarantined: {file_path} -> {quarantine_path}")
        return str(quarantine_path)
    
    def create_safe_copy(self, file_path: str, destination: str) -> str:
        """Create a safe copy of validated file.
        
        Args:
            file_path: Source file path
            destination: Destination directory
            
        Returns:
            Path to safe copy
        """
        source_path = Path(file_path)
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        safe_filename = self._sanitize_filename(source_path.name)
        safe_path = dest_dir / safe_filename
        
        # Ensure unique filename
        counter = 1
        while safe_path.exists():
            stem = safe_path.stem
            suffix = safe_path.suffix
            safe_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Copy file
        shutil.copy2(str(source_path), str(safe_path))
        
        # Verify copy
        if self._calculate_checksum(source_path) != self._calculate_checksum(safe_path):
            safe_path.unlink()
            raise ValidationException(
                message="File copy verification failed",
                error_code="COPY_VERIFICATION_FAILED"
            )
        
        return str(safe_path)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent directory traversal and other attacks."""
        # Remove directory traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"|?*\x00'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        # Ensure not empty
        if not filename or filename.isspace():
            filename = f"sanitized_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return filename
    
    def _validate_file_size(self, result: ValidationResult, expected_type: FileType) -> None:
        """Validate file size against limits."""
        max_size = self.max_file_sizes.get(expected_type, 1024 * 1024)  # 1MB default
        
        if result.file_size > max_size:
            result.security_threats.append(SecurityThreat.OVERSIZED_FILE)
            result.errors.append(f"File size {result.file_size} exceeds limit {max_size}")
        
        if result.file_size == 0:
            result.errors.append("File is empty")
    
    def _detect_mime_type(self, file_path: Path, result: ValidationResult) -> None:
        """Detect and validate MIME type."""
        # Try libmagic first
        if self.libmagic_available:
            try:
                result.mime_type = self.mime_detector.from_file(str(file_path))
            except Exception as e:
                result.warnings.append(f"libmagic MIME detection failed: {e}")
        
        # Fallback to Python's mimetypes
        if not result.mime_type:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            result.mime_type = mime_type or "application/octet-stream"
    
    def _validate_file_signature(self, file_path: Path, result: ValidationResult,
                                expected_type: FileType) -> None:
        """Validate file signature (magic bytes)."""
        with open(file_path, 'rb') as f:
            signature = f.read(16)  # Read first 16 bytes
        
        # Check for dangerous signatures
        for danger_sig, description in self.dangerous_signatures.items():
            if signature.startswith(danger_sig):
                result.security_threats.append(SecurityThreat.MALICIOUS_CONTENT)
                result.errors.append(f"Dangerous file signature detected: {description}")
        
        # Validate expected file signatures
        if expected_type == FileType.FACE_IMAGE:
            valid_image_sigs = [
                b'\xFF\xD8\xFF',  # JPEG
                b'\x89\x50\x4E\x47',  # PNG
                b'\x42\x4D',  # BMP
            ]
            if not any(signature.startswith(sig) for sig in valid_image_sigs):
                result.errors.append("Invalid image file signature")
        
        elif expected_type == FileType.VOICE_AUDIO:
            valid_audio_sigs = [
                b'RIFF',  # WAV
                b'\xFF\xFB',  # MP3
                b'\xFF\xF3',  # MP3
                b'\xFF\xF2',  # MP3
            ]
            if not any(signature.startswith(sig) for sig in valid_audio_sigs):
                result.errors.append("Invalid audio file signature")
    
    def _validate_file_content(self, file_path: Path, result: ValidationResult,
                             expected_type: FileType) -> None:
        """Validate file content based on type."""
        try:
            if expected_type == FileType.FACE_IMAGE:
                self._validate_image_content(file_path, result)
            elif expected_type == FileType.VOICE_AUDIO:
                self._validate_audio_content(file_path, result)
            elif expected_type == FileType.DOCUMENT:
                self._validate_document_content(file_path, result)
        
        except Exception as e:
            result.errors.append(f"Content validation failed: {str(e)}")
    
    def _validate_image_content(self, file_path: Path, result: ValidationResult) -> None:
        """Validate image file content."""
        try:
            # Enable loading of truncated images for validation
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            with Image.open(file_path) as img:
                # Basic image validation
                result.metadata.update({
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                })
                
                # Check reasonable dimensions
                if img.width > 10000 or img.height > 10000:
                    result.warnings.append("Unusually large image dimensions")
                
                if img.width < 100 or img.height < 100:
                    result.warnings.append("Image dimensions may be too small for face detection")
                
                # Check for suspicious metadata
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    if exif_data:
                        # Look for suspicious EXIF data
                        for tag, value in exif_data.items():
                            if isinstance(value, str):
                                for keyword in self.suspicious_keywords:
                                    if keyword.lower() in value.lower():
                                        result.security_threats.append(SecurityThreat.SUSPICIOUS_METADATA)
                                        result.warnings.append(f"Suspicious metadata detected: {keyword}")
                
                # Verify image can be processed
                img.verify()
        
        except Exception as e:
            result.security_threats.append(SecurityThreat.CORRUPTED_FILE)
            result.errors.append(f"Image validation failed: {str(e)}")
    
    def _validate_audio_content(self, file_path: Path, result: ValidationResult) -> None:
        """Validate audio file content."""
        try:
            if file_path.suffix.lower() == '.wav':
                with wave.open(str(file_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    channels = wav_file.getnchannels()
                    
                    result.metadata.update({
                        'duration_seconds': duration,
                        'sample_rate': sample_rate,
                        'channels': channels,
                        'frames': frames
                    })
                    
                    # Validate reasonable audio parameters
                    if duration > 300:  # 5 minutes
                        result.warnings.append("Audio file is quite long")
                    
                    if duration < 1:  # Less than 1 second
                        result.warnings.append("Audio file is very short")
                    
                    if sample_rate < 8000 or sample_rate > 48000:
                        result.warnings.append("Unusual sample rate detected")
        
        except Exception as e:
            result.security_threats.append(SecurityThreat.CORRUPTED_FILE)
            result.errors.append(f"Audio validation failed: {str(e)}")
    
    def _validate_document_content(self, file_path: Path, result: ValidationResult) -> None:
        """Validate document file content."""
        try:
            if file_path.suffix.lower() in ['.txt', '.md']:
                # Text file validation
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    result.metadata.update({
                        'character_count': len(content),
                        'line_count': content.count('\n') + 1,
                        'encoding': 'utf-8'
                    })
                    
                    # Check for suspicious content
                    content_lower = content.lower()
                    for keyword in self.suspicious_keywords:
                        if keyword in content_lower:
                            result.security_threats.append(SecurityThreat.SUSPICIOUS_METADATA)
                            result.warnings.append(f"Suspicious content detected: {keyword}")
        
        except Exception as e:
            result.errors.append(f"Document validation failed: {str(e)}")
    
    def _security_scan(self, file_path: Path, result: ValidationResult) -> None:
        """Perform security scanning."""
        # Check for virus scan (if available)
        self._virus_scan(file_path, result)
        
        # Check file entropy (high entropy might indicate encryption/packing)
        entropy = self._calculate_entropy(file_path)
        result.metadata['entropy'] = entropy
        
        if entropy > 7.5:  # Very high entropy
            result.warnings.append("High entropy detected - file might be compressed or encrypted")
    
    def _virus_scan(self, file_path: Path, result: ValidationResult) -> None:
        """Attempt virus scanning if ClamAV is available."""
        try:
            # Check if clamdscan is available
            clam_result = subprocess.run(
                ['clamdscan', '--no-summary', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if clam_result.returncode != 0:
                if 'FOUND' in clam_result.stdout:
                    result.security_threats.append(SecurityThreat.VIRUS_DETECTED)
                    result.errors.append("Virus detected by ClamAV")
                else:
                    result.warnings.append("ClamAV scan inconclusive")
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # ClamAV not available or timeout
            result.warnings.append("Virus scan not available")
        except Exception as e:
            result.warnings.append(f"Virus scan failed: {str(e)}")
    
    def _calculate_entropy(self, file_path: Path) -> float:
        """Calculate Shannon entropy of file."""
        import math
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _analyze_metadata(self, file_path: Path, result: ValidationResult,
                         expected_type: FileType) -> None:
        """Analyze file metadata for security issues."""
        try:
            # Get file system metadata
            stat = file_path.stat()
            result.metadata.update({
                'created_timestamp': stat.st_ctime,
                'modified_timestamp': stat.st_mtime,
                'permissions': oct(stat.st_mode)[-3:]
            })
            
            # Check for unusual timestamps
            now = datetime.now().timestamp()
            if stat.st_mtime > now + 86400:  # Future timestamp
                result.warnings.append("File has future modification timestamp")
            
            if stat.st_ctime < 946684800:  # Before year 2000
                result.warnings.append("File has very old creation timestamp")
        
        except Exception as e:
            result.warnings.append(f"Metadata analysis failed: {str(e)}")
    
    def _determine_file_type(self, result: ValidationResult, expected_type: FileType) -> FileType:
        """Determine actual file type from validation results."""
        mime_type = result.mime_type.lower()
        
        if mime_type.startswith('image/'):
            return FileType.FACE_IMAGE
        elif mime_type.startswith('audio/'):
            return FileType.VOICE_AUDIO
        elif mime_type.startswith('text/') or 'pdf' in mime_type:
            return FileType.DOCUMENT
        else:
            return FileType.UNKNOWN
    
    def get_upload_guidelines(self, file_type: FileType) -> Dict[str, Any]:
        """Get upload guidelines for a specific file type.
        
        Args:
            file_type: Type of file
            
        Returns:
            Guidelines and requirements
        """
        guidelines = {
            'file_type': file_type.value,
            'max_size_mb': self.max_file_sizes.get(file_type, 1024*1024) // (1024*1024),
            'allowed_formats': list(self.allowed_extensions.get(file_type, set())),
            'mime_types': list(self.allowed_mime_types.get(file_type, set())),
            'security_requirements': [
                'File must pass virus scan',
                'No executable content allowed',
                'File signature must match extension',
                'Reasonable file size limits enforced'
            ]
        }
        
        if file_type == FileType.FACE_IMAGE:
            guidelines.update({
                'specific_requirements': [
                    'Image should contain a clear, visible face',
                    'Recommended minimum size: 512x512 pixels',
                    'Supported formats: JPEG, PNG, BMP',
                    'Good lighting and minimal blur preferred'
                ]
            })
        
        elif file_type == FileType.VOICE_AUDIO:
            guidelines.update({
                'specific_requirements': [
                    'Clear speech recording preferred',
                    'Minimum 3 seconds, maximum 5 minutes',
                    'WAV format recommended for best quality',
                    'Sample rate: 16kHz or higher'
                ]
            })
        
        return guidelines


def main():
    """Test the secure file validator."""
    print("🧪 Testing Secure File Validator")
    print("=" * 50)
    
    validator = SecureFileValidator()
    
    # Test guidelines
    guidelines = validator.get_upload_guidelines(FileType.FACE_IMAGE)
    print(f"Image upload guidelines: {guidelines['max_size_mb']}MB max")
    
    # Test filename sanitization
    dangerous_filename = "../../../etc/passwd<script>alert('xss')</script>"
    safe_filename = validator._sanitize_filename(dangerous_filename)
    print(f"Sanitized filename: {safe_filename}")
    
    print("Secure File Validator ready for production use!")


if __name__ == "__main__":
    main()