#!/usr/bin/env python3
"""
Centralized Metadata System for Video Synthesis Pipeline
Enhanced metadata-driven architecture with path management and validation
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging
import hashlib
import uuid


class CentralizedMetadataSystem:
    """Enhanced centralized metadata management system."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the centralized metadata system.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            # Find NEW directory relative to this file
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.metadata_dir = self.base_dir / "metadata"
        self.user_assets_dir = self.base_dir / "user_assets"
        self.processed_dir = self.base_dir / "processed"
        self.enhanced_dir = self.base_dir / "enhanced"
        self.final_dir = self.base_dir / "final"
        self.thumbnails_dir = self.base_dir / "thumbnails"
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize metadata schema
        self._initialize_metadata_schema()
        
        self.logger.info("STARTING Centralized Metadata System initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Ensure base directory exists before creating log file
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.base_dir / "metadata_system.log")
            ]
        )
    
    def _create_directory_structure(self):
        """Create standardized directory structure."""
        directories = [
            # Core directories
            self.metadata_dir,
            self.user_assets_dir,
            self.thumbnails_dir,
            
            # User asset subdirectories
            self.user_assets_dir / "voices",
            self.user_assets_dir / "faces",
            self.user_assets_dir / "documents",
            
            # Processing directories
            self.processed_dir,
            self.processed_dir / "voice_chunks",
            self.processed_dir / "face_crops",
            self.processed_dir / "video_chunks",
            self.processed_dir / "manim_scripts",
            
            # Enhanced processing directories
            self.enhanced_dir,
            self.enhanced_dir / "video_chunks",
            self.enhanced_dir / "audio_enhanced",
            
            # Final output directories
            self.final_dir,
            self.final_dir / "talking_head",
            self.final_dir / "background_animation",
            self.final_dir / "complete_video",
            
            # Backup and temp directories
            self.metadata_dir / "backups",
            self.base_dir / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def _initialize_metadata_schema(self):
        """Initialize the metadata schema structure."""
        self.metadata_schema = {
            "session_id": str,
            "created_at": str,
            "last_modified": str,
            "version": str,
            
            # User inputs
            "user_inputs": {
                "title": str,
                "topic": str,
                "audience": str,
                "tone": str,
                "emotion": str,
                "content_type": str,
                "additional_context": str
            },
            
            # User assets with paths
            "user_assets": {
                "face_image": str,
                "voice_sample": str,
                "document_path": str,
                "added_at": str
            },
            
            # Generated content
            "generated_content": {
                "clean_script": str,
                "manim_script": str,
                "thumbnail_prompts": list,
                "generated_at": str
            },
            
            # Pipeline stages with paths
            "pipeline_stages": {
                "voice_cloning": {
                    "status": str,
                    "input_paths": dict,
                    "output_paths": dict,
                    "processing_data": dict,
                    "error_info": dict,
                    "timestamps": dict
                },
                "face_processing": {
                    "status": str,
                    "input_paths": dict,
                    "output_paths": dict,
                    "processing_data": dict,
                    "error_info": dict,
                    "timestamps": dict
                },
                "video_generation": {
                    "status": str,
                    "input_paths": dict,
                    "output_paths": dict,
                    "processing_data": dict,
                    "error_info": dict,
                    "timestamps": dict
                },
                "video_enhancement": {
                    "status": str,
                    "input_paths": dict,
                    "output_paths": dict,
                    "processing_data": dict,
                    "error_info": dict,
                    "timestamps": dict
                },
                "background_animation": {
                    "status": str,
                    "input_paths": dict,
                    "output_paths": dict,
                    "processing_data": dict,
                    "error_info": dict,
                    "timestamps": dict
                },
                "final_assembly": {
                    "status": str,
                    "input_paths": dict,
                    "output_paths": dict,
                    "processing_data": dict,
                    "error_info": dict,
                    "timestamps": dict
                }
            },
            
            # Pipeline configuration
            "pipeline_config": {
                "auto_chunking": bool,
                "chunk_duration": int,
                "quality_preset": str,
                "gpu_acceleration": bool,
                "emotion_mapping": bool,
                "max_retries": int,
                "parallel_processing": bool
            },
            
            # Quality metrics
            "quality_metrics": {
                "processing_times": dict,
                "file_sizes": dict,
                "success_rates": dict,
                "error_counts": dict
            }
        }
    
    def create_new_session(self, user_inputs: Dict[str, Any]) -> str:
        """Create a new metadata session.
        
        Args:
            user_inputs: User input data (title, topic, audience, tone, emotion, content_type)
            
        Returns:
            Session ID for the new session
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create new metadata structure
        metadata = {
            "session_id": session_id,
            "created_at": timestamp,
            "last_modified": timestamp,
            "version": "1.0",
            
            # Store user inputs
            "user_inputs": {
                "title": user_inputs.get("title", ""),
                "topic": user_inputs.get("topic", ""),
                "audience": user_inputs.get("audience", ""),
                "tone": user_inputs.get("tone", "professional"),
                "emotion": user_inputs.get("emotion", "inspired"),
                "content_type": user_inputs.get("content_type", "Short-Form Video Reel"),
                "additional_context": user_inputs.get("additional_context", "")
            },
            
            # Initialize empty user assets
            "user_assets": {
                "face_image": None,
                "voice_sample": None,
                "document_path": None,
                "added_at": None
            },
            
            # Initialize empty generated content
            "generated_content": {
                "clean_script": None,
                "manim_script": None,
                "thumbnail_prompts": [],
                "generated_at": None
            },
            
            # Initialize pipeline stages
            "pipeline_stages": self._initialize_pipeline_stages(),
            
            # Default pipeline configuration
            "pipeline_config": {
                "auto_chunking": True,
                "chunk_duration": 10,
                "quality_preset": "high",
                "gpu_acceleration": True,
                "emotion_mapping": True,
                "max_retries": 3,
                "parallel_processing": False
            },
            
            # Initialize quality metrics
            "quality_metrics": {
                "processing_times": {},
                "file_sizes": {},
                "success_rates": {},
                "error_counts": {}
            }
        }
        
        # Save the new metadata
        if self._save_metadata(metadata):
            self.logger.info(f"[SUCCESS] New session created: {session_id}")
            return session_id
        else:
            self.logger.error("[ERROR] Failed to create new session")
            return None
    
    def _initialize_pipeline_stages(self) -> Dict[str, Any]:
        """Initialize pipeline stages structure."""
        stages = {}
        stage_names = [
            "voice_cloning", "face_processing", "video_generation", 
            "video_enhancement", "background_animation", "final_assembly"
        ]
        
        for stage_name in stage_names:
            stages[stage_name] = {
                "status": "pending",  # pending, processing, completed, failed
                "input_paths": {},
                "output_paths": {},
                "processing_data": {},
                "error_info": {},
                "timestamps": {
                    "started_at": None,
                    "completed_at": None,
                    "processing_time": 0
                }
            }
        
        return stages
    
    def load_metadata(self, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Load metadata for a session.
        
        Args:
            session_id: Session ID to load. If None, loads latest session.
            
        Returns:
            Metadata dictionary or None if not found
        """
        if session_id is None:
            # Load latest metadata
            metadata_file = self.metadata_dir / "latest_metadata.json"
        else:
            # Load specific session
            metadata_file = self.metadata_dir / f"session_{session_id}.json"
        
        if not metadata_file.exists():
            # Try backup location
            backup_files = list(self.metadata_dir.glob("latest_metadata.backup_*.json"))
            if backup_files:
                metadata_file = max(backup_files, key=os.path.getmtime)
                self.logger.warning(f"Using backup metadata: {metadata_file.name}")
            else:
                self.logger.error(f"Metadata file not found: {metadata_file}")
                return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate metadata structure
            if self._validate_metadata(metadata):
                self.logger.info(f"[SUCCESS] Metadata loaded: {metadata.get('session_id', 'unknown')}")
                return metadata
            else:
                self.logger.error("[ERROR] Invalid metadata structure")
                return None
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading metadata: {e}")
            return None
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata structure.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["session_id", "created_at", "user_inputs", "pipeline_stages"]
        
        for key in required_keys:
            if key not in metadata:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        # Validate pipeline stages
        required_stages = [
            "voice_cloning", "face_processing", "video_generation",
            "video_enhancement", "background_animation", "final_assembly"
        ]
        
        for stage in required_stages:
            if stage not in metadata["pipeline_stages"]:
                self.logger.warning(f"Missing pipeline stage: {stage}")
                # Initialize missing stage
                metadata["pipeline_stages"][stage] = {
                    "status": "pending",
                    "input_paths": {},
                    "output_paths": {},
                    "processing_data": {},
                    "error_info": {},
                    "timestamps": {
                        "started_at": None,
                        "completed_at": None,
                        "processing_time": 0
                    }
                }
        
        return True
    
    def _save_metadata(self, metadata: Dict[str, Any], create_backup: bool = True) -> bool:
        """Save metadata to file.
        
        Args:
            metadata: Metadata dictionary to save
            create_backup: Whether to create a backup
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Update last modified timestamp
            metadata["last_modified"] = datetime.now().isoformat()
            
            # Main metadata file
            metadata_file = self.metadata_dir / "latest_metadata.json"
            
            # Create backup if requested and file exists
            if create_backup and metadata_file.exists():
                backup_file = self.metadata_dir / "backups" / f"latest_metadata.backup_{int(datetime.now().timestamp())}.json"
                shutil.copy2(metadata_file, backup_file)
                self.logger.debug(f"Created backup: {backup_file.name}")
            
            # Save metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Also save session-specific file
            session_id = metadata.get("session_id")
            if session_id:
                session_file = self.metadata_dir / f"session_{session_id}.json"
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"[SUCCESS] Metadata saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error saving metadata: {e}")
            return False
    
    def update_user_assets(self, face_image_path: str = None, 
                          voice_sample_path: str = None,
                          document_path: str = None) -> bool:
        """Update user assets in metadata.
        
        Args:
            face_image_path: Path to face image
            voice_sample_path: Path to voice sample
            document_path: Path to document
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        # Update user assets
        if face_image_path:
            metadata["user_assets"]["face_image"] = self._get_relative_path(face_image_path)
        
        if voice_sample_path:
            metadata["user_assets"]["voice_sample"] = self._get_relative_path(voice_sample_path)
        
        if document_path:
            metadata["user_assets"]["document_path"] = self._get_relative_path(document_path)
        
        if any([face_image_path, voice_sample_path, document_path]):
            metadata["user_assets"]["added_at"] = datetime.now().isoformat()
        
        return self._save_metadata(metadata)
    
    def update_generated_content(self, clean_script: str = None,
                                manim_script: str = None,
                                thumbnail_prompts: List[Dict] = None) -> bool:
        """Update generated content in metadata.
        
        Args:
            clean_script: Clean script for voice cloning
            manim_script: Manim Python script
            thumbnail_prompts: List of thumbnail prompts
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        # Update generated content
        if clean_script:
            metadata["generated_content"]["clean_script"] = clean_script
        
        if manim_script:
            metadata["generated_content"]["manim_script"] = manim_script
        
        if thumbnail_prompts:
            metadata["generated_content"]["thumbnail_prompts"] = thumbnail_prompts
        
        if any([clean_script, manim_script, thumbnail_prompts]):
            metadata["generated_content"]["generated_at"] = datetime.now().isoformat()
        
        return self._save_metadata(metadata)
    
    def update_stage_status(self, stage_name: str, status: str,
                           input_paths: Dict[str, str] = None,
                           output_paths: Dict[str, str] = None,
                           processing_data: Dict[str, Any] = None,
                           error_info: Dict[str, Any] = None) -> bool:
        """Update pipeline stage status and data.
        
        Args:
            stage_name: Name of the pipeline stage
            status: New status (pending, processing, completed, failed)
            input_paths: Dictionary of input paths
            output_paths: Dictionary of output paths
            processing_data: Processing data and results
            error_info: Error information if failed
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        if stage_name not in metadata["pipeline_stages"]:
            self.logger.error(f"Unknown pipeline stage: {stage_name}")
            return False
        
        stage = metadata["pipeline_stages"][stage_name]
        
        # Update status
        old_status = stage["status"]
        stage["status"] = status
        
        # Update timestamps
        current_time = datetime.now().isoformat()
        
        if old_status == "pending" and status == "processing":
            stage["timestamps"]["started_at"] = current_time
        elif status in ["completed", "failed"]:
            stage["timestamps"]["completed_at"] = current_time
            
            # Calculate processing time
            if stage["timestamps"]["started_at"]:
                start_time = datetime.fromisoformat(stage["timestamps"]["started_at"])
                end_time = datetime.fromisoformat(current_time)
                processing_time = (end_time - start_time).total_seconds()
                stage["timestamps"]["processing_time"] = processing_time
        
        # Update paths (convert to relative paths)
        if input_paths:
            for key, path in input_paths.items():
                stage["input_paths"][key] = self._get_relative_path(path)
        
        if output_paths:
            for key, path in output_paths.items():
                stage["output_paths"][key] = self._get_relative_path(path)
        
        # Update processing data
        if processing_data:
            stage["processing_data"].update(processing_data)
        
        # Update error info
        if error_info:
            stage["error_info"].update(error_info)
        
        self.logger.info(f"[SUCCESS] Stage {stage_name} updated: {old_status} → {status}")
        return self._save_metadata(metadata)
    
    def get_stage_input_path(self, stage_name: str, input_key: str) -> Optional[str]:
        """Get absolute input path for a stage.
        
        Args:
            stage_name: Name of the pipeline stage
            input_key: Key for the input
            
        Returns:
            Absolute path to input file or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        stage = metadata["pipeline_stages"].get(stage_name, {})
        relative_path = stage.get("input_paths", {}).get(input_key)
        
        if relative_path:
            absolute_path = self.base_dir / relative_path
            # Return path even if file doesn't exist yet - it might be created later
            return str(absolute_path)
        
        return None
    
    def get_stage_output_path(self, stage_name: str, output_key: str) -> Optional[str]:
        """Get absolute output path for a stage.
        
        Args:
            stage_name: Name of the pipeline stage
            output_key: Key for the output
            
        Returns:
            Absolute path to output file or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        stage = metadata["pipeline_stages"].get(stage_name, {})
        relative_path = stage.get("output_paths", {}).get(output_key)
        
        if relative_path:
            absolute_path = self.base_dir / relative_path
            return str(absolute_path)
        
        return None
    
    def _get_relative_path(self, path: Union[str, Path]) -> str:
        """Convert absolute path to relative path from base directory.
        
        Args:
            path: Absolute or relative path
            
        Returns:
            Relative path from base directory
        """
        path = Path(path)
        
        if path.is_absolute():
            try:
                relative_path = path.relative_to(self.base_dir)
                return str(relative_path).rstrip('/')
            except ValueError:
                # Path is outside base directory
                return str(path)
        else:
            return str(path).rstrip('/')
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get comprehensive pipeline progress information.
        
        Returns:
            Dictionary with detailed progress information
        """
        metadata = self.load_metadata()
        if metadata is None:
            return {"progress": 0.0, "status": "error", "error": "No metadata found"}
        
        stages = metadata["pipeline_stages"]
        stage_names = [
            "voice_cloning", "face_processing", "video_generation",
            "video_enhancement", "background_animation", "final_assembly"
        ]
        
        completed_stages = 0
        current_stage = None
        failed_stages = []
        processing_stages = []
        
        for stage_name in stage_names:
            stage = stages.get(stage_name, {})
            status = stage.get("status", "pending")
            
            if status == "completed":
                completed_stages += 1
            elif status == "processing":
                processing_stages.append(stage_name)
                if current_stage is None:
                    current_stage = stage_name
            elif status == "failed":
                failed_stages.append(stage_name)
            elif status == "pending" and current_stage is None:
                current_stage = stage_name
        
        progress = completed_stages / len(stage_names)
        
        # Determine overall status
        if failed_stages:
            overall_status = "failed"
        elif progress == 1.0:
            overall_status = "completed"
        elif processing_stages:
            overall_status = "processing"
        else:
            overall_status = "pending"
        
        return {
            "progress": progress,
            "progress_percent": progress * 100,
            "current_stage": current_stage,
            "completed_stages": completed_stages,
            "total_stages": len(stage_names),
            "failed_stages": failed_stages,
            "processing_stages": processing_stages,
            "status": overall_status,
            "session_id": metadata.get("session_id"),
            "created_at": metadata.get("created_at"),
            "last_modified": metadata.get("last_modified")
        }
    
    def get_next_stage(self) -> Optional[str]:
        """Get the next stage to execute based on dependencies.
        
        Returns:
            Name of next stage or None if pipeline is complete
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        stages = metadata["pipeline_stages"]
        
        # Define stage dependencies
        stage_dependencies = {
            "voice_cloning": [],
            "face_processing": [],
            "video_generation": ["voice_cloning", "face_processing"],
            "video_enhancement": ["video_generation"],
            "background_animation": [],
            "final_assembly": ["video_enhancement", "background_animation"]
        }
        
        # Find next stage to execute
        for stage_name, dependencies in stage_dependencies.items():
            stage = stages.get(stage_name, {})
            status = stage.get("status", "pending")
            
            if status in ["pending", "failed"]:
                # Check if all dependencies are completed
                dependencies_met = all(
                    stages.get(dep, {}).get("status") == "completed"
                    for dep in dependencies
                )
                
                if dependencies_met:
                    return stage_name
        
        return None
    
    def validate_paths(self) -> Dict[str, Any]:
        """Validate all paths in metadata.
        
        Returns:
            Dictionary with validation results
        """
        metadata = self.load_metadata()
        if metadata is None:
            return {"valid": False, "error": "No metadata found"}
        
        validation_results = {
            "valid": True,
            "missing_files": [],
            "invalid_paths": [],
            "warnings": []
        }
        
        # Check user assets
        user_assets = metadata.get("user_assets", {})
        for asset_type, relative_path in user_assets.items():
            if relative_path and asset_type != "added_at":
                absolute_path = self.base_dir / relative_path
                if not absolute_path.exists():
                    validation_results["missing_files"].append(f"{asset_type}: {relative_path}")
                    validation_results["valid"] = False
        
        # Check pipeline stage paths
        stages = metadata.get("pipeline_stages", {})
        for stage_name, stage_data in stages.items():
            # Check input paths
            for input_key, relative_path in stage_data.get("input_paths", {}).items():
                if relative_path:
                    absolute_path = self.base_dir / relative_path
                    if not absolute_path.exists():
                        validation_results["missing_files"].append(f"{stage_name}.{input_key}: {relative_path}")
                        validation_results["valid"] = False
            
            # Check output paths
            for output_key, relative_path in stage_data.get("output_paths", {}).items():
                if relative_path:
                    absolute_path = self.base_dir / relative_path
                    if not absolute_path.exists():
                        validation_results["warnings"].append(f"{stage_name}.{output_key}: {relative_path}")
        
        return validation_results
    
    def cleanup_session(self, session_id: str = None, keep_final: bool = True) -> bool:
        """Clean up session files.
        
        Args:
            session_id: Session ID to clean up. If None, cleans current session.
            keep_final: Whether to keep final output files
            
        Returns:
            True if cleaned successfully, False otherwise
        """
        try:
            # Clean up intermediate directories
            cleanup_dirs = [
                self.processed_dir,
                self.enhanced_dir,
                self.base_dir / "temp"
            ]
            
            if not keep_final:
                cleanup_dirs.append(self.final_dir)
            
            for cleanup_dir in cleanup_dirs:
                if cleanup_dir.exists():
                    shutil.rmtree(cleanup_dir)
                    cleanup_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Cleaned: {cleanup_dir}")
            
            # Recreate necessary subdirectories
            self._create_directory_structure()
            
            self.logger.info("[SUCCESS] Session cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error during cleanup: {e}")
            return False


def main():
    """Test the centralized metadata system."""
    cms = CentralizedMetadataSystem()
    
    # Test user inputs
    test_user_inputs = {
        "title": "Test Video",
        "topic": "Machine Learning Basics",
        "audience": "junior engineers",
        "tone": "professional",
        "emotion": "inspired",
        "content_type": "Short-Form Video Reel"
    }
    
    print("🧪 Testing Centralized Metadata System")
    print("=" * 50)
    
    # Test session creation
    session_id = cms.create_new_session(test_user_inputs)
    if session_id:
        print(f"[SUCCESS] Session created: {session_id}")
    else:
        print("[ERROR] Failed to create session")
        return
    
    # Test metadata loading
    metadata = cms.load_metadata()
    if metadata:
        print(f"[SUCCESS] Metadata loaded: {metadata['user_inputs']['title']}")
    else:
        print("[ERROR] Failed to load metadata")
        return
    
    # Test pipeline progress
    progress = cms.get_pipeline_progress()
    print(f"[SUCCESS] Pipeline progress: {progress['progress_percent']:.1f}%")
    
    # Test next stage
    next_stage = cms.get_next_stage()
    print(f"[SUCCESS] Next stage: {next_stage}")
    
    # Test path validation
    validation = cms.validate_paths()
    print(f"[SUCCESS] Path validation: {'Valid' if validation['valid'] else 'Invalid'}")
    
    print("\nSUCCESS All tests completed successfully!")


if __name__ == "__main__":
    main()