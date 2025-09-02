#!/usr/bin/env python3
"""
Enhanced Metadata Manager - Integration Layer
Bridge between CentralizedMetadataSystem and existing video_pipeline components
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from centralized_metadata_system import CentralizedMetadataSystem


class EnhancedMetadataManager:
    """Enhanced metadata manager that integrates with the centralized system."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the enhanced metadata manager.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize centralized metadata system
        self.cms = CentralizedMetadataSystem(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Current session ID
        self.current_session_id = None
        
        self.logger.info("STARTING Enhanced Metadata Manager initialized")
    
    def create_session(self, user_inputs: Dict[str, Any]) -> str:
        """Create a new session with user inputs.
        
        Args:
            user_inputs: Dictionary containing user inputs
            
        Returns:
            Session ID for the new session
        """
        session_id = self.cms.create_new_session(user_inputs)
        if session_id:
            self.current_session_id = session_id
            self.logger.info(f"[SUCCESS] New session created: {session_id}")
        return session_id
    
    def load_metadata(self, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Load metadata for a session.
        
        Args:
            session_id: Session ID to load. If None, loads current session.
            
        Returns:
            Metadata dictionary or None if not found
        """
        if session_id is None:
            session_id = self.current_session_id
        
        metadata = self.cms.load_metadata(session_id)
        if metadata:
            self.current_session_id = metadata.get("session_id")
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], create_backup: bool = True) -> bool:
        """Save metadata (compatibility method).
        
        Args:
            metadata: Metadata dictionary to save
            create_backup: Whether to create backup
            
        Returns:
            True if saved successfully, False otherwise
        """
        return self.cms._save_metadata(metadata, create_backup)
    
    def update_user_assets(self, face_image: str = None, voice_sample: str = None, 
                          document: str = None) -> bool:
        """Update user assets in metadata.
        
        Args:
            face_image: Path to face image file
            voice_sample: Path to voice sample file
            document: Path to document file
            
        Returns:
            True if updated successfully, False otherwise
        """
        return self.cms.update_user_assets(
            face_image_path=face_image,
            voice_sample_path=voice_sample,
            document_path=document
        )
    
    def update_generated_content(self, clean_script: str = None, 
                               manim_script: str = None,
                               thumbnail_prompts: List[Dict] = None) -> bool:
        """Update generated content in metadata.
        
        Args:
            clean_script: Clean script text for voice cloning
            manim_script: Manim Python script code
            thumbnail_prompts: List of thumbnail prompt dictionaries
            
        Returns:
            True if updated successfully, False otherwise
        """
        return self.cms.update_generated_content(
            clean_script=clean_script,
            manim_script=manim_script,
            thumbnail_prompts=thumbnail_prompts
        )
    
    def update_stage_status(self, stage_name: str, status: str, 
                           update_data: Dict[str, Any] = None) -> bool:
        """Update pipeline stage status (compatibility method).
        
        Args:
            stage_name: Name of the pipeline stage
            status: New status (pending, processing, completed, failed)
            update_data: Additional data to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Extract paths and data from update_data
        input_paths = {}
        output_paths = {}
        processing_data = {}
        error_info = {}
        
        if update_data:
            for key, value in update_data.items():
                if key.startswith("input_"):
                    input_paths[key] = value
                elif key.startswith("output_"):
                    output_paths[key] = value
                elif key == "error":
                    error_info["error"] = value
                else:
                    processing_data[key] = value
        
        return self.cms.update_stage_status(
            stage_name=stage_name,
            status=status,
            input_paths=input_paths if input_paths else None,
            output_paths=output_paths if output_paths else None,
            processing_data=processing_data if processing_data else None,
            error_info=error_info if error_info else None
        )
    
    def get_stage_status(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get pipeline stage status and data.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Stage data dictionary or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        return metadata.get("pipeline_stages", {}).get(stage_name)
    
    def get_stage_input(self, stage_name: str, input_key: str) -> Optional[str]:
        """Get input path for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            input_key: Key for the input
            
        Returns:
            Absolute path to input file or None if not found
        """
        return self.cms.get_stage_input_path(stage_name, input_key)
    
    def get_stage_output(self, stage_name: str, output_key: str) -> Optional[str]:
        """Get output path for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            output_key: Key for the output
            
        Returns:
            Absolute path to output file or None if not found
        """
        return self.cms.get_stage_output_path(stage_name, output_key)
    
    def set_stage_input(self, stage_name: str, input_key: str, input_path: str) -> bool:
        """Set input path for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            input_key: Key for the input
            input_path: Path to input file
            
        Returns:
            True if set successfully, False otherwise
        """
        return self.cms.update_stage_status(
            stage_name=stage_name,
            status=None,  # Don't change status
            input_paths={input_key: input_path}
        )
    
    def set_stage_output(self, stage_name: str, output_key: str, output_path: str) -> bool:
        """Set output path for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            output_key: Key for the output
            output_path: Path to output file
            
        Returns:
            True if set successfully, False otherwise
        """
        return self.cms.update_stage_status(
            stage_name=stage_name,
            status=None,  # Don't change status
            output_paths={output_key: output_path}
        )
    
    def get_next_stage(self) -> Optional[str]:
        """Get the next pipeline stage to execute.
        
        Returns:
            Name of next stage or None if all completed
        """
        return self.cms.get_next_stage()
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get overall pipeline progress.
        
        Returns:
            Dictionary with progress information
        """
        return self.cms.get_pipeline_progress()
    
    def get_final_output(self) -> Optional[str]:
        """Get path to final video output.
        
        Returns:
            Absolute path to final video or None if not available
        """
        return self.cms.get_stage_output_path("final_assembly", "final_video")
    
    def cleanup_intermediate_files(self, keep_final: bool = True) -> bool:
        """Clean up intermediate processing files.
        
        Args:
            keep_final: Whether to keep the final output video
            
        Returns:
            True if cleaned successfully, False otherwise
        """
        return self.cms.cleanup_session(keep_final=keep_final)
    
    def validate_prerequisites(self) -> Dict[str, Any]:
        """Validate that all prerequisites are met for pipeline execution.
        
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Load metadata
        metadata = self.load_metadata()
        if metadata is None:
            validation_results["valid"] = False
            validation_results["errors"].append("No metadata found. Please generate script first.")
            return validation_results
        
        # Check script generation
        generated_content = metadata.get("generated_content", {})
        if not generated_content.get("clean_script"):
            validation_results["valid"] = False
            validation_results["errors"].append("No clean script found in generated content")
        
        # Check user assets
        user_assets = metadata.get("user_assets", {})
        
        face_image = user_assets.get("face_image")
        if not face_image:
            validation_results["valid"] = False
            validation_results["errors"].append("No face image provided. Please add face image.")
        elif not (self.base_dir / face_image).exists():
            validation_results["valid"] = False
            validation_results["errors"].append(f"Face image not found: {face_image}")
        
        voice_sample = user_assets.get("voice_sample")
        if not voice_sample:
            validation_results["valid"] = False
            validation_results["errors"].append("No voice sample provided. Please add voice sample.")
        elif not (self.base_dir / voice_sample).exists():
            validation_results["valid"] = False
            validation_results["errors"].append(f"Voice sample not found: {voice_sample}")
        
        # Check user inputs
        user_inputs = metadata.get("user_inputs", {})
        tone = user_inputs.get("tone")
        emotion = user_inputs.get("emotion")
        
        if not tone:
            validation_results["warnings"].append("No tone specified, using 'professional'")
        
        if not emotion:
            validation_results["warnings"].append("No emotion specified, using 'inspired'")
        
        # Path validation
        path_validation = self.cms.validate_paths()
        if not path_validation["valid"]:
            validation_results["valid"] = False
            validation_results["errors"].extend(path_validation["missing_files"])
        
        if path_validation["warnings"]:
            validation_results["warnings"].extend(path_validation["warnings"])
        
        return validation_results
    
    def get_user_inputs(self) -> Optional[Dict[str, Any]]:
        """Get user inputs from metadata.
        
        Returns:
            User inputs dictionary or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        return metadata.get("user_inputs", {})
    
    def get_user_assets(self) -> Optional[Dict[str, Any]]:
        """Get user assets from metadata.
        
        Returns:
            User assets dictionary or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        return metadata.get("user_assets", {})
    
    def get_generated_content(self) -> Optional[Dict[str, Any]]:
        """Get generated content from metadata.
        
        Returns:
            Generated content dictionary or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        return metadata.get("generated_content", {})
    
    def get_pipeline_config(self) -> Optional[Dict[str, Any]]:
        """Get pipeline configuration from metadata.
        
        Returns:
            Pipeline configuration dictionary or None if not found
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        return metadata.get("pipeline_config", {})
    
    def update_pipeline_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update pipeline configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        if "pipeline_config" not in metadata:
            metadata["pipeline_config"] = {}
        
        metadata["pipeline_config"].update(config_updates)
        return self.save_metadata(metadata)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information.
        
        Returns:
            Dictionary with session information
        """
        metadata = self.load_metadata()
        if metadata is None:
            return {"session_id": None, "created_at": None, "last_modified": None}
        
        return {
            "session_id": metadata.get("session_id"),
            "created_at": metadata.get("created_at"),
            "last_modified": metadata.get("last_modified"),
            "version": metadata.get("version", "1.0")
        }
    
    def export_session_data(self, output_path: str) -> bool:
        """Export session data to a file.
        
        Args:
            output_path: Path to export the session data
            
        Returns:
            True if exported successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        try:
            import json
            output_path = Path(output_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"[SUCCESS] Session data exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error exporting session data: {e}")
            return False
    
    def import_session_data(self, input_path: str) -> bool:
        """Import session data from a file.
        
        Args:
            input_path: Path to import the session data from
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            import json
            input_path = Path(input_path)
            
            if not input_path.exists():
                self.logger.error(f"[ERROR] Import file not found: {input_path}")
                return False
            
            with open(input_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if self.save_metadata(metadata):
                self.current_session_id = metadata.get("session_id")
                self.logger.info(f"[SUCCESS] Session data imported from: {input_path}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error importing session data: {e}")
            return False
    
    # Compatibility methods for existing code
    def get_latest_metadata_file(self) -> Optional[Path]:
        """Get latest metadata file path (compatibility method)."""
        return self.cms.metadata_dir / "latest_metadata.json"
    
    def update_metadata(self, update_data: Dict[str, Any], create_backup: bool = True) -> bool:
        """Update metadata with new data (compatibility method).
        
        Args:
            update_data: Dictionary of data to update
            create_backup: Whether to create backup
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        metadata.update(update_data)
        return self.save_metadata(metadata, create_backup)


def main():
    """Test the enhanced metadata manager."""
    manager = EnhancedMetadataManager()
    
    # Test user inputs
    test_user_inputs = {
        "title": "Enhanced Test Video",
        "topic": "Advanced Machine Learning",
        "audience": "senior engineers",
        "tone": "professional",
        "emotion": "confident",
        "content_type": "Full Training Module"
    }
    
    print("🧪 Testing Enhanced Metadata Manager")
    print("=" * 50)
    
    # Test session creation
    session_id = manager.create_session(test_user_inputs)
    if session_id:
        print(f"[SUCCESS] Session created: {session_id}")
    else:
        print("[ERROR] Failed to create session")
        return
    
    # Test metadata operations
    metadata = manager.load_metadata()
    if metadata:
        print(f"[SUCCESS] Metadata loaded: {metadata['user_inputs']['title']}")
    else:
        print("[ERROR] Failed to load metadata")
        return
    
    # Test user asset update
    success = manager.update_user_assets(
        face_image="user_assets/faces/test_face.jpg",
        voice_sample="user_assets/voices/test_voice.wav"
    )
    print(f"[SUCCESS] User assets updated: {'Success' if success else 'Failed'}")
    
    # Test generated content update
    success = manager.update_generated_content(
        clean_script="This is a test clean script for voice cloning.",
        manim_script="# Manim script for background animation"
    )
    print(f"[SUCCESS] Generated content updated: {'Success' if success else 'Failed'}")
    
    # Test stage status update
    success = manager.update_stage_status(
        "voice_cloning", 
        "processing",
        {
            "input_audio": "user_assets/voices/test_voice.wav",
            "chunks": ["chunk1.wav", "chunk2.wav"]
        }
    )
    print(f"[SUCCESS] Stage status updated: {'Success' if success else 'Failed'}")
    
    # Test pipeline progress
    progress = manager.get_pipeline_progress()
    print(f"[SUCCESS] Pipeline progress: {progress['progress_percent']:.1f}%")
    
    # Test next stage
    next_stage = manager.get_next_stage()
    print(f"[SUCCESS] Next stage: {next_stage}")
    
    # Test prerequisites validation
    validation = manager.validate_prerequisites()
    print(f"[SUCCESS] Prerequisites validation: {'Valid' if validation['valid'] else 'Invalid'}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")
    
    # Test session info
    session_info = manager.get_session_info()
    print(f"[SUCCESS] Session info: {session_info['session_id']}")
    
    print("\nSUCCESS All enhanced metadata manager tests completed!")


if __name__ == "__main__":
    main()