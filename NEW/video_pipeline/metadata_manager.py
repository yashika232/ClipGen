#!/usr/bin/env python3
"""
Centralized Metadata Manager for NEW Video Pipeline
Manages metadata operations across all video synthesis stages
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging


class MetadataManager:
    """Centralized metadata management for NEW video pipeline."""
    
    def __init__(self, new_dir: str = None):
        """Initialize metadata manager.
        
        Args:
            new_dir: Path to NEW directory. Defaults to parent of script location.
        """
        if new_dir is None:
            self.new_dir = Path(__file__).parent.parent
        else:
            self.new_dir = Path(new_dir)
        
        self.metadata_dir = self.new_dir / "metadata"
        self.assets_dir = self.new_dir / "user_assets" 
        self.output_dir = self.new_dir / "video_outputs"
        
        # Create directories if they don't exist
        self.assets_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "voice_cloning").mkdir(exist_ok=True)
        (self.output_dir / "face_processing").mkdir(exist_ok=True)
        (self.output_dir / "video_generation").mkdir(exist_ok=True)
        (self.output_dir / "video_enhancement").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_latest_metadata_file(self) -> Optional[Path]:
        """Find the latest metadata file."""
        if not self.metadata_dir.exists():
            self.logger.error(f"Metadata directory not found: {self.metadata_dir}")
            return None
        
        # Look for latest_metadata.json first
        latest_file = self.metadata_dir / "latest_metadata.json"
        if latest_file.exists():
            return latest_file
        
        # Fallback to script_ files
        metadata_files = list(self.metadata_dir.glob("script_*.json"))
        if not metadata_files:
            self.logger.error(f"No metadata files found in {self.metadata_dir}")
            return None
        
        # Get the most recent file
        latest_file = max(metadata_files, key=os.path.getmtime)
        self.logger.info(f"Using metadata file: {latest_file.name}")
        return latest_file
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load the latest metadata."""
        metadata_file = self.get_latest_metadata_file()
        if metadata_file is None:
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Ensure pipeline stages exist
            if "pipeline_stages" not in metadata:
                metadata = self._initialize_pipeline_stages(metadata)
                self.save_metadata(metadata, create_backup=False)
            
            self.logger.info(f"Loaded metadata: {metadata.get('title', 'Unknown')}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return None
    
    def _initialize_pipeline_stages(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize pipeline stages in metadata if not present."""
        metadata["pipeline_stages"] = {
            "voice_cloning": {
                "status": "pending",
                "input_audio": metadata.get("user_assets", {}).get("voice_sample"),
                "output_voice": None,
                "chunks": [],
                "processing_time": 0,
                "duration": 0,
                "error": None
            },
            "face_processing": {
                "status": "pending", 
                "input_image": metadata.get("user_assets", {}).get("face_image"),
                "face_crop": None,
                "face_data": {},
                "processing_time": 0,
                "error": None
            },
            "video_generation": {
                "status": "pending",
                "input_script": None,
                "input_voice": None,
                "input_face": None,
                "video_chunks": [],
                "combined_video": None,
                "chunk_count": 0,
                "processing_time": 0,
                "error": None
            },
            "video_enhancement": {
                "status": "pending",
                "enhanced_chunks": [],
                "final_video": None,
                "processing_time": 0,
                "error": None
            }
        }
        
        # Add pipeline configuration
        if "pipeline_config" not in metadata:
            metadata["pipeline_config"] = {
                "auto_chunking": True,
                "chunk_duration": 10,
                "quality_preset": "high", 
                "gpu_acceleration": True,
                "emotion_mapping": True,
                "max_retries": 3
            }
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], create_backup: bool = True) -> bool:
        """Save updated metadata.
        
        Args:
            metadata: Updated metadata dictionary
            create_backup: Whether to create backup of original
            
        Returns:
            True if saved successfully, False otherwise
        """
        metadata_file = self.get_latest_metadata_file()
        if metadata_file is None:
            return False
        
        try:
            # Create backup if requested
            if create_backup and metadata_file.exists():
                backup_file = metadata_file.with_suffix(f".backup_{int(datetime.now().timestamp())}.json")
                shutil.copy2(metadata_file, backup_file)
                self.logger.info(f"Created backup: {backup_file.name}")
            
            # Update last modified timestamp
            metadata["last_modified"] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metadata updated: {metadata_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            return False
    
    def update_stage_status(self, stage_name: str, status: str, 
                           update_data: Dict[str, Any] = None) -> bool:
        """Update pipeline stage status and data.
        
        Args:
            stage_name: Name of the pipeline stage
            status: New status (pending, processing, completed, failed)
            update_data: Additional data to update in the stage
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        if "pipeline_stages" not in metadata:
            metadata = self._initialize_pipeline_stages(metadata)
        
        if stage_name not in metadata["pipeline_stages"]:
            self.logger.error(f"Unknown pipeline stage: {stage_name}")
            return False
        
        # Update status
        metadata["pipeline_stages"][stage_name]["status"] = status
        metadata["pipeline_stages"][stage_name]["last_updated"] = datetime.now().isoformat()
        
        # Update additional data if provided
        if update_data:
            metadata["pipeline_stages"][stage_name].update(update_data)
        
        return self.save_metadata(metadata)
    
    def update_metadata(self, update_data: Dict[str, Any], create_backup: bool = True) -> bool:
        """Update metadata with new data.
        
        Args:
            update_data: Dictionary of data to update in metadata
            create_backup: Whether to create backup before updating
            
        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.load_metadata()
        if metadata is None:
            return False
        
        # Update metadata with new data
        metadata.update(update_data)
        
        return self.save_metadata(metadata, create_backup)
    
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
            input_key: Key for the input (e.g., 'input_audio', 'input_image')
            
        Returns:
            Absolute path to input file or None if not found
        """
        stage_data = self.get_stage_status(stage_name)
        if stage_data is None:
            return None
        
        input_path = stage_data.get(input_key)
        if input_path is None:
            return None
        
        # Convert to absolute path
        if not Path(input_path).is_absolute():
            input_path = self.new_dir / input_path
        
        return str(input_path) if Path(input_path).exists() else None
    
    def set_stage_output(self, stage_name: str, output_key: str, output_path: str) -> bool:
        """Set output path for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            output_key: Key for the output (e.g., 'output_voice', 'face_crop')
            output_path: Path to output file (absolute or relative to NEW dir)
            
        Returns:
            True if set successfully, False otherwise
        """
        # Convert to relative path from NEW directory
        output_path = Path(output_path)
        if output_path.is_absolute():
            try:
                output_path = output_path.relative_to(self.new_dir)
            except ValueError:
                # Path is outside NEW directory, keep absolute
                pass
        
        return self.update_stage_status(stage_name, None, {output_key: str(output_path)})
    
    def get_next_stage(self) -> Optional[str]:
        """Get the next pipeline stage to execute.
        
        Returns:
            Name of next stage or None if all completed
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        stages = metadata.get("pipeline_stages", {})
        stage_order = ["voice_cloning", "face_processing", "video_generation", "video_enhancement"]
        
        for stage_name in stage_order:
            stage = stages.get(stage_name, {})
            status = stage.get("status", "pending")
            
            if status in ["pending", "failed"]:
                # Check if inputs are available
                if self._stage_inputs_ready(stage_name, metadata):
                    return stage_name
        
        return None
    
    def _stage_inputs_ready(self, stage_name: str, metadata: Dict[str, Any]) -> bool:
        """Check if inputs are ready for a stage.
        
        Args:
            stage_name: Name of the pipeline stage
            metadata: Full metadata dictionary
            
        Returns:
            True if inputs are ready, False otherwise
        """
        stages = metadata.get("pipeline_stages", {})
        
        if stage_name == "voice_cloning":
            # Needs user voice sample
            return metadata.get("user_assets", {}).get("voice_sample") is not None
        
        elif stage_name == "face_processing":
            # Needs user face image
            return metadata.get("user_assets", {}).get("face_image") is not None
        
        elif stage_name == "video_generation":
            # Needs script, voice, and face
            script_ready = metadata.get("script_generated", {}).get("core_content") is not None
            voice_ready = stages.get("voice_cloning", {}).get("status") == "completed"
            face_ready = stages.get("face_processing", {}).get("status") == "completed"
            return script_ready and voice_ready and face_ready
        
        elif stage_name == "video_enhancement":
            # Needs completed video generation
            return stages.get("video_generation", {}).get("status") == "completed"
        
        return False
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get overall pipeline progress.
        
        Returns:
            Dictionary with progress information
        """
        metadata = self.load_metadata()
        if metadata is None:
            return {"progress": 0.0, "current_stage": None, "status": "error"}
        
        stages = metadata.get("pipeline_stages", {})
        stage_names = ["voice_cloning", "face_processing", "video_generation", "video_enhancement"]
        
        completed_stages = 0
        current_stage = None
        failed_stages = []
        
        for stage_name in stage_names:
            stage = stages.get(stage_name, {})
            status = stage.get("status", "pending")
            
            if status == "completed":
                completed_stages += 1
            elif status == "processing":
                current_stage = stage_name
            elif status == "failed":
                failed_stages.append(stage_name)
            elif status == "pending" and current_stage is None:
                current_stage = stage_name
        
        progress = completed_stages / len(stage_names)
        
        return {
            "progress": progress,
            "current_stage": current_stage,
            "completed_stages": completed_stages,
            "total_stages": len(stage_names),
            "failed_stages": failed_stages,
            "status": "failed" if failed_stages else ("completed" if progress == 1.0 else "processing")
        }
    
    def get_final_output(self) -> Optional[str]:
        """Get path to final video output.
        
        Returns:
            Absolute path to final video or None if not available
        """
        stage_data = self.get_stage_status("video_enhancement")
        if stage_data is None or stage_data.get("status") != "completed":
            return None
        
        final_video = stage_data.get("final_video")
        if final_video is None:
            return None
        
        # Convert to absolute path
        if not Path(final_video).is_absolute():
            final_video = self.new_dir / final_video
        
        return str(final_video) if Path(final_video).exists() else None
    
    def cleanup_intermediate_files(self, keep_final: bool = True) -> None:
        """Clean up intermediate processing files.
        
        Args:
            keep_final: Whether to keep the final output video
        """
        self.logger.info("Cleaning up intermediate files...")
        
        # Define directories to clean
        cleanup_dirs = [
            self.output_dir / "voice_cloning",
            self.output_dir / "face_processing", 
            self.output_dir / "video_generation"
        ]
        
        if not keep_final:
            cleanup_dirs.append(self.output_dir / "video_enhancement")
        
        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                try:
                    shutil.rmtree(cleanup_dir)
                    cleanup_dir.mkdir(exist_ok=True)
                    self.logger.info(f"Cleaned: {cleanup_dir}")
                except Exception as e:
                    self.logger.error(f"Error cleaning {cleanup_dir}: {e}")


def main():
    """Test the metadata manager."""
    manager = MetadataManager()
    
    # Test loading metadata
    metadata = manager.load_metadata()
    if metadata:
        print("[SUCCESS] Metadata loaded successfully")
        
        # Show pipeline progress
        progress = manager.get_pipeline_progress()
        print(f"Pipeline Progress: {progress['progress']:.1%}")
        print(f"Current Stage: {progress['current_stage']}")
        
        # Show next stage
        next_stage = manager.get_next_stage()
        print(f"Next Stage: {next_stage}")
    else:
        print("[ERROR] Failed to load metadata")


if __name__ == "__main__":
    main()