#!/usr/bin/env python3
"""
Corrected Pipeline Integration - Metadata-Driven Architecture
Properly integrates with existing conda environments and INTEGRATED_PIPELINE structure
Based on methodology.md, projectplan.md, and technical_stack.md analysis
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager


class CorrectedPipelineIntegration:
    """Corrected pipeline integration using existing conda environments and proven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize corrected pipeline integration.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to project root.
        """
        if base_dir is None:
            # Use the actual project root (video-synthesis-pipeline copy)
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Set up paths to match existing working structure
        self.integrated_pipeline_dir = self.base_dir / "INTEGRATED_PIPELINE"
        self.src_dir = self.integrated_pipeline_dir / "src"
        self.new_dir = self.base_dir / "NEW"
        
        # Initialize metadata manager for NEW directory (our enhancement)
        self.metadata_manager = EnhancedMetadataManager(str(self.new_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Define conda environments from technical_stack.md
        self.conda_environments = {
            'xtts': 'xtts',  # Voice synthesis environment
            'sadtalker': 'sadtalker',  # Animation environment  
            'enhancement': 'enhancement',  # Enhancement environment
            'manim': 'manim',  # Educational animation environment
            'video_audio_processing': 'video-audio-processing'  # Audio processing environment
        }
        
        # Define working Python paths for each environment
        self.env_python_paths = {}
        for env_name, conda_name in self.conda_environments.items():
            # Check for both miniforge and anaconda paths
            possible_paths = [
                Path.home() / "miniforge3" / "envs" / conda_name / "bin" / "python",
                Path.home() / "anaconda3" / "envs" / conda_name / "bin" / "python",
                Path.home() / ".conda" / "envs" / conda_name / "bin" / "python"
            ]
            
            for py_path in possible_paths:
                if py_path.exists():
                    self.env_python_paths[env_name] = str(py_path)
                    break
            
            if env_name not in self.env_python_paths:
                self.logger.warning(f"Conda environment '{conda_name}' not found")
        
        # Define stage script paths in INTEGRATED_PIPELINE/src
        self.stage_scripts = {
            'metadata_reader': self.src_dir / "metadata_reader.py",
            'script_processor': self.src_dir / "script_processor.py", 
            'emotion_mapper': self.src_dir / "emotion_mapper.py",
            'xtts_stage': self.src_dir / "xtts_stage.py",
            'insightface_stage': self.src_dir / "insightface_stage.py",
            'sadtalker_stage': self.src_dir / "sadtalker_stage.py",
            'realesrgan_stage': self.src_dir / "realesrgan_stage.py",
            'codeformer_stage': self.src_dir / "codeformer_stage.py",
            'final_assembly_stage': self.src_dir / "final_assembly_stage.py",
            'main_pipeline': self.integrated_pipeline_dir / "main_pipeline.py"  # main_pipeline.py is in root, not src
        }
        
        self.logger.info("Tools Corrected Pipeline Integration initialized")
        self.logger.info(f"   Project root: {self.base_dir}")
        self.logger.info(f"   INTEGRATED_PIPELINE: {self.integrated_pipeline_dir}")
        self.logger.info(f"   Available conda environments: {list(self.env_python_paths.keys())}")
    
    def validate_working_environment(self) -> Dict[str, Any]:
        """Validate that the existing working environment is properly set up.
        
        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'environment_status': {}
        }
        
        # Check INTEGRATED_PIPELINE directory structure
        if not self.integrated_pipeline_dir.exists():
            validation['valid'] = False
            validation['errors'].append(f"INTEGRATED_PIPELINE directory not found: {self.integrated_pipeline_dir}")
        else:
            validation['environment_status']['integrated_pipeline'] = True
        
        if not self.src_dir.exists():
            validation['valid'] = False
            validation['errors'].append(f"INTEGRATED_PIPELINE/src directory not found: {self.src_dir}")
        else:
            validation['environment_status']['src_directory'] = True
        
        # Check conda environments
        for env_name, conda_name in self.conda_environments.items():
            if env_name in self.env_python_paths:
                validation['environment_status'][f'conda_{env_name}'] = True
            else:
                validation['warnings'].append(f"Conda environment '{conda_name}' not found")
                validation['environment_status'][f'conda_{env_name}'] = False
        
        # Check stage scripts
        missing_scripts = []
        for script_name, script_path in self.stage_scripts.items():
            if script_path.exists():
                validation['environment_status'][f'script_{script_name}'] = True
            else:
                missing_scripts.append(str(script_path))
                validation['environment_status'][f'script_{script_name}'] = False
        
        if missing_scripts:
            validation['warnings'].append(f"Stage scripts not found: {missing_scripts}")
        
        return validation
    
    def run_stage_in_conda_env(self, stage_name: str, env_name: str, 
                              script_args: List[str] = None) -> Dict[str, Any]:
        """Run a stage script in its designated conda environment.
        
        Args:
            stage_name: Name of the stage script to run
            env_name: Name of the conda environment
            script_args: Additional arguments for the script
            
        Returns:
            Execution results
        """
        try:
            if stage_name not in self.stage_scripts:
                return {'success': False, 'error': f'Unknown stage: {stage_name}'}
            
            if env_name not in self.env_python_paths:
                return {'success': False, 'error': f'Conda environment not available: {env_name}'}
            
            script_path = self.stage_scripts[stage_name]
            if not script_path.exists():
                return {'success': False, 'error': f'Stage script not found: {script_path}'}
            
            python_path = self.env_python_paths[env_name]
            
            # Build command
            cmd = [python_path, str(script_path)]
            if script_args:
                cmd.extend(script_args)
            
            self.logger.info(f"STARTING Running {stage_name} in {env_name} environment")
            self.logger.info(f"   Command: {' '.join(cmd)}")
            
            # Execute in conda environment
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(self.integrated_pipeline_dir)
            )
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] {stage_name} completed successfully in {processing_time:.1f}s")
                return {
                    'success': True,
                    'stage': stage_name,
                    'environment': env_name,
                    'processing_time': processing_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                self.logger.error(f"[ERROR] {stage_name} failed with return code {result.returncode}")
                return {
                    'success': False,
                    'stage': stage_name,
                    'environment': env_name,
                    'processing_time': processing_time,
                    'error': f"Return code {result.returncode}",
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stage': stage_name,
                'error': 'Process timeout (30 minutes)'
            }
        except Exception as e:
            return {
                'success': False,
                'stage': stage_name,
                'error': str(e)
            }
    
    def process_metadata_reading(self) -> Dict[str, Any]:
        """Process metadata reading using the existing metadata_reader.py."""
        # Create metadata file path from NEW directory for INTEGRATED_PIPELINE to read
        new_metadata_path = self.new_dir / "metadata" / "latest_metadata.json"
        
        script_args = []
        if new_metadata_path.exists():
            script_args = ['--metadata-path', str(new_metadata_path)]
        
        return self.run_stage_in_conda_env('metadata_reader', 'xtts', script_args)
    
    def process_script_processing(self) -> Dict[str, Any]:
        """Process script processing using the existing script_processor.py."""
        return self.run_stage_in_conda_env('script_processor', 'xtts')
    
    def process_xtts_voice_cloning(self, emotion: str = 'confident', tone: str = 'professional') -> Dict[str, Any]:
        """Process XTTS voice cloning using the existing xtts_stage.py."""
        script_args = ['--emotion', emotion, '--tone', tone]
        return self.run_stage_in_conda_env('xtts_stage', 'xtts', script_args)
    
    def process_insightface_detection(self, emotion: str = 'confident') -> Dict[str, Any]:
        """Process InsightFace face detection using the existing insightface_stage.py."""
        script_args = ['--emotion', emotion]
        return self.run_stage_in_conda_env('insightface_stage', 'sadtalker', script_args)
    
    def process_sadtalker_animation(self, emotion: str = 'confident', tone: str = 'professional') -> Dict[str, Any]:
        """Process SadTalker animation using the existing sadtalker_stage.py."""
        script_args = ['--emotion', emotion, '--tone', tone]
        return self.run_stage_in_conda_env('sadtalker_stage', 'sadtalker', script_args)
    
    def process_realesrgan_enhancement(self, tone: str = 'professional') -> Dict[str, Any]:
        """Process Real-ESRGAN enhancement using the existing realesrgan_stage.py."""
        script_args = ['--tone', tone]
        return self.run_stage_in_conda_env('realesrgan_stage', 'enhancement', script_args)
    
    def process_codeformer_enhancement(self, emotion: str = 'confident') -> Dict[str, Any]:
        """Process CodeFormer enhancement using the existing codeformer_stage.py."""
        script_args = ['--emotion', emotion]
        return self.run_stage_in_conda_env('codeformer_stage', 'enhancement', script_args)
    
    def process_final_assembly(self) -> Dict[str, Any]:
        """Process final assembly using the existing final_assembly_stage.py."""
        return self.run_stage_in_conda_env('final_assembly_stage', 'video_audio_processing')
    
    def process_complete_pipeline(self, emotion: str = 'confident', tone: str = 'professional') -> Dict[str, Any]:
        """Process the complete pipeline using existing main_pipeline.py with emotion/tone parameters.
        
        Args:
            emotion: Emotion parameter (inspired, confident, curious, excited, calm)
            tone: Tone parameter (professional, friendly, motivational, casual)
            
        Returns:
            Complete pipeline results
        """
        try:
            # Update NEW metadata with processing start
            start_time = time.time()
            
            # Pass emotion and tone parameters to main pipeline
            script_args = ['--emotion', emotion, '--tone', tone]
            
            # Add metadata path for integration
            new_metadata_path = self.new_dir / "metadata" / "latest_metadata.json"
            if new_metadata_path.exists():
                script_args.extend(['--metadata-path', str(new_metadata_path)])
            
            self.logger.info(f"VIDEO PIPELINE Starting complete pipeline with emotion: {emotion}, tone: {tone}")
            
            # Try to run with available conda environments
            # First preference: xtts environment (recommended)
            result = None
            environments_to_try = ['xtts', 'video_audio_processing', 'sadtalker']
            
            for env_name in environments_to_try:
                if env_name in self.env_python_paths:
                    self.logger.info(f"   Attempting to run main_pipeline in {env_name} environment")
                    result = self.run_stage_in_conda_env('main_pipeline', env_name, script_args)
                    break
            
            if result is None:
                # If no conda environments available, try with system Python
                self.logger.warning("No conda environments available, attempting with system Python")
                result = self._run_with_system_python('main_pipeline', script_args)
            
            total_time = time.time() - start_time
            
            if result['success']:
                # Try to update NEW metadata with results
                try:
                    processing_data = {
                        'complete_pipeline_processing_time': total_time,
                        'emotion': emotion,
                        'tone': tone,
                        'integrated_pipeline_result': result,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    # Update all stages to completed in NEW metadata (since main pipeline handles everything)
                    for stage_name in ['voice_cloning', 'face_processing', 'video_generation', 
                                     'video_enhancement', 'final_assembly']:
                        self.metadata_manager.update_stage_status(stage_name, 'completed', processing_data)
                    
                    self.logger.info(f"[SUCCESS] Complete pipeline processed successfully in {total_time:.1f}s")
                    self.logger.info(f"   INTEGRATED_PIPELINE result: {result.get('stdout', '')[:200]}...")
                    
                except Exception as metadata_error:
                    self.logger.warning(f"Could not update NEW metadata: {metadata_error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[ERROR] Complete pipeline processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_with_system_python(self, stage_name: str, script_args: List[str] = None) -> Dict[str, Any]:
        """Run a stage script with system Python as fallback.
        
        Args:
            stage_name: Name of the stage script to run
            script_args: Additional arguments for the script
            
        Returns:
            Execution results
        """
        try:
            if stage_name not in self.stage_scripts:
                return {'success': False, 'error': f'Unknown stage: {stage_name}'}
            
            script_path = self.stage_scripts[stage_name]
            if not script_path.exists():
                return {'success': False, 'error': f'Stage script not found: {script_path}'}
            
            # Build command with system Python
            cmd = [sys.executable, str(script_path)]
            if script_args:
                cmd.extend(script_args)
            
            self.logger.info(f"STARTING Running {stage_name} with system Python")
            self.logger.info(f"   Command: {' '.join(cmd)}")
            
            # Execute with system Python
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(self.integrated_pipeline_dir)
            )
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] {stage_name} completed successfully in {processing_time:.1f}s")
                return {
                    'success': True,
                    'stage': stage_name,
                    'environment': 'system_python',
                    'processing_time': processing_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                self.logger.error(f"[ERROR] {stage_name} failed with return code {result.returncode}")
                return {
                    'success': False,
                    'stage': stage_name,
                    'environment': 'system_python',
                    'processing_time': processing_time,
                    'error': f"Return code {result.returncode}",
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stage': stage_name,
                'error': 'Process timeout (30 minutes)'
            }
        except Exception as e:
            return {
                'success': False,
                'stage': stage_name,
                'error': str(e)
            }
    
    def sync_metadata_to_integrated_pipeline(self) -> Dict[str, Any]:
        """Sync NEW metadata to format expected by INTEGRATED_PIPELINE."""
        try:
            # Load NEW metadata
            new_metadata = self.metadata_manager.load_metadata()
            if not new_metadata:
                return {'success': False, 'error': 'No NEW metadata found'}
            
            # Convert to INTEGRATED_PIPELINE format
            integrated_metadata = {
                'title': new_metadata.get('user_inputs', {}).get('title', ''),
                'topic': new_metadata.get('user_inputs', {}).get('topic', ''),
                'audience': new_metadata.get('user_inputs', {}).get('audience', ''),
                'tone': new_metadata.get('user_inputs', {}).get('tone', 'professional'),
                'emotion': new_metadata.get('user_inputs', {}).get('emotion', 'confident'),
                'content_type': new_metadata.get('user_inputs', {}).get('content_type', ''),
                'additional_context': new_metadata.get('user_inputs', {}).get('additional_context', ''),
                'script_generated': new_metadata.get('generated_content', {}),
                'face_image_path': new_metadata.get('user_assets', {}).get('face_image', ''),
                'voice_sample_path': new_metadata.get('user_assets', {}).get('voice_sample', ''),
                'session_id': new_metadata.get('session_id', ''),
                'created_at': new_metadata.get('created_at', ''),
                'new_metadata_source': True  # Flag to indicate this came from NEW system
            }
            
            # Save to INTEGRATED_PIPELINE expected location
            integrated_metadata_path = self.integrated_pipeline_dir / "output" / "metadata" / "latest_metadata.json"
            integrated_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(integrated_metadata_path, 'w') as f:
                json.dump(integrated_metadata, f, indent=2)
            
            self.logger.info(f"[SUCCESS] Metadata synced to INTEGRATED_PIPELINE: {integrated_metadata_path}")
            
            return {
                'success': True,
                'integrated_metadata_path': str(integrated_metadata_path),
                'synced_fields': list(integrated_metadata.keys())
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_integrated_pipeline_outputs(self) -> Dict[str, Any]:
        """Get outputs from INTEGRATED_PIPELINE processing."""
        try:
            output_dir = self.integrated_pipeline_dir / "output"
            outputs = {}
            
            # Check for common output directories
            output_subdirs = ['xtts', 'insightface', 'sadtalker', 'realesrgan', 'codeformer', 'final']
            
            for subdir in output_subdirs:
                subdir_path = output_dir / subdir
                if subdir_path.exists():
                    # Get most recent files in each directory
                    files = list(subdir_path.glob('*'))
                    if files:
                        # Sort by modification time, get most recent
                        latest_file = max(files, key=lambda f: f.stat().st_mtime)
                        outputs[subdir] = {
                            'latest_file': str(latest_file),
                            'file_size': latest_file.stat().st_size,
                            'modified_time': datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
                        }
            
            return {
                'success': True,
                'output_directory': str(output_dir),
                'outputs': outputs
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


def main():
    """Test the corrected pipeline integration."""
    print("🧪 Testing Corrected Pipeline Integration")
    print("=" * 50)
    
    # Initialize corrected integration
    integration = CorrectedPipelineIntegration()
    
    # Validate working environment
    validation = integration.validate_working_environment()
    print(f"Search Environment Validation:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Errors: {len(validation['errors'])}")
    print(f"   Warnings: {len(validation['warnings'])}")
    
    if validation['errors']:
        print(f"[ERROR] Errors found:")
        for error in validation['errors']:
            print(f"   - {error}")
    
    if validation['warnings']:
        print(f"[WARNING] Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    # Show environment status
    print(f"\nStatus: Environment Status:")
    for component, status in validation['environment_status'].items():
        status_icon = "[SUCCESS]" if status else "[ERROR]"
        print(f"   {status_icon} {component}")
    
    # Test metadata sync
    print(f"\n[EMOJI] Testing metadata sync...")
    sync_result = integration.sync_metadata_to_integrated_pipeline()
    if sync_result['success']:
        print(f"[SUCCESS] Metadata sync successful")
        print(f"   Synced to: {sync_result['integrated_metadata_path']}")
        print(f"   Fields: {len(sync_result['synced_fields'])}")
    else:
        print(f"[ERROR] Metadata sync failed: {sync_result['error']}")
    
    # Check outputs
    print(f"\nAssets: Checking INTEGRATED_PIPELINE outputs...")
    outputs = integration.get_integrated_pipeline_outputs()
    if outputs['success']:
        print(f"[SUCCESS] Output directory: {outputs['output_directory']}")
        if outputs['outputs']:
            for subdir, info in outputs['outputs'].items():
                file_size_mb = info['file_size'] / (1024 * 1024)
                print(f"   File: {subdir}: {Path(info['latest_file']).name} ({file_size_mb:.1f} MB)")
        else:
            print(f"   No outputs found")
    else:
        print(f"[ERROR] Could not check outputs: {outputs['error']}")
    
    print(f"\nSUCCESS Corrected Pipeline Integration testing completed!")


if __name__ == "__main__":
    main()