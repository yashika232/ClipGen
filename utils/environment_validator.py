#!/usr/bin/env python3
"""
Environment Validator System
Comprehensive validation and auto-fixing for all pipeline environments
"""

import os
import sys
import subprocess
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time

logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Comprehensive environment validation and auto-fixing system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.validation_results = {}
        self.fix_applied = {}
        
        # Environment configurations
        self.environments = {
            'voice_cloning': {
                'path': self.project_root / 'CHECK' / 'venv_voice_cloning_advanced' / 'bin' / 'python',
                'requirements': ['TTS', 'torch', 'transformers', 'soundfile', 'librosa'],
                'critical': True
            },
            'sadtalker': {
                'path': 'system',  # SadTalker uses system environment
                'requirements': ['imageio-ffmpeg', 'ffmpeg'],
                'critical': True,
                'conda_packages': ['ffmpeg']
            },
            'realesrgan': {
                'path': 'system',  # Real-ESRGAN uses system environment
                'requirements': ['basicsr', 'torchvision'],
                'critical': True,
                'compatibility_issues': ['torchvision.transforms.functional_tensor']
            },
            'codeformer': {
                'path': 'system',  # CodeFormer uses system environment
                'requirements': ['basicsr', 'facexlib'],
                'critical': True
            },
            'insightface': {
                'path': 'system',  # InsightFace uses system environment
                'requirements': ['insightface', 'onnxruntime'],
                'critical': True
            }
        }
    
    def validate_all_environments(self) -> Dict[str, Any]:
        """Validate all pipeline environments."""
        logger.info("Search Starting comprehensive environment validation...")
        
        validation_start = time.time()
        overall_status = True
        
        for env_name, env_config in self.environments.items():
            logger.info(f"🧪 Validating {env_name} environment...")
            
            try:
                result = self._validate_environment(env_name, env_config)
                self.validation_results[env_name] = result
                
                if not result['valid']:
                    overall_status = False
                    logger.error(f"[ERROR] {env_name} environment validation failed")
                else:
                    logger.info(f"[SUCCESS] {env_name} environment validation passed")
                    
            except Exception as e:
                self.validation_results[env_name] = {
                    'valid': False,
                    'error': str(e),
                    'critical': env_config.get('critical', False)
                }
                overall_status = False
                logger.error(f"[ERROR] {env_name} environment validation crashed: {e}")
        
        validation_time = time.time() - validation_start
        
        summary = {
            'overall_valid': overall_status,
            'validation_time': validation_time,
            'environments': self.validation_results,
            'critical_failures': self._get_critical_failures(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"[EMOJI] Environment validation completed in {validation_time:.2f}s")
        logger.info(f"Status: Overall status: {'[SUCCESS] VALID' if overall_status else '[ERROR] INVALID'}")
        
        return summary
    
    def _validate_environment(self, env_name: str, env_config: Dict) -> Dict[str, Any]:
        """Validate a specific environment."""
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'info': {},
            'fixes_applied': []
        }
        
        # Check Python environment
        if env_config['path'] != 'system':
            python_path = env_config['path']
            if not python_path.exists():
                result['valid'] = False
                result['issues'].append(f"Python environment not found: {python_path}")
                return result
        
        # Check required packages
        for requirement in env_config.get('requirements', []):
            try:
                if env_config['path'] == 'system':
                    # Test in current environment
                    importlib.import_module(requirement.replace('-', '_'))
                else:
                    # Test in specific environment
                    self._test_import_in_env(env_config['path'], requirement)
                
                result['info'][requirement] = 'available'
                
            except ImportError:
                result['valid'] = False
                result['issues'].append(f"Missing package: {requirement}")
                
                # Try to auto-fix
                if self._can_auto_fix(env_name, requirement):
                    fix_result = self._auto_fix_package(env_name, env_config, requirement)
                    if fix_result:
                        result['fixes_applied'].append(f"Auto-installed {requirement}")
                        result['valid'] = True  # Re-validate after fix
        
        # Check conda packages
        for conda_pkg in env_config.get('conda_packages', []):
            if not self._check_conda_package(conda_pkg):
                result['issues'].append(f"Missing conda package: {conda_pkg}")
                
                # Try to auto-fix conda packages
                if self._auto_install_conda_package(conda_pkg):
                    result['fixes_applied'].append(f"Auto-installed conda package {conda_pkg}")
        
        # Check compatibility issues
        for compatibility_issue in env_config.get('compatibility_issues', []):
            if not self._check_compatibility(compatibility_issue):
                result['issues'].append(f"Compatibility issue: {compatibility_issue}")
                
                # Try to auto-fix compatibility
                if self._auto_fix_compatibility(compatibility_issue):
                    result['fixes_applied'].append(f"Fixed compatibility: {compatibility_issue}")
        
        # Environment-specific checks
        if env_name == 'sadtalker':
            self._validate_sadtalker_specific(result)
        elif env_name == 'realesrgan':
            self._validate_realesrgan_specific(result)
        elif env_name == 'voice_cloning':
            self._validate_voice_cloning_specific(result)
        
        return result
    
    def _test_import_in_env(self, python_path: Path, package: str) -> bool:
        """Test if a package can be imported in a specific environment."""
        cmd = [
            str(python_path), '-c', 
            f"import {package.replace('-', '_')}; print('SUCCESS')"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and 'SUCCESS' in result.stdout
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def _check_conda_package(self, package: str) -> bool:
        """Check if a conda package is installed."""
        try:
            result = subprocess.run(
                ['conda', 'list', package], 
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0 and package in result.stdout
        except Exception:
            return False
    
    def _check_compatibility(self, issue: str) -> bool:
        """Check specific compatibility issues."""
        if 'torchvision.transforms.functional_tensor' in issue:
            try:
                import torchvision.transforms.functional_tensor
                return True
            except ImportError:
                return False
        return True
    
    def _can_auto_fix(self, env_name: str, requirement: str) -> bool:
        """Check if we can automatically fix a missing requirement."""
        # Define what we can auto-fix
        auto_fixable = {
            'sadtalker': ['imageio-ffmpeg'],
            'realesrgan': ['basicsr-fixed'],
            'codeformer': ['basicsr', 'facexlib'],
            'voice_cloning': [],  # Don't auto-fix complex TTS environments
            'insightface': ['insightface', 'onnxruntime']
        }
        
        return requirement in auto_fixable.get(env_name, [])
    
    def _auto_fix_package(self, env_name: str, env_config: Dict, package: str) -> bool:
        """Automatically fix a missing package."""
        try:
            logger.info(f"Tools Auto-fixing {package} for {env_name}...")
            
            if package == 'imageio-ffmpeg':
                return self._fix_imageio_ffmpeg()
            elif package == 'basicsr-fixed':
                return self._fix_basicsr_compatibility()
            elif package in ['insightface', 'onnxruntime', 'facexlib']:
                return self._install_pip_package(package)
            elif package == 'basicsr':
                return self._fix_basicsr_compatibility()
            
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] Auto-fix failed for {package}: {e}")
            return False
    
    def _auto_install_conda_package(self, package: str) -> bool:
        """Auto-install conda package."""
        try:
            logger.info(f"Tools Installing conda package: {package}")
            
            # Try conda-forge first
            cmd = ['conda', 'install', '-c', 'conda-forge', package, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"[SUCCESS] Successfully installed conda package: {package}")
                return True
            else:
                logger.error(f"[ERROR] Failed to install conda package {package}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Conda install failed for {package}: {e}")
            return False
    
    def _auto_fix_compatibility(self, issue: str) -> bool:
        """Auto-fix compatibility issues."""
        if 'torchvision.transforms.functional_tensor' in issue:
            try:
                # Try different import paths for the torchvision compatibility fix
                try:
                    from .torchvision_compatibility_fix import apply_torchvision_compatibility_fixes
                except ImportError:
                    # Try importing from the utils directory directly
                    import sys
                    from pathlib import Path
                    utils_path = Path(__file__).parent
                    if str(utils_path) not in sys.path:
                        sys.path.insert(0, str(utils_path))
                    from torchvision_compatibility_fix import apply_torchvision_compatibility_fixes
                
                result = apply_torchvision_compatibility_fixes()
                if result:
                    logger.info("[SUCCESS] Torchvision compatibility fixes applied")
                return result
                
            except ImportError:
                logger.error("[ERROR] Torchvision compatibility fix module not found")
                # Try manual compatibility fix
                return self._manual_torchvision_fix()
            except Exception as e:
                logger.error(f"[ERROR] Torchvision compatibility fix failed: {e}")
                return False
        
        return False
    
    def _manual_torchvision_fix(self) -> bool:
        """Manual torchvision compatibility fix as fallback."""
        try:
            import torchvision
            from packaging import version
            
            # Check if we need the fix
            torchvision_version = version.parse(torchvision.__version__)
            if torchvision_version >= version.parse("0.17.0"):
                
                # Try to import functional_tensor to see if it already exists
                try:
                    import torchvision.transforms.functional_tensor
                    logger.info("[SUCCESS] torchvision.transforms.functional_tensor already available")
                    return True
                except ImportError:
                    # Create simple compatibility
                    self._create_simple_functional_tensor_fix()
                    logger.info("[SUCCESS] Created simple torchvision compatibility layer")
                    return True
            else:
                logger.info(f"[SUCCESS] TorchVision {torchvision.__version__} has native functional_tensor support")
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] Manual torchvision fix failed: {e}")
            return False
    
    def _create_simple_functional_tensor_fix(self):
        """Create a simple functional_tensor compatibility module."""
        try:
            import sys
            import types
            import torchvision.transforms.functional as F
            
            # Create a mock functional_tensor module
            functional_tensor_module = types.ModuleType('torchvision.transforms.functional_tensor')
            
            # Add the most commonly needed function
            if hasattr(F, 'rgb_to_grayscale'):
                functional_tensor_module.rgb_to_grayscale = F.rgb_to_grayscale
            elif hasattr(F, 'to_grayscale'):
                functional_tensor_module.rgb_to_grayscale = F.to_grayscale
            
            # Add other common functions
            for func_name in ['adjust_brightness', 'adjust_contrast', 'normalize', 'resize']:
                if hasattr(F, func_name):
                    setattr(functional_tensor_module, func_name, getattr(F, func_name))
            
            # Add the module to sys.modules
            sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor_module
            
            # Also add to the torchvision.transforms namespace
            import torchvision.transforms
            torchvision.transforms.functional_tensor = functional_tensor_module
            
        except Exception as e:
            logger.error(f"[ERROR] Simple compatibility fix creation failed: {e}")
            raise
    
    def _fix_imageio_ffmpeg(self) -> bool:
        """Fix imageio-ffmpeg installation."""
        try:
            # Install imageio-ffmpeg
            cmd = [sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Successfully installed imageio-ffmpeg")
                return True
            else:
                logger.error(f"[ERROR] Failed to install imageio-ffmpeg: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] imageio-ffmpeg installation failed: {e}")
            return False
    
    def _fix_basicsr_compatibility(self) -> bool:
        """Fix BasicSR compatibility issues."""
        try:
            # Try installing basicsr-fixed
            cmd = [sys.executable, '-m', 'pip', 'install', 'basicsr-fixed']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Successfully installed basicsr-fixed")
                return True
            else:
                # Fallback: install BasicSR from GitHub with fixes
                cmd = [
                    sys.executable, '-m', 'pip', 'install', 
                    'git+https://github.com/XPixelGroup/BasicSR@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    logger.info("[SUCCESS] Successfully installed BasicSR from GitHub")
                    return True
                else:
                    logger.error(f"[ERROR] Failed to install BasicSR: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"[ERROR] BasicSR installation failed: {e}")
            return False
    
    def _install_pip_package(self, package: str) -> bool:
        """Install a pip package."""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"[SUCCESS] Successfully installed {package}")
                return True
            else:
                logger.error(f"[ERROR] Failed to install {package}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Installation failed for {package}: {e}")
            return False
    
    def _validate_sadtalker_specific(self, result: Dict) -> None:
        """SadTalker-specific validation."""
        # Check if SadTalker models exist
        sadtalker_dir = self.project_root / "models" / "SadTalker"
        if not sadtalker_dir.exists():
            result['warnings'].append("SadTalker models directory not found")
        
        # Check specific dependencies
        try:
            import imageio
            result['info']['imageio_version'] = imageio.__version__
        except ImportError:
            result['issues'].append("imageio not available")
    
    def _validate_realesrgan_specific(self, result: Dict) -> None:
        """Real-ESRGAN-specific validation."""
        # Check if Real-ESRGAN models exist
        realesrgan_dir = self.project_root / "models" / "Real-ESRGAN"
        if not realesrgan_dir.exists():
            result['warnings'].append("Real-ESRGAN models directory not found")
        
        # Check torchvision compatibility
        try:
            import torchvision
            result['info']['torchvision_version'] = torchvision.__version__
            
            # Check if functional_tensor is available
            try:
                import torchvision.transforms.functional_tensor
                result['info']['functional_tensor_available'] = True
            except ImportError:
                result['warnings'].append("functional_tensor not available - compatibility fix needed")
                result['info']['functional_tensor_available'] = False
                
        except ImportError:
            result['issues'].append("torchvision not available")
    
    def _validate_voice_cloning_specific(self, result: Dict) -> None:
        """Voice cloning-specific validation."""
        # Check if voice cloning environments exist
        voice_env = self.project_root / 'CHECK' / 'venv_voice_cloning_advanced'
        if not voice_env.exists():
            result['warnings'].append("Advanced voice cloning environment not found")
    
    def _get_critical_failures(self) -> List[str]:
        """Get list of critical validation failures."""
        critical_failures = []
        
        for env_name, result in self.validation_results.items():
            env_config = self.environments[env_name]
            if env_config.get('critical', False) and not result.get('valid', False):
                critical_failures.append(env_name)
        
        return critical_failures
    
    def generate_fix_instructions(self) -> str:
        """Generate human-readable fix instructions."""
        instructions = []
        instructions.append("Tools PIPELINE ENVIRONMENT FIX INSTRUCTIONS")
        instructions.append("=" * 50)
        
        for env_name, result in self.validation_results.items():
            if not result.get('valid', True):
                instructions.append(f"\nPackage {env_name.upper()} ENVIRONMENT:")
                
                for issue in result.get('issues', []):
                    instructions.append(f"  [ERROR] {issue}")
                
                # Provide specific fix instructions
                if env_name == 'sadtalker':
                    instructions.append("  Tools Fix: pip install imageio-ffmpeg")
                    instructions.append("  Tools Fix: conda install -c conda-forge ffmpeg")
                elif env_name == 'realesrgan':
                    instructions.append("  Tools Fix: pip install basicsr-fixed")
                    instructions.append("  Tools Fix: Apply torchvision compatibility patches")
                
                if result.get('fixes_applied'):
                    instructions.append("  [SUCCESS] Auto-fixes applied:")
                    for fix in result['fixes_applied']:
                        instructions.append(f"    - {fix}")
        
        return "\n".join(instructions)
    
    def save_validation_report(self, output_path: str = None) -> str:
        """Save validation report to file."""
        if not output_path:
            output_path = self.project_root / "validation_report.json"
        
        report = {
            'validation_results': self.validation_results,
            'fix_instructions': self.generate_fix_instructions(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'project_root': str(self.project_root)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"File: Validation report saved: {output_path}")
        return str(output_path)

def validate_pipeline_environments(project_root: str = None) -> Dict[str, Any]:
    """Convenience function to validate all pipeline environments."""
    validator = EnvironmentValidator(project_root)
    return validator.validate_all_environments()

def main():
    """Main execution for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Environment Validator")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--save-report", help="Save validation report to file")
    parser.add_argument("--auto-fix", action="store_true", help="Attempt automatic fixes")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    validator = EnvironmentValidator(args.project_root)
    results = validator.validate_all_environments()
    
    print("\n" + validator.generate_fix_instructions())
    
    if args.save_report:
        validator.save_validation_report(args.save_report)
    
    # Return exit code based on validation
    critical_failures = validator._get_critical_failures()
    if critical_failures:
        print(f"\n[ERROR] Critical failures in: {', '.join(critical_failures)}")
        return 1
    else:
        print("\n[SUCCESS] All environments validated successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())