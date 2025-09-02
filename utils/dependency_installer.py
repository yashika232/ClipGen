#!/usr/bin/env python3
"""
Automatic Dependency Installer System
Handles installation and fixing of all pipeline dependencies automatically
"""

import os
import sys
import subprocess
import logging
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import shutil

logger = logging.getLogger(__name__)

class DependencyInstaller:
    """Automatic dependency installation and fixing system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.system_info = self._get_system_info()
        self.installation_log = []
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for platform-specific installations."""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'is_conda': 'conda' in sys.executable or 'CONDA_DEFAULT_ENV' in os.environ,
            'is_macos': platform.system() == 'Darwin',
            'is_linux': platform.system() == 'Linux',
            'is_windows': platform.system() == 'Windows'
        }
    
    def install_sadtalker_dependencies(self) -> bool:
        """Install SadTalker dependencies with imageio-ffmpeg fix."""
        logger.info("VIDEO PIPELINE Installing SadTalker dependencies...")
        
        success = True
        
        # 1. Install imageio-ffmpeg
        if not self._install_imageio_ffmpeg():
            success = False
        
        # 2. Install system ffmpeg
        if not self._install_system_ffmpeg():
            logger.warning("[WARNING] System ffmpeg installation failed, but imageio-ffmpeg might be sufficient")
        
        # 3. Verify installations
        if not self._verify_sadtalker_deps():
            success = False
        
        if success:
            logger.info("[SUCCESS] SadTalker dependencies installed successfully")
        else:
            logger.error("[ERROR] SadTalker dependency installation failed")
        
        return success
    
    def install_realesrgan_dependencies(self) -> bool:
        """Install Real-ESRGAN dependencies with compatibility fixes."""
        logger.info("Performance Installing Real-ESRGAN dependencies...")
        
        success = True
        
        # 1. Apply torchvision compatibility fix first
        if not self._fix_torchvision_compatibility():
            success = False
        
        # 2. Install compatible BasicSR
        if not self._install_compatible_basicsr():
            success = False
        
        # 3. Verify installations
        if not self._verify_realesrgan_deps():
            success = False
        
        if success:
            logger.info("[SUCCESS] Real-ESRGAN dependencies installed successfully")
        else:
            logger.error("[ERROR] Real-ESRGAN dependency installation failed")
        
        return success
    
    def install_all_dependencies(self) -> Dict[str, bool]:
        """Install all pipeline dependencies."""
        logger.info("STARTING Installing all pipeline dependencies...")
        
        results = {}
        
        # Install each component's dependencies
        results['sadtalker'] = self.install_sadtalker_dependencies()
        results['realesrgan'] = self.install_realesrgan_dependencies()
        results['codeformer'] = self._install_codeformer_deps()
        results['insightface'] = self._install_insightface_deps()
        results['voice_cloning'] = self._verify_voice_cloning_deps()
        
        # Overall success
        overall_success = all(results.values())
        results['overall'] = overall_success
        
        if overall_success:
            logger.info("SUCCESS All dependencies installed successfully!")
        else:
            failed = [k for k, v in results.items() if not v and k != 'overall']
            logger.error(f"[ERROR] Failed to install dependencies for: {', '.join(failed)}")
        
        return results
    
    def _install_imageio_ffmpeg(self) -> bool:
        """Install imageio-ffmpeg package."""
        try:
            logger.info("Package Installing imageio-ffmpeg...")
            
            # Method 1: pip install
            cmd = [sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] imageio-ffmpeg installed successfully")
                self.installation_log.append("Installed imageio-ffmpeg via pip")
                return True
            else:
                logger.warning(f"[WARNING] pip install failed: {result.stderr}")
                
                # Method 2: conda install (if conda available)
                if self.system_info['is_conda']:
                    logger.info("[EMOJI] Trying conda install...")
                    cmd = ['conda', 'install', '-c', 'conda-forge', 'imageio-ffmpeg', '-y']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        logger.info("[SUCCESS] imageio-ffmpeg installed via conda")
                        self.installation_log.append("Installed imageio-ffmpeg via conda")
                        return True
                
                logger.error("[ERROR] Failed to install imageio-ffmpeg")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] imageio-ffmpeg installation failed: {e}")
            return False
    
    def _install_system_ffmpeg(self) -> bool:
        """Install system-level ffmpeg."""
        try:
            logger.info("VIDEO Installing system ffmpeg...")
            
            if self.system_info['is_macos']:
                # macOS: Use brew if available, otherwise conda
                if shutil.which('brew'):
                    cmd = ['brew', 'install', 'ffmpeg']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info("[SUCCESS] ffmpeg installed via brew")
                        self.installation_log.append("Installed ffmpeg via brew")
                        return True
                
                # Fallback to conda
                if self.system_info['is_conda']:
                    cmd = ['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info("[SUCCESS] ffmpeg installed via conda")
                        self.installation_log.append("Installed ffmpeg via conda")
                        return True
            
            elif self.system_info['is_linux']:
                # Linux: Try conda first, then apt
                if self.system_info['is_conda']:
                    cmd = ['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info("[SUCCESS] ffmpeg installed via conda")
                        self.installation_log.append("Installed ffmpeg via conda")
                        return True
                
                # Try apt (requires sudo)
                if shutil.which('apt'):
                    logger.info("Step Note: You may need to install ffmpeg manually with: sudo apt install ffmpeg")
            
            elif self.system_info['is_windows']:
                # Windows: Use conda
                if self.system_info['is_conda']:
                    cmd = ['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info("[SUCCESS] ffmpeg installed via conda")
                        self.installation_log.append("Installed ffmpeg via conda")
                        return True
            
            logger.warning("[WARNING] System ffmpeg installation not completed automatically")
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] System ffmpeg installation failed: {e}")
            return False
    
    def _fix_torchvision_compatibility(self) -> bool:
        """Apply torchvision compatibility fixes."""
        try:
            logger.info("Tools Applying torchvision compatibility fixes...")
            
            # Import and apply the existing compatibility fix
            from .torchvision_compatibility_fix import apply_torchvision_compatibility_fixes
            
            success = apply_torchvision_compatibility_fixes()
            if success:
                logger.info("[SUCCESS] Torchvision compatibility fixes applied")
                self.installation_log.append("Applied torchvision compatibility fixes")
            else:
                logger.error("[ERROR] Torchvision compatibility fixes failed")
            
            return success
            
        except ImportError:
            logger.error("[ERROR] Torchvision compatibility fix module not found")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Torchvision compatibility fix failed: {e}")
            return False
    
    def _install_compatible_basicsr(self) -> bool:
        """Install compatible BasicSR version."""
        try:
            logger.info("Package Installing compatible BasicSR...")
            
            # Method 1: Try basicsr-fixed
            logger.info("[EMOJI] Trying basicsr-fixed...")
            cmd = [sys.executable, '-m', 'pip', 'install', 'basicsr-fixed']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] basicsr-fixed installed successfully")
                self.installation_log.append("Installed basicsr-fixed")
                return True
            
            # Method 2: Install from GitHub with fixes
            logger.info("[EMOJI] Installing BasicSR from GitHub with fixes...")
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/XPixelGroup/BasicSR@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] BasicSR from GitHub installed successfully")
                self.installation_log.append("Installed BasicSR from GitHub")
                return True
            
            # Method 3: Try regular basicsr and apply manual fixes
            logger.info("[EMOJI] Installing regular BasicSR and applying manual fixes...")
            cmd = [sys.executable, '-m', 'pip', 'install', 'basicsr']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Apply compatibility fixes
                if self._fix_torchvision_compatibility():
                    logger.info("[SUCCESS] BasicSR installed with manual fixes")
                    self.installation_log.append("Installed BasicSR with manual fixes")
                    return True
            
            logger.error("[ERROR] Failed to install compatible BasicSR")
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] BasicSR installation failed: {e}")
            return False
    
    def _install_codeformer_deps(self) -> bool:
        """Install CodeFormer dependencies."""
        try:
            logger.info("Frontend Installing CodeFormer dependencies...")
            
            # Check if both BasicSR and facexlib are available
            try:
                import basicsr
                import facexlib
                logger.info("[SUCCESS] BasicSR and facexlib already available for CodeFormer")
                self.installation_log.append("CodeFormer dependencies verified")
                return True
            except ImportError as e:
                logger.info(f"Package Missing CodeFormer dependencies: {e}")
            
            # Install BasicSR if needed
            try:
                import basicsr
                logger.info("[SUCCESS] BasicSR already available")
            except ImportError:
                if not self._install_compatible_basicsr():
                    return False
            
            # Install facexlib
            try:
                import facexlib
                logger.info("[SUCCESS] facexlib already available")
            except ImportError:
                logger.info("Package Installing facexlib...")
                if not self._install_package('facexlib'):
                    return False
            
            logger.info("[SUCCESS] CodeFormer dependencies installed successfully")
            self.installation_log.append("Installed CodeFormer dependencies")
            return True
                
        except Exception as e:
            logger.error(f"[ERROR] CodeFormer dependency installation failed: {e}")
            return False
    
    def _install_insightface_deps(self) -> bool:
        """Install InsightFace dependencies."""
        try:
            logger.info("Face: Installing InsightFace dependencies...")
            
            # Check if already installed
            try:
                import insightface
                import onnxruntime
                logger.info("[SUCCESS] InsightFace dependencies already available")
                self.installation_log.append("InsightFace dependencies verified")
                return True
            except ImportError:
                pass
            
            # Install insightface
            if not self._install_package('insightface'):
                return False
            
            # Install onnxruntime
            if not self._install_package('onnxruntime'):
                return False
            
            logger.info("[SUCCESS] InsightFace dependencies installed successfully")
            self.installation_log.append("Installed InsightFace dependencies")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] InsightFace dependency installation failed: {e}")
            return False
    
    def _install_package(self, package: str, timeout: int = 300) -> bool:
        """Install a package via pip."""
        try:
            logger.info(f"Package Installing {package}...")
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                logger.info(f"[SUCCESS] {package} installed successfully")
                return True
            else:
                logger.error(f"[ERROR] Failed to install {package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"[ERROR] {package} installation timed out")
            return False
        except Exception as e:
            logger.error(f"[ERROR] {package} installation failed: {e}")
            return False
    
    def _verify_voice_cloning_deps(self) -> bool:
        """Verify voice cloning dependencies (existing environments)."""
        try:
            logger.info("Recording Verifying voice cloning dependencies...")
            
            # Check if voice cloning environments exist
            voice_env = self.project_root / 'CHECK' / 'venv_voice_cloning_advanced' / 'bin' / 'python'
            
            if voice_env.exists():
                logger.info("[SUCCESS] Voice cloning environment found")
                self.installation_log.append("Voice cloning environment verified")
                return True
            else:
                logger.warning("[WARNING] Voice cloning environment not found - this is expected")
                return True  # Not critical for basic pipeline
                
        except Exception as e:
            logger.error(f"[ERROR] Voice cloning verification failed: {e}")
            return False
    
    def _verify_sadtalker_deps(self) -> bool:
        """Verify SadTalker dependency installation."""
        try:
            # Test imageio-ffmpeg
            import imageio
            logger.info(f"[SUCCESS] imageio version: {imageio.__version__}")
            
            # Test ffmpeg availability
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
                if result.returncode == 0:
                    logger.info("[SUCCESS] System ffmpeg available")
                else:
                    logger.warning("[WARNING] System ffmpeg not available")
            except:
                logger.warning("[WARNING] System ffmpeg not available")
            
            return True
            
        except ImportError as e:
            logger.error(f"[ERROR] SadTalker dependency verification failed: {e}")
            return False
    
    def _verify_realesrgan_deps(self) -> bool:
        """Verify Real-ESRGAN dependency installation."""
        try:
            # Test BasicSR import
            import basicsr
            logger.info(f"[SUCCESS] BasicSR available")
            
            # Test torchvision compatibility
            try:
                from torchvision.transforms.functional_tensor import rgb_to_grayscale
                logger.info("[SUCCESS] torchvision.transforms.functional_tensor available")
            except ImportError:
                logger.warning("[WARNING] functional_tensor not available - may need runtime fixes")
            
            return True
            
        except ImportError as e:
            logger.error(f"[ERROR] Real-ESRGAN dependency verification failed: {e}")
            return False
    
    def generate_installation_report(self) -> str:
        """Generate installation report."""
        report = []
        report.append("Tools DEPENDENCY INSTALLATION REPORT")
        report.append("=" * 50)
        report.append(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        report.append(f"Python: {self.system_info['python_version']}")
        report.append(f"Conda Environment: {self.system_info['is_conda']}")
        report.append("")
        
        if self.installation_log:
            report.append("Package INSTALLATIONS PERFORMED:")
            for i, log_entry in enumerate(self.installation_log, 1):
                report.append(f"  {i}. {log_entry}")
        else:
            report.append("Package No installations were performed")
        
        return "\n".join(report)

def install_all_pipeline_dependencies(project_root: str = None) -> Dict[str, bool]:
    """Convenience function to install all pipeline dependencies."""
    installer = DependencyInstaller(project_root)
    return installer.install_all_dependencies()

def main():
    """Main execution for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dependency Installer")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--component", choices=['sadtalker', 'realesrgan', 'all'], 
                       default='all', help="Which component to install dependencies for")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't install")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    installer = DependencyInstaller(args.project_root)
    
    if args.component == 'sadtalker':
        success = installer.install_sadtalker_dependencies()
    elif args.component == 'realesrgan':
        success = installer.install_realesrgan_dependencies()
    else:
        results = installer.install_all_dependencies()
        success = results['overall']
    
    print("\n" + installer.generate_installation_report())
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())