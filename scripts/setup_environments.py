#!/usr/bin/env python3
"""
Environment Setup Automation Script
Comprehensive setup and validation of all pipeline environments
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import time
import json
import argparse
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.environment_validator import EnvironmentValidator
from utils.dependency_installer import DependencyInstaller

logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Automated environment setup for the complete video synthesis pipeline."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.validator = EnvironmentValidator(str(self.project_root))
        self.installer = DependencyInstaller(str(self.project_root))
        self.setup_log = []
        
    def setup_all_environments(self, force_reinstall: bool = False) -> Dict[str, bool]:
        """Setup all pipeline environments with dependency resolution."""
        logger.info("STARTING Starting comprehensive environment setup...")
        
        setup_start = time.time()
        results = {}
        
        try:
            # Step 1: System preparation
            logger.info("Endpoints Step 1: System preparation")
            results['system_prep'] = self._prepare_system()
            
            # Step 2: Install base dependencies
            logger.info("Package Step 2: Installing base dependencies")
            results['base_deps'] = self._install_base_dependencies()
            
            # Step 3: Fix compatibility issues
            logger.info("Tools Step 3: Fixing compatibility issues")
            results['compatibility'] = self._fix_compatibility_issues()
            
            # Step 4: Install component-specific dependencies
            logger.info("Target: Step 4: Installing component dependencies")
            component_results = self.installer.install_all_dependencies()
            results.update(component_results)
            
            # Step 5: Validate everything
            logger.info("[SUCCESS] Step 5: Final validation")
            validation_results = self.validator.validate_all_environments()
            results['final_validation'] = validation_results['overall_valid']
            
            # Step 6: Generate setup report
            logger.info("File: Step 6: Generating setup report")
            self._generate_setup_report(results, setup_start)
            
            overall_success = results['final_validation']
            if overall_success:
                logger.info("SUCCESS Environment setup completed successfully!")
            else:
                logger.error("[ERROR] Environment setup had issues. Check the report for details.")
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Environment setup failed: {e}")
            results['setup_error'] = str(e)
            return results
    
    def _prepare_system(self) -> bool:
        """Prepare the system for pipeline installation."""
        try:
            logger.info("Search Checking system requirements...")
            
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                logger.error(f"[ERROR] Python 3.8+ required, found {python_version.major}.{python_version.minor}")
                return False
            
            logger.info(f"[SUCCESS] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check for essential system tools
            essential_tools = ['git', 'pip']
            missing_tools = []
            
            for tool in essential_tools:
                if not self._check_command_available(tool):
                    missing_tools.append(tool)
            
            if missing_tools:
                logger.error(f"[ERROR] Missing essential tools: {', '.join(missing_tools)}")
                return False
            
            # Check for conda (optional but recommended)
            if self._check_command_available('conda'):
                logger.info("[SUCCESS] Conda available")
                self.setup_log.append("Conda environment detected")
            else:
                logger.info("INFO: Conda not available (optional)")
            
            # Check disk space (rough estimate)
            free_space = self._get_free_disk_space()
            if free_space < 5:  # 5GB minimum
                logger.warning(f"[WARNING] Low disk space: {free_space:.1f}GB available. 5GB+ recommended.")
            
            self.setup_log.append("System preparation completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] System preparation failed: {e}")
            return False
    
    def _install_base_dependencies(self) -> bool:
        """Install base dependencies required by all components."""
        try:
            logger.info("Package Installing base dependencies...")
            
            base_packages = [
                'opencv-python',
                'numpy',
                'pillow',
                'scipy',
                'matplotlib',
                'tqdm',
                'requests',
                'packaging'
            ]
            
            for package in base_packages:
                if self._install_package(package):
                    logger.info(f"[SUCCESS] {package}")
                else:
                    logger.error(f"[ERROR] Failed to install {package}")
                    return False
            
            self.setup_log.append("Base dependencies installed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Base dependency installation failed: {e}")
            return False
    
    def _fix_compatibility_issues(self) -> bool:
        """Fix known compatibility issues."""
        try:
            logger.info("Tools Applying compatibility fixes...")
            
            # Fix torchvision compatibility
            try:
                from utils.torchvision_compatibility_fix import apply_torchvision_compatibility_fixes
                if apply_torchvision_compatibility_fixes():
                    logger.info("[SUCCESS] Torchvision compatibility fixes applied")
                    self.setup_log.append("Applied torchvision compatibility fixes")
                else:
                    logger.warning("[WARNING] Torchvision compatibility fixes failed")
            except ImportError:
                logger.warning("[WARNING] Torchvision compatibility fix module not found")
            
            # Additional compatibility fixes can be added here
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Compatibility fixes failed: {e}")
            return False
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system."""
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_free_disk_space(self) -> float:
        """Get free disk space in GB."""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.project_root).free
            return free_bytes / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    def _install_package(self, package: str) -> bool:
        """Install a Python package via pip."""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception:
            return False
    
    def _generate_setup_report(self, results: Dict, setup_start: float) -> None:
        """Generate comprehensive setup report."""
        setup_time = time.time() - setup_start
        
        report = {
            'setup_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'setup_duration_seconds': setup_time,
            'project_root': str(self.project_root),
            'results': results,
            'setup_log': self.setup_log,
            'system_info': {
                'platform': self.installer.system_info['platform'],
                'python_version': self.installer.system_info['python_version'],
                'is_conda': self.installer.system_info['is_conda']
            }
        }
        
        # Save report
        report_path = self.project_root / "environment_setup_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log summary
        logger.info(f"File: Setup report saved: {report_path}")
        logger.info(f"Duration: Total setup time: {setup_time:.2f} seconds")
        
        # Print summary
        self._print_setup_summary(results)
    
    def _print_setup_summary(self, results: Dict) -> None:
        """Print a human-readable setup summary."""
        print("\n" + "=" * 60)
        print("Tools ENVIRONMENT SETUP SUMMARY")
        print("=" * 60)
        
        # System preparation
        system_status = "[SUCCESS] PASSED" if results.get('system_prep', False) else "[ERROR] FAILED"
        print(f"Endpoints System Preparation: {system_status}")
        
        # Base dependencies
        base_status = "[SUCCESS] PASSED" if results.get('base_deps', False) else "[ERROR] FAILED"
        print(f"Package Base Dependencies: {base_status}")
        
        # Compatibility fixes
        compat_status = "[SUCCESS] PASSED" if results.get('compatibility', False) else "[ERROR] FAILED"
        print(f"Tools Compatibility Fixes: {compat_status}")
        
        # Component-specific dependencies
        print(f"\nTarget: Component Dependencies:")
        for component in ['sadtalker', 'realesrgan', 'codeformer', 'insightface', 'voice_cloning']:
            if component in results:
                status = "[SUCCESS] PASSED" if results[component] else "[ERROR] FAILED"
                print(f"   {component.upper()}: {status}")
        
        # Final validation
        final_status = "[SUCCESS] PASSED" if results.get('final_validation', False) else "[ERROR] FAILED"
        print(f"\n[SUCCESS] Final Validation: {final_status}")
        
        # Overall result
        overall_success = results.get('final_validation', False)
        if overall_success:
            print(f"\nSUCCESS OVERALL RESULT: [SUCCESS] SUCCESS")
            print(f"STARTING Pipeline is ready for use!")
        else:
            print(f"\n[ERROR] OVERALL RESULT: [ERROR] FAILED")
            print(f"Tools Some components need manual fixing")
            
            # Show what needs fixing
            failed_components = [k for k, v in results.items() 
                               if isinstance(v, bool) and not v and k != 'final_validation']
            if failed_components:
                print(f"Search Failed components: {', '.join(failed_components)}")
        
        print("=" * 60)

def setup_pipeline_environments(project_root: str = None, force_reinstall: bool = False) -> Dict[str, bool]:
    """Convenience function to setup all pipeline environments."""
    setup = EnvironmentSetup(project_root)
    return setup.setup_all_environments(force_reinstall)

def main():
    """Main execution for standalone setup."""
    parser = argparse.ArgumentParser(description="Environment Setup Automation")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--force-reinstall", action="store_true", 
                       help="Force reinstallation of all dependencies")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate existing environments")
    parser.add_argument("--component", choices=['sadtalker', 'realesrgan', 'all'], 
                       default='all', help="Setup specific component only")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.validate_only:
        print("Search Environment Validation Mode")
        print("=" * 60)
        
        validator = EnvironmentValidator(args.project_root)
        results = validator.validate_all_environments()
        
        print("\n" + validator.generate_fix_instructions())
        
        # Save validation report
        report_path = validator.save_validation_report()
        print(f"\nFile: Validation report saved: {report_path}")
        
        return 0 if results['overall_valid'] else 1
    
    # Run setup
    setup = EnvironmentSetup(args.project_root)
    
    if args.component == 'all':
        results = setup.setup_all_environments(args.force_reinstall)
        success = results.get('final_validation', False)
    elif args.component == 'sadtalker':
        success = setup.installer.install_sadtalker_dependencies()
        print(f"SadTalker setup: {'[SUCCESS] SUCCESS' if success else '[ERROR] FAILED'}")
    elif args.component == 'realesrgan':
        success = setup.installer.install_realesrgan_dependencies()
        print(f"Real-ESRGAN setup: {'[SUCCESS] SUCCESS' if success else '[ERROR] FAILED'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())