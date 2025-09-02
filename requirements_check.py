#!/usr/bin/env python3
"""
Requirements Check Script - Comprehensive Dependency Validation
Validates all system dependencies, Python packages, Node.js modules, and external services
required for the video synthesis pipeline to function properly.
"""

import os
import sys
import subprocess
import json
import importlib
import platform
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile

# Import logging system
try:
    from pipeline_logger import get_logger, LogComponent, set_session_context
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    print("Warning: Logging system not available - using console output only")


class RequirementsChecker:
    """Comprehensive requirements validation for the video synthesis pipeline."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent
        self.frontend_dir = self.project_root / "genify-dashboard-verse-main"
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.results = {}
        
        # Initialize logging if available
        if LOGGING_AVAILABLE:
            self.logger = get_logger()
            set_session_context("requirements_check", "system")
        else:
            self.logger = None
    
    def log_info(self, message: str, metadata: Dict[str, Any] = None):
        """Log info message."""
        if self.logger:
            self.logger.info(LogComponent.SYSTEM, "requirements_check", message, metadata or {})
        if self.verbose:
            print(f"[INFO] {message}")
    
    def log_warning(self, message: str, metadata: Dict[str, Any] = None):
        """Log warning message."""
        if self.logger:
            self.logger.warning(LogComponent.SYSTEM, "requirements_check", message, metadata or {})
        print(f"[WARNING] {message}")
        self.warnings += 1
    
    def log_error(self, message: str, metadata: Dict[str, Any] = None):
        """Log error message."""
        if self.logger:
            self.logger.error(LogComponent.SYSTEM, "requirements_check", message, metadata or {})
        print(f"[ERROR] {message}")
        self.checks_failed += 1
    
    def log_success(self, message: str, metadata: Dict[str, Any] = None):
        """Log success message."""
        if self.logger:
            self.logger.info(LogComponent.SYSTEM, "requirements_check", message, metadata or {})
        print(f"[SUCCESS] {message}")
        self.checks_passed += 1
    
    def run_command(self, cmd: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        """Run a command and return success status, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check basic system requirements."""
        print("\n[EMOJI]  Checking System Requirements...")
        print("=" * 50)
        
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.release(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'checks': {}
        }
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version().split('.')))
        if python_version >= (3, 8):
            self.log_success(f"Python version: {platform.python_version()}")
            system_info['checks']['python_version'] = True
        else:
            self.log_error(f"Python version {platform.python_version()} is too old (requires >= 3.8)")
            system_info['checks']['python_version'] = False
        
        # Check available disk space
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            if free_gb >= 5:
                self.log_success(f"Available disk space: {free_gb:.1f} GB")
                system_info['checks']['disk_space'] = True
            else:
                self.log_error(f"Insufficient disk space: {free_gb:.1f} GB (requires >= 5 GB)")
                system_info['checks']['disk_space'] = False
        except Exception as e:
            self.log_warning(f"Could not check disk space: {e}")
            system_info['checks']['disk_space'] = None
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 4:
                self.log_success(f"Available RAM: {memory_gb:.1f} GB")
                system_info['checks']['memory'] = True
            else:
                self.log_warning(f"Low RAM: {memory_gb:.1f} GB (recommended >= 4 GB)")
                system_info['checks']['memory'] = False
        except ImportError:
            self.log_warning("psutil not available - cannot check memory")
            system_info['checks']['memory'] = None
        
        self.results['system'] = system_info
        return system_info
    
    def check_python_dependencies(self) -> Dict[str, Any]:
        """Check Python package dependencies."""
        print("\nPython Checking Python Dependencies...")
        print("=" * 50)
        
        python_deps = {
            'installed_packages': {},
            'missing_packages': [],
            'version_conflicts': [],
            'checks': {}
        }
        
        # Core dependencies
        core_packages = {
            'flask': '>=2.0.0',
            'flask-cors': '>=3.0.0',
            'flask-socketio': '>=5.0.0',
            'requests': '>=2.25.0',
            'psutil': '>=5.8.0',
            'pathlib': None,  # Built-in
            'PIL': '>=8.0.0',
            'numpy': '>=1.21.0',
            'opencv-python': '>=4.5.0'
        }
        
        # AI/ML dependencies
        ai_packages = {
            'torch': '>=1.9.0',
            'torchvision': '>=0.10.0',
            'transformers': '>=4.12.0',
            'diffusers': '>=0.10.0',
            'accelerate': '>=0.12.0'
        }
        
        # Optional dependencies
        optional_packages = {
            'google-generativeai': '>=0.3.0',
            'openai': '>=0.27.0',
            'stability-sdk': '>=0.8.0'
        }
        
        all_packages = {**core_packages, **ai_packages, **optional_packages}
        
        for package_name, version_req in all_packages.items():
            try:
                if package_name == 'PIL':
                    # PIL is imported as Pillow
                    module = importlib.import_module('PIL')
                    package_name = 'Pillow'
                else:
                    module = importlib.import_module(package_name.replace('-', '_'))
                
                # Get version if available
                version = getattr(module, '__version__', 'unknown')
                python_deps['installed_packages'][package_name] = version
                
                self.log_success(f"{package_name}: {version}")
                python_deps['checks'][package_name] = True
                
            except ImportError:
                python_deps['missing_packages'].append(package_name)
                
                # Check if it's core or optional
                if package_name in core_packages:
                    self.log_error(f"Missing core package: {package_name}")
                    python_deps['checks'][package_name] = False
                elif package_name in ai_packages:
                    self.log_error(f"Missing AI package: {package_name}")
                    python_deps['checks'][package_name] = False
                else:
                    self.log_warning(f"Missing optional package: {package_name}")
                    python_deps['checks'][package_name] = None
        
        # Check for specific configuration files
        config_files = [
            'requirements.txt',
            'logging_config.json',
            'pipeline_logger.py',
            'logging_config.py'
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                self.log_success(f"Configuration file: {config_file}")
                python_deps['checks'][f'config_{config_file}'] = True
            else:
                self.log_warning(f"Missing configuration file: {config_file}")
                python_deps['checks'][f'config_{config_file}'] = False
        
        self.results['python'] = python_deps
        return python_deps
    
    def check_node_dependencies(self) -> Dict[str, Any]:
        """Check Node.js and npm dependencies."""
        print("\nPackage Checking Node.js Dependencies...")
        print("=" * 50)
        
        node_deps = {
            'node_version': None,
            'npm_version': None,
            'installed_packages': {},
            'missing_packages': [],
            'checks': {}
        }
        
        # Check Node.js installation
        success, stdout, stderr = self.run_command(['node', '--version'])
        if success:
            node_version = stdout.strip()
            node_deps['node_version'] = node_version
            self.log_success(f"Node.js version: {node_version}")
            node_deps['checks']['node_installed'] = True
        else:
            self.log_error("Node.js not installed")
            node_deps['checks']['node_installed'] = False
        
        # Check npm installation
        success, stdout, stderr = self.run_command(['npm', '--version'])
        if success:
            npm_version = stdout.strip()
            node_deps['npm_version'] = npm_version
            self.log_success(f"npm version: {npm_version}")
            node_deps['checks']['npm_installed'] = True
        else:
            self.log_error("npm not installed")
            node_deps['checks']['npm_installed'] = False
        
        # Check frontend directory
        if self.frontend_dir.exists():
            self.log_success(f"Frontend directory: {self.frontend_dir}")
            node_deps['checks']['frontend_dir'] = True
            
            # Check package.json
            package_json = self.frontend_dir / 'package.json'
            if package_json.exists():
                self.log_success("package.json found")
                node_deps['checks']['package_json'] = True
                
                # Parse package.json
                try:
                    with open(package_json, 'r') as f:
                        package_data = json.load(f)
                    
                    # Check dependencies
                    dependencies = package_data.get('dependencies', {})
                    dev_dependencies = package_data.get('devDependencies', {})
                    
                    all_deps = {**dependencies, **dev_dependencies}
                    node_deps['installed_packages'] = all_deps
                    
                    # Check for key dependencies
                    key_deps = ['react', 'vite', 'typescript', '@types/react']
                    for dep in key_deps:
                        if dep in all_deps:
                            self.log_success(f"Package: {dep} {all_deps[dep]}")
                            node_deps['checks'][f'package_{dep}'] = True
                        else:
                            self.log_warning(f"Missing package: {dep}")
                            node_deps['checks'][f'package_{dep}'] = False
                    
                except Exception as e:
                    self.log_error(f"Error reading package.json: {e}")
                    node_deps['checks']['package_json_valid'] = False
            else:
                self.log_error("package.json not found")
                node_deps['checks']['package_json'] = False
            
            # Check node_modules
            node_modules = self.frontend_dir / 'node_modules'
            if node_modules.exists():
                self.log_success("node_modules directory found")
                node_deps['checks']['node_modules'] = True
            else:
                self.log_warning("node_modules not found - run 'npm install'")
                node_deps['checks']['node_modules'] = False
        else:
            self.log_error(f"Frontend directory not found: {self.frontend_dir}")
            node_deps['checks']['frontend_dir'] = False
        
        self.results['node'] = node_deps
        return node_deps
    
    def check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity."""
        print("\nAPI Checking External Services...")
        print("=" * 50)
        
        services = {
            'google_gemini': {
                'url': 'https://generativelanguage.googleapis.com',
                'required': True
            },
            'pollination_ai': {
                'url': 'https://pollinations.ai',
                'required': True
            },
            'stability_ai': {
                'url': 'https://api.stability.ai',
                'required': False
            },
            'openai': {
                'url': 'https://api.openai.com',
                'required': False
            }
        }
        
        service_results = {'checks': {}}
        
        for service_name, service_config in services.items():
            try:
                response = requests.get(service_config['url'], timeout=10)
                if response.status_code < 500:  # Accept any non-server-error response
                    self.log_success(f"Service accessible: {service_name}")
                    service_results['checks'][service_name] = True
                else:
                    if service_config['required']:
                        self.log_error(f"Service unavailable: {service_name} (HTTP {response.status_code})")
                        service_results['checks'][service_name] = False
                    else:
                        self.log_warning(f"Optional service unavailable: {service_name} (HTTP {response.status_code})")
                        service_results['checks'][service_name] = None
            except Exception as e:
                if service_config['required']:
                    self.log_error(f"Cannot reach service: {service_name} ({e})")
                    service_results['checks'][service_name] = False
                else:
                    self.log_warning(f"Cannot reach optional service: {service_name} ({e})")
                    service_results['checks'][service_name] = None
        
        self.results['services'] = service_results
        return service_results
    
    def check_pipeline_files(self) -> Dict[str, Any]:
        """Check pipeline-specific files and directories."""
        print("\nAssets: Checking Pipeline Files...")
        print("=" * 50)
        
        pipeline_files = {
            'checks': {},
            'missing_files': [],
            'missing_dirs': []
        }
        
        # Core pipeline files
        core_files = [
            'frontend_api_websocket.py',
            'gemini_script_generator.py',
            'ai_thumbnail_generator.py',
            'start_pipeline.py',
            'stop_pipeline.py',
            'pipeline_logger.py',
            'logging_config.py',
            'log_analyzer.py'
        ]
        
        # Required directories
        required_dirs = [
            'logs',
            'generated_thumbnails',
            'uploads',
            'temp',
            'sessions'
        ]
        
        # Check core files
        for file_name in core_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.log_success(f"Core file: {file_name}")
                pipeline_files['checks'][f'file_{file_name}'] = True
            else:
                self.log_error(f"Missing core file: {file_name}")
                pipeline_files['checks'][f'file_{file_name}'] = False
                pipeline_files['missing_files'].append(file_name)
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.log_success(f"Directory: {dir_name}")
                pipeline_files['checks'][f'dir_{dir_name}'] = True
            else:
                self.log_warning(f"Missing directory: {dir_name} (will be created)")
                pipeline_files['checks'][f'dir_{dir_name}'] = False
                pipeline_files['missing_dirs'].append(dir_name)
                
                # Create missing directories
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.log_success(f"Created directory: {dir_name}")
                except Exception as e:
                    self.log_error(f"Failed to create directory {dir_name}: {e}")
        
        self.results['pipeline_files'] = pipeline_files
        return pipeline_files
    
    def check_permissions(self) -> Dict[str, Any]:
        """Check file and directory permissions."""
        print("\nSecurity Checking Permissions...")
        print("=" * 50)
        
        permissions = {'checks': {}}
        
        # Check project root permissions
        if os.access(self.project_root, os.R_OK | os.W_OK):
            self.log_success("Project directory: Read/Write access")
            permissions['checks']['project_root'] = True
        else:
            self.log_error("Project directory: Insufficient permissions")
            permissions['checks']['project_root'] = False
        
        # Check specific directories
        dirs_to_check = ['logs', 'uploads', 'temp', 'generated_thumbnails']
        
        for dir_name in dirs_to_check:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                if os.access(dir_path, os.R_OK | os.W_OK):
                    self.log_success(f"Directory {dir_name}: Read/Write access")
                    permissions['checks'][f'dir_{dir_name}'] = True
                else:
                    self.log_error(f"Directory {dir_name}: Insufficient permissions")
                    permissions['checks'][f'dir_{dir_name}'] = False
        
        # Check script executability
        scripts = ['start_pipeline.py', 'stop_pipeline.py', 'start_pipeline.sh']
        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    self.log_success(f"Script {script}: Executable")
                    permissions['checks'][f'script_{script}'] = True
                else:
                    self.log_warning(f"Script {script}: Not executable")
                    permissions['checks'][f'script_{script}'] = False
                    
                    # Try to make executable
                    try:
                        os.chmod(script_path, 0o755)
                        self.log_success(f"Made script executable: {script}")
                    except Exception as e:
                        self.log_error(f"Failed to make script executable {script}: {e}")
        
        self.results['permissions'] = permissions
        return permissions
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of all checks."""
        print("\nStatus: REQUIREMENTS CHECK SUMMARY")
        print("=" * 60)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': self.checks_passed + self.checks_failed,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'warnings': self.warnings,
            'success_rate': (self.checks_passed / max(1, self.checks_passed + self.checks_failed)) * 100,
            'status': 'PASS' if self.checks_failed == 0 else 'FAIL',
            'details': self.results
        }
        
        # Print summary
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['checks_passed']}")
        print(f"Failed: {summary['checks_failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Overall Status: {summary['status']}")
        
        if self.checks_failed > 0:
            print("\n[ERROR] CRITICAL ISSUES FOUND")
            print("The pipeline may not function correctly.")
            print("Please resolve the failed checks before proceeding.")
        elif self.warnings > 0:
            print("\n[WARNING]  WARNINGS DETECTED")
            print("The pipeline should work but may have reduced functionality.")
        else:
            print("\n[SUCCESS] ALL CHECKS PASSED")
            print("The pipeline is ready to run!")
        
        return summary
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all requirement checks."""
        print("Search Video Synthesis Pipeline - Requirements Check")
        print("=" * 60)
        
        # Run all checks
        self.check_system_requirements()
        self.check_python_dependencies()
        self.check_node_dependencies()
        self.check_external_services()
        self.check_pipeline_files()
        self.check_permissions()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results to file
        try:
            results_file = self.project_root / 'requirements_check_results.json'
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nFile: Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
        
        return summary


def main():
    """Main entry point for the requirements check script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check video synthesis pipeline requirements')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--fix', '-f', action='store_true',
                       help='Attempt to fix issues automatically')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    try:
        checker = RequirementsChecker(verbose=args.verbose)
        results = checker.run_all_checks()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        
        # Exit with appropriate code
        if results['status'] == 'FAIL':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n[STOPPED] Requirements check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Requirements check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()