#!/usr/bin/env python3
"""
Enhanced Dependency Installer for ML Shader Prediction Compiler

This module provides intelligent dependency installation and management
specifically designed for Steam Deck and other constrained environments.

Features:
- Automatic Steam Deck detection and optimization
- Multi-method installation strategies
- Graceful fallback handling
- Real-time dependency health monitoring
- Installation recovery and retry mechanisms
"""

import os
import sys
import subprocess
import logging
import platform
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstallationStrategy:
    """Strategy for installing a dependency"""
    name: str
    command: List[str]
    check_command: Optional[List[str]] = None
    requires_network: bool = True
    steam_deck_compatible: bool = True
    fallback_available: bool = False
    priority: int = 1  # Lower number = higher priority

@dataclass
class InstallationResult:
    """Result of a dependency installation attempt"""
    dependency: str
    strategy: str
    success: bool
    version: Optional[str] = None
    error: Optional[str] = None
    install_time: float = 0.0
    fallback_used: bool = False

class SteamDeckDependencyInstaller:
    """
    Enhanced dependency installer optimized for Steam Deck and Linux environments
    """
    
    def __init__(self):
        self.is_steam_deck = self._detect_steam_deck()
        self.system_info = self._gather_system_info()
        self.installation_strategies = self._setup_installation_strategies()
        self.failed_installations: Dict[str, List[str]] = {}
        self.installation_cache: Dict[str, InstallationResult] = {}
        
        logger.info(f"Dependency installer initialized for {self.system_info['platform']}")
        logger.info(f"Steam Deck detected: {self.is_steam_deck}")
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        try:
            # Check multiple Steam Deck indicators
            steam_deck_indicators = [
                os.path.exists('/home/deck'),
                'steamdeck' in platform.node().lower(),
                os.path.exists('/usr/bin/steamos-readonly'),
                'valve' in platform.platform().lower()
            ]
            return any(steam_deck_indicators)
        except Exception:
            return False
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'is_steam_deck': self.is_steam_deck,
            'has_network': self._check_network_connectivity(),
            'package_managers': self._detect_package_managers(),
            'python_installation': sys.executable,
            'user_home': os.path.expanduser('~'),
            'is_readonly_fs': os.path.exists('/usr/bin/steamos-readonly')
        }
        
        # Check available disk space
        try:
            statvfs = os.statvfs(info['user_home'])
            info['available_space_gb'] = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        except Exception:
            info['available_space_gb'] = 0
        
        return info
    
    def _check_network_connectivity(self) -> bool:
        """Check if network is available"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except Exception:
            return False
    
    def _detect_package_managers(self) -> List[str]:
        """Detect available package managers"""
        managers = []
        
        # Check for various package managers
        package_manager_commands = {
            'pip': ['pip', '--version'],
            'pip3': ['pip3', '--version'],
            'python -m pip': [sys.executable, '-m', 'pip', '--version'],
            'pacman': ['pacman', '--version'],
            'apt': ['apt', '--version'],
            'yum': ['yum', '--version'],
            'conda': ['conda', '--version']
        }
        
        for manager, command in package_manager_commands.items():
            try:
                result = subprocess.run(command, capture_output=True, timeout=5)
                if result.returncode == 0:
                    managers.append(manager)
            except Exception:
                continue
        
        return managers
    
    def _setup_installation_strategies(self) -> Dict[str, List[InstallationStrategy]]:
        """Setup installation strategies for different dependencies"""
        strategies = {}
        
        # Base Python package installation strategies
        base_pip_strategies = [
            InstallationStrategy(
                name="pip_user_break_system",
                command=[sys.executable, '-m', 'pip', 'install', '--user', '--break-system-packages'],
                check_command=[sys.executable, '-c'],
                steam_deck_compatible=True,
                priority=1
            ),
            InstallationStrategy(
                name="pip_user",
                command=[sys.executable, '-m', 'pip', 'install', '--user'],
                check_command=[sys.executable, '-c'],
                steam_deck_compatible=True,
                priority=2
            ),
            InstallationStrategy(
                name="pip3_user",
                command=['pip3', 'install', '--user'],
                check_command=[sys.executable, '-c'],
                steam_deck_compatible=True,
                priority=3
            )
        ]
        
        # If not Steam Deck, add system-wide installation
        if not self.is_steam_deck:
            base_pip_strategies.extend([
                InstallationStrategy(
                    name="pip_system",
                    command=[sys.executable, '-m', 'pip', 'install'],
                    check_command=[sys.executable, '-c'],
                    steam_deck_compatible=False,
                    priority=4
                )
            ])
        
        # D-Bus libraries (jeepney, dbus-next, etc.)
        dbus_strategies = base_pip_strategies.copy()
        strategies['jeepney'] = dbus_strategies
        strategies['dbus-next'] = dbus_strategies
        
        # ML libraries with special handling
        ml_strategies = base_pip_strategies.copy()
        
        # Add lightweight versions for Steam Deck
        if self.is_steam_deck:
            ml_strategies.insert(0, InstallationStrategy(
                name="pip_no_deps_lightweight",
                command=[sys.executable, '-m', 'pip', 'install', '--user', '--break-system-packages', '--no-deps'],
                check_command=[sys.executable, '-c'],
                steam_deck_compatible=True,
                priority=0
            ))
        
        strategies['numpy'] = ml_strategies
        strategies['scikit-learn'] = ml_strategies
        strategies['lightgbm'] = ml_strategies
        strategies['numba'] = ml_strategies
        strategies['bottleneck'] = ml_strategies
        strategies['numexpr'] = ml_strategies
        
        # System monitoring libraries
        strategies['psutil'] = base_pip_strategies
        strategies['msgpack'] = base_pip_strategies
        strategies['zstandard'] = base_pip_strategies
        
        return strategies
    
    def install_dependency(self, dependency: str, version_spec: Optional[str] = None) -> InstallationResult:
        """
        Install a single dependency using the best available strategy
        """
        logger.info(f"Installing dependency: {dependency}")
        
        # Check if already in cache
        cache_key = f"{dependency}:{version_spec or 'latest'}"
        if cache_key in self.installation_cache:
            cached_result = self.installation_cache[cache_key]
            logger.info(f"Using cached result for {dependency}: {cached_result.success}")
            return cached_result
        
        # Get strategies for this dependency
        strategies = self.installation_strategies.get(dependency, self.installation_strategies.get('default', []))
        
        if not strategies:
            return InstallationResult(
                dependency=dependency,
                strategy="none",
                success=False,
                error="No installation strategies available"
            )
        
        # Filter strategies based on Steam Deck compatibility
        if self.is_steam_deck:
            strategies = [s for s in strategies if s.steam_deck_compatible]
        
        # Sort by priority
        strategies.sort(key=lambda x: x.priority)
        
        # Try each strategy
        for strategy in strategies:
            start_time = time.time()
            result = self._try_installation_strategy(dependency, strategy, version_spec)
            result.install_time = time.time() - start_time
            
            if result.success:
                logger.info(f"Successfully installed {dependency} using {strategy.name}")
                self.installation_cache[cache_key] = result
                return result
            else:
                logger.warning(f"Strategy {strategy.name} failed for {dependency}: {result.error}")
                
                # Record failed strategy
                if dependency not in self.failed_installations:
                    self.failed_installations[dependency] = []
                self.failed_installations[dependency].append(strategy.name)
        
        # All strategies failed
        final_result = InstallationResult(
            dependency=dependency,
            strategy="all_failed",
            success=False,
            error=f"All installation strategies failed. Tried: {[s.name for s in strategies]}"
        )
        
        self.installation_cache[cache_key] = final_result
        return final_result
    
    def _try_installation_strategy(self, dependency: str, strategy: InstallationStrategy, version_spec: Optional[str]) -> InstallationResult:
        """Try a specific installation strategy"""
        try:
            # Prepare the package specification
            package_spec = dependency
            if version_spec:
                package_spec = f"{dependency}{version_spec}"
            
            # Build the command
            command = strategy.command.copy()
            command.append(package_spec)
            
            logger.debug(f"Executing: {' '.join(command)}")
            
            # Execute installation
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                # Verify installation
                if self._verify_installation(dependency):
                    version = self._get_installed_version(dependency)
                    return InstallationResult(
                        dependency=dependency,
                        strategy=strategy.name,
                        success=True,
                        version=version
                    )
                else:
                    return InstallationResult(
                        dependency=dependency,
                        strategy=strategy.name,
                        success=False,
                        error="Installation completed but verification failed"
                    )
            else:
                return InstallationResult(
                    dependency=dependency,
                    strategy=strategy.name,
                    success=False,
                    error=f"Installation failed: {result.stderr}"
                )
        
        except subprocess.TimeoutExpired:
            return InstallationResult(
                dependency=dependency,
                strategy=strategy.name,
                success=False,
                error="Installation timed out"
            )
        except Exception as e:
            return InstallationResult(
                dependency=dependency,
                strategy=strategy.name,
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _verify_installation(self, dependency: str) -> bool:
        """Verify that a dependency is properly installed"""
        try:
            # Map dependency names to import names
            import_mapping = {
                'scikit-learn': 'sklearn',
                'dbus-next': 'dbus_next',
                'lightgbm': 'lightgbm',
                'msgpack': 'msgpack',
                'zstandard': 'zstandard',
                'jeepney': 'jeepney',
                'psutil': 'psutil',
                'numpy': 'numpy',
                'numba': 'numba',
                'bottleneck': 'bottleneck',
                'numexpr': 'numexpr'
            }
            
            import_name = import_mapping.get(dependency, dependency)
            
            # Try to import the module
            result = subprocess.run(
                [sys.executable, '-c', f'import {import_name}; print("OK")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0 and 'OK' in result.stdout
            
        except Exception as e:
            logger.debug(f"Verification failed for {dependency}: {e}")
            return False
    
    def _get_installed_version(self, dependency: str) -> Optional[str]:
        """Get the version of an installed dependency"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', dependency],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':', 1)[1].strip()
            
            return None
            
        except Exception:
            return None
    
    def install_multiple_dependencies(self, dependencies: List[str], parallel: bool = True) -> Dict[str, InstallationResult]:
        """
        Install multiple dependencies with optional parallel execution
        """
        logger.info(f"Installing {len(dependencies)} dependencies (parallel={parallel})")
        
        results = {}
        
        if parallel and len(dependencies) > 1:
            # Use ThreadPoolExecutor for parallel installation
            with ThreadPoolExecutor(max_workers=min(4, len(dependencies))) as executor:
                futures = {
                    executor.submit(self.install_dependency, dep): dep
                    for dep in dependencies
                }
                
                for future in as_completed(futures):
                    dep = futures[future]
                    try:
                        result = future.result()
                        results[dep] = result
                    except Exception as e:
                        results[dep] = InstallationResult(
                            dependency=dep,
                            strategy="parallel_error",
                            success=False,
                            error=f"Parallel execution error: {str(e)}"
                        )
        else:
            # Sequential installation
            for dep in dependencies:
                results[dep] = self.install_dependency(dep)
        
        return results
    
    def create_dependency_health_check(self) -> Dict[str, Any]:
        """
        Create a comprehensive dependency health check
        """
        logger.info("Running dependency health check...")
        
        # Critical dependencies for ML Shader Prediction
        critical_dependencies = [
            'numpy', 'scikit-learn', 'lightgbm', 'psutil'
        ]
        
        # Optional but recommended dependencies
        optional_dependencies = [
            'jeepney', 'dbus-next', 'msgpack', 'zstandard', 
            'numba', 'bottleneck', 'numexpr'
        ]
        
        health_report = {
            'system_info': self.system_info,
            'critical_status': {},
            'optional_status': {},
            'overall_health': 0.0,
            'recommendations': [],
            'installation_summary': {
                'total_attempted': 0,
                'successful': 0,
                'failed': 0,
                'using_fallbacks': 0
            }
        }
        
        # Check critical dependencies
        critical_available = 0
        for dep in critical_dependencies:
            available = self._verify_installation(dep)
            version = self._get_installed_version(dep) if available else None
            
            health_report['critical_status'][dep] = {
                'available': available,
                'version': version,
                'status': 'healthy' if available else 'missing'
            }
            
            if available:
                critical_available += 1
        
        # Check optional dependencies
        optional_available = 0
        for dep in optional_dependencies:
            available = self._verify_installation(dep)
            version = self._get_installed_version(dep) if available else None
            
            health_report['optional_status'][dep] = {
                'available': available,
                'version': version,
                'status': 'healthy' if available else 'missing'
            }
            
            if available:
                optional_available += 1
        
        # Calculate overall health
        critical_health = critical_available / len(critical_dependencies)
        optional_health = optional_available / len(optional_dependencies)
        health_report['overall_health'] = (0.7 * critical_health) + (0.3 * optional_health)
        
        # Generate recommendations
        if critical_health < 1.0:
            missing_critical = [
                dep for dep in critical_dependencies 
                if not health_report['critical_status'][dep]['available']
            ]
            health_report['recommendations'].append({
                'type': 'critical',
                'message': f"Install missing critical dependencies: {', '.join(missing_critical)}",
                'dependencies': missing_critical
            })
        
        if optional_health < 0.5:
            missing_optional = [
                dep for dep in optional_dependencies 
                if not health_report['optional_status'][dep]['available']
            ]
            health_report['recommendations'].append({
                'type': 'performance',
                'message': f"Consider installing optional dependencies for better performance: {', '.join(missing_optional[:3])}",
                'dependencies': missing_optional[:3]  # Top 3 recommendations
            })
        
        # Steam Deck specific recommendations
        if self.is_steam_deck:
            if not health_report['optional_status'].get('jeepney', {}).get('available', False):
                health_report['recommendations'].append({
                    'type': 'steam_deck',
                    'message': "Install jeepney for better Steam integration on Steam Deck",
                    'dependencies': ['jeepney']
                })
        
        logger.info(f"Health check complete. Overall health: {health_report['overall_health']:.1%}")
        
        return health_report
    
    def auto_install_missing_dependencies(self, include_optional: bool = True) -> Dict[str, Any]:
        """
        Automatically install missing dependencies based on health check
        """
        logger.info("Starting automatic dependency installation...")
        
        health_check = self.create_dependency_health_check()
        dependencies_to_install = []
        
        # Always install missing critical dependencies
        for dep, status in health_check['critical_status'].items():
            if not status['available']:
                dependencies_to_install.append(dep)
        
        # Optionally install missing optional dependencies
        if include_optional:
            for dep, status in health_check['optional_status'].items():
                if not status['available']:
                    dependencies_to_install.append(dep)
        
        if not dependencies_to_install:
            logger.info("All dependencies are already installed")
            return {
                'status': 'complete',
                'message': 'All dependencies already available',
                'installed': {},
                'health_improvement': 0.0
            }
        
        logger.info(f"Installing {len(dependencies_to_install)} missing dependencies...")
        
        # Install dependencies
        installation_results = self.install_multiple_dependencies(dependencies_to_install)
        
        # Check health improvement
        new_health_check = self.create_dependency_health_check()
        health_improvement = new_health_check['overall_health'] - health_check['overall_health']
        
        # Summarize results
        successful_installations = {
            dep: result for dep, result in installation_results.items() 
            if result.success
        }
        
        failed_installations = {
            dep: result for dep, result in installation_results.items() 
            if not result.success
        }
        
        result = {
            'status': 'complete' if not failed_installations else 'partial',
            'installed': successful_installations,
            'failed': failed_installations,
            'health_improvement': health_improvement,
            'new_health_score': new_health_check['overall_health'],
            'recommendations': new_health_check['recommendations']
        }
        
        logger.info(f"Installation complete. Health improved by {health_improvement:.1%}")
        
        return result
    
    def export_installation_report(self, path: Path) -> None:
        """Export detailed installation report"""
        report = {
            'system_info': self.system_info,
            'installation_cache': {
                key: {
                    'dependency': result.dependency,
                    'strategy': result.strategy,
                    'success': result.success,
                    'version': result.version,
                    'error': result.error,
                    'install_time': result.install_time
                }
                for key, result in self.installation_cache.items()
            },
            'failed_installations': self.failed_installations,
            'health_check': self.create_dependency_health_check(),
            'timestamp': time.time()
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Installation report exported to {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_install_jeepney() -> bool:
    """Quick installation of jeepney specifically"""
    installer = SteamDeckDependencyInstaller()
    result = installer.install_dependency('jeepney')
    
    if result.success:
        print(f"✅ jeepney {result.version} installed successfully using {result.strategy}")
        return True
    else:
        print(f"❌ Failed to install jeepney: {result.error}")
        return False

def install_all_ml_dependencies() -> Dict[str, bool]:
    """Install all ML dependencies for the shader prediction system"""
    installer = SteamDeckDependencyInstaller()
    
    dependencies = [
        'numpy', 'scikit-learn', 'lightgbm', 'psutil', 
        'jeepney', 'dbus-next', 'msgpack', 'zstandard',
        'numba', 'bottleneck', 'numexpr'
    ]
    
    results = installer.install_multiple_dependencies(dependencies)
    
    # Return simplified success status
    return {dep: result.success for dep, result in results.items()}

def create_installation_summary() -> Dict[str, Any]:
    """Create a summary of current installation status"""
    installer = SteamDeckDependencyInstaller()
    return installer.create_dependency_health_check()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n🔧 Enhanced Dependency Installer for ML Shader Prediction Compiler")
    print("=" * 70)
    
    installer = SteamDeckDependencyInstaller()
    
    print(f"\n🖥️  System Information:")
    print(f"   Platform: {installer.system_info['platform']} {installer.system_info['machine']}")
    print(f"   Python: {installer.system_info['python_version']}")
    print(f"   Steam Deck: {installer.system_info['is_steam_deck']}")
    print(f"   Network: {'Available' if installer.system_info['has_network'] else 'Unavailable'}")
    print(f"   Available Space: {installer.system_info['available_space_gb']:.1f} GB")
    
    # Run health check
    print(f"\n🔍 Running dependency health check...")
    health_check = installer.create_dependency_health_check()
    
    print(f"\n📊 Health Report:")
    print(f"   Overall Health: {health_check['overall_health']:.1%}")
    
    print(f"\n   Critical Dependencies:")
    for dep, status in health_check['critical_status'].items():
        emoji = "✅" if status['available'] else "❌"
        version = f" v{status['version']}" if status['version'] else ""
        print(f"     {emoji} {dep}{version}")
    
    print(f"\n   Optional Dependencies:")
    for dep, status in health_check['optional_status'].items():
        emoji = "✅" if status['available'] else "⚠️"
        version = f" v{status['version']}" if status['version'] else ""
        print(f"     {emoji} {dep}{version}")
    
    # Show recommendations
    if health_check['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in health_check['recommendations']:
            print(f"   {rec['type'].upper()}: {rec['message']}")
    
    # Auto-install missing dependencies
    if health_check['overall_health'] < 1.0:
        print(f"\n🚀 Auto-installing missing dependencies...")
        auto_result = installer.auto_install_missing_dependencies()
        
        print(f"\n📈 Installation Results:")
        print(f"   Status: {auto_result['status']}")
        print(f"   Health Improvement: +{auto_result['health_improvement']:.1%}")
        print(f"   New Health Score: {auto_result['new_health_score']:.1%}")
        
        if auto_result['installed']:
            print(f"\n   Successfully Installed:")
            for dep, result in auto_result['installed'].items():
                print(f"     ✅ {dep} v{result.version} ({result.strategy})")
        
        if auto_result['failed']:
            print(f"\n   Failed Installations:")
            for dep, result in auto_result['failed'].items():
                print(f"     ❌ {dep}: {result.error}")
    
    # Export report
    report_path = Path("/tmp/dependency_installation_report.json")
    installer.export_installation_report(report_path)
    print(f"\n💾 Detailed report exported to {report_path}")
    
    print(f"\n✅ Dependency installer completed successfully!")