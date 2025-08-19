#!/usr/bin/env python3
"""
Comprehensive Dependency Version Manager and Compatibility Matrix

This module provides advanced dependency version checking, compatibility verification,
and fallback coordination for the Steam Deck shader prediction system.

Features:
- Version compatibility matrix with detailed constraints
- Platform-specific dependency validation
- Steam Deck hardware detection and optimization
- Graceful fallback chain management
- Performance-based dependency selection
- Thread-safe version checking with timeout handling
- Comprehensive error recovery and reporting
"""

import os
import sys
import time
import json
import logging
import platform
import threading
import subprocess
import importlib
from typing import Dict, List, Any, Optional, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import lru_cache, wraps
from contextlib import contextmanager
from packaging import version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# COMPATIBILITY MATRIX DEFINITIONS
# =============================================================================

class VersionConstraint(NamedTuple):
    """Version constraint specification"""
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    excluded_versions: List[str] = []
    preferred_version: Optional[str] = None
    platform_specific: Dict[str, str] = {}

class HardwareCompatibility(NamedTuple):
    """Hardware compatibility specification"""
    min_memory_mb: int = 512
    min_cpu_cores: int = 1
    gpu_required: bool = False
    steam_deck_compatible: bool = True
    arm_compatible: bool = True
    compilation_required: bool = False

class DependencyRisk(NamedTuple):
    """Dependency risk assessment"""
    compilation_risk: int = 0  # 0-10 scale
    breaking_changes_risk: int = 0
    maintenance_risk: int = 0
    performance_impact: float = 1.0
    memory_footprint_mb: int = 10

@dataclass
class DependencyProfile:
    """Complete dependency profile with all compatibility information"""
    name: str
    import_names: List[str]
    version_constraint: VersionConstraint
    hardware_compatibility: HardwareCompatibility
    risk_assessment: DependencyRisk
    category: str = "optional"
    priority: int = 5  # 1-10
    fallback_available: bool = True
    test_commands: List[str] = field(default_factory=list)
    install_commands: Dict[str, List[str]] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)

# =============================================================================
# COMPREHENSIVE DEPENDENCY COMPATIBILITY MATRIX
# =============================================================================

DEPENDENCY_MATRIX = {
    # Core ML Dependencies
    'numpy': DependencyProfile(
        name='numpy',
        import_names=['numpy'],
        version_constraint=VersionConstraint(
            min_version='1.19.0',
            max_version='2.0.0',
            excluded_versions=['1.20.0', '1.20.1'],  # Known Steam Deck issues
            preferred_version='1.24.3',
            platform_specific={
                'steam_deck': '1.24.3',
                'aarch64': '1.21.0'
            }
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=256,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=False
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=2,
            breaking_changes_risk=3,
            maintenance_risk=1,
            performance_impact=5.0,
            memory_footprint_mb=50
        ),
        category='ml_core',
        priority=9,
        fallback_available=True,
        test_commands=['python -c "import numpy; numpy.array([1,2,3]).mean()"'],
        install_commands={
            'pip': ['pip', 'install', '--no-compile', 'numpy>=1.19.0,<2.0.0'],
            'conda': ['conda', 'install', '-y', 'numpy'],
            'steam_deck': ['pip', 'install', '--user', '--no-compile', 'numpy==1.24.3']
        }
    ),

    'scikit-learn': DependencyProfile(
        name='scikit-learn',
        import_names=['sklearn'],
        version_constraint=VersionConstraint(
            min_version='1.0.0',
            max_version='1.5.0',
            excluded_versions=[],
            preferred_version='1.3.0',
            platform_specific={
                'steam_deck': '1.3.0'
            }
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=512,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=False
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=3,
            breaking_changes_risk=4,
            maintenance_risk=2,
            performance_impact=4.0,
            memory_footprint_mb=120
        ),
        category='ml_core',
        priority=8,
        fallback_available=True,
        test_commands=['python -c "from sklearn.ensemble import RandomForestRegressor"'],
        install_commands={
            'pip': ['pip', 'install', '--no-compile', 'scikit-learn>=1.0.0,<1.5.0'],
            'steam_deck': ['pip', 'install', '--user', '--no-compile', 'scikit-learn==1.3.0']
        }
    ),

    'lightgbm': DependencyProfile(
        name='lightgbm',
        import_names=['lightgbm'],
        version_constraint=VersionConstraint(
            min_version='3.0.0',
            max_version='4.5.0',
            excluded_versions=['4.0.0', '4.1.0'],  # Known issues
            preferred_version='4.3.0',
            platform_specific={
                'steam_deck': '4.3.0',
                'aarch64': None  # Skip on ARM due to compilation complexity
            }
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=256,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=False,  # Compilation issues on ARM
            compilation_required=True
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=7,
            breaking_changes_risk=5,
            maintenance_risk=3,
            performance_impact=6.0,
            memory_footprint_mb=80
        ),
        category='ml_advanced',
        priority=7,
        fallback_available=True,
        test_commands=['python -c "import lightgbm; lightgbm.LGBMRegressor()"'],
        install_commands={
            'pip': ['pip', 'install', '--no-compile', 'lightgbm>=3.0.0,<4.5.0'],
            'steam_deck': ['pip', 'install', '--user', 'lightgbm==4.3.0']
        }
    ),

    # Performance Dependencies
    'numba': DependencyProfile(
        name='numba',
        import_names=['numba'],
        version_constraint=VersionConstraint(
            min_version='0.56.0',
            max_version='0.60.0',
            excluded_versions=['0.57.0'],  # Performance regression
            preferred_version='0.59.1',
            platform_specific={
                'steam_deck': '0.59.1',
                'python3.13': None  # Not yet compatible with Python 3.13
            }
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=512,
            min_cpu_cores=2,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=True
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=8,
            breaking_changes_risk=6,
            maintenance_risk=4,
            performance_impact=10.0,
            memory_footprint_mb=200
        ),
        category='performance',
        priority=8,
        fallback_available=True,
        test_commands=['python -c "from numba import njit; njit(lambda x: x+1)(5)"'],
        install_commands={
            'pip': ['pip', 'install', 'numba>=0.56.0,<0.60.0'],
            'steam_deck': ['pip', 'install', '--user', 'numba==0.59.1']
        },
        environment_variables={
            'NUMBA_THREADING_LAYER': 'workqueue',
            'NUMBA_NUM_THREADS': '4'
        }
    ),

    'numexpr': DependencyProfile(
        name='numexpr',
        import_names=['numexpr'],
        version_constraint=VersionConstraint(
            min_version='2.8.0',
            max_version='3.0.0',
            preferred_version='2.8.7'
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=128,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=False
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=3,
            breaking_changes_risk=2,
            maintenance_risk=2,
            performance_impact=3.0,
            memory_footprint_mb=30
        ),
        category='performance',
        priority=6,
        fallback_available=True,
        test_commands=['python -c "import numexpr; numexpr.evaluate(\'2*3\')"'],
        install_commands={
            'pip': ['pip', 'install', 'numexpr>=2.8.0,<3.0.0'],
            'steam_deck': ['pip', 'install', '--user', 'numexpr==2.8.7']
        }
    ),

    # System Dependencies
    'psutil': DependencyProfile(
        name='psutil',
        import_names=['psutil'],
        version_constraint=VersionConstraint(
            min_version='5.8.0',
            max_version='6.0.0',
            preferred_version='5.9.5'
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=64,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=False
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=2,
            breaking_changes_risk=2,
            maintenance_risk=1,
            performance_impact=2.0,
            memory_footprint_mb=20
        ),
        category='system',
        priority=7,
        fallback_available=True,
        test_commands=['python -c "import psutil; psutil.cpu_count()"'],
        install_commands={
            'pip': ['pip', 'install', 'psutil>=5.8.0,<6.0.0'],
            'steam_deck': ['pip', 'install', '--user', 'psutil==5.9.5']
        }
    ),

    # Serialization Dependencies
    'msgpack': DependencyProfile(
        name='msgpack',
        import_names=['msgpack'],
        version_constraint=VersionConstraint(
            min_version='1.0.0',
            max_version='2.0.0',
            preferred_version='1.0.7'
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=32,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=False
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=1,
            breaking_changes_risk=2,
            maintenance_risk=1,
            performance_impact=2.5,
            memory_footprint_mb=15
        ),
        category='serialization',
        priority=5,
        fallback_available=True,
        test_commands=['python -c "import msgpack; msgpack.packb({\'test\': 123})"'],
        install_commands={
            'pip': ['pip', 'install', 'msgpack>=1.0.0,<2.0.0'],
            'steam_deck': ['pip', 'install', '--user', 'msgpack==1.0.7']
        }
    ),

    'zstandard': DependencyProfile(
        name='zstandard',
        import_names=['zstandard'],
        version_constraint=VersionConstraint(
            min_version='0.20.0',
            max_version='0.23.0',
            preferred_version='0.22.0',
            platform_specific={
                'aarch64': None  # Skip on ARM64 - compilation issues
            }
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=128,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=False,
            compilation_required=True
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=6,
            breaking_changes_risk=3,
            maintenance_risk=2,
            performance_impact=4.0,
            memory_footprint_mb=40
        ),
        category='compression',
        priority=4,
        fallback_available=True,
        test_commands=['python -c "import zstandard; zstandard.ZstdCompressor()"'],
        install_commands={
            'pip': ['pip', 'install', 'zstandard>=0.20.0,<0.23.0'],
            'steam_deck': ['pip', 'install', '--user', 'zstandard==0.22.0']
        }
    ),

    # D-Bus Dependencies (Linux-specific)
    'dbus-next': DependencyProfile(
        name='dbus-next',
        import_names=['dbus_next'],
        version_constraint=VersionConstraint(
            min_version='0.2.0',
            max_version='1.0.0',
            preferred_version='0.2.3',
            platform_specific={
                'windows': None,
                'darwin': None
            }
        ),
        hardware_compatibility=HardwareCompatibility(
            min_memory_mb=32,
            min_cpu_cores=1,
            steam_deck_compatible=True,
            arm_compatible=True,
            compilation_required=False
        ),
        risk_assessment=DependencyRisk(
            compilation_risk=1,
            breaking_changes_risk=3,
            maintenance_risk=3,
            performance_impact=1.5,
            memory_footprint_mb=25
        ),
        category='system_integration',
        priority=3,
        fallback_available=True,
        test_commands=['python -c "import dbus_next; dbus_next.BaseInterface"'],
        install_commands={
            'pip': ['pip', 'install', 'dbus-next>=0.2.0,<1.0.0'],
            'steam_deck': ['pip', 'install', '--user', 'dbus-next==0.2.3']
        }
    )
}

# =============================================================================
# DEPENDENCY VERSION MANAGER
# =============================================================================

class DependencyVersionManager:
    """
    Advanced dependency version manager with comprehensive compatibility checking
    """

    def __init__(self):
        self.system_info = self._detect_system_info()
        self.compatibility_cache: Dict[str, Dict[str, Any]] = {}
        self.version_cache: Dict[str, str] = {}
        self.detection_lock = threading.RLock()
        self.failed_dependencies: Dict[str, str] = {}
        self.successful_dependencies: Dict[str, str] = {}
        
        logger.info(f"DependencyVersionManager initialized for {self.system_info}")

    def _detect_system_info(self) -> Dict[str, Any]:
        """Detect comprehensive system information"""
        system_info = {
            'platform': platform.system().lower(),
            'machine': platform.machine().lower(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'cpu_count': os.cpu_count() or 1,
            'is_steam_deck': self._is_steam_deck(),
            'memory_gb': self._estimate_memory_gb(),
            'is_arm': platform.machine().lower() in ['aarch64', 'arm64', 'armv7l'],
            'package_managers': self._detect_package_managers()
        }
        
        return system_info

    def _is_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        indicators = [
            lambda: 'jupiter' in platform.node().lower(),
            lambda: 'galileo' in platform.node().lower(),
            lambda: os.path.exists('/home/deck'),
            lambda: os.path.exists('/sys/devices/virtual/dmi/id/product_name') and 
                   'steam deck' in open('/sys/devices/virtual/dmi/id/product_name').read().lower()
        ]
        
        return any(self._safe_check(check) for check in indicators)

    def _safe_check(self, check_func) -> bool:
        """Safely execute a check function"""
        try:
            return check_func()
        except Exception:
            return False

    def _estimate_memory_gb(self) -> int:
        """Estimate total system memory in GB"""
        try:
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            gb = max(1, int(kb / 1024 / 1024))
                            return gb
        except Exception:
            pass
        
        # Default assumption for Steam Deck
        return 16 if self.system_info.get('is_steam_deck', False) else 8

    def _detect_package_managers(self) -> List[str]:
        """Detect available package managers"""
        managers = []
        commands = {
            'pip': ['pip', '--version'],
            'pip3': ['pip3', '--version'],
            'conda': ['conda', '--version'],
            'mamba': ['mamba', '--version']
        }
        
        for manager, cmd in commands.items():
            try:
                subprocess.run(cmd, capture_output=True, timeout=5, check=True)
                managers.append(manager)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        return managers

    @lru_cache(maxsize=256)
    def check_version_compatibility(self, dependency_name: str, installed_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Check version compatibility for a dependency
        
        Returns comprehensive compatibility report
        """
        if dependency_name not in DEPENDENCY_MATRIX:
            return {
                'compatible': False,
                'error': f'Unknown dependency: {dependency_name}',
                'recommendations': []
            }

        profile = DEPENDENCY_MATRIX[dependency_name]
        result = {
            'dependency': dependency_name,
            'compatible': True,
            'installed_version': installed_version,
            'preferred_version': profile.version_constraint.preferred_version,
            'platform_compatible': True,
            'hardware_compatible': True,
            'risk_assessment': profile.risk_assessment._asdict(),
            'recommendations': [],
            'warnings': [],
            'install_command': None,
            'fallback_available': profile.fallback_available
        }

        # Check platform compatibility
        platform_check = self._check_platform_compatibility(profile)
        if not platform_check['compatible']:
            result['compatible'] = False
            result['platform_compatible'] = False
            result['warnings'].extend(platform_check['warnings'])
            result['recommendations'].extend(platform_check['recommendations'])

        # Check hardware compatibility
        hardware_check = self._check_hardware_compatibility(profile)
        if not hardware_check['compatible']:
            result['compatible'] = False
            result['hardware_compatible'] = False
            result['warnings'].extend(hardware_check['warnings'])
            result['recommendations'].extend(hardware_check['recommendations'])

        # Check version constraints if installed
        if installed_version and result['compatible']:
            version_check = self._check_version_constraints(profile, installed_version)
            result.update(version_check)

        # Add installation recommendation
        if result['compatible'] or result['fallback_available']:
            install_cmd = self._get_install_command(profile)
            if install_cmd:
                result['install_command'] = install_cmd

        return result

    def _check_platform_compatibility(self, profile: DependencyProfile) -> Dict[str, Any]:
        """Check if dependency is compatible with current platform"""
        result = {'compatible': True, 'warnings': [], 'recommendations': []}
        
        platform_constraints = profile.version_constraint.platform_specific
        current_platform = self.system_info['platform']
        
        # Check for platform exclusions
        if current_platform in platform_constraints:
            preferred_version = platform_constraints[current_platform]
            if preferred_version is None:
                result['compatible'] = False
                result['warnings'].append(f'Not compatible with {current_platform}')
                if profile.fallback_available:
                    result['recommendations'].append('Use pure Python fallback implementation')
                return result

        # Check ARM compatibility
        if self.system_info['is_arm'] and not profile.hardware_compatibility.arm_compatible:
            result['compatible'] = False
            result['warnings'].append('Not compatible with ARM architecture')
            result['recommendations'].append('Use alternative dependency or fallback')

        # Check Steam Deck compatibility
        if self.system_info['is_steam_deck'] and not profile.hardware_compatibility.steam_deck_compatible:
            result['warnings'].append('May have issues on Steam Deck')
            result['recommendations'].append('Monitor for stability issues')

        # Check compilation requirements
        if profile.hardware_compatibility.compilation_required:
            if self.system_info['is_steam_deck']:
                result['warnings'].append('Requires compilation on Steam Deck (may be slow)')
                result['recommendations'].append('Use pre-compiled wheel if available')

        return result

    def _check_hardware_compatibility(self, profile: DependencyProfile) -> Dict[str, Any]:
        """Check hardware resource compatibility"""
        result = {'compatible': True, 'warnings': [], 'recommendations': []}
        
        hw_reqs = profile.hardware_compatibility
        
        # Check memory requirements
        required_memory_gb = hw_reqs.min_memory_mb / 1024
        if self.system_info['memory_gb'] < required_memory_gb:
            result['compatible'] = False
            result['warnings'].append(
                f'Insufficient memory: requires {required_memory_gb:.1f}GB, '
                f'have {self.system_info["memory_gb"]}GB'
            )
            result['recommendations'].append('Free memory or use lighter alternatives')

        # Check CPU requirements
        if self.system_info['cpu_count'] < hw_reqs.min_cpu_cores:
            result['warnings'].append(
                f'May run slowly with {self.system_info["cpu_count"]} cores '
                f'(recommended: {hw_reqs.min_cpu_cores})'
            )

        return result

    def _check_version_constraints(self, profile: DependencyProfile, installed_version: str) -> Dict[str, Any]:
        """Check if installed version meets constraints"""
        result = {'version_compatible': True, 'version_warnings': [], 'version_recommendations': []}
        
        constraints = profile.version_constraint
        
        try:
            installed_ver = version.Version(installed_version)
            
            # Check minimum version
            if constraints.min_version:
                min_ver = version.Version(constraints.min_version)
                if installed_ver < min_ver:
                    result['version_compatible'] = False
                    result['version_warnings'].append(
                        f'Version {installed_version} below minimum {constraints.min_version}'
                    )
                    result['version_recommendations'].append(f'Upgrade to >= {constraints.min_version}')

            # Check maximum version
            if constraints.max_version:
                max_ver = version.Version(constraints.max_version)
                if installed_ver >= max_ver:
                    result['version_compatible'] = False
                    result['version_warnings'].append(
                        f'Version {installed_version} at or above maximum {constraints.max_version}'
                    )
                    result['version_recommendations'].append(f'Downgrade to < {constraints.max_version}')

            # Check excluded versions
            if installed_version in constraints.excluded_versions:
                result['version_compatible'] = False
                result['version_warnings'].append(f'Version {installed_version} is known to have issues')
                result['version_recommendations'].append(
                    f'Use preferred version {constraints.preferred_version} instead'
                )

            # Recommend preferred version if different
            if (constraints.preferred_version and 
                installed_version != constraints.preferred_version and
                result['version_compatible']):
                result['version_recommendations'].append(
                    f'Consider upgrading to preferred version {constraints.preferred_version}'
                )

        except Exception as e:
            result['version_warnings'].append(f'Could not parse version {installed_version}: {e}')

        return result

    def _get_install_command(self, profile: DependencyProfile) -> Optional[List[str]]:
        """Get appropriate install command for current environment"""
        install_commands = profile.install_commands
        
        # Priority order for Steam Deck
        if self.system_info['is_steam_deck'] and 'steam_deck' in install_commands:
            return install_commands['steam_deck']
        
        # Check available package managers
        for manager in self.system_info['package_managers']:
            if manager in install_commands:
                return install_commands[manager]
        
        # Fallback to pip if available
        if 'pip' in install_commands:
            return install_commands['pip']
        
        return None

    def detect_installed_version(self, dependency_name: str, timeout: float = 10.0) -> Optional[str]:
        """
        Detect installed version of a dependency with timeout
        """
        if dependency_name in self.version_cache:
            return self.version_cache[dependency_name]

        if dependency_name not in DEPENDENCY_MATRIX:
            return None

        profile = DEPENDENCY_MATRIX[dependency_name]
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._detect_version_sync, profile)
            try:
                version_str = future.result(timeout=timeout)
                if version_str:
                    self.version_cache[dependency_name] = version_str
                return version_str
            except TimeoutError:
                logger.warning(f"Version detection for {dependency_name} timed out after {timeout}s")
                return None
            except Exception as e:
                logger.error(f"Error detecting version for {dependency_name}: {e}")
                return None

    def _detect_version_sync(self, profile: DependencyProfile) -> Optional[str]:
        """Synchronously detect dependency version"""
        for import_name in profile.import_names:
            try:
                module = importlib.import_module(import_name)
                
                # Try common version attributes
                for attr in ['__version__', 'version', 'VERSION']:
                    if hasattr(module, attr):
                        version_obj = getattr(module, attr)
                        if isinstance(version_obj, str):
                            return version_obj
                        elif hasattr(version_obj, '__str__'):
                            return str(version_obj)
                
                # Try version info tuple
                if hasattr(module, 'version_info'):
                    version_info = module.version_info
                    if isinstance(version_info, tuple):
                        return '.'.join(map(str, version_info))

            except ImportError:
                continue
            except Exception as e:
                logger.debug(f"Error getting version from {import_name}: {e}")
                continue
        
        return None

    def get_dependency_recommendations(self, include_optional: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive dependency recommendations for current environment
        """
        recommendations = {
            'critical': [],
            'recommended': [],
            'optional': [],
            'avoid': [],
            'system_optimization': [],
            'installation_order': []
        }

        # Analyze all dependencies
        dependency_analysis = {}
        for dep_name, profile in DEPENDENCY_MATRIX.items():
            installed_version = self.detect_installed_version(dep_name)
            compatibility = self.check_version_compatibility(dep_name, installed_version)
            
            dependency_analysis[dep_name] = {
                'profile': profile,
                'installed_version': installed_version,
                'compatibility': compatibility,
                'is_installed': installed_version is not None
            }

        # Generate recommendations based on analysis
        for dep_name, analysis in dependency_analysis.items():
            profile = analysis['profile']
            compatibility = analysis['compatibility']
            
            if not compatibility['platform_compatible']:
                recommendations['avoid'].append({
                    'dependency': dep_name,
                    'reason': 'Platform incompatible',
                    'details': compatibility['warnings']
                })
                continue

            if not analysis['is_installed']:
                if profile.category in ['ml_core', 'system'] and profile.priority >= 7:
                    recommendations['critical'].append({
                        'dependency': dep_name,
                        'reason': 'High-priority missing dependency',
                        'install_command': compatibility.get('install_command'),
                        'performance_benefit': profile.risk_assessment.performance_impact
                    })
                elif profile.priority >= 5:
                    recommendations['recommended'].append({
                        'dependency': dep_name,
                        'reason': 'Significant performance benefit',
                        'install_command': compatibility.get('install_command'),
                        'performance_benefit': profile.risk_assessment.performance_impact
                    })
                elif include_optional:
                    recommendations['optional'].append({
                        'dependency': dep_name,
                        'reason': 'Minor performance benefit',
                        'install_command': compatibility.get('install_command'),
                        'performance_benefit': profile.risk_assessment.performance_impact
                    })

            elif not compatibility['compatible']:
                if compatibility['version_compatible'] is False:
                    recommendations['critical'].append({
                        'dependency': dep_name,
                        'reason': 'Version compatibility issue',
                        'current_version': analysis['installed_version'],
                        'recommended_action': compatibility['recommendations']
                    })

        # Generate system-specific optimizations
        if self.system_info['is_steam_deck']:
            recommendations['system_optimization'].extend([
                'Use --user flag for pip installations to avoid permission issues',
                'Consider using --no-compile flag to speed up installation',
                'Monitor thermal state during heavy ML operations'
            ])

        if self.system_info['memory_gb'] <= 8:
            recommendations['system_optimization'].append(
                'Consider using memory-efficient alternatives for large datasets'
            )

        # Generate installation order based on dependencies and priorities
        installable_deps = []
        for dep_name, analysis in dependency_analysis.items():
            if (not analysis['is_installed'] and 
                analysis['compatibility']['platform_compatible'] and
                analysis['profile'].priority >= 5):
                installable_deps.append((dep_name, analysis['profile']))

        # Sort by priority and risk (install safer dependencies first)
        installable_deps.sort(key=lambda x: (
            -x[1].priority,  # Higher priority first
            x[1].risk_assessment.compilation_risk  # Lower risk first
        ))

        recommendations['installation_order'] = [dep[0] for dep in installable_deps]

        return recommendations

    def create_compatibility_report(self) -> Dict[str, Any]:
        """Create comprehensive compatibility report"""
        report = {
            'system_info': self.system_info,
            'dependency_status': {},
            'compatibility_summary': {
                'total': len(DEPENDENCY_MATRIX),
                'compatible': 0,
                'installed': 0,
                'platform_incompatible': 0,
                'version_issues': 0
            },
            'recommendations': self.get_dependency_recommendations(include_optional=True),
            'risk_assessment': {
                'high_risk': [],
                'medium_risk': [],
                'low_risk': []
            },
            'timestamp': time.time()
        }

        # Analyze each dependency
        for dep_name in DEPENDENCY_MATRIX.keys():
            installed_version = self.detect_installed_version(dep_name)
            compatibility = self.check_version_compatibility(dep_name, installed_version)
            
            status = {
                'installed_version': installed_version,
                'compatibility': compatibility,
                'profile': DEPENDENCY_MATRIX[dep_name]._asdict()
            }
            
            report['dependency_status'][dep_name] = status
            
            # Update summary counters
            if compatibility['compatible']:
                report['compatibility_summary']['compatible'] += 1
            if installed_version:
                report['compatibility_summary']['installed'] += 1
            if not compatibility['platform_compatible']:
                report['compatibility_summary']['platform_incompatible'] += 1
            if compatibility.get('version_compatible') is False:
                report['compatibility_summary']['version_issues'] += 1

            # Categorize by risk
            profile = DEPENDENCY_MATRIX[dep_name]
            total_risk = (
                profile.risk_assessment.compilation_risk +
                profile.risk_assessment.breaking_changes_risk +
                profile.risk_assessment.maintenance_risk
            )
            
            if total_risk >= 15:
                report['risk_assessment']['high_risk'].append(dep_name)
            elif total_risk >= 8:
                report['risk_assessment']['medium_risk'].append(dep_name)
            else:
                report['risk_assessment']['low_risk'].append(dep_name)

        return report

    def validate_environment(self) -> Dict[str, Any]:
        """Validate current environment for ML workloads"""
        validation_result = {
            'overall_health': 0.0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'environment_score': {
                'platform_compatibility': 0.0,
                'resource_adequacy': 0.0,
                'dependency_health': 0.0,
                'performance_potential': 0.0
            }
        }

        compatibility_report = self.create_compatibility_report()
        
        # Calculate platform compatibility score
        total_deps = compatibility_report['compatibility_summary']['total']
        compatible_deps = compatibility_report['compatibility_summary']['compatible']
        platform_score = compatible_deps / max(total_deps, 1)
        validation_result['environment_score']['platform_compatibility'] = platform_score

        # Calculate resource adequacy
        resource_score = 1.0
        if self.system_info['memory_gb'] < 4:
            resource_score *= 0.7
            validation_result['warnings'].append('Low memory may impact performance')
        if self.system_info['cpu_count'] < 2:
            resource_score *= 0.8
            validation_result['warnings'].append('Single CPU core may limit parallel processing')
        validation_result['environment_score']['resource_adequacy'] = resource_score

        # Calculate dependency health
        installed_deps = compatibility_report['compatibility_summary']['installed']
        dependency_health = installed_deps / max(total_deps, 1) * 0.7  # Not all deps are critical
        validation_result['environment_score']['dependency_health'] = dependency_health

        # Calculate performance potential
        high_impact_available = 0
        high_impact_total = 0
        
        for dep_name, profile in DEPENDENCY_MATRIX.items():
            if profile.risk_assessment.performance_impact >= 4.0:
                high_impact_total += 1
                installed_version = self.detect_installed_version(dep_name)
                if installed_version:
                    high_impact_available += 1
        
        performance_potential = high_impact_available / max(high_impact_total, 1)
        validation_result['environment_score']['performance_potential'] = performance_potential

        # Calculate overall health
        scores = validation_result['environment_score'].values()
        validation_result['overall_health'] = sum(scores) / len(scores)

        # Generate critical issues and recommendations
        if platform_score < 0.5:
            validation_result['critical_issues'].append(
                'Many dependencies incompatible with current platform'
            )

        if resource_score < 0.7:
            validation_result['critical_issues'].append(
                'System resources may be insufficient for ML workloads'
            )

        if dependency_health < 0.3:
            validation_result['recommendations'].append(
                'Install key dependencies to improve system capability'
            )

        if performance_potential < 0.5:
            validation_result['recommendations'].append(
                'Install high-performance dependencies for better ML performance'
            )

        # Steam Deck specific recommendations
        if self.system_info['is_steam_deck']:
            validation_result['recommendations'].extend([
                'Use Steam Deck optimized dependency versions when available',
                'Monitor thermal state during intensive operations',
                'Consider battery impact when selecting performance dependencies'
            ])

        return validation_result

    def export_compatibility_matrix(self, filepath: Path) -> None:
        """Export compatibility matrix and current status to file"""
        export_data = {
            'system_info': self.system_info,
            'compatibility_matrix': {
                name: profile._asdict() for name, profile in DEPENDENCY_MATRIX.items()
            },
            'current_status': self.create_compatibility_report(),
            'validation': self.validate_environment(),
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Compatibility matrix exported to {filepath}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_version_manager: Optional[DependencyVersionManager] = None

def get_version_manager() -> DependencyVersionManager:
    """Get or create global version manager instance"""
    global _version_manager
    if _version_manager is None:
        _version_manager = DependencyVersionManager()
    return _version_manager

def quick_compatibility_check(dependency: str) -> bool:
    """Quick compatibility check for a dependency"""
    manager = get_version_manager()
    installed_version = manager.detect_installed_version(dependency)
    compatibility = manager.check_version_compatibility(dependency, installed_version)
    return compatibility['compatible'] or compatibility['fallback_available']

def get_installation_recommendations() -> List[str]:
    """Get list of dependencies recommended for installation"""
    manager = get_version_manager()
    recommendations = manager.get_dependency_recommendations()
    
    install_list = []
    for rec_type in ['critical', 'recommended']:
        for rec in recommendations[rec_type]:
            install_list.append(rec['dependency'])
    
    return install_list

@contextmanager
def version_check_timeout(seconds: float = 30.0):
    """Context manager for version checking with timeout"""
    original_timeout = threading.current_thread()._timeout if hasattr(threading.current_thread(), '_timeout') else None
    threading.current_thread()._timeout = seconds
    try:
        yield
    finally:
        threading.current_thread()._timeout = original_timeout


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🔍 Dependency Version Manager Test Suite")
    print("=" * 55)
    
    # Initialize manager
    manager = DependencyVersionManager()
    
    print("\n🖥️  System Information:")
    for key, value in manager.system_info.items():
        print(f"  {key}: {value}")
    
    print("\n📦 Dependency Version Detection:")
    for dep_name in list(DEPENDENCY_MATRIX.keys())[:5]:  # Test first 5
        version_str = manager.detect_installed_version(dep_name)
        status = "✅" if version_str else "❌"
        print(f"  {status} {dep_name}: {version_str or 'Not installed'}")
    
    print("\n🔍 Compatibility Analysis:")
    for dep_name in ['numpy', 'scikit-learn', 'lightgbm']:
        if dep_name in DEPENDENCY_MATRIX:
            installed_version = manager.detect_installed_version(dep_name)
            compatibility = manager.check_version_compatibility(dep_name, installed_version)
            
            status = "✅" if compatibility['compatible'] else "🔄" if compatibility['fallback_available'] else "❌"
            print(f"  {status} {dep_name}: {'Compatible' if compatibility['compatible'] else 'Needs fallback' if compatibility['fallback_available'] else 'Incompatible'}")
            
            if compatibility['warnings']:
                for warning in compatibility['warnings']:
                    print(f"      ⚠️  {warning}")
    
    print("\n💡 Installation Recommendations:")
    recommendations = manager.get_dependency_recommendations()
    
    for rec_type, recs in recommendations.items():
        if recs and rec_type in ['critical', 'recommended']:
            print(f"  {rec_type.title()}:")
            for rec in recs[:3]:  # Show top 3
                print(f"    📦 {rec['dependency']}: {rec.get('reason', 'No reason provided')}")
    
    print("\n🏥 Environment Validation:")
    validation = manager.validate_environment()
    print(f"  Overall Health: {validation['overall_health']:.1%}")
    
    for score_name, score in validation['environment_score'].items():
        print(f"  {score_name.replace('_', ' ').title()}: {score:.1%}")
    
    if validation['critical_issues']:
        print("  Critical Issues:")
        for issue in validation['critical_issues']:
            print(f"    🚨 {issue}")
    
    # Export compatibility matrix
    export_path = Path("/tmp/dependency_compatibility_matrix.json")
    manager.export_compatibility_matrix(export_path)
    print(f"\n💾 Compatibility matrix exported to {export_path}")
    
    print(f"\n✅ Dependency Version Manager test completed!")
    print(f"🎯 System compatibility: {validation['overall_health']:.1%}")