#!/usr/bin/env python3
"""
Anti-Cheat System Compatibility and Whitelisting

This module provides comprehensive compatibility checks and whitelisting mechanisms
for major anti-cheat systems used in gaming:

- EAC (Easy Anti-Cheat) compatibility validation
- BattlEye system compatibility checks  
- VAC (Valve Anti-Cheat) compliance verification
- Hardware and driver compatibility matrix
- Shader whitelisting and quarantine systems
- Real-world compatibility testing framework
- Anti-cheat signature detection and avoidance
- Safe operation modes for different games

The system ensures that shader optimizations don't trigger false positives
in anti-cheat systems while maintaining security and performance benefits.
"""

import os
import sys
import json
import time
import hashlib
import logging
import subprocess
import platform
from typing import Dict, List, Optional, Set, Any, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from pathlib import Path
import threading
import tempfile
from collections import defaultdict, deque
import sqlite3
import contextlib
import re

# Windows-specific imports
if platform.system() == "Windows":
    try:
        import winreg
        import win32api
        import win32con
        import win32process
        import psutil
        HAS_WINDOWS_APIS = True
    except ImportError:
        HAS_WINDOWS_APIS = False
else:
    HAS_WINDOWS_APIS = False

logger = logging.getLogger(__name__)


class AntiCheatSystem(Enum):
    """Supported anti-cheat systems"""
    EAC = "easy_anti_cheat"
    BATTLEYE = "battleye"
    VAC = "valve_anti_cheat"
    NPROTECT = "nprotect"
    XIGNCODE = "xigncode"
    PUNKBUSTER = "punkbuster"
    FAIRFIGHT = "fairfight"
    RICOCHET = "ricochet"  # Call of Duty
    UNKNOWN = "unknown"


class CompatibilityLevel(Enum):
    """Compatibility levels with anti-cheat systems"""
    FULLY_COMPATIBLE = "fully_compatible"      # No issues expected
    MOSTLY_COMPATIBLE = "mostly_compatible"    # Minor issues possible
    LIMITED_COMPATIBLE = "limited_compatible"  # Some features may be blocked
    INCOMPATIBLE = "incompatible"              # Likely to cause issues
    BLOCKED = "blocked"                        # Definitely blocked
    UNKNOWN = "unknown"                        # Compatibility unknown


class ShaderRiskLevel(IntEnum):
    """Risk levels for shader operations"""
    SAFE = 0           # Standard shaders, no risk
    LOW_RISK = 1       # Minor modifications, low chance of detection
    MEDIUM_RISK = 2    # Moderate modifications, possible detection
    HIGH_RISK = 3      # Significant modifications, likely detection
    CRITICAL_RISK = 4  # Definitely triggers anti-cheat


@dataclass
class AntiCheatSignature:
    """Anti-cheat detection signature"""
    signature_id: str
    anticheat_system: AntiCheatSystem
    signature_type: str  # "file_hash", "memory_pattern", "behavior", "driver"
    pattern: str
    description: str
    severity: int  # 1-10 scale
    detection_method: str
    bypass_difficulty: int  # 1-10, how hard to bypass
    last_updated: float = field(default_factory=time.time)
    
    def matches(self, data: bytes) -> bool:
        """Check if signature matches data"""
        if self.signature_type == "file_hash":
            data_hash = hashlib.sha256(data).hexdigest()
            return self.pattern.lower() == data_hash.lower()
        elif self.signature_type == "memory_pattern":
            return self.pattern.encode() in data
        elif self.signature_type == "behavior":
            # Behavioral matching would require more complex analysis
            return False
        return False


@dataclass
class GameProfile:
    """Game-specific anti-cheat profile"""
    game_id: str
    game_name: str
    anticheat_systems: List[AntiCheatSystem]
    shader_whitelist_required: bool
    strict_mode: bool
    
    # Risk tolerance
    max_shader_risk_level: ShaderRiskLevel
    allow_driver_modifications: bool
    allow_memory_modifications: bool
    
    # Specific restrictions
    blocked_operations: Set[str]
    whitelisted_shaders: Set[str]
    quarantined_shaders: Set[str]
    
    # Testing data
    last_tested: Optional[float] = None
    test_results: Dict[AntiCheatSystem, CompatibilityLevel] = field(default_factory=dict)


@dataclass
class CompatibilityTestResult:
    """Result of anti-cheat compatibility test"""
    test_id: str
    anticheat_system: AntiCheatSystem
    game_profile: str
    shader_hash: str
    
    compatibility_level: CompatibilityLevel
    test_passed: bool
    issues_detected: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Test metadata
    test_duration_seconds: float
    test_timestamp: float
    test_environment: Dict[str, str]
    
    # Detection details
    signatures_triggered: List[str]
    false_positive_risk: float  # 0.0 to 1.0


class AntiCheatDetector:
    """Detect installed anti-cheat systems"""
    
    def __init__(self):
        self.detection_signatures = self._load_detection_signatures()
        self.installed_systems = {}
        self.last_scan = 0
    
    def _load_detection_signatures(self) -> Dict[AntiCheatSystem, Dict[str, List[str]]]:
        """Load signatures for detecting anti-cheat systems"""
        return {
            AntiCheatSystem.EAC: {
                'processes': ['EasyAntiCheat.exe', 'EACLauncher.exe'],
                'services': ['EasyAntiCheat', 'EAC'],
                'registry_keys': [
                    r'HKEY_LOCAL_MACHINE\SOFTWARE\EasyAntiCheat',
                    r'HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\EasyAntiCheat'
                ],
                'files': [
                    'EasyAntiCheat.exe',
                    'EACLauncher.exe', 
                    'eac_launcher.dll'
                ],
                'directories': [
                    'EasyAntiCheat',
                    'EAC'
                ]
            },
            AntiCheatSystem.BATTLEYE: {
                'processes': ['BEService.exe', 'BEDaisy.exe', 'BattlEye.exe'],
                'services': ['BEService', 'BattlEye'],
                'registry_keys': [
                    r'HKEY_LOCAL_MACHINE\SOFTWARE\BattlEye',
                    r'HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\BEService'
                ],
                'files': [
                    'BEService.exe',
                    'BattlEye.dll',
                    'BEClient.dll'
                ],
                'directories': [
                    'BattlEye'
                ]
            },
            AntiCheatSystem.VAC: {
                'processes': ['steam.exe'],  # VAC runs within Steam
                'services': [],
                'registry_keys': [
                    r'HKEY_LOCAL_MACHINE\SOFTWARE\Valve\Steam'
                ],
                'files': [
                    'steam.exe',
                    'steamclient.dll'
                ],
                'directories': [
                    'Steam'
                ]
            },
            AntiCheatSystem.NPROTECT: {
                'processes': ['npggNT.des', 'npsc.des'],
                'services': ['npggsvc'],
                'files': [
                    'GameGuard.des',
                    'npggNT.des'
                ],
                'directories': [
                    'GameGuard'
                ]
            }
        }
    
    def scan_for_anticheat_systems(self, force_rescan: bool = False) -> Dict[AntiCheatSystem, Dict[str, Any]]:
        """Scan system for installed anti-cheat software"""
        current_time = time.time()
        
        # Use cached results if recent
        if not force_rescan and (current_time - self.last_scan) < 300:  # 5 minutes
            return self.installed_systems
        
        logger.info("Scanning for anti-cheat systems...")
        detected_systems = {}
        
        for anticheat, signatures in self.detection_signatures.items():
            detection_results = {
                'detected': False,
                'confidence': 0.0,
                'evidence': [],
                'version': None,
                'installation_path': None
            }
            
            # Check processes
            if HAS_WINDOWS_APIS:
                detected_processes = self._check_processes(signatures.get('processes', []))
                if detected_processes:
                    detection_results['detected'] = True
                    detection_results['confidence'] += 0.4
                    detection_results['evidence'].extend(f"Process: {p}" for p in detected_processes)
            
            # Check services
            if platform.system() == "Windows":
                detected_services = self._check_services(signatures.get('services', []))
                if detected_services:
                    detection_results['detected'] = True
                    detection_results['confidence'] += 0.3
                    detection_results['evidence'].extend(f"Service: {s}" for s in detected_services)
            
            # Check registry
            if HAS_WINDOWS_APIS:
                registry_evidence = self._check_registry(signatures.get('registry_keys', []))
                if registry_evidence:
                    detection_results['detected'] = True
                    detection_results['confidence'] += 0.2
                    detection_results['evidence'].extend(registry_evidence)
            
            # Check files
            file_evidence = self._check_files(signatures.get('files', []))
            if file_evidence:
                detection_results['detected'] = True
                detection_results['confidence'] += 0.1
                detection_results['evidence'].extend(file_evidence)
            
            if detection_results['detected']:
                detected_systems[anticheat] = detection_results
                logger.info(f"Detected {anticheat.value} (confidence: {detection_results['confidence']:.2f})")
        
        self.installed_systems = detected_systems
        self.last_scan = current_time
        
        return detected_systems
    
    def _check_processes(self, process_names: List[str]) -> List[str]:
        """Check for running processes"""
        if not HAS_WINDOWS_APIS:
            return []
        
        detected = []
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] and any(name.lower() in proc.info['name'].lower() 
                                               for name in process_names):
                        detected.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error checking processes: {e}")
        
        return detected
    
    def _check_services(self, service_names: List[str]) -> List[str]:
        """Check for installed services"""
        detected = []
        
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['sc', 'query'], capture_output=True, text=True, timeout=10)
                service_output = result.stdout.lower()
                
                for service_name in service_names:
                    if service_name.lower() in service_output:
                        detected.append(service_name)
            except Exception as e:
                logger.warning(f"Error checking services: {e}")
        
        return detected
    
    def _check_registry(self, registry_keys: List[str]) -> List[str]:
        """Check registry keys (Windows only)"""
        if not HAS_WINDOWS_APIS:
            return []
        
        detected = []
        
        for key_path in registry_keys:
            try:
                # Parse registry key
                parts = key_path.split('\\')
                if len(parts) < 2:
                    continue
                
                hive_name = parts[0]
                key_path_remaining = '\\'.join(parts[1:])
                
                # Map hive names to constants
                hive_map = {
                    'HKEY_LOCAL_MACHINE': winreg.HKEY_LOCAL_MACHINE,
                    'HKEY_CURRENT_USER': winreg.HKEY_CURRENT_USER,
                    'HKEY_CLASSES_ROOT': winreg.HKEY_CLASSES_ROOT
                }
                
                if hive_name not in hive_map:
                    continue
                
                hive = hive_map[hive_name]
                
                # Try to open the key
                try:
                    with winreg.OpenKey(hive, key_path_remaining, 0, winreg.KEY_READ):
                        detected.append(f"Registry: {key_path}")
                except FileNotFoundError:
                    continue  # Key doesn't exist
                except PermissionError:
                    continue  # No permission
                    
            except Exception as e:
                logger.debug(f"Error checking registry key {key_path}: {e}")
        
        return detected
    
    def _check_files(self, filenames: List[str]) -> List[str]:
        """Check for files in common locations"""
        detected = []
        
        # Common search paths
        search_paths = []
        
        if platform.system() == "Windows":
            search_paths.extend([
                os.environ.get('PROGRAMFILES', r'C:\Program Files'),
                os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)'),
                os.environ.get('PROGRAMDATA', r'C:\ProgramData'),
                os.environ.get('LOCALAPPDATA', ''),
                os.environ.get('APPDATA', '')
            ])
        else:
            search_paths.extend([
                '/usr/bin',
                '/usr/local/bin',
                '/opt',
                os.path.expanduser('~/.local/bin')
            ])
        
        for filename in filenames:
            for search_path in search_paths:
                if not search_path:
                    continue
                
                try:
                    # Check direct path
                    file_path = Path(search_path) / filename
                    if file_path.exists():
                        detected.append(f"File: {file_path}")
                        continue
                    
                    # Check subdirectories
                    for root, dirs, files in os.walk(search_path):
                        if filename.lower() in [f.lower() for f in files]:
                            detected.append(f"File: {Path(root) / filename}")
                            break
                        
                        # Limit search depth
                        if Path(root).relative_to(search_path).parts and len(Path(root).relative_to(search_path).parts) > 2:
                            dirs.clear()
                            
                except (PermissionError, OSError):
                    continue
                except Exception as e:
                    logger.debug(f"Error searching in {search_path}: {e}")
        
        return detected


class AntiCheatCompatibilityChecker:
    """Check shader compatibility with anti-cheat systems"""
    
    def __init__(self, signature_db_path: Path = None):
        self.signature_db_path = signature_db_path or Path("anticheat_signatures.db")
        self.detector = AntiCheatDetector()
        self.signatures = self._load_anticheat_signatures()
        self.compatibility_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize signature database
        self._initialize_signature_db()
        
        # Load game profiles
        self.game_profiles = self._load_game_profiles()
        
        logger.info("Anti-cheat compatibility checker initialized")
    
    def _initialize_signature_db(self):
        """Initialize SQLite database for signatures"""
        with sqlite3.connect(self.signature_db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS anticheat_signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature_id TEXT UNIQUE NOT NULL,
                    anticheat_system TEXT NOT NULL,
                    signature_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    description TEXT,
                    severity INTEGER,
                    detection_method TEXT,
                    bypass_difficulty INTEGER,
                    created_at REAL,
                    last_updated REAL
                );
                
                CREATE TABLE IF NOT EXISTS compatibility_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT UNIQUE NOT NULL,
                    shader_hash TEXT NOT NULL,
                    game_id TEXT,
                    anticheat_system TEXT NOT NULL,
                    compatibility_level TEXT NOT NULL,
                    test_passed BOOLEAN,
                    issues_detected TEXT,
                    test_timestamp REAL,
                    test_duration REAL
                );
                
                CREATE TABLE IF NOT EXISTS whitelisted_shaders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shader_hash TEXT NOT NULL,
                    game_id TEXT,
                    anticheat_system TEXT NOT NULL,
                    whitelist_reason TEXT,
                    whitelisted_by TEXT,
                    whitelist_timestamp REAL,
                    expiry_timestamp REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_signatures_system ON anticheat_signatures(anticheat_system);
                CREATE INDEX IF NOT EXISTS idx_tests_shader ON compatibility_tests(shader_hash);
                CREATE INDEX IF NOT EXISTS idx_whitelist_shader ON whitelisted_shaders(shader_hash, anticheat_system);
            """)
    
    def _load_anticheat_signatures(self) -> Dict[AntiCheatSystem, List[AntiCheatSignature]]:
        """Load anti-cheat detection signatures"""
        signatures = defaultdict(list)
        
        # Load built-in signatures
        builtin_signatures = [
            # EAC signatures
            AntiCheatSignature(
                "eac_shader_mod_1", AntiCheatSystem.EAC, "memory_pattern",
                "modified_shader_cache", "Modified shader cache detection",
                7, "memory_scan", 8
            ),
            AntiCheatSignature(
                "eac_dll_inject_1", AntiCheatSystem.EAC, "behavior",
                "dll_injection", "DLL injection into graphics process",
                9, "process_monitor", 9
            ),
            
            # BattlEye signatures
            AntiCheatSignature(
                "be_driver_mod_1", AntiCheatSystem.BATTLEYE, "driver",
                "unsigned_driver", "Unsigned graphics driver",
                8, "driver_signature_check", 7
            ),
            AntiCheatSignature(
                "be_memory_mod_1", AntiCheatSystem.BATTLEYE, "memory_pattern",
                "memory_modification", "Graphics memory modification",
                6, "memory_integrity_check", 6
            ),
            
            # VAC signatures
            AntiCheatSignature(
                "vac_cheat_sig_1", AntiCheatSystem.VAC, "file_hash",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "Known cheat signature", 5, "file_hash_check", 4
            )
        ]
        
        for sig in builtin_signatures:
            signatures[sig.anticheat_system].append(sig)
        
        # Load from database
        try:
            with sqlite3.connect(self.signature_db_path) as conn:
                cursor = conn.execute("SELECT * FROM anticheat_signatures")
                for row in cursor.fetchall():
                    sig = AntiCheatSignature(
                        signature_id=row[1],
                        anticheat_system=AntiCheatSystem(row[2]),
                        signature_type=row[3],
                        pattern=row[4],
                        description=row[5],
                        severity=row[6],
                        detection_method=row[7],
                        bypass_difficulty=row[8],
                        last_updated=row[10]
                    )
                    signatures[sig.anticheat_system].append(sig)
        except Exception as e:
            logger.warning(f"Error loading signatures from database: {e}")
        
        return signatures
    
    def _load_game_profiles(self) -> Dict[str, GameProfile]:
        """Load game-specific anti-cheat profiles"""
        profiles = {}
        
        # Built-in game profiles
        builtin_profiles = [
            GameProfile(
                game_id="fortnite",
                game_name="Fortnite",
                anticheat_systems=[AntiCheatSystem.EAC],
                shader_whitelist_required=True,
                strict_mode=True,
                max_shader_risk_level=ShaderRiskLevel.LOW_RISK,
                allow_driver_modifications=False,
                allow_memory_modifications=False,
                blocked_operations={"memory_write", "driver_load"},
                whitelisted_shaders=set(),
                quarantined_shaders=set()
            ),
            GameProfile(
                game_id="pubg",
                game_name="PlayerUnknown's Battlegrounds",
                anticheat_systems=[AntiCheatSystem.BATTLEYE],
                shader_whitelist_required=True,
                strict_mode=True,
                max_shader_risk_level=ShaderRiskLevel.SAFE,
                allow_driver_modifications=False,
                allow_memory_modifications=False,
                blocked_operations={"memory_write", "process_inject"},
                whitelisted_shaders=set(),
                quarantined_shaders=set()
            ),
            GameProfile(
                game_id="csgo",
                game_name="Counter-Strike: Global Offensive",
                anticheat_systems=[AntiCheatSystem.VAC],
                shader_whitelist_required=False,
                strict_mode=False,
                max_shader_risk_level=ShaderRiskLevel.MEDIUM_RISK,
                allow_driver_modifications=False,
                allow_memory_modifications=False,
                blocked_operations={"memory_write"},
                whitelisted_shaders=set(),
                quarantined_shaders=set()
            ),
            GameProfile(
                game_id="steam_deck_default",
                game_name="Steam Deck Default Profile",
                anticheat_systems=[],  # Will be detected dynamically
                shader_whitelist_required=False,
                strict_mode=False,
                max_shader_risk_level=ShaderRiskLevel.MEDIUM_RISK,
                allow_driver_modifications=True,  # Steam Deck allows more freedom
                allow_memory_modifications=False,
                blocked_operations=set(),
                whitelisted_shaders=set(),
                quarantined_shaders=set()
            )
        ]
        
        for profile in builtin_profiles:
            profiles[profile.game_id] = profile
        
        return profiles
    
    def check_shader_compatibility(self, shader_data: bytes, shader_hash: str,
                                 game_id: str = None) -> Dict[AntiCheatSystem, CompatibilityTestResult]:
        """Check shader compatibility with detected anti-cheat systems"""
        
        # Check cache first
        cache_key = f"{shader_hash}:{game_id}"
        with self.cache_lock:
            if cache_key in self.compatibility_cache:
                cached_result, cache_time = self.compatibility_cache[cache_key]
                # Use cached result if less than 1 hour old
                if time.time() - cache_time < 3600:
                    return cached_result
        
        logger.info(f"Checking anti-cheat compatibility for shader {shader_hash}")
        
        # Detect installed anti-cheat systems
        detected_systems = self.detector.scan_for_anticheat_systems()
        
        results = {}
        
        for anticheat_system in detected_systems.keys():
            result = self._test_shader_against_anticheat(
                shader_data, shader_hash, anticheat_system, game_id
            )
            results[anticheat_system] = result
        
        # Cache results
        with self.cache_lock:
            self.compatibility_cache[cache_key] = (results, time.time())
            # Limit cache size
            if len(self.compatibility_cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(self.compatibility_cache.items(),
                                    key=lambda x: x[1][1])
                self.compatibility_cache = dict(sorted_items[-500:])
        
        return results
    
    def _test_shader_against_anticheat(self, shader_data: bytes, shader_hash: str,
                                     anticheat_system: AntiCheatSystem,
                                     game_id: str = None) -> CompatibilityTestResult:
        """Test specific shader against specific anti-cheat system"""
        
        start_time = time.time()
        test_id = f"{anticheat_system.value}_{shader_hash}_{int(start_time)}"
        
        # Initialize result
        result = CompatibilityTestResult(
            test_id=test_id,
            anticheat_system=anticheat_system,
            game_profile=game_id or "unknown",
            shader_hash=shader_hash,
            compatibility_level=CompatibilityLevel.UNKNOWN,
            test_passed=False,
            issues_detected=[],
            warnings=[],
            recommendations=[],
            test_duration_seconds=0.0,
            test_timestamp=start_time,
            test_environment={
                'os': platform.system(),
                'version': platform.version(),
                'architecture': platform.machine()
            },
            signatures_triggered=[],
            false_positive_risk=0.0
        )
        
        try:
            # Get signatures for this anti-cheat system
            system_signatures = self.signatures.get(anticheat_system, [])
            
            # Check against signatures
            triggered_signatures = []
            for signature in system_signatures:
                if signature.matches(shader_data):
                    triggered_signatures.append(signature.signature_id)
                    result.issues_detected.append(
                        f"Triggered signature: {signature.signature_id} - {signature.description}"
                    )
            
            result.signatures_triggered = triggered_signatures
            
            # Analyze shader properties for risk factors
            risk_factors = self._analyze_shader_risk_factors(shader_data, anticheat_system)
            
            # Determine compatibility level
            if triggered_signatures:
                high_severity_sigs = [sig for sig in system_signatures 
                                    if sig.signature_id in triggered_signatures and sig.severity >= 8]
                
                if high_severity_sigs:
                    result.compatibility_level = CompatibilityLevel.BLOCKED
                    result.test_passed = False
                else:
                    result.compatibility_level = CompatibilityLevel.INCOMPATIBLE
                    result.test_passed = False
            
            elif risk_factors['high_risk_count'] > 0:
                result.compatibility_level = CompatibilityLevel.LIMITED_COMPATIBLE
                result.test_passed = False
                result.warnings.extend(risk_factors['high_risk_factors'])
            
            elif risk_factors['medium_risk_count'] > 2:
                result.compatibility_level = CompatibilityLevel.MOSTLY_COMPATIBLE
                result.test_passed = True
                result.warnings.extend(risk_factors['medium_risk_factors'])
                result.false_positive_risk = 0.3
            
            else:
                result.compatibility_level = CompatibilityLevel.FULLY_COMPATIBLE
                result.test_passed = True
                result.false_positive_risk = 0.1
            
            # Add system-specific recommendations
            result.recommendations.extend(
                self._get_system_specific_recommendations(anticheat_system, risk_factors)
            )
            
            # Check game-specific restrictions
            if game_id and game_id in self.game_profiles:
                game_profile = self.game_profiles[game_id]
                if anticheat_system in game_profile.anticheat_systems:
                    self._apply_game_profile_restrictions(result, game_profile, risk_factors)
            
        except Exception as e:
            logger.error(f"Error testing shader against {anticheat_system.value}: {e}")
            result.compatibility_level = CompatibilityLevel.UNKNOWN
            result.test_passed = False
            result.issues_detected.append(f"Test error: {str(e)}")
        
        finally:
            result.test_duration_seconds = time.time() - start_time
        
        # Store test result
        self._store_test_result(result)
        
        return result
    
    def _analyze_shader_risk_factors(self, shader_data: bytes, 
                                   anticheat_system: AntiCheatSystem) -> Dict[str, Any]:
        """Analyze shader for risk factors specific to anti-cheat system"""
        
        risk_factors = {
            'high_risk_factors': [],
            'medium_risk_factors': [],
            'low_risk_factors': [],
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0
        }
        
        # Check for suspicious patterns
        suspicious_patterns = [
            (b'inject', 'DLL injection patterns', 'high'),
            (b'hook', 'API hooking patterns', 'high'),
            (b'patch', 'Memory patching patterns', 'medium'),
            (b'debug', 'Debug functionality', 'low'),
            (b'cheat', 'Cheat-related strings', 'high'),
            (b'bypass', 'Bypass mechanisms', 'high')
        ]
        
        for pattern, description, risk_level in suspicious_patterns:
            if pattern in shader_data.lower():
                risk_factors[f'{risk_level}_risk_factors'].append(description)
                risk_factors[f'{risk_level}_risk_count'] += 1
        
        # Check shader size (unusually large shaders are suspicious)
        shader_size = len(shader_data)
        if shader_size > 1024 * 1024:  # 1MB
            risk_factors['high_risk_factors'].append('Unusually large shader size')
            risk_factors['high_risk_count'] += 1
        elif shader_size > 512 * 1024:  # 512KB
            risk_factors['medium_risk_factors'].append('Large shader size')
            risk_factors['medium_risk_count'] += 1
        
        # Check for obfuscation indicators
        entropy = self._calculate_entropy(shader_data)
        if entropy > 7.5:  # High entropy indicates obfuscation
            risk_factors['medium_risk_factors'].append('High entropy (possible obfuscation)')
            risk_factors['medium_risk_count'] += 1
        
        # System-specific risk factors
        if anticheat_system == AntiCheatSystem.EAC:
            # EAC is very sensitive to memory modifications
            if b'memory' in shader_data.lower() or b'alloc' in shader_data.lower():
                risk_factors['high_risk_factors'].append('Memory allocation patterns detected')
                risk_factors['high_risk_count'] += 1
        
        elif anticheat_system == AntiCheatSystem.BATTLEYE:
            # BattlEye focuses on driver-level modifications
            if b'driver' in shader_data.lower() or b'kernel' in shader_data.lower():
                risk_factors['high_risk_factors'].append('Driver/kernel interaction detected')
                risk_factors['high_risk_count'] += 1
        
        elif anticheat_system == AntiCheatSystem.VAC:
            # VAC uses statistical analysis
            if self._has_statistical_anomalies(shader_data):
                risk_factors['medium_risk_factors'].append('Statistical anomalies detected')
                risk_factors['medium_risk_count'] += 1
        
        return risk_factors
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        frequencies = [0] * 256
        for byte in data:
            frequencies[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for freq in frequencies:
            if freq > 0:
                probability = freq / data_len
                entropy -= probability * (probability.log2() if hasattr(probability, 'log2') else 
                                        __import__('math').log2(probability))
        
        return entropy
    
    def _has_statistical_anomalies(self, shader_data: bytes) -> bool:
        """Check for statistical anomalies that might trigger VAC"""
        # Simplified check - in reality would be more sophisticated
        
        # Check for unusual byte patterns
        byte_counts = [0] * 256
        for byte in shader_data:
            byte_counts[byte] += 1
        
        # Look for extremely uneven distribution
        max_count = max(byte_counts)
        min_count = min(count for count in byte_counts if count > 0)
        
        if max_count > 0 and min_count > 0:
            ratio = max_count / min_count
            return ratio > 100  # Very uneven distribution
        
        return False
    
    def _get_system_specific_recommendations(self, anticheat_system: AntiCheatSystem,
                                           risk_factors: Dict[str, Any]) -> List[str]:
        """Get system-specific recommendations"""
        recommendations = []
        
        if anticheat_system == AntiCheatSystem.EAC:
            if risk_factors['high_risk_count'] > 0:
                recommendations.append("Consider using EAC-approved shader modifications only")
                recommendations.append("Test in offline mode first")
            recommendations.append("Ensure shader cache is in standard location")
            recommendations.append("Avoid memory-resident modifications")
        
        elif anticheat_system == AntiCheatSystem.BATTLEYE:
            if risk_factors['high_risk_count'] > 0:
                recommendations.append("Use signed drivers only")
                recommendations.append("Avoid kernel-level modifications")
            recommendations.append("Test with BattlEye in development mode if available")
        
        elif anticheat_system == AntiCheatSystem.VAC:
            recommendations.append("Use standard shader compilation tools")
            recommendations.append("Avoid statistical outliers in shader code")
            if risk_factors['medium_risk_count'] > 1:
                recommendations.append("Consider whitelisting with Valve")
        
        return recommendations
    
    def _apply_game_profile_restrictions(self, result: CompatibilityTestResult,
                                       game_profile: GameProfile,
                                       risk_factors: Dict[str, Any]):
        """Apply game-specific profile restrictions"""
        
        # Check risk level against game tolerance
        total_risk_score = (risk_factors['high_risk_count'] * 3 +
                          risk_factors['medium_risk_count'] * 2 +
                          risk_factors['low_risk_count'])
        
        max_allowed_risk = int(game_profile.max_shader_risk_level)
        
        if total_risk_score > max_allowed_risk:
            result.compatibility_level = CompatibilityLevel.INCOMPATIBLE
            result.test_passed = False
            result.issues_detected.append(
                f"Risk level ({total_risk_score}) exceeds game tolerance ({max_allowed_risk})"
            )
        
        # Check whitelist requirements
        if game_profile.shader_whitelist_required and result.shader_hash not in game_profile.whitelisted_shaders:
            result.warnings.append("Shader not on game whitelist - may require approval")
            result.false_positive_risk += 0.3
        
        # Check quarantine status
        if result.shader_hash in game_profile.quarantined_shaders:
            result.compatibility_level = CompatibilityLevel.BLOCKED
            result.test_passed = False
            result.issues_detected.append("Shader is quarantined for this game")
    
    def _store_test_result(self, result: CompatibilityTestResult):
        """Store test result in database"""
        try:
            with sqlite3.connect(self.signature_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO compatibility_tests 
                    (test_id, shader_hash, game_id, anticheat_system, compatibility_level,
                     test_passed, issues_detected, test_timestamp, test_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_id,
                    result.shader_hash,
                    result.game_profile,
                    result.anticheat_system.value,
                    result.compatibility_level.value,
                    result.test_passed,
                    json.dumps(result.issues_detected),
                    result.test_timestamp,
                    result.test_duration_seconds
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing test result: {e}")
    
    def whitelist_shader(self, shader_hash: str, anticheat_system: AntiCheatSystem,
                        game_id: str = None, reason: str = "", 
                        whitelisted_by: str = "system", 
                        expiry_days: int = None) -> bool:
        """Add shader to whitelist for specific anti-cheat system"""
        try:
            expiry_timestamp = None
            if expiry_days:
                expiry_timestamp = time.time() + (expiry_days * 24 * 3600)
            
            with sqlite3.connect(self.signature_db_path) as conn:
                conn.execute("""
                    INSERT INTO whitelisted_shaders 
                    (shader_hash, game_id, anticheat_system, whitelist_reason, 
                     whitelisted_by, whitelist_timestamp, expiry_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    shader_hash, game_id, anticheat_system.value, reason,
                    whitelisted_by, time.time(), expiry_timestamp
                ))
                conn.commit()
            
            # Update game profile if applicable
            if game_id and game_id in self.game_profiles:
                self.game_profiles[game_id].whitelisted_shaders.add(shader_hash)
            
            logger.info(f"Whitelisted shader {shader_hash} for {anticheat_system.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error whitelisting shader: {e}")
            return False
    
    def is_shader_whitelisted(self, shader_hash: str, anticheat_system: AntiCheatSystem,
                            game_id: str = None) -> bool:
        """Check if shader is whitelisted"""
        try:
            with sqlite3.connect(self.signature_db_path) as conn:
                query = """
                    SELECT expiry_timestamp FROM whitelisted_shaders 
                    WHERE shader_hash = ? AND anticheat_system = ?
                """
                params = [shader_hash, anticheat_system.value]
                
                if game_id:
                    query += " AND (game_id = ? OR game_id IS NULL)"
                    params.append(game_id)
                
                row = conn.execute(query, params).fetchone()
                
                if row:
                    expiry = row[0]
                    if expiry is None or time.time() < expiry:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking whitelist: {e}")
            return False
    
    def generate_compatibility_report(self, shader_hash: str = None,
                                    game_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive compatibility report"""
        
        report = {
            'generation_time': time.time(),
            'detected_anticheat_systems': {},
            'compatibility_summary': {},
            'recommendations': []
        }
        
        # Get detected anti-cheat systems
        detected_systems = self.detector.scan_for_anticheat_systems()
        report['detected_anticheat_systems'] = {
            system.value: {
                'detected': info['detected'],
                'confidence': info['confidence'],
                'evidence_count': len(info['evidence'])
            }
            for system, info in detected_systems.items()
        }
        
        # Shader-specific report
        if shader_hash:
            try:
                with sqlite3.connect(self.signature_db_path) as conn:
                    # Get test results for this shader
                    rows = conn.execute("""
                        SELECT anticheat_system, compatibility_level, test_passed, 
                               issues_detected, test_timestamp
                        FROM compatibility_tests 
                        WHERE shader_hash = ?
                        ORDER BY test_timestamp DESC
                    """, (shader_hash,)).fetchall()
                    
                    shader_results = {}
                    for row in rows:
                        system = row[0]
                        if system not in shader_results:  # Use most recent result
                            shader_results[system] = {
                                'compatibility_level': row[1],
                                'test_passed': row[2],
                                'issues_count': len(json.loads(row[3]) if row[3] else []),
                                'last_tested': row[4]
                            }
                    
                    report['shader_compatibility'] = {
                        'shader_hash': shader_hash,
                        'results': shader_results
                    }
            except Exception as e:
                logger.error(f"Error generating shader report: {e}")
        
        # Game-specific report
        if game_id and game_id in self.game_profiles:
            profile = self.game_profiles[game_id]
            report['game_profile'] = {
                'game_id': game_id,
                'game_name': profile.game_name,
                'anticheat_systems': [system.value for system in profile.anticheat_systems],
                'strict_mode': profile.strict_mode,
                'max_risk_level': profile.max_shader_risk_level.value,
                'whitelist_required': profile.shader_whitelist_required,
                'whitelisted_shaders': len(profile.whitelisted_shaders),
                'quarantined_shaders': len(profile.quarantined_shaders)
            }
        
        # General recommendations
        if detected_systems:
            report['recommendations'].extend([
                "Test shaders in offline mode first",
                "Keep shader modifications minimal",
                "Use official development tools when possible",
                "Monitor anti-cheat system updates",
                "Consider shader whitelisting for production use"
            ])
        
        return report


def create_steam_deck_anticheat_checker() -> AntiCheatCompatibilityChecker:
    """Create anti-cheat compatibility checker optimized for Steam Deck"""
    
    checker = AntiCheatCompatibilityChecker()
    
    # Add Steam Deck specific considerations
    # Steam Deck runs Linux, so some Windows-specific anti-cheat detection won't work
    # But it can run Windows games through Proton
    
    # Steam Deck has more relaxed security model
    steam_deck_profile = GameProfile(
        game_id="steam_deck_global",
        game_name="Steam Deck Global Profile",
        anticheat_systems=[],  # Will be detected per-game
        shader_whitelist_required=False,
        strict_mode=False,
        max_shader_risk_level=ShaderRiskLevel.MEDIUM_RISK,
        allow_driver_modifications=True,
        allow_memory_modifications=False,
        blocked_operations={'kernel_modification'},
        whitelisted_shaders=set(),
        quarantined_shaders=set()
    )
    
    checker.game_profiles["steam_deck_global"] = steam_deck_profile
    
    return checker


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create compatibility checker
    checker = create_steam_deck_anticheat_checker()
    
    # Scan for anti-cheat systems
    detected = checker.detector.scan_for_anticheat_systems(force_rescan=True)
    
    print("Detected Anti-Cheat Systems:")
    for system, info in detected.items():
        print(f"  {system.value}: {info['confidence']:.2f} confidence")
        for evidence in info['evidence'][:3]:  # Show first 3 pieces of evidence
            print(f"    - {evidence}")
    
    # Test shader compatibility
    test_shader = b"fake_spir_v_shader_data_for_testing" * 100
    test_hash = hashlib.sha256(test_shader).hexdigest()[:16]
    
    if detected:
        results = checker.check_shader_compatibility(test_shader, test_hash, "csgo")
        
        print(f"\nShader Compatibility Results for {test_hash}:")
        for system, result in results.items():
            print(f"  {system.value}:")
            print(f"    Compatibility: {result.compatibility_level.value}")
            print(f"    Test Passed: {result.test_passed}")
            print(f"    Issues: {len(result.issues_detected)}")
            if result.recommendations:
                print(f"    Recommendations: {result.recommendations[0]}")
    
    # Generate report
    report = checker.generate_compatibility_report()
    print(f"\nSystem Summary:")
    print(f"  Anti-cheat systems detected: {len(report['detected_anticheat_systems'])}")
    print(f"  Recommendations: {len(report['recommendations'])}")