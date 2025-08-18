#!/usr/bin/env python3
"""
Hardware Fingerprinting System for Anti-Cheat Compatibility

This module creates robust hardware identification systems for compatibility verification
while maintaining user privacy. It generates hardware fingerprints that:

- Uniquely identify hardware configurations for shader compatibility
- Remain stable across system updates and configuration changes  
- Protect user privacy by using cryptographic hashing
- Support anti-cheat system requirements (EAC, BattlEye, VAC)
- Enable hardware-specific shader optimizations
- Detect hardware spoofing attempts

The fingerprinting system balances uniqueness with privacy protection, ensuring
that community shader caches can be properly validated for hardware compatibility
without compromising user anonymity.
"""

import os
import sys
import hashlib
import platform
import subprocess
import json
import logging
import time
import struct
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import base64
from pathlib import Path
import tempfile
import re
import cpuinfo

# Try to import GPU detection libraries
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import wmi  # Windows only
    HAS_WMI = True
except ImportError:
    HAS_WMI = False

logger = logging.getLogger(__name__)


class FingerprintingLevel(Enum):
    """Levels of hardware fingerprinting detail"""
    MINIMAL = "minimal"        # Basic GPU and CPU info only
    STANDARD = "standard"      # Include memory, storage, motherboard
    DETAILED = "detailed"      # Include all detectable hardware
    ANTI_CHEAT = "anti_cheat"  # Include anti-cheat specific identifiers


class PrivacyMode(Enum):
    """Privacy protection levels"""
    NONE = "none"            # Raw hardware identifiers
    HASHED = "hashed"        # Cryptographically hashed identifiers
    SALTED = "salted"        # Salted hashes with per-system salt
    ANONYMOUS = "anonymous"  # Maximum privacy with randomized elements


@dataclass
class HardwareComponent:
    """Individual hardware component information"""
    component_type: str
    identifier: str
    model: str
    vendor: str
    driver_version: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 1.0  # How confident we are in this identification


@dataclass  
class HardwareFingerprint:
    """Complete hardware fingerprint"""
    fingerprint_id: str
    generation_time: float
    system_uuid: str
    
    # Core components
    cpu_fingerprint: str
    gpu_fingerprint: str
    memory_fingerprint: str
    storage_fingerprint: str
    motherboard_fingerprint: str
    
    # System information
    os_fingerprint: str
    driver_fingerprint: str
    
    # Steam Deck specific
    is_steam_deck: bool
    steam_deck_model: Optional[str] = None
    steam_deck_revision: Optional[str] = None
    
    # Anti-cheat compatibility markers
    eac_compatible_hardware: bool = True
    battleye_compatible_hardware: bool = True
    vac_compatible_hardware: bool = True
    
    # Privacy protection
    privacy_level: PrivacyMode = PrivacyMode.HASHED
    salt_used: Optional[str] = None
    
    # Validation
    components: List[HardwareComponent] = field(default_factory=list)
    detection_confidence: float = 1.0
    spoofing_indicators: List[str] = field(default_factory=list)
    
    def get_compatibility_hash(self, shader_requirements: Dict[str, Any] = None) -> str:
        """Get compatibility hash for shader matching"""
        components = [
            self.cpu_fingerprint,
            self.gpu_fingerprint,
            self.memory_fingerprint,
            self.os_fingerprint
        ]
        
        # Add shader-specific requirements if provided
        if shader_requirements:
            req_hash = hashlib.sha256(json.dumps(shader_requirements, sort_keys=True).encode()).hexdigest()
            components.append(req_hash)
        
        combined = '|'.join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def is_compatible_with(self, other: 'HardwareFingerprint', 
                          strict_matching: bool = False) -> bool:
        """Check compatibility with another hardware fingerprint"""
        if strict_matching:
            return (self.cpu_fingerprint == other.cpu_fingerprint and
                   self.gpu_fingerprint == other.gpu_fingerprint and
                   self.memory_fingerprint == other.memory_fingerprint)
        
        # Relaxed matching - allow some differences
        cpu_match = self.cpu_fingerprint == other.cpu_fingerprint
        gpu_match = self.gpu_fingerprint == other.gpu_fingerprint
        
        # At minimum, GPU must match for shader compatibility
        return gpu_match and (cpu_match or self._cpu_family_compatible(other))
    
    def _cpu_family_compatible(self, other: 'HardwareFingerprint') -> bool:
        """Check if CPUs are from compatible families"""
        # Extract CPU family info from fingerprints (simplified)
        self_family = self.cpu_fingerprint[:8]  # First 8 chars often indicate family
        other_family = other.cpu_fingerprint[:8]
        
        return self_family == other_family


class AntiCheatHardwareChecker:
    """Check hardware compatibility with anti-cheat systems"""
    
    def __init__(self):
        self.eac_blocked_hardware = {
            # Known problematic hardware for EAC
            'gpu_vendors': set(),
            'cpu_models': set(),
            'driver_versions': set()
        }
        
        self.battleye_blocked_hardware = {
            'gpu_vendors': set(),
            'cpu_models': set(),
            'virtualization_indicators': {
                'vmware', 'virtualbox', 'qemu', 'xen', 'hyper-v'
            }
        }
        
        self.vac_suspicious_patterns = {
            'excessive_virtualization',
            'debugger_present',
            'modified_drivers',
            'unsigned_drivers'
        }
    
    def check_eac_compatibility(self, fingerprint: HardwareFingerprint) -> Tuple[bool, List[str]]:
        """Check EAC compatibility"""
        issues = []
        
        # Check for virtualization (EAC can be strict about VMs)
        if any(indicator in fingerprint.spoofing_indicators 
               for indicator in ['vm_detected', 'hypervisor_present']):
            issues.append("Virtualization detected - may cause EAC issues")
        
        # Check driver signatures
        for component in fingerprint.components:
            if (component.component_type == 'gpu' and 
                component.driver_version and 
                'unsigned' in component.additional_data.get('driver_status', '')):
                issues.append("Unsigned GPU driver detected")
        
        return len(issues) == 0, issues
    
    def check_battleye_compatibility(self, fingerprint: HardwareFingerprint) -> Tuple[bool, List[str]]:
        """Check BattlEye compatibility"""
        issues = []
        
        # BattlEye is very sensitive to virtualization
        vm_indicators = {'vmware', 'virtualbox', 'qemu', 'vm_detected'}
        if any(indicator in fingerprint.spoofing_indicators for indicator in vm_indicators):
            issues.append("Virtual machine detected - BattlEye incompatible")
        
        # Check for debugging tools
        if 'debugger_present' in fingerprint.spoofing_indicators:
            issues.append("Debugging tools detected")
        
        return len(issues) == 0, issues
    
    def check_vac_compatibility(self, fingerprint: HardwareFingerprint) -> Tuple[bool, List[str]]:
        """Check VAC compatibility (generally more lenient)"""
        issues = []
        
        # VAC mainly checks for obvious cheating patterns
        if 'memory_injection' in fingerprint.spoofing_indicators:
            issues.append("Memory injection detected")
        
        if 'dll_injection' in fingerprint.spoofing_indicators:
            issues.append("DLL injection detected")
        
        return len(issues) == 0, issues


class HardwareDetector:
    """Comprehensive hardware detection system"""
    
    def __init__(self, fingerprinting_level: FingerprintingLevel = FingerprintingLevel.STANDARD):
        self.fingerprinting_level = fingerprinting_level
        self.detected_components = []
        self.spoofing_indicators = []
        
    def detect_cpu(self) -> HardwareComponent:
        """Detect CPU information"""
        try:
            # Use cpuinfo library if available
            cpu_info = cpuinfo.get_cpu_info()
            
            cpu_id = cpu_info.get('brand_raw', cpu_info.get('brand', 'Unknown'))
            vendor = cpu_info.get('vendor_id_raw', 'Unknown')
            
            # Get additional CPU details
            additional_data = {
                'architecture': cpu_info.get('arch_string_raw', platform.machine()),
                'cores': cpu_info.get('count', os.cpu_count()),
                'frequency': cpu_info.get('hz_advertised_friendly', ''),
                'cache_info': cpu_info.get('cache_info', {}),
                'flags': cpu_info.get('flags', [])
            }
            
            # Check for virtualization indicators
            if any(flag in cpu_info.get('flags', []) for flag in ['hypervisor', 'vmx', 'svm']):
                if 'hypervisor' in cpu_info.get('flags', []):
                    self.spoofing_indicators.append('hypervisor_present')
            
            return HardwareComponent(
                component_type='cpu',
                identifier=cpu_id,
                model=cpu_id,
                vendor=vendor,
                additional_data=additional_data,
                confidence_level=0.9
            )
            
        except Exception as e:
            logger.warning(f"CPU detection error: {e}")
            
            # Fallback to platform module
            return HardwareComponent(
                component_type='cpu',
                identifier=platform.processor(),
                model=platform.processor(),
                vendor='Unknown',
                additional_data={'architecture': platform.machine()},
                confidence_level=0.5
            )
    
    def detect_gpu(self) -> List[HardwareComponent]:
        """Detect GPU information"""
        gpus = []
        
        try:
            if HAS_GPUTIL:
                # Use GPUtil for detection
                gpu_list = GPUtil.getGPUs()
                
                for gpu in gpu_list:
                    additional_data = {
                        'memory_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'load_percent': gpu.load * 100,
                        'temperature_c': gpu.temperature,
                        'uuid': getattr(gpu, 'uuid', None)
                    }
                    
                    # Detect driver version
                    driver_version = None
                    try:
                        if platform.system() == "Windows":
                            driver_version = self._get_windows_gpu_driver_version(gpu.name)
                        elif platform.system() == "Linux":
                            driver_version = self._get_linux_gpu_driver_version()
                    except:
                        pass
                    
                    gpu_component = HardwareComponent(
                        component_type='gpu',
                        identifier=f"{gpu.name}_{gpu.id}",
                        model=gpu.name,
                        vendor=self._extract_gpu_vendor(gpu.name),
                        driver_version=driver_version,
                        additional_data=additional_data,
                        confidence_level=0.95
                    )
                    
                    gpus.append(gpu_component)
            
            else:
                # Fallback detection methods
                gpus.extend(self._fallback_gpu_detection())
                
        except Exception as e:
            logger.warning(f"GPU detection error: {e}")
            gpus.extend(self._fallback_gpu_detection())
        
        # Steam Deck specific GPU detection
        if self._is_steam_deck():
            steam_deck_gpu = self._detect_steam_deck_gpu()
            if steam_deck_gpu:
                gpus.append(steam_deck_gpu)
        
        return gpus if gpus else [self._create_unknown_gpu()]
    
    def detect_memory(self) -> HardwareComponent:
        """Detect memory information"""
        try:
            if HAS_PSUTIL:
                memory = psutil.virtual_memory()
                
                additional_data = {
                    'total_mb': memory.total // (1024 * 1024),
                    'available_mb': memory.available // (1024 * 1024),
                    'used_percent': memory.percent,
                    'swap_total_mb': psutil.swap_memory().total // (1024 * 1024) if hasattr(psutil, 'swap_memory') else 0
                }
                
                # Detect memory type and speed if possible
                if platform.system() == "Windows" and HAS_WMI:
                    try:
                        additional_data.update(self._get_windows_memory_details())
                    except:
                        pass
                elif platform.system() == "Linux":
                    try:
                        additional_data.update(self._get_linux_memory_details())
                    except:
                        pass
                
                # Create a stable identifier based on memory size and configuration
                memory_id = f"memory_{memory.total // (1024 * 1024)}mb"
                
                return HardwareComponent(
                    component_type='memory',
                    identifier=memory_id,
                    model=f"{memory.total // (1024 * 1024 * 1024)}GB RAM",
                    vendor='Unknown',
                    additional_data=additional_data,
                    confidence_level=0.9
                )
                
        except Exception as e:
            logger.warning(f"Memory detection error: {e}")
        
        # Fallback
        return HardwareComponent(
            component_type='memory',
            identifier='memory_unknown',
            model='Unknown',
            vendor='Unknown',
            confidence_level=0.3
        )
    
    def detect_storage(self) -> List[HardwareComponent]:
        """Detect storage devices"""
        storage_devices = []
        
        try:
            if HAS_PSUTIL:
                # Get disk usage for all mounted disks
                disk_partitions = psutil.disk_partitions()
                
                processed_devices = set()
                
                for partition in disk_partitions:
                    try:
                        # Avoid duplicates and skip special filesystems
                        if (partition.device in processed_devices or 
                            partition.fstype in ['', 'tmpfs', 'devtmpfs', 'sysfs', 'proc']):
                            continue
                        
                        processed_devices.add(partition.device)
                        
                        # Get disk usage
                        usage = psutil.disk_usage(partition.mountpoint)
                        
                        additional_data = {
                            'mountpoint': partition.mountpoint,
                            'filesystem': partition.fstype,
                            'total_gb': usage.total // (1024 ** 3),
                            'used_gb': usage.used // (1024 ** 3),
                            'free_gb': usage.free // (1024 ** 3),
                            'used_percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                        }
                        
                        # Try to get more detailed device info
                        device_model = partition.device
                        if platform.system() == "Linux":
                            device_model = self._get_linux_disk_model(partition.device)
                        elif platform.system() == "Windows":
                            device_model = self._get_windows_disk_model(partition.device)
                        
                        storage_component = HardwareComponent(
                            component_type='storage',
                            identifier=f"storage_{hashlib.sha256(partition.device.encode()).hexdigest()[:16]}",
                            model=device_model,
                            vendor='Unknown',
                            additional_data=additional_data,
                            confidence_level=0.8
                        )
                        
                        storage_devices.append(storage_component)
                        
                    except (PermissionError, FileNotFoundError):
                        continue
                        
        except Exception as e:
            logger.warning(f"Storage detection error: {e}")
        
        return storage_devices if storage_devices else [self._create_unknown_storage()]
    
    def detect_motherboard(self) -> HardwareComponent:
        """Detect motherboard information"""
        try:
            if platform.system() == "Windows" and HAS_WMI:
                return self._get_windows_motherboard()
            elif platform.system() == "Linux":
                return self._get_linux_motherboard()
            else:
                return self._get_generic_motherboard()
                
        except Exception as e:
            logger.warning(f"Motherboard detection error: {e}")
            return self._create_unknown_motherboard()
    
    def detect_system_info(self) -> Dict[str, str]:
        """Detect system information"""
        system_info = {
            'os_name': platform.system(),
            'os_version': platform.version(),
            'os_release': platform.release(),
            'hostname': platform.node(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version()
        }
        
        # Add Steam Deck specific detection
        if self._is_steam_deck():
            system_info['steam_deck_model'] = self._detect_steam_deck_model()
            system_info['steam_deck_revision'] = self._detect_steam_deck_revision()
        
        return system_info
    
    def _is_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        try:
            # Check for Steam Deck specific identifiers
            if platform.system() == "Linux":
                # Check DMI information
                try:
                    with open('/sys/class/dmi/id/product_name', 'r') as f:
                        product_name = f.read().strip()
                        if 'steam deck' in product_name.lower():
                            return True
                except (FileNotFoundError, PermissionError):
                    pass
                
                # Check for Steam Deck specific files
                steam_deck_indicators = [
                    '/home/deck',
                    '/usr/bin/steamos-session-select',
                    '/etc/steamos-release'
                ]
                
                if any(os.path.exists(indicator) for indicator in steam_deck_indicators):
                    return True
            
            # Check hostname patterns
            hostname = platform.node().lower()
            if 'steamdeck' in hostname or 'deck' in hostname:
                return True
                
        except Exception:
            pass
        
        return False
    
    def _detect_steam_deck_model(self) -> Optional[str]:
        """Detect Steam Deck model (LCD/OLED)"""
        try:
            if platform.system() == "Linux":
                # Check DMI information for model details
                try:
                    with open('/sys/class/dmi/id/product_version', 'r') as f:
                        version = f.read().strip()
                        if 'oled' in version.lower():
                            return 'OLED'
                        else:
                            return 'LCD'
                except (FileNotFoundError, PermissionError):
                    pass
                
                # Check display information
                try:
                    result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=5)
                    if 'OLED' in result.stdout:
                        return 'OLED'
                    elif 'eDP' in result.stdout:
                        return 'LCD'
                except:
                    pass
        except Exception:
            pass
        
        return 'LCD'  # Default assumption
    
    def _detect_steam_deck_revision(self) -> Optional[str]:
        """Detect Steam Deck hardware revision"""
        try:
            if platform.system() == "Linux":
                try:
                    with open('/sys/class/dmi/id/board_version', 'r') as f:
                        return f.read().strip()
                except (FileNotFoundError, PermissionError):
                    pass
        except Exception:
            pass
        
        return None
    
    def _detect_steam_deck_gpu(self) -> Optional[HardwareComponent]:
        """Detect Steam Deck GPU specifics"""
        return HardwareComponent(
            component_type='gpu',
            identifier='steamdeck_rdna2',
            model='AMD RDNA2 (Steam Deck)',
            vendor='AMD',
            additional_data={
                'compute_units': 8,
                'architecture': 'RDNA2',
                'integrated': True,
                'steam_deck_optimized': True
            },
            confidence_level=1.0
        )
    
    # Helper methods for platform-specific detection
    def _get_windows_gpu_driver_version(self, gpu_name: str) -> Optional[str]:
        """Get Windows GPU driver version"""
        if not HAS_WMI:
            return None
        
        try:
            c = wmi.WMI()
            for gpu in c.Win32_VideoController():
                if gpu_name.lower() in gpu.Name.lower():
                    return gpu.DriverVersion
        except:
            pass
        
        return None
    
    def _get_linux_gpu_driver_version(self) -> Optional[str]:
        """Get Linux GPU driver version"""
        try:
            # Try nvidia-smi for NVIDIA cards
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        try:
            # Try for AMD cards
            with open('/sys/module/amdgpu/version', 'r') as f:
                return f.read().strip()
        except:
            pass
        
        return None
    
    def _fallback_gpu_detection(self) -> List[HardwareComponent]:
        """Fallback GPU detection methods"""
        gpus = []
        
        try:
            if platform.system() == "Windows":
                gpus.extend(self._windows_gpu_fallback())
            elif platform.system() == "Linux":
                gpus.extend(self._linux_gpu_fallback())
        except Exception as e:
            logger.warning(f"Fallback GPU detection failed: {e}")
        
        return gpus
    
    def _windows_gpu_fallback(self) -> List[HardwareComponent]:
        """Windows GPU fallback detection"""
        gpus = []
        
        if HAS_WMI:
            try:
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name and gpu.Name != "Microsoft Basic Display Adapter":
                        gpus.append(HardwareComponent(
                            component_type='gpu',
                            identifier=gpu.Name,
                            model=gpu.Name,
                            vendor=self._extract_gpu_vendor(gpu.Name),
                            driver_version=gpu.DriverVersion,
                            confidence_level=0.8
                        ))
            except:
                pass
        
        return gpus
    
    def _linux_gpu_fallback(self) -> List[HardwareComponent]:
        """Linux GPU fallback detection"""
        gpus = []
        
        try:
            # Try lspci
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA compatible controller' in line or 'Display controller' in line:
                        # Extract GPU name
                        parts = line.split(': ')
                        if len(parts) > 1:
                            gpu_name = parts[1].split('[')[0].strip()
                            gpus.append(HardwareComponent(
                                component_type='gpu',
                                identifier=gpu_name,
                                model=gpu_name,
                                vendor=self._extract_gpu_vendor(gpu_name),
                                confidence_level=0.7
                            ))
        except:
            pass
        
        return gpus
    
    def _extract_gpu_vendor(self, gpu_name: str) -> str:
        """Extract GPU vendor from name"""
        gpu_name_lower = gpu_name.lower()
        
        if any(vendor in gpu_name_lower for vendor in ['nvidia', 'geforce', 'gtx', 'rtx', 'tesla', 'quadro']):
            return 'NVIDIA'
        elif any(vendor in gpu_name_lower for vendor in ['amd', 'radeon', 'rx ', 'vega', 'rdna']):
            return 'AMD'
        elif any(vendor in gpu_name_lower for vendor in ['intel', 'iris', 'uhd']):
            return 'Intel'
        else:
            return 'Unknown'
    
    def _create_unknown_gpu(self) -> HardwareComponent:
        """Create unknown GPU component"""
        return HardwareComponent(
            component_type='gpu',
            identifier='gpu_unknown',
            model='Unknown GPU',
            vendor='Unknown',
            confidence_level=0.1
        )
    
    def _create_unknown_storage(self) -> HardwareComponent:
        """Create unknown storage component"""
        return HardwareComponent(
            component_type='storage',
            identifier='storage_unknown',
            model='Unknown Storage',
            vendor='Unknown',
            confidence_level=0.1
        )
    
    def _create_unknown_motherboard(self) -> HardwareComponent:
        """Create unknown motherboard component"""
        return HardwareComponent(
            component_type='motherboard',
            identifier='motherboard_unknown',
            model='Unknown Motherboard',
            vendor='Unknown',
            confidence_level=0.1
        )
    
    def _get_windows_memory_details(self) -> Dict[str, Any]:
        """Get Windows memory details"""
        details = {}
        if HAS_WMI:
            try:
                c = wmi.WMI()
                for memory in c.Win32_PhysicalMemory():
                    details['memory_type'] = getattr(memory, 'MemoryType', None)
                    details['memory_speed'] = getattr(memory, 'Speed', None)
                    break  # Just get first module details
            except:
                pass
        return details
    
    def _get_linux_memory_details(self) -> Dict[str, Any]:
        """Get Linux memory details"""
        details = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                # Extract memory info (simplified)
                if 'MemTotal' in meminfo:
                    for line in meminfo.split('\n'):
                        if line.startswith('MemTotal'):
                            total_kb = int(line.split()[1])
                            details['total_kb'] = total_kb
                            break
        except:
            pass
        return details
    
    def _get_windows_motherboard(self) -> HardwareComponent:
        """Get Windows motherboard info"""
        if HAS_WMI:
            try:
                c = wmi.WMI()
                for board in c.Win32_BaseBoard():
                    return HardwareComponent(
                        component_type='motherboard',
                        identifier=f"{board.Manufacturer}_{board.Product}",
                        model=board.Product,
                        vendor=board.Manufacturer,
                        additional_data={'serial': board.SerialNumber},
                        confidence_level=0.9
                    )
            except:
                pass
        
        return self._create_unknown_motherboard()
    
    def _get_linux_motherboard(self) -> HardwareComponent:
        """Get Linux motherboard info"""
        try:
            board_vendor = "Unknown"
            board_name = "Unknown"
            
            try:
                with open('/sys/class/dmi/id/board_vendor', 'r') as f:
                    board_vendor = f.read().strip()
            except:
                pass
            
            try:
                with open('/sys/class/dmi/id/board_name', 'r') as f:
                    board_name = f.read().strip()
            except:
                pass
            
            return HardwareComponent(
                component_type='motherboard',
                identifier=f"{board_vendor}_{board_name}",
                model=board_name,
                vendor=board_vendor,
                confidence_level=0.9
            )
        except:
            return self._create_unknown_motherboard()
    
    def _get_generic_motherboard(self) -> HardwareComponent:
        """Get generic motherboard info"""
        return HardwareComponent(
            component_type='motherboard',
            identifier='motherboard_generic',
            model=f"{platform.system()} System",
            vendor=platform.system(),
            confidence_level=0.5
        )
    
    def _get_linux_disk_model(self, device: str) -> str:
        """Get Linux disk model"""
        try:
            # Extract device name (e.g., sda from /dev/sda1)
            device_name = os.path.basename(device).rstrip('0123456789')
            
            model_path = f'/sys/block/{device_name}/device/model'
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        
        return os.path.basename(device)
    
    def _get_windows_disk_model(self, device: str) -> str:
        """Get Windows disk model"""
        if HAS_WMI:
            try:
                c = wmi.WMI()
                for disk in c.Win32_DiskDrive():
                    if device.startswith(disk.Caption[:2]):  # Match drive letter
                        return disk.Model
            except:
                pass
        
        return device


class HardwareFingerprintGenerator:
    """Generate hardware fingerprints with privacy protection"""
    
    def __init__(self, privacy_mode: PrivacyMode = PrivacyMode.HASHED,
                 fingerprinting_level: FingerprintingLevel = FingerprintingLevel.STANDARD):
        self.privacy_mode = privacy_mode
        self.fingerprinting_level = fingerprinting_level
        self.detector = HardwareDetector(fingerprinting_level)
        self.anticheat_checker = AntiCheatHardwareChecker()
        
        # Generate system salt for privacy protection
        self.system_salt = self._generate_system_salt()
        
        logger.info(f"Hardware fingerprinting initialized: {privacy_mode.value} privacy, "
                   f"{fingerprinting_level.value} detail")
    
    def generate_fingerprint(self) -> HardwareFingerprint:
        """Generate complete hardware fingerprint"""
        start_time = time.time()
        
        # Detect all hardware components
        cpu_component = self.detector.detect_cpu()
        gpu_components = self.detector.detect_gpu()
        memory_component = self.detector.detect_memory()
        storage_components = self.detector.detect_storage()
        motherboard_component = self.detector.detect_motherboard()
        
        # Collect all components
        all_components = [cpu_component, memory_component, motherboard_component]
        all_components.extend(gpu_components)
        all_components.extend(storage_components)
        
        # Detect system information
        system_info = self.detector.detect_system_info()
        
        # Generate component fingerprints
        cpu_fingerprint = self._generate_component_fingerprint(cpu_component)
        gpu_fingerprint = self._generate_gpu_fingerprint(gpu_components)
        memory_fingerprint = self._generate_component_fingerprint(memory_component)
        storage_fingerprint = self._generate_storage_fingerprint(storage_components)
        motherboard_fingerprint = self._generate_component_fingerprint(motherboard_component)
        
        # Generate system fingerprints
        os_fingerprint = self._generate_os_fingerprint(system_info)
        driver_fingerprint = self._generate_driver_fingerprint(all_components)
        
        # Generate system UUID (stable across reboots)
        system_uuid = self._generate_system_uuid(all_components)
        
        # Steam Deck detection
        is_steam_deck = self.detector._is_steam_deck()
        steam_deck_model = system_info.get('steam_deck_model')
        steam_deck_revision = system_info.get('steam_deck_revision')
        
        # Calculate detection confidence
        confidence = sum(comp.confidence_level for comp in all_components) / len(all_components)
        
        # Check anti-cheat compatibility
        eac_compatible, eac_issues = self.anticheat_checker.check_eac_compatibility(None)
        battleye_compatible, battleye_issues = self.anticheat_checker.check_battleye_compatibility(None)
        vac_compatible, vac_issues = self.anticheat_checker.check_vac_compatibility(None)
        
        # Create fingerprint
        fingerprint = HardwareFingerprint(
            fingerprint_id=self._generate_fingerprint_id(all_components),
            generation_time=start_time,
            system_uuid=system_uuid,
            cpu_fingerprint=cpu_fingerprint,
            gpu_fingerprint=gpu_fingerprint,
            memory_fingerprint=memory_fingerprint,
            storage_fingerprint=storage_fingerprint,
            motherboard_fingerprint=motherboard_fingerprint,
            os_fingerprint=os_fingerprint,
            driver_fingerprint=driver_fingerprint,
            is_steam_deck=is_steam_deck,
            steam_deck_model=steam_deck_model,
            steam_deck_revision=steam_deck_revision,
            eac_compatible_hardware=eac_compatible,
            battleye_compatible_hardware=battleye_compatible,
            vac_compatible_hardware=vac_compatible,
            privacy_level=self.privacy_mode,
            salt_used=self.system_salt if self.privacy_mode == PrivacyMode.SALTED else None,
            components=all_components,
            detection_confidence=confidence,
            spoofing_indicators=self.detector.spoofing_indicators
        )
        
        # Update anti-cheat compatibility with full fingerprint
        fingerprint.eac_compatible_hardware, _ = self.anticheat_checker.check_eac_compatibility(fingerprint)
        fingerprint.battleye_compatible_hardware, _ = self.anticheat_checker.check_battleye_compatibility(fingerprint)  
        fingerprint.vac_compatible_hardware, _ = self.anticheat_checker.check_vac_compatibility(fingerprint)
        
        logger.info(f"Generated hardware fingerprint {fingerprint.fingerprint_id} "
                   f"(confidence: {confidence:.2f})")
        
        return fingerprint
    
    def _generate_component_fingerprint(self, component: HardwareComponent) -> str:
        """Generate fingerprint for a hardware component"""
        data = f"{component.component_type}:{component.vendor}:{component.model}:{component.identifier}"
        
        # Add additional data if detailed fingerprinting
        if self.fingerprinting_level in [FingerprintingLevel.DETAILED, FingerprintingLevel.ANTI_CHEAT]:
            additional = json.dumps(component.additional_data, sort_keys=True)
            data += f":{additional}"
        
        return self._apply_privacy_protection(data)
    
    def _generate_gpu_fingerprint(self, gpu_components: List[HardwareComponent]) -> str:
        """Generate combined GPU fingerprint"""
        if not gpu_components:
            return self._apply_privacy_protection("gpu:unknown")
        
        # Sort GPUs for consistent fingerprinting
        sorted_gpus = sorted(gpu_components, key=lambda g: g.identifier)
        
        gpu_data = []
        for gpu in sorted_gpus:
            gpu_info = f"{gpu.vendor}:{gpu.model}"
            
            # Add GPU-specific details for shader compatibility
            if gpu.additional_data:
                memory_mb = gpu.additional_data.get('memory_mb', 0)
                architecture = gpu.additional_data.get('architecture', '')
                gpu_info += f":{memory_mb}:{architecture}"
            
            gpu_data.append(gpu_info)
        
        combined_data = "|".join(gpu_data)
        return self._apply_privacy_protection(combined_data)
    
    def _generate_storage_fingerprint(self, storage_components: List[HardwareComponent]) -> str:
        """Generate combined storage fingerprint"""
        if not storage_components:
            return self._apply_privacy_protection("storage:unknown")
        
        # Only include total storage capacity for privacy
        total_storage_gb = 0
        storage_types = set()
        
        for storage in storage_components:
            if storage.additional_data:
                total_storage_gb += storage.additional_data.get('total_gb', 0)
                filesystem = storage.additional_data.get('filesystem', '')
                if filesystem:
                    storage_types.add(filesystem)
        
        storage_data = f"total_gb:{total_storage_gb}:types:{':'.join(sorted(storage_types))}"
        return self._apply_privacy_protection(storage_data)
    
    def _generate_os_fingerprint(self, system_info: Dict[str, str]) -> str:
        """Generate OS fingerprint"""
        os_data = (f"{system_info.get('os_name', '')}:"
                  f"{system_info.get('os_version', '')}:"
                  f"{system_info.get('architecture', '')}")
        
        return self._apply_privacy_protection(os_data)
    
    def _generate_driver_fingerprint(self, components: List[HardwareComponent]) -> str:
        """Generate driver fingerprint"""
        driver_versions = []
        
        for component in components:
            if component.driver_version:
                driver_versions.append(f"{component.component_type}:{component.driver_version}")
        
        driver_data = "|".join(sorted(driver_versions))
        return self._apply_privacy_protection(driver_data)
    
    def _generate_system_uuid(self, components: List[HardwareComponent]) -> str:
        """Generate stable system UUID"""
        # Use stable hardware identifiers
        stable_identifiers = []
        
        for component in components:
            if component.component_type in ['cpu', 'motherboard']:
                stable_identifiers.append(f"{component.component_type}:{component.identifier}")
        
        stable_data = "|".join(sorted(stable_identifiers))
        uuid_hash = hashlib.sha256(stable_data.encode()).hexdigest()
        
        # Format as UUID
        return f"{uuid_hash[:8]}-{uuid_hash[8:12]}-{uuid_hash[12:16]}-{uuid_hash[16:20]}-{uuid_hash[20:32]}"
    
    def _generate_fingerprint_id(self, components: List[HardwareComponent]) -> str:
        """Generate unique fingerprint ID"""
        all_identifiers = [comp.identifier for comp in components]
        combined = "|".join(sorted(all_identifiers))
        
        fingerprint_hash = hashlib.sha256(combined.encode()).hexdigest()
        return fingerprint_hash[:16]  # 16 character ID
    
    def _apply_privacy_protection(self, data: str) -> str:
        """Apply privacy protection to data"""
        if self.privacy_mode == PrivacyMode.NONE:
            return data
        elif self.privacy_mode == PrivacyMode.HASHED:
            return hashlib.sha256(data.encode()).hexdigest()
        elif self.privacy_mode == PrivacyMode.SALTED:
            salted_data = f"{self.system_salt}:{data}"
            return hashlib.sha256(salted_data.encode()).hexdigest()
        elif self.privacy_mode == PrivacyMode.ANONYMOUS:
            # Add randomization while preserving compatibility info
            hash_value = hashlib.sha256(data.encode()).hexdigest()
            # Keep first 16 chars for compatibility, randomize rest
            anonymous_suffix = hashlib.sha256(f"{hash_value}:{time.time()}".encode()).hexdigest()[:16]
            return f"{hash_value[:16]}{anonymous_suffix}"
        
        return data
    
    def _generate_system_salt(self) -> str:
        """Generate system-specific salt for privacy protection"""
        try:
            # Try to use machine-specific identifier
            if hasattr(uuid, 'getnode'):
                mac_address = uuid.getnode()
                salt_source = f"system_salt_{mac_address}_{platform.node()}"
            else:
                salt_source = f"system_salt_{platform.node()}_{os.path.expanduser('~')}"
            
            return hashlib.sha256(salt_source.encode()).hexdigest()[:16]
            
        except Exception:
            # Fallback to random salt (less stable but still functional)
            import random
            return hashlib.sha256(f"fallback_salt_{random.random()}".encode()).hexdigest()[:16]
    
    def save_fingerprint(self, fingerprint: HardwareFingerprint, output_path: Path):
        """Save fingerprint to JSON file"""
        fingerprint_data = {
            'fingerprint_id': fingerprint.fingerprint_id,
            'generation_time': fingerprint.generation_time,
            'system_uuid': fingerprint.system_uuid,
            'fingerprints': {
                'cpu': fingerprint.cpu_fingerprint,
                'gpu': fingerprint.gpu_fingerprint,
                'memory': fingerprint.memory_fingerprint,
                'storage': fingerprint.storage_fingerprint,
                'motherboard': fingerprint.motherboard_fingerprint,
                'os': fingerprint.os_fingerprint,
                'driver': fingerprint.driver_fingerprint
            },
            'steam_deck': {
                'is_steam_deck': fingerprint.is_steam_deck,
                'model': fingerprint.steam_deck_model,
                'revision': fingerprint.steam_deck_revision
            },
            'anticheat_compatibility': {
                'eac_compatible': fingerprint.eac_compatible_hardware,
                'battleye_compatible': fingerprint.battleye_compatible_hardware,
                'vac_compatible': fingerprint.vac_compatible_hardware
            },
            'privacy': {
                'privacy_level': fingerprint.privacy_level.value,
                'salt_used': fingerprint.salt_used
            },
            'validation': {
                'detection_confidence': fingerprint.detection_confidence,
                'spoofing_indicators': fingerprint.spoofing_indicators,
                'component_count': len(fingerprint.components)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(fingerprint_data, f, indent=2)
        
        logger.info(f"Saved hardware fingerprint to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate hardware fingerprint with different privacy levels
    for privacy_mode in [PrivacyMode.HASHED, PrivacyMode.SALTED]:
        generator = HardwareFingerprintGenerator(
            privacy_mode=privacy_mode,
            fingerprinting_level=FingerprintingLevel.STANDARD
        )
        
        fingerprint = generator.generate_fingerprint()
        
        print(f"\n{privacy_mode.value.upper()} Fingerprint:")
        print(f"ID: {fingerprint.fingerprint_id}")
        print(f"System UUID: {fingerprint.system_uuid}")
        print(f"Steam Deck: {fingerprint.is_steam_deck}")
        print(f"GPU: {fingerprint.gpu_fingerprint[:32]}...")
        print(f"CPU: {fingerprint.cpu_fingerprint[:32]}...")
        print(f"Anti-cheat: EAC={fingerprint.eac_compatible_hardware}, "
              f"BE={fingerprint.battleye_compatible_hardware}, VAC={fingerprint.vac_compatible_hardware}")
        print(f"Confidence: {fingerprint.detection_confidence:.2f}")
        
        # Save to file
        output_file = Path(f"hardware_fingerprint_{privacy_mode.value}.json")
        generator.save_fingerprint(fingerprint, output_file)