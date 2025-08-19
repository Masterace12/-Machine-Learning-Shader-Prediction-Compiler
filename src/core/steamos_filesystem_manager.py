#!/usr/bin/env python3
"""
SteamOS Filesystem Manager for Immutable System Integration

This module handles proper file placement and permissions for SteamOS's
immutable filesystem architecture, ensuring the ML shader predictor works
correctly within SteamOS constraints while maintaining system integrity.

Features:
- Immutable filesystem detection and handling
- User-writable path management and optimization
- Flatpak compatibility and sandboxing support
- Steam directory structure integration
- Atomic file operations for system safety
- Proper permission management for SteamOS
- Background service integration without system modification
- Cache management optimized for immutable systems

SteamOS 3.0+ uses an immutable root filesystem with specific writable areas.
This manager ensures all ML predictor files are placed in appropriate locations
while maintaining compatibility with system updates and immutable constraints.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import subprocess
import logging
import stat
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import pwd
import grp

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# STEAMOS FILESYSTEM DETECTION
# =============================================================================

class SteamOSVersion(Enum):
    """SteamOS versions"""
    STEAMOS_2 = "2.x"
    STEAMOS_3 = "3.x"
    HOLO_OS = "holo"
    NON_STEAMOS = "non_steamos"

class FilesystemType(Enum):
    """Filesystem types"""
    IMMUTABLE = "immutable"
    TRADITIONAL = "traditional"
    FLATPAK_SANDBOX = "flatpak_sandbox"

@dataclass
class SteamOSInfo:
    """SteamOS system information"""
    version: SteamOSVersion
    filesystem_type: FilesystemType
    is_readonly_root: bool
    boot_mode: str  # 'desktop', 'gaming', 'recovery'
    deck_variant: Optional[str]  # 'lcd', 'oled', None
    build_id: Optional[str]
    update_channel: str  # 'stable', 'beta', 'preview'

@dataclass
class WritableLocation:
    """Writable location information"""
    path: Path
    purpose: str
    size_limit_mb: Optional[int]
    temporary: bool
    persistent: bool
    requires_sudo: bool
    flatpak_accessible: bool

# =============================================================================
# STEAMOS DETECTION AND ANALYSIS
# =============================================================================

class SteamOSDetector:
    """
    Detects SteamOS version, filesystem type, and system characteristics
    """
    
    def __init__(self):
        self.os_info = self._detect_steamos_info()
        self.writable_locations = self._discover_writable_locations()
        self.steam_paths = self._detect_steam_paths()
        
    def _detect_steamos_info(self) -> SteamOSInfo:
        """Detect SteamOS version and configuration"""
        
        # Default values
        version = SteamOSVersion.NON_STEAMOS
        filesystem_type = FilesystemType.TRADITIONAL
        is_readonly_root = False
        boot_mode = "desktop"
        deck_variant = None
        build_id = None
        update_channel = "stable"
        
        try:
            # Check /etc/os-release
            os_release_path = Path('/etc/os-release')
            if os_release_path.exists():
                with open(os_release_path, 'r') as f:
                    os_release_content = f.read().lower()
                
                # Detect SteamOS
                if 'steamos' in os_release_content:
                    if 'version_id="3.' in os_release_content:
                        version = SteamOSVersion.STEAMOS_3
                        filesystem_type = FilesystemType.IMMUTABLE
                        is_readonly_root = True
                    elif 'version_id="2.' in os_release_content:
                        version = SteamOSVersion.STEAMOS_2
                        filesystem_type = FilesystemType.TRADITIONAL
                elif 'holo' in os_release_content:
                    version = SteamOSVersion.HOLO_OS
                    filesystem_type = FilesystemType.IMMUTABLE
                    is_readonly_root = True
                
                # Extract build ID
                for line in os_release_content.split('\n'):
                    if 'build_id=' in line:
                        build_id = line.split('=')[1].strip('"')
                        break
            
            # Check for immutable filesystem
            if not is_readonly_root:
                is_readonly_root = not os.access('/usr', os.W_OK)
                if is_readonly_root:
                    filesystem_type = FilesystemType.IMMUTABLE
            
            # Detect boot mode
            boot_mode = self._detect_boot_mode()
            
            # Detect Steam Deck variant
            deck_variant = self._detect_deck_variant()
            
            # Detect update channel
            update_channel = self._detect_update_channel()
            
            # Check for Flatpak environment
            if os.environ.get('FLATPAK_ID'):
                filesystem_type = FilesystemType.FLATPAK_SANDBOX
        
        except Exception as e:
            logger.error(f"SteamOS detection error: {e}")
        
        logger.info(f"Detected SteamOS: {version.value}, filesystem: {filesystem_type.value}")
        
        return SteamOSInfo(
            version=version,
            filesystem_type=filesystem_type,
            is_readonly_root=is_readonly_root,
            boot_mode=boot_mode,
            deck_variant=deck_variant,
            build_id=build_id,
            update_channel=update_channel
        )
    
    def _detect_boot_mode(self) -> str:
        """Detect current boot mode"""
        try:
            # Check for gamescope (Gaming Mode indicator)
            result = subprocess.run(['pgrep', '-f', 'gamescope'], 
                                  capture_output=True, timeout=3)
            if result.returncode == 0:
                return "gaming"
            
            # Check for desktop environment
            desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
            if desktop_env:
                return "desktop"
            
            # Check systemd target
            result = subprocess.run(['systemctl', 'get-default'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                target = result.stdout.strip()
                if 'gaming' in target or 'gamescope' in target:
                    return "gaming"
                elif 'graphical' in target:
                    return "desktop"
        
        except Exception:
            pass
        
        return "desktop"
    
    def _detect_deck_variant(self) -> Optional[str]:
        """Detect Steam Deck hardware variant"""
        try:
            # Check DMI information
            dmi_files = [
                '/sys/devices/virtual/dmi/id/product_name',
                '/sys/devices/virtual/dmi/id/board_name'
            ]
            
            for dmi_file in dmi_files:
                if os.path.exists(dmi_file):
                    with open(dmi_file, 'r') as f:
                        content = f.read().strip().lower()
                        if 'galileo' in content:
                            return "oled"
                        elif 'jupiter' in content:
                            return "lcd"
            
            # Fallback: check for Steam Deck user
            if os.path.exists('/home/deck'):
                return "lcd"  # Default to LCD if uncertain
                
        except Exception:
            pass
        
        return None
    
    def _detect_update_channel(self) -> str:
        """Detect SteamOS update channel"""
        try:
            # Check system configuration
            channel_files = [
                '/etc/steamos-release',
                '/usr/share/steamos/steamos-release'
            ]
            
            for channel_file in channel_files:
                if os.path.exists(channel_file):
                    with open(channel_file, 'r') as f:
                        content = f.read().lower()
                        if 'beta' in content:
                            return "beta"
                        elif 'preview' in content:
                            return "preview"
            
            # Check Steam client beta participation
            steam_config_path = Path.home() / ".steam" / "config" / "config.vdf"
            if steam_config_path.exists():
                try:
                    with open(steam_config_path, 'r', encoding='utf-8', errors='ignore') as f:
                        config_content = f.read().lower()
                        if '"beta"' in config_content and '"1"' in config_content:
                            return "beta"
                except Exception:
                    pass
        
        except Exception:
            pass
        
        return "stable"
    
    def _discover_writable_locations(self) -> List[WritableLocation]:
        """Discover all writable filesystem locations"""
        locations = []
        
        # User home directory and subdirectories
        home_locations = [
            (Path.home(), "user_home", None, False, True, False, True),
            (Path.home() / ".local", "user_local", None, False, True, False, True),
            (Path.home() / ".config", "user_config", 100, False, True, False, True),
            (Path.home() / ".cache", "user_cache", 500, True, False, False, True),
            (Path.home() / ".local" / "share", "user_data", None, False, True, False, True),
            (Path.home() / ".local" / "bin", "user_binaries", 50, False, True, False, False),
        ]
        
        for path, purpose, size_limit, temporary, persistent, requires_sudo, flatpak_accessible in home_locations:
            if path.exists() or self._can_create_directory(path):
                locations.append(WritableLocation(
                    path=path,
                    purpose=purpose,
                    size_limit_mb=size_limit,
                    temporary=temporary,
                    persistent=persistent,
                    requires_sudo=requires_sudo,
                    flatpak_accessible=flatpak_accessible
                ))
        
        # Temporary directories
        temp_locations = [
            (Path("/tmp"), "system_temp", 1000, True, False, False, False),
            (Path("/var/tmp"), "persistent_temp", 500, True, True, False, False),
            (Path.home() / ".local" / "tmp", "user_temp", 200, True, False, False, True),
        ]
        
        for path, purpose, size_limit, temporary, persistent, requires_sudo, flatpak_accessible in temp_locations:
            if path.exists() or self._can_create_directory(path):
                locations.append(WritableLocation(
                    path=path,
                    purpose=purpose,
                    size_limit_mb=size_limit,
                    temporary=temporary,
                    persistent=persistent,
                    requires_sudo=requires_sudo,
                    flatpak_accessible=flatpak_accessible
                ))
        
        # Runtime directories
        runtime_dir = Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}'))
        if runtime_dir.exists():
            locations.append(WritableLocation(
                path=runtime_dir,
                purpose="runtime",
                size_limit_mb=100,
                temporary=True,
                persistent=False,
                requires_sudo=False,
                flatpak_accessible=False
            ))
        
        # Special SteamOS locations
        if self.os_info.version in [SteamOSVersion.STEAMOS_3, SteamOSVersion.HOLO_OS]:
            steamos_locations = [
                (Path("/home") / ".steamos" / "offload", "steamos_offload", None, False, True, False, False),
                (Path("/var") / "lib" / "steamos", "steamos_var", None, False, True, True, False),
            ]
            
            for path, purpose, size_limit, temporary, persistent, requires_sudo, flatpak_accessible in steamos_locations:
                if path.exists() or (requires_sudo and self._can_create_with_sudo(path)):
                    locations.append(WritableLocation(
                        path=path,
                        purpose=purpose,
                        size_limit_mb=size_limit,
                        temporary=temporary,
                        persistent=persistent,
                        requires_sudo=requires_sudo,
                        flatpak_accessible=flatpak_accessible
                    ))
        
        logger.info(f"Discovered {len(locations)} writable locations")
        return locations
    
    def _detect_steam_paths(self) -> Dict[str, Path]:
        """Detect Steam installation paths"""
        steam_paths = {}
        
        # Common Steam locations
        possible_steam_roots = [
            Path.home() / ".steam" / "steam",
            Path.home() / ".local" / "share" / "Steam",
            Path("/home") / "deck" / ".steam" / "steam",
            Path("/usr") / "share" / "steam",
            Path("/opt") / "steam"
        ]
        
        for steam_root in possible_steam_roots:
            if steam_root.exists() and (steam_root / "steamapps").exists():
                steam_paths['steam_root'] = steam_root
                steam_paths['steamapps'] = steam_root / "steamapps"
                steam_paths['shader_cache'] = steam_root / "steamapps" / "shadercache"
                steam_paths['compatdata'] = steam_root / "steamapps" / "compatdata"
                steam_paths['common'] = steam_root / "steamapps" / "common"
                break
        
        # Steam configuration and logs
        steam_config_locations = [
            Path.home() / ".steam",
            Path.home() / ".local" / "share" / "Steam"
        ]
        
        for config_root in steam_config_locations:
            if config_root.exists():
                steam_paths['steam_config'] = config_root
                steam_paths['steam_logs'] = config_root / "logs"
                steam_paths['steam_userdata'] = config_root / "userdata"
                break
        
        return steam_paths
    
    def _can_create_directory(self, path: Path) -> bool:
        """Check if directory can be created"""
        try:
            parent = path.parent
            return parent.exists() and os.access(parent, os.W_OK)
        except Exception:
            return False
    
    def _can_create_with_sudo(self, path: Path) -> bool:
        """Check if directory can be created with sudo"""
        try:
            result = subprocess.run(['sudo', '-n', 'test', '-w', str(path.parent)], 
                                  capture_output=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False

# =============================================================================
# FILESYSTEM OPERATIONS MANAGER
# =============================================================================

class SteamOSFilesystemManager:
    """
    Manages filesystem operations for SteamOS immutable systems
    """
    
    def __init__(self):
        self.detector = SteamOSDetector()
        self.ml_predictor_paths = self._setup_ml_predictor_paths()
        self.file_locks: Dict[str, bool] = {}
        
    def _setup_ml_predictor_paths(self) -> Dict[str, Path]:
        """Setup ML predictor specific paths"""
        base_config = self._get_optimal_location("user_config")
        base_cache = self._get_optimal_location("user_cache")
        base_data = self._get_optimal_location("user_data")
        
        paths = {
            'config_root': base_config / "ml-shader-predictor",
            'cache_root': base_cache / "ml-shader-predictor",
            'data_root': base_data / "ml-shader-predictor",
            'models': base_data / "ml-shader-predictor" / "models",
            'shader_cache': base_cache / "ml-shader-predictor" / "shader_cache",
            'predictions': base_cache / "ml-shader-predictor" / "predictions",
            'logs': base_cache / "ml-shader-predictor" / "logs",
            'temp': self._get_optimal_location("user_temp") / "ml-shader-predictor",
            'runtime': Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / "ml-shader-predictor"
        }
        
        # Ensure directories exist
        for name, path in paths.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created ML predictor directory: {name} -> {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {name} at {path}: {e}")
        
        return paths
    
    def _get_optimal_location(self, purpose: str) -> Path:
        """Get optimal location for a specific purpose"""
        for location in self.detector.writable_locations:
            if location.purpose == purpose and location.path.exists():
                return location.path
        
        # Fallback to home directory
        return Path.home()
    
    def get_cache_path(self, cache_type: str) -> Path:
        """Get cache path for specific cache type"""
        cache_mapping = {
            'shader_predictions': self.ml_predictor_paths['predictions'],
            'ml_models': self.ml_predictor_paths['models'],
            'compiled_shaders': self.ml_predictor_paths['shader_cache'],
            'fossilize_integration': self.ml_predictor_paths['cache_root'] / "fossilize",
            'proton_configs': self.ml_predictor_paths['cache_root'] / "proton",
            'temp_compilations': self.ml_predictor_paths['temp']
        }
        
        cache_path = cache_mapping.get(cache_type, self.ml_predictor_paths['cache_root'] / cache_type)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def get_config_path(self, config_type: str) -> Path:
        """Get configuration path for specific config type"""
        config_mapping = {
            'main_config': self.ml_predictor_paths['config_root'] / "config.json",
            'steam_integration': self.ml_predictor_paths['config_root'] / "steam_integration.json",
            'hardware_profiles': self.ml_predictor_paths['config_root'] / "hardware_profiles.json",
            'user_preferences': self.ml_predictor_paths['config_root'] / "user_preferences.json"
        }
        
        return config_mapping.get(config_type, self.ml_predictor_paths['config_root'] / f"{config_type}.json")
    
    @contextmanager
    def atomic_write(self, file_path: Path):
        """Context manager for atomic file writes"""
        file_path = Path(file_path)
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary file
            with open(temp_path, 'w') as temp_file:
                yield temp_file
            
            # Atomic move
            if os.name == 'nt':
                # Windows doesn't support atomic moves with existing files
                if file_path.exists():
                    file_path.unlink()
            
            temp_path.replace(file_path)
            logger.debug(f"Atomic write completed: {file_path}")
            
        except Exception as e:
            # Cleanup on failure
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Atomic write failed for {file_path}: {e}")
            raise
    
    @contextmanager
    def atomic_write_json(self, file_path: Path):
        """Context manager for atomic JSON writes"""
        with self.atomic_write(file_path) as f:
            yield f
    
    def safe_copy(self, src: Path, dst: Path, preserve_permissions: bool = True) -> bool:
        """Safely copy file with atomic operation"""
        try:
            src = Path(src)
            dst = Path(dst)
            
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Use temporary file for atomic copy
            temp_dst = dst.with_suffix(dst.suffix + '.tmp')
            
            # Copy file
            shutil.copy2(src, temp_dst)
            
            # Preserve permissions if requested
            if preserve_permissions:
                src_stat = src.stat()
                os.chmod(temp_dst, src_stat.st_mode)
            
            # Atomic move
            temp_dst.replace(dst)
            
            logger.debug(f"Safe copy completed: {src} -> {dst}")
            return True
            
        except Exception as e:
            logger.error(f"Safe copy failed {src} -> {dst}: {e}")
            if 'temp_dst' in locals() and temp_dst.exists():
                temp_dst.unlink()
            return False
    
    def safe_remove(self, file_path: Path) -> bool:
        """Safely remove file"""
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Safe remove completed: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Safe remove failed for {file_path}: {e}")
            return False
    
    def cleanup_old_files(self, directory: Path, max_age_days: int = 30) -> int:
        """Cleanup old files from directory"""
        if not directory.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            removed_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to remove old file {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Cleanup error in {directory}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old files from {directory}")
        
        return removed_count
    
    def get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory}: {e}")
        
        return total_size
    
    def enforce_size_limits(self) -> Dict[str, Any]:
        """Enforce size limits on cache directories"""
        enforcement_results = {}
        
        for location in self.detector.writable_locations:
            if location.size_limit_mb is None:
                continue
            
            try:
                # Check if any of our paths are in this location
                relevant_paths = []
                for name, path in self.ml_predictor_paths.items():
                    if self._is_path_under(path, location.path):
                        relevant_paths.append((name, path))
                
                if not relevant_paths:
                    continue
                
                # Calculate total size
                total_size_mb = 0
                for name, path in relevant_paths:
                    if path.exists():
                        size_bytes = self.get_directory_size(path)
                        total_size_mb += size_bytes / (1024 * 1024)
                
                # Check if over limit
                if total_size_mb > location.size_limit_mb:
                    excess_mb = total_size_mb - location.size_limit_mb
                    logger.warning(f"Size limit exceeded for {location.purpose}: {total_size_mb:.1f}MB > {location.size_limit_mb}MB")
                    
                    # Cleanup strategies
                    cleaned_mb = 0
                    for name, path in relevant_paths:
                        if cleaned_mb >= excess_mb:
                            break
                        
                        if 'cache' in name or 'temp' in name:
                            # More aggressive cleanup for cache directories
                            removed_files = self.cleanup_old_files(path, max_age_days=7)
                            new_size = self.get_directory_size(path) / (1024 * 1024)
                            cleaned_mb += (total_size_mb - new_size)
                        elif 'predictions' in name:
                            # Less aggressive for predictions
                            removed_files = self.cleanup_old_files(path, max_age_days=14)
                    
                    enforcement_results[location.purpose] = {
                        'size_limit_mb': location.size_limit_mb,
                        'initial_size_mb': total_size_mb,
                        'cleaned_mb': cleaned_mb,
                        'final_size_mb': total_size_mb - cleaned_mb,
                        'action_taken': 'cleanup'
                    }
                else:
                    enforcement_results[location.purpose] = {
                        'size_limit_mb': location.size_limit_mb,
                        'current_size_mb': total_size_mb,
                        'action_taken': 'none'
                    }
                    
            except Exception as e:
                logger.error(f"Size enforcement error for {location.purpose}: {e}")
                enforcement_results[location.purpose] = {
                    'error': str(e),
                    'action_taken': 'failed'
                }
        
        return enforcement_results
    
    def _is_path_under(self, path: Path, parent: Path) -> bool:
        """Check if path is under parent directory"""
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False
    
    def setup_systemd_integration(self) -> Dict[str, bool]:
        """Setup systemd user service integration"""
        integration_results = {
            'service_file_created': False,
            'service_enabled': False,
            'timer_created': False,
            'timer_enabled': False
        }
        
        try:
            systemd_user_dir = Path.home() / ".config" / "systemd" / "user"
            systemd_user_dir.mkdir(parents=True, exist_ok=True)
            
            # Create service file
            service_content = self._generate_systemd_service()
            service_file = systemd_user_dir / "ml-shader-predictor.service"
            
            with self.atomic_write(service_file) as f:
                f.write(service_content)
            
            integration_results['service_file_created'] = True
            
            # Create timer for periodic tasks
            timer_content = self._generate_systemd_timer()
            timer_file = systemd_user_dir / "ml-shader-predictor.timer"
            
            with self.atomic_write(timer_file) as f:
                f.write(timer_content)
            
            integration_results['timer_created'] = True
            
            # Reload systemd and enable services
            try:
                subprocess.run(['systemctl', '--user', 'daemon-reload'], 
                              check=True, timeout=10)
                
                subprocess.run(['systemctl', '--user', 'enable', 'ml-shader-predictor.service'], 
                              check=True, timeout=10)
                integration_results['service_enabled'] = True
                
                subprocess.run(['systemctl', '--user', 'enable', 'ml-shader-predictor.timer'], 
                              check=True, timeout=10)
                integration_results['timer_enabled'] = True
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Systemd service management failed: {e}")
        
        except Exception as e:
            logger.error(f"Systemd integration setup failed: {e}")
        
        return integration_results
    
    def _generate_systemd_service(self) -> str:
        """Generate systemd service file content"""
        python_path = sys.executable
        service_script = Path(__file__).parent / "steam_platform_integration.py"
        
        return f"""[Unit]
Description=ML Shader Predictor for Steam
Documentation=https://github.com/your-repo/ml-shader-predictor
After=steam.service graphical-session.target
Wants=steam.service

[Service]
Type=notify
ExecStart={python_path} {service_script} --service-mode
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
TimeoutStartSec=30
TimeoutStopSec=30

# Environment
Environment=PYTHONPATH={Path(__file__).parent}
Environment=XDG_RUNTIME_DIR=%i
Environment=STEAMOS_ML_PREDICTOR=1

# Resource limits
MemoryMax=512M
CPUQuota=50%

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={self.ml_predictor_paths['cache_root']} {self.ml_predictor_paths['data_root']} {self.ml_predictor_paths['config_root']}

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ml-shader-predictor

[Install]
WantedBy=default.target
Also=ml-shader-predictor.timer
"""
    
    def _generate_systemd_timer(self) -> str:
        """Generate systemd timer file content"""
        return """[Unit]
Description=ML Shader Predictor Maintenance Timer
Documentation=https://github.com/your-repo/ml-shader-predictor
Requires=ml-shader-predictor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
"""
    
    def setup_flatpak_integration(self) -> Dict[str, bool]:
        """Setup Flatpak sandbox integration"""
        if self.detector.os_info.filesystem_type != FilesystemType.FLATPAK_SANDBOX:
            return {'not_in_flatpak': True}
        
        integration_results = {
            'permissions_verified': False,
            'steam_access_configured': False,
            'cache_directories_configured': False
        }
        
        try:
            # Verify Flatpak permissions
            flatpak_info = os.environ.get('FLATPAK_ID', '')
            if flatpak_info:
                logger.info(f"Running in Flatpak: {flatpak_info}")
                
                # Check filesystem permissions
                steam_accessible = self._check_steam_access_in_flatpak()
                integration_results['steam_access_configured'] = steam_accessible
                
                # Setup cache directories in Flatpak-accessible locations
                cache_setup = self._setup_flatpak_cache_directories()
                integration_results['cache_directories_configured'] = cache_setup
                
                integration_results['permissions_verified'] = True
        
        except Exception as e:
            logger.error(f"Flatpak integration error: {e}")
        
        return integration_results
    
    def _check_steam_access_in_flatpak(self) -> bool:
        """Check if Steam directories are accessible in Flatpak"""
        steam_paths = [
            Path.home() / ".steam",
            Path.home() / ".local" / "share" / "Steam"
        ]
        
        for steam_path in steam_paths:
            if steam_path.exists() and os.access(steam_path, os.R_OK):
                return True
        
        return False
    
    def _setup_flatpak_cache_directories(self) -> bool:
        """Setup cache directories for Flatpak environment"""
        try:
            # Use XDG directories that are accessible in Flatpak
            xdg_cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
            xdg_data = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
            
            flatpak_cache = xdg_cache / "ml-shader-predictor"
            flatpak_data = xdg_data / "ml-shader-predictor"
            
            flatpak_cache.mkdir(parents=True, exist_ok=True)
            flatpak_data.mkdir(parents=True, exist_ok=True)
            
            # Update ML predictor paths for Flatpak
            self.ml_predictor_paths.update({
                'cache_root': flatpak_cache,
                'data_root': flatpak_data,
                'models': flatpak_data / "models",
                'shader_cache': flatpak_cache / "shader_cache",
                'predictions': flatpak_cache / "predictions"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Flatpak cache setup error: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'steamos_info': {
                'version': self.detector.os_info.version.value,
                'filesystem_type': self.detector.os_info.filesystem_type.value,
                'is_readonly_root': self.detector.os_info.is_readonly_root,
                'boot_mode': self.detector.os_info.boot_mode,
                'deck_variant': self.detector.os_info.deck_variant,
                'build_id': self.detector.os_info.build_id,
                'update_channel': self.detector.os_info.update_channel
            },
            'writable_locations': [
                {
                    'path': str(loc.path),
                    'purpose': loc.purpose,
                    'size_limit_mb': loc.size_limit_mb,
                    'temporary': loc.temporary,
                    'persistent': loc.persistent,
                    'requires_sudo': loc.requires_sudo,
                    'flatpak_accessible': loc.flatpak_accessible
                }
                for loc in self.detector.writable_locations
            ],
            'steam_paths': {name: str(path) for name, path in self.detector.steam_paths.items()},
            'ml_predictor_paths': {name: str(path) for name, path in self.ml_predictor_paths.items()},
            'disk_usage': {
                name: {
                    'size_mb': self.get_directory_size(path) / (1024 * 1024) if path.exists() else 0,
                    'exists': path.exists()
                }
                for name, path in self.ml_predictor_paths.items()
            }
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global filesystem manager instance
_filesystem_manager: Optional[SteamOSFilesystemManager] = None

def get_filesystem_manager() -> SteamOSFilesystemManager:
    """Get or create the global filesystem manager"""
    global _filesystem_manager
    if _filesystem_manager is None:
        _filesystem_manager = SteamOSFilesystemManager()
    return _filesystem_manager

def get_cache_path(cache_type: str) -> Path:
    """Get cache path for specific cache type"""
    manager = get_filesystem_manager()
    return manager.get_cache_path(cache_type)

def get_config_path(config_type: str) -> Path:
    """Get configuration path for specific config type"""
    manager = get_filesystem_manager()
    return manager.get_config_path(config_type)

@contextmanager
def atomic_json_write(file_path: Path, data: Any):
    """Context manager for atomic JSON writes"""
    manager = get_filesystem_manager()
    with manager.atomic_write_json(file_path) as f:
        json.dump(data, f, indent=2)

# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def main():
    """Main entry point for testing"""
    print("\n💾 SteamOS Filesystem Manager Test")
    print("=" * 45)
    
    manager = SteamOSFilesystemManager()
    
    # Display system information
    system_info = manager.get_system_info()
    
    print(f"\n🔍 SteamOS Detection:")
    steamos_info = system_info['steamos_info']
    print(f"  Version: {steamos_info['version']}")
    print(f"  Filesystem: {steamos_info['filesystem_type']}")
    print(f"  Read-only root: {steamos_info['is_readonly_root']}")
    print(f"  Boot mode: {steamos_info['boot_mode']}")
    print(f"  Steam Deck variant: {steamos_info['deck_variant']}")
    print(f"  Update channel: {steamos_info['update_channel']}")
    
    print(f"\n📁 Writable Locations:")
    for location in system_info['writable_locations']:
        print(f"  {location['purpose']}: {location['path']}")
        if location['size_limit_mb']:
            print(f"    Size limit: {location['size_limit_mb']}MB")
        print(f"    Flatpak accessible: {location['flatpak_accessible']}")
    
    print(f"\n🎮 Steam Paths:")
    for name, path in system_info['steam_paths'].items():
        print(f"  {name}: {path}")
    
    print(f"\n🧠 ML Predictor Paths:")
    for name, path in system_info['ml_predictor_paths'].items():
        disk_info = system_info['disk_usage'][name]
        status = "✅" if disk_info['exists'] else "❌"
        size_mb = disk_info['size_mb']
        print(f"  {status} {name}: {path} ({size_mb:.1f}MB)")
    
    # Test atomic operations
    print(f"\n⚛️ Testing Atomic Operations:")
    test_file = manager.get_cache_path('test') / "atomic_test.json"
    test_data = {"test": True, "timestamp": time.time()}
    
    try:
        with atomic_json_write(test_file, test_data):
            pass
        print(f"  ✅ Atomic JSON write: {test_file}")
        
        # Verify the file
        if test_file.exists():
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
                if loaded_data == test_data:
                    print(f"  ✅ Data verification successful")
                else:
                    print(f"  ❌ Data verification failed")
        
        # Cleanup
        manager.safe_remove(test_file)
        print(f"  ✅ Safe file removal")
        
    except Exception as e:
        print(f"  ❌ Atomic operations test failed: {e}")
    
    # Test size enforcement
    print(f"\n📏 Testing Size Enforcement:")
    enforcement_results = manager.enforce_size_limits()
    for location, result in enforcement_results.items():
        print(f"  {location}: {result.get('action_taken', 'unknown')}")
        if 'current_size_mb' in result:
            print(f"    Current size: {result['current_size_mb']:.1f}MB")
    
    # Test system integration
    print(f"\n🔧 Testing System Integration:")
    systemd_results = manager.setup_systemd_integration()
    for component, success in systemd_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {component}")
    
    flatpak_results = manager.setup_flatpak_integration()
    for component, success in flatpak_results.items():
        if component != 'not_in_flatpak':
            status = "✅" if success else "❌"
            print(f"  {status} Flatpak {component}")
    
    print(f"\n✅ SteamOS Filesystem Manager test completed")

if __name__ == "__main__":
    main()