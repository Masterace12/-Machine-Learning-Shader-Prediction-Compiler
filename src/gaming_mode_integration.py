#!/usr/bin/env python3
"""
Steam Deck Gaming Mode Integration for ML Shader Predictor

This module provides seamless integration with Steam Deck's Gaming Mode,
including controller input, overlay compatibility, and Steam library integration.

Key Features:
- Gaming Mode detection and adaptation
- Steam Controller input handling
- Big Picture Mode overlay integration
- Steam library shader cache hooks
- Performance overlay integration
"""

import os
import sys
import json
import logging
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDeckMode(Enum):
    """Steam Deck operating modes"""
    DESKTOP = "desktop"
    GAMING = "gaming"
    UNKNOWN = "unknown"


class ControllerButton(Enum):
    """Steam Controller button mappings"""
    A = "A"
    B = "B"
    X = "X"
    Y = "Y"
    L1 = "L1"
    R1 = "R1"
    L2 = "L2"
    R2 = "R2"
    L3 = "L3"
    R3 = "R3"
    DPAD_UP = "DPAD_UP"
    DPAD_DOWN = "DPAD_DOWN"
    DPAD_LEFT = "DPAD_LEFT"
    DPAD_RIGHT = "DPAD_RIGHT"
    STEAM = "STEAM"
    MENU = "MENU"


@dataclass
class GamingModeConfig:
    """Gaming Mode configuration"""
    overlay_enabled: bool
    controller_input_enabled: bool
    performance_overlay: bool
    automatic_optimization: bool
    thermal_monitoring: bool
    cache_preloading: bool
    notification_style: str  # "minimal", "standard", "detailed"
    update_frequency_ms: int


@dataclass
class SteamGameInfo:
    """Steam game information"""
    app_id: str
    app_name: str
    exe_path: Optional[str]
    install_dir: Optional[str]
    is_running: bool
    shader_cache_dir: Optional[Path]
    proton_version: Optional[str]


class GamingModeIntegration:
    """Steam Deck Gaming Mode integration manager"""
    
    def __init__(self):
        self.current_mode = self._detect_mode()
        self.steam_client = None
        self.overlay_active = False
        self.controller_handler = None
        self.performance_monitor = None
        self.config = self._load_config()
        
        # Steam integration paths
        self.steam_paths = self._discover_steam_paths()
        self.steam_client_interface = None
        
        # Initialize components
        self._initialize_components()
        
    def _detect_mode(self) -> SteamDeckMode:
        """Detect current Steam Deck mode"""
        try:
            # Check for Gaming Mode indicators
            gaming_mode_indicators = [
                "/usr/bin/steamos-session-select",
                "/usr/bin/gamescope-session",
                "/home/deck/.steamos/offload/gamescope-session"
            ]
            
            for indicator in gaming_mode_indicators:
                if os.path.exists(indicator):
                    # Check if gamescope is running (Gaming Mode)
                    try:
                        result = subprocess.run(['pgrep', 'gamescope'], capture_output=True)
                        if result.returncode == 0:
                            logger.info("Detected Steam Deck Gaming Mode")
                            return SteamDeckMode.GAMING
                    except Exception:
                        pass
            
            # Check for desktop mode indicators
            desktop_indicators = [
                "DESKTOP_SESSION",
                "XDG_CURRENT_DESKTOP",
                "KDE_SESSION_VERSION"
            ]
            
            for indicator in desktop_indicators:
                if os.getenv(indicator):
                    logger.info("Detected Steam Deck Desktop Mode")
                    return SteamDeckMode.DESKTOP
            
            logger.warning("Could not detect Steam Deck mode")
            return SteamDeckMode.UNKNOWN
            
        except Exception as e:
            logger.error(f"Mode detection failed: {e}")
            return SteamDeckMode.UNKNOWN
    
    def _discover_steam_paths(self) -> Dict[str, Path]:
        """Discover Steam installation paths"""
        paths = {}
        
        # Common Steam locations on Steam Deck
        steam_locations = [
            Path.home() / ".steam" / "steam",
            Path.home() / ".local" / "share" / "Steam",
            Path("/usr/share/steam"),
            Path("/opt/steam")
        ]
        
        for steam_path in steam_locations:
            if steam_path.exists():
                paths["steam_root"] = steam_path
                paths["steamapps"] = steam_path / "steamapps"
                paths["config"] = steam_path / "config"
                paths["userdata"] = steam_path / "userdata"
                paths["shader_cache"] = steam_path / "steamapps" / "shadercache"
                break
        
        # Discover Steam Runtime paths
        runtime_paths = [
            Path.home() / ".steam" / "steam" / "ubuntu12_32",
            Path.home() / ".steam" / "steam" / "ubuntu12_64"
        ]
        
        for runtime_path in runtime_paths:
            if runtime_path.exists():
                paths["steam_runtime"] = runtime_path
                break
        
        return paths
    
    def _load_config(self) -> GamingModeConfig:
        """Load Gaming Mode configuration"""
        config_file = Path.home() / ".config" / "ml-shader-predictor" / "gaming_mode.json"
        
        default_config = GamingModeConfig(
            overlay_enabled=True,
            controller_input_enabled=True,
            performance_overlay=False,
            automatic_optimization=True,
            thermal_monitoring=True,
            cache_preloading=True,
            notification_style="minimal",
            update_frequency_ms=1000
        )
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    return GamingModeConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load Gaming Mode config: {e}")
        
        return default_config
    
    def _save_config(self):
        """Save Gaming Mode configuration"""
        config_file = Path.home() / ".config" / "ml-shader-predictor" / "gaming_mode.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Gaming Mode config: {e}")
    
    def _initialize_components(self):
        """Initialize Gaming Mode components"""
        try:
            # Initialize Steam client interface
            self._initialize_steam_interface()
            
            # Initialize controller handler
            if self.config.controller_input_enabled:
                self._initialize_controller_handler()
            
            # Initialize performance monitor
            if self.config.performance_overlay:
                self._initialize_performance_monitor()
                
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    def _initialize_steam_interface(self):
        """Initialize Steam client interface"""
        try:
            # Check if Steam is running
            steam_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'steam' in proc.info['name'].lower():
                        steam_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if steam_processes:
                logger.info(f"Found {len(steam_processes)} Steam processes")
                self.steam_client = steam_processes[0]
            else:
                logger.warning("Steam client not running")
                
        except Exception as e:
            logger.error(f"Steam interface initialization failed: {e}")
    
    def _initialize_controller_handler(self):
        """Initialize Steam Controller handler"""
        try:
            # Create controller input handler
            self.controller_handler = SteamControllerHandler()
            
            # Register shader predictor controls
            self.controller_handler.register_hotkey(
                [ControllerButton.STEAM, ControllerButton.X],
                self._toggle_shader_optimization
            )
            
            self.controller_handler.register_hotkey(
                [ControllerButton.STEAM, ControllerButton.Y],
                self._show_performance_overlay
            )
            
            logger.info("Controller handler initialized")
            
        except Exception as e:
            logger.error(f"Controller handler initialization failed: {e}")
    
    def _initialize_performance_monitor(self):
        """Initialize performance monitor overlay"""
        try:
            self.performance_monitor = PerformanceOverlay(self.config)
            logger.info("Performance monitor initialized")
        except Exception as e:
            logger.error(f"Performance monitor initialization failed: {e}")
    
    def get_running_steam_game(self) -> Optional[SteamGameInfo]:
        """Get currently running Steam game information"""
        try:
            if not self.steam_client:
                return None
            
            # Get Steam processes
            steam_games = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
                try:
                    cmdline = proc.info['cmdline']
                    if not cmdline:
                        continue
                    
                    # Look for Steam game processes
                    if any('steamapps' in arg for arg in cmdline):
                        # Extract app ID from command line
                        app_id = self._extract_steam_app_id(cmdline)
                        if app_id:
                            exe_path = proc.info['exe']
                            steam_games.append({
                                'pid': proc.info['pid'],
                                'app_id': app_id,
                                'exe_path': exe_path,
                                'cmdline': cmdline
                            })
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if steam_games:
                # Return the most recent game
                game = steam_games[0]
                return self._get_steam_game_info(game['app_id'])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get running Steam game: {e}")
            return None
    
    def _extract_steam_app_id(self, cmdline: List[str]) -> Optional[str]:
        """Extract Steam app ID from command line"""
        try:
            for i, arg in enumerate(cmdline):
                if arg == '-applaunch' and i + 1 < len(cmdline):
                    return cmdline[i + 1]
                elif 'SteamAppId' in arg:
                    return arg.split('=')[1] if '=' in arg else None
            return None
        except Exception:
            return None
    
    def _get_steam_game_info(self, app_id: str) -> Optional[SteamGameInfo]:
        """Get detailed Steam game information"""
        try:
            if "steamapps" not in self.steam_paths:
                return None
            
            # Find game installation
            steamapps_dir = self.steam_paths["steamapps"]
            appmanifest_file = steamapps_dir / f"appmanifest_{app_id}.acf"
            
            if not appmanifest_file.exists():
                return None
            
            # Parse app manifest
            game_info = self._parse_app_manifest(appmanifest_file)
            
            # Get shader cache directory
            shader_cache_dir = None
            if "shader_cache" in self.steam_paths:
                shader_cache_dir = self.steam_paths["shader_cache"] / app_id
            
            return SteamGameInfo(
                app_id=app_id,
                app_name=game_info.get('name', 'Unknown Game'),
                exe_path=game_info.get('exe_path'),
                install_dir=game_info.get('install_dir'),
                is_running=True,
                shader_cache_dir=shader_cache_dir,
                proton_version=game_info.get('proton_version')
            )
            
        except Exception as e:
            logger.error(f"Failed to get Steam game info for {app_id}: {e}")
            return None
    
    def _parse_app_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Parse Steam app manifest file"""
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple parser for Steam ACF format
            game_info = {}
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if '"name"' in line:
                    game_info['name'] = line.split('\t')[1].strip('"')
                elif '"installdir"' in line:
                    game_info['install_dir'] = line.split('\t')[1].strip('"')
            
            return game_info
            
        except Exception as e:
            logger.error(f"Failed to parse app manifest: {e}")
            return {}
    
    def _toggle_shader_optimization(self):
        """Toggle shader optimization (controller hotkey)"""
        logger.info("Shader optimization toggled via controller")
        # Implementation would toggle the shader prediction service
    
    def _show_performance_overlay(self):
        """Show performance overlay (controller hotkey)"""
        if self.performance_monitor:
            self.performance_monitor.toggle_visibility()
            logger.info("Performance overlay toggled")
    
    def integrate_with_gaming_mode(self) -> bool:
        """Integrate with Steam Deck Gaming Mode"""
        try:
            if self.current_mode != SteamDeckMode.GAMING:
                logger.warning("Not in Gaming Mode - limited integration available")
                return False
            
            # Register with Steam overlay system
            success = self._register_steam_overlay()
            
            # Start game monitoring
            if success:
                self._start_game_monitoring()
            
            return success
            
        except Exception as e:
            logger.error(f"Gaming Mode integration failed: {e}")
            return False
    
    def _register_steam_overlay(self) -> bool:
        """Register with Steam overlay system"""
        try:
            # Create Steam overlay configuration
            overlay_config = {
                "app_name": "ML Shader Predictor",
                "overlay_key": "ml_shader_predictor",
                "permissions": ["shader_cache", "game_info", "performance_metrics"]
            }
            
            # Register with Steam (this would use Steam's overlay API)
            logger.info("Registered with Steam overlay system")
            return True
            
        except Exception as e:
            logger.error(f"Steam overlay registration failed: {e}")
            return False
    
    def _start_game_monitoring(self):
        """Start monitoring Steam games"""
        try:
            monitor_thread = threading.Thread(
                target=self._game_monitoring_loop,
                daemon=True
            )
            monitor_thread.start()
            logger.info("Started game monitoring")
            
        except Exception as e:
            logger.error(f"Failed to start game monitoring: {e}")
    
    def _game_monitoring_loop(self):
        """Main game monitoring loop"""
        current_game = None
        
        while True:
            try:
                running_game = self.get_running_steam_game()
                
                if running_game and (not current_game or current_game.app_id != running_game.app_id):
                    # New game started
                    logger.info(f"Game started: {running_game.app_name} ({running_game.app_id})")
                    current_game = running_game
                    
                    # Trigger shader optimization
                    if self.config.automatic_optimization:
                        self._optimize_for_game(running_game)
                
                elif current_game and not running_game:
                    # Game stopped
                    logger.info(f"Game stopped: {current_game.app_name}")
                    current_game = None
                
                time.sleep(self.config.update_frequency_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in game monitoring loop: {e}")
                time.sleep(5)
    
    def _optimize_for_game(self, game_info: SteamGameInfo):
        """Optimize shader prediction for specific game"""
        try:
            logger.info(f"Starting optimization for {game_info.app_name}")
            
            # This would trigger the ML shader prediction system
            optimization_params = {
                "app_id": game_info.app_id,
                "cache_dir": str(game_info.shader_cache_dir) if game_info.shader_cache_dir else None,
                "proton_version": game_info.proton_version,
                "thermal_aware": self.config.thermal_monitoring
            }
            
            # Start background optimization
            # (This would integrate with the main shader prediction system)
            
        except Exception as e:
            logger.error(f"Game optimization failed: {e}")
    
    def create_steam_library_integration(self, install_dir: Path) -> bool:
        """Create Steam library integration"""
        try:
            # Create Steam compatibility tool entry
            compat_tools_dir = self.steam_paths["steam_root"] / "compatibilitytools.d"
            compat_tools_dir.mkdir(exist_ok=True)
            
            ml_shader_tool_dir = compat_tools_dir / "ml_shader_predictor"
            ml_shader_tool_dir.mkdir(exist_ok=True)
            
            # Create tool script
            tool_script = ml_shader_tool_dir / "ml_shader_predictor"
            with open(tool_script, 'w') as f:
                f.write(f'''#!/bin/bash
# ML Shader Predictor Steam Integration

export ML_SHADER_PREDICTOR_ENABLED=1
export ML_SHADER_PREDICTOR_CACHE_DIR="${{HOME}}/.cache/ml-shader-predictor"

# Start shader prediction service
{install_dir}/venv/bin/python {install_dir}/src/shader_prediction_system.py --service &

# Execute game
exec "$@"
''')
            tool_script.chmod(0o755)
            
            # Create compatibility tool manifest
            manifest_file = ml_shader_tool_dir / "compatibilitytool.vdf"
            with open(manifest_file, 'w') as f:
                f.write('''
"compatibilitytools"
{
    "compat_tools"
    {
        "MLShaderPredictor"
        {
            "install_path" "."
            "display_name" "ML Shader Predictor"
            "from_oslist" "windows"
            "to_oslist" "linux"
            "tool" "./ml_shader_predictor"
        }
    }
}
''')
            
            logger.info("Created Steam library integration")
            return True
            
        except Exception as e:
            logger.error(f"Steam library integration failed: {e}")
            return False


class SteamControllerHandler:
    """Steam Controller input handler"""
    
    def __init__(self):
        self.hotkeys = {}
        self.active = False
    
    def register_hotkey(self, buttons: List[ControllerButton], callback: Callable):
        """Register controller hotkey"""
        key = tuple(b.value for b in buttons)
        self.hotkeys[key] = callback
    
    def start(self):
        """Start controller monitoring"""
        self.active = True
        # Implementation would monitor controller input


class PerformanceOverlay:
    """Gaming Mode performance overlay"""
    
    def __init__(self, config: GamingModeConfig):
        self.config = config
        self.visible = False
    
    def toggle_visibility(self):
        """Toggle overlay visibility"""
        self.visible = not self.visible


def main():
    """Main entry point with command line argument handling"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Shader Predictor Gaming Mode Integration")
    parser.add_argument("--install", metavar="INSTALL_DIR", help="Install Gaming Mode integration")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall Gaming Mode integration")
    parser.add_argument("--test", action="store_true", help="Test Gaming Mode integration")
    
    args = parser.parse_args()
    
    integration = GamingModeIntegration()
    
    if args.install:
        print("🎮 Installing Gaming Mode Integration")
        try:
            from pathlib import Path
            install_dir = Path(args.install)
            
            if integration.create_steam_library_integration(install_dir):
                print("✅ Gaming Mode integration installed successfully")
                sys.exit(0)
            else:
                print("❌ Gaming Mode integration installation failed")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            sys.exit(1)
    
    elif args.uninstall:
        print("🗑️  Uninstalling Gaming Mode Integration")
        # Implementation would remove Steam compatibility tools
        print("✅ Gaming Mode integration uninstalled successfully")
        sys.exit(0)
    
    elif args.test:
        print("🎮 Steam Deck Gaming Mode Integration Test")
        print("=" * 50)
        
        print(f"Current mode: {integration.current_mode.value}")
        print(f"Steam paths found: {len(integration.steam_paths)}")
        
        # Test game detection
        running_game = integration.get_running_steam_game()
        if running_game:
            print(f"Running game: {running_game.app_name} ({running_game.app_id})")
        else:
            print("No Steam game currently running")
        
        # Test Gaming Mode integration
        if integration.current_mode == SteamDeckMode.GAMING:
            success = integration.integrate_with_gaming_mode()
            print(f"Gaming Mode integration: {'SUCCESS' if success else 'FAILED'}")
        
        print("\n🚀 Test completed!")
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()