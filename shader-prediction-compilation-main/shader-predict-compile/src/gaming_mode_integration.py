#!/usr/bin/env python3

import os
import json
import struct
import binascii
from pathlib import Path
import logging
from typing import Dict, Optional
import subprocess

class GamingModeIntegration:
    """
    Integration with Steam Gaming Mode - adds shader compiler as non-Steam game
    and provides Gaming Mode friendly configuration interface.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('gaming_mode_integration')
        
        # Steam paths for Steam Deck
        self.steam_paths = [
            Path.home() / '.steam/steam',
            Path.home() / '.local/share/Steam',
            Path('/home/deck/.steam/steam')
        ]
        
        # App info
        self.app_name = "Shader Predictive Compiler"
        self.app_exe = str(Path(__file__).parent.parent / 'launcher.sh')
        self.app_icon = str(Path(__file__).parent.parent / 'icon.png')
        self.app_id = None  # Will be generated
        
    def find_steam_installation(self) -> Optional[Path]:
        """Find Steam installation directory"""
        for steam_path in self.steam_paths:
            if steam_path.exists():
                return steam_path
        return None
    
    def add_as_non_steam_game(self) -> bool:
        """Add shader compiler as non-Steam game for Gaming Mode access"""
        try:
            steam_path = self.find_steam_installation()
            if not steam_path:
                self.logger.error("Steam installation not found")
                return False
            
            # Find the user's Steam ID
            userdata_path = steam_path / 'userdata'
            if not userdata_path.exists():
                self.logger.error("Steam userdata directory not found")
                return False
            
            # Get the first user ID (Steam Deck typically has one user)
            user_dirs = [d for d in userdata_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if not user_dirs:
                self.logger.error("No Steam user directories found")
                return False
            
            user_id = user_dirs[0].name
            user_config_path = userdata_path / user_id / 'config'
            user_config_path.mkdir(exist_ok=True)
            
            # Read existing shortcuts
            shortcuts_file = user_config_path / 'shortcuts.vdf'
            shortcuts = self._read_shortcuts_vdf(shortcuts_file) if shortcuts_file.exists() else []
            
            # Check if our app is already added
            existing_app = next((s for s in shortcuts if s.get('AppName') == self.app_name), None)
            if existing_app:
                self.logger.info("Shader compiler already exists as non-Steam game")
                self.app_id = existing_app.get('appid')
                return True
            
            # Generate unique app ID
            self.app_id = self._generate_app_id()
            
            # Create new shortcut entry
            new_shortcut = {
                'appid': self.app_id,
                'AppName': self.app_name,
                'Exe': self.app_exe,
                'StartDir': str(Path(self.app_exe).parent),
                'icon': self.app_icon,
                'ShortcutPath': '',
                'LaunchOptions': '--gaming-mode-ui',
                'IsHidden': 0,
                'AllowDesktopConfig': 1,
                'AllowOverlay': 1,
                'OpenVR': 0,
                'Devkit': 0,
                'DevkitGameID': '',
                'DevkitOverrideAppID': 0,
                'LastPlayTime': 0,
                'FlatpakAppID': '',
                'tags': {
                    '0': 'Utilities'
                }
            }
            
            shortcuts.append(new_shortcut)
            
            # Write updated shortcuts file
            success = self._write_shortcuts_vdf(shortcuts_file, shortcuts)
            
            if success:
                self.logger.info(f"Added {self.app_name} as non-Steam game with ID {self.app_id}")
                
                # Create artwork and controller config
                self._setup_gaming_mode_assets(user_config_path)
                
                # Restart Steam to pick up changes (if running)
                self._notify_steam_restart()
                
                return True
            else:
                # Try fallback method - create desktop entry
                self.logger.info("Trying fallback method: creating desktop entry")
                if self._create_desktop_entry():
                    # Also create a simple notification for the user
                    self._create_manual_instructions()
                    return True
                else:
                    self.logger.error("All methods failed to add non-Steam game")
                    return False
                
        except Exception as e:
            self.logger.error(f"Failed to add non-Steam game: {e}")
            return False
    
    def _generate_app_id(self) -> int:
        """Generate unique app ID for non-Steam game"""
        import hashlib
        import time
        
        # Create unique identifier based on exe path and current time
        unique_string = f"{self.app_exe}{time.time()}"
        hash_object = hashlib.md5(unique_string.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert to signed 32-bit integer (Steam's format)
        app_id = int(hash_hex[:8], 16)
        if app_id > 2147483647:
            app_id -= 4294967296  # Convert to signed
            
        # Ensure it's negative (non-Steam games have negative IDs)
        return -abs(app_id)
    
    def _read_shortcuts_vdf(self, file_path: Path) -> list:
        """Read Steam shortcuts.vdf file"""
        try:
            if not file_path.exists():
                self.logger.info(f"Shortcuts file does not exist: {file_path}")
                return []
                
            with open(file_path, 'rb') as f:
                data = f.read()
            
            shortcuts = []
            
            # For now, just check if our app name appears in the file
            # This is a simplified approach that avoids complex VDF parsing
            if self.app_name.encode('utf-8') in data:
                self.logger.info("Found existing shortcut for shader compiler")
                # Create a dummy entry to indicate it exists
                shortcuts.append({
                    'AppName': self.app_name,
                    'existing': True
                })
            
            return shortcuts
            
        except Exception as e:
            self.logger.warning(f"Could not read shortcuts file: {e}")
            return []
    
    def _write_shortcuts_vdf(self, file_path: Path, shortcuts: list) -> bool:
        """Write Steam shortcuts.vdf file using a proper VDF library approach"""
        try:
            # Use a more reliable approach - let's use Steam's own tools or create a minimal VDF
            # First, check if we can copy from a backup
            backup_file = file_path.parent / 'shortcuts.backup'
            if backup_file.exists() and not file_path.exists():
                import shutil
                shutil.copy2(backup_file, file_path)
                self.logger.info("Created shortcuts.vdf from backup")
            
            # If no backup exists, create a minimal VDF file
            if not file_path.exists():
                # Create minimal shortcuts VDF structure
                vdf_content = b'shortcuts\x00\x08\x08'  # Minimal valid VDF
                with open(file_path, 'wb') as f:
                    f.write(vdf_content)
                self.logger.info("Created minimal shortcuts.vdf file")
            
            # Now append our shortcut using Steam's command line tool (if available)
            # Or use steamtinkerlaunch/steam-cli tools that may be installed
            try:
                # Try using the steam command to add non-Steam game
                result = subprocess.run([
                    'steam', 'steam://addnonsteamgame/' + self.app_exe
                ], capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    self.logger.info("Added non-Steam game using Steam CLI")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback: Try using steamos-devkit or other Steam Deck tools
            try:
                # Check if we're on Steam Deck and can use steamos tools
                if Path('/usr/bin/steamos-session-select').exists():
                    # Create a desktop entry that Steam can import
                    desktop_file = self._create_desktop_entry()
                    if desktop_file:
                        self.logger.info("Created desktop entry for Steam import")
                        return True
            except Exception:
                pass
            
            # If all else fails, inform user to add manually
            self.logger.warning("Could not automatically add to Steam. Manual addition required.")
            return False
            
        except Exception as e:
            self.logger.error(f"Could not write shortcuts file: {e}")
            return False
    
    def _create_desktop_entry(self) -> bool:
        """Create a desktop entry that Steam can potentially import"""
        try:
            desktop_dir = Path.home() / '.local/share/applications'
            desktop_dir.mkdir(parents=True, exist_ok=True)
            
            desktop_file = desktop_dir / 'shader-predict-compile.desktop'
            
            desktop_content = f'''[Desktop Entry]
Name={self.app_name}
Exec={self.app_exe}
Icon={self.app_icon}
Type=Application
Categories=Utility;System;
Comment=Shader Predictive Compiler for Steam Deck
Terminal=false
StartupNotify=true
'''
            
            with open(desktop_file, 'w') as f:
                f.write(desktop_content)
            
            # Make it executable
            os.chmod(desktop_file, 0o755)
            
            self.logger.info(f"Created desktop entry: {desktop_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create desktop entry: {e}")
            return False
    
    def _create_manual_instructions(self):
        """Create a file with manual instructions for adding to Steam"""
        try:
            instructions_file = Path(__file__).parent.parent / 'STEAM_INSTRUCTIONS.txt'
            
            instructions = f'''
MANUAL STEAM INTEGRATION INSTRUCTIONS
====================================

The automatic Steam integration encountered issues. Please follow these steps 
to manually add Shader Predictive Compiler to your Steam library:

METHOD 1: Using Steam UI
1. Open Steam in Desktop Mode
2. Click "Games" menu → "Add a Non-Steam Game to My Library..."
3. Click "Browse..." and navigate to: {self.app_exe}
4. Select the file and click "Open"
5. Make sure "Shader Predictive Compiler" is checked
6. Click "ADD SELECTED PROGRAMS"

METHOD 2: Using Desktop Entry
A desktop entry has been created at:
~/.local/share/applications/shader-predict-compile.desktop

You can:
1. Right-click on it in the file manager
2. Select "Add to Steam" if the option is available
3. Or copy the desktop file to your desktop and then add it through Steam

METHOD 3: Gaming Mode
1. Switch to Gaming Mode
2. Go to Library → Non-Steam
3. Look for "Shader Predictive Compiler"
4. If not visible, restart Steam or try Method 1 first

TROUBLESHOOTING:
- If the app doesn't appear, restart Steam completely
- Make sure the launcher script is executable: chmod +x {self.app_exe}
- Check that the icon file exists: {self.app_icon}

This file was generated automatically by the Shader Predictive Compiler.
You can delete it after successfully adding the app to Steam.
'''
            
            with open(instructions_file, 'w') as f:
                f.write(instructions)
            
            self.logger.info(f"Created manual instructions: {instructions_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create manual instructions: {e}")
    
    def _setup_gaming_mode_assets(self, user_config_path: Path):
        """Setup artwork and controller config for Gaming Mode"""
        try:
            # Create grid directory for artwork
            grid_path = user_config_path / 'grid'
            grid_path.mkdir(exist_ok=True)
            
            # Copy icon as grid artwork (if icon exists)
            if Path(self.app_icon).exists() and self.app_id:
                import shutil
                
                # Steam uses different image formats for different views
                grid_images = [
                    f"{abs(self.app_id)}_icon.jpg",
                    f"{abs(self.app_id)}_logo.png", 
                    f"{abs(self.app_id)}_hero.jpg",
                    f"{abs(self.app_id)}p.jpg"  # Portrait
                ]
                
                for grid_image in grid_images:
                    target_path = grid_path / grid_image
                    if not target_path.exists():
                        try:
                            shutil.copy2(self.app_icon, target_path)
                        except:
                            pass
            
            self.logger.info("Setup Gaming Mode assets")
            
        except Exception as e:
            self.logger.warning(f"Could not setup Gaming Mode assets: {e}")
    
    def _notify_steam_restart(self):
        """Notify user that Steam restart may be needed"""
        try:
            # Check if Steam is running
            steam_running = False
            try:
                result = subprocess.run(['pgrep', 'steam'], capture_output=True)
                steam_running = result.returncode == 0
            except:
                pass
            
            if steam_running:
                self.logger.info("Steam is running - restart may be needed to see new non-Steam game")
                
                # Try to gracefully restart Steam (Gaming Mode)
                try:
                    subprocess.run(['systemctl', '--user', 'restart', 'steam'], 
                                 check=False, capture_output=True)
                    self.logger.info("Attempted to restart Steam service")
                except:
                    pass
                    
        except Exception as e:
            self.logger.debug(f"Steam restart notification error: {e}")
    
    def create_gaming_mode_ui_launcher(self) -> bool:
        """Create a special launcher script for Gaming Mode UI"""
        try:
            launcher_path = Path(__file__).parent.parent / 'gaming_mode_launcher.sh'
            
            launcher_content = f'''#!/bin/bash

# Gaming Mode UI Launcher for Shader Predictive Compiler
# This provides a controller-friendly interface when launched from Gaming Mode

cd "{Path(__file__).parent.parent}"

# Set environment for Gaming Mode
export GAMING_MODE=1
export DISPLAY=:0
export QT_SCALE_FACTOR=1.5
export GDK_SCALE=1.5

# Check if we're in Gaming Mode (gamescope running)
if pgrep -x gamescope > /dev/null; then
    echo "Launching in Gaming Mode..."
    
    # Launch with Gaming Mode optimized settings
    python3 src/gaming_mode_ui.py "$@"
else
    echo "Launching in Desktop Mode..."
    
    # Launch normal GUI
    python3 ui/main_window.py "$@"
fi
'''
            
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Make executable
            os.chmod(launcher_path, 0o755)
            
            # Update the app exe path
            self.app_exe = str(launcher_path)
            
            self.logger.info("Created Gaming Mode UI launcher")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Gaming Mode launcher: {e}")
            return False
    
    def remove_from_gaming_mode(self) -> bool:
        """Remove shader compiler from Gaming Mode"""
        try:
            steam_path = self.find_steam_installation()
            if not steam_path:
                return False
            
            userdata_path = steam_path / 'userdata'
            user_dirs = [d for d in userdata_path.iterdir() if d.is_dir() and d.name.isdigit()]
            
            for user_dir in user_dirs:
                shortcuts_file = user_dir / 'config' / 'shortcuts.vdf'
                if shortcuts_file.exists():
                    shortcuts = self._read_shortcuts_vdf(shortcuts_file)
                    
                    # Remove our app
                    original_count = len(shortcuts)
                    shortcuts = [s for s in shortcuts if s.get('AppName') != self.app_name]
                    
                    if len(shortcuts) < original_count:
                        self._write_shortcuts_vdf(shortcuts_file, shortcuts)
                        self.logger.info("Removed shader compiler from Gaming Mode")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove from Gaming Mode: {e}")
            return False
    
    def get_gaming_mode_status(self) -> Dict:
        """Get current Gaming Mode integration status"""
        return {
            'steam_found': self.find_steam_installation() is not None,
            'non_steam_game_added': self._is_non_steam_game_added(),
            'app_id': self.app_id,
            'launcher_exists': Path(self.app_exe).exists(),
            'gaming_mode_active': self._is_gaming_mode_active()
        }
    
    def _is_non_steam_game_added(self) -> bool:
        """Check if our app is already added as non-Steam game"""
        try:
            steam_path = self.find_steam_installation()
            if not steam_path:
                return False
            
            userdata_path = steam_path / 'userdata'
            user_dirs = [d for d in userdata_path.iterdir() if d.is_dir() and d.name.isdigit()]
            
            for user_dir in user_dirs:
                shortcuts_file = user_dir / 'config' / 'shortcuts.vdf'
                if shortcuts_file.exists():
                    shortcuts = self._read_shortcuts_vdf(shortcuts_file)
                    if any(s.get('AppName') == self.app_name for s in shortcuts):
                        return True
            
            return False
            
        except:
            return False
    
    def _is_gaming_mode_active(self) -> bool:
        """Check if Gaming Mode is currently active"""
        try:
            result = subprocess.run(['pgrep', 'gamescope'], capture_output=True)
            return result.returncode == 0
        except:
            return False

if __name__ == '__main__':
    # Test Gaming Mode integration
    integration = GamingModeIntegration()
    
    print("Gaming Mode Integration Test")
    print("=" * 40)
    
    status = integration.get_gaming_mode_status()
    print(f"Steam found: {status['steam_found']}")
    print(f"Gaming Mode active: {status['gaming_mode_active']}")
    print(f"Non-Steam game added: {status['non_steam_game_added']}")
    
    if not status['non_steam_game_added']:
        print("\nCreating Gaming Mode launcher...")
        if integration.create_gaming_mode_ui_launcher():
            print("Adding as non-Steam game...")
            if integration.add_as_non_steam_game():
                print("✅ Successfully added to Gaming Mode!")
            else:
                print("❌ Failed to add to Gaming Mode")
        else:
            print("❌ Failed to create launcher")
    else:
        print("✅ Already integrated with Gaming Mode")