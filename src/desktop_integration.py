#!/usr/bin/env python3
"""
Desktop Integration for ML Shader Predictor

This module handles desktop environment integration including:
- System tray/notification area integration
- Desktop notifications
- File manager integration
- Application menu registration
- Autostart management
- KDE/GNOME specific integrations
"""

import os
import sys
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DesktopEnvironment(Enum):
    """Supported desktop environments"""
    KDE = "kde"
    GNOME = "gnome" 
    XFCE = "xfce"
    UNKNOWN = "unknown"


class NotificationUrgency(Enum):
    """Notification urgency levels"""
    LOW = "low"
    NORMAL = "normal"
    CRITICAL = "critical"


@dataclass
class NotificationConfig:
    """Desktop notification configuration"""
    enabled: bool
    urgency: NotificationUrgency
    timeout_ms: int
    show_progress: bool
    sound_enabled: bool


@dataclass
class SystemTrayConfig:
    """System tray configuration"""
    enabled: bool
    show_status: bool
    show_progress: bool
    menu_items: List[str]


class DesktopIntegration:
    """Desktop environment integration manager"""
    
    def __init__(self, install_dir: Path):
        self.install_dir = install_dir
        self.desktop_env = self._detect_desktop_environment()
        self.notification_config = self._load_notification_config()
        self.tray_config = self._load_tray_config()
        
        # Integration components
        self.system_tray = None
        self.notification_handler = None
        
        # Desktop paths
        self.desktop_paths = self._get_desktop_paths()
        
        # Initialize components
        self._initialize_components()
    
    def _detect_desktop_environment(self) -> DesktopEnvironment:
        """Detect current desktop environment"""
        try:
            # Check environment variables
            desktop_env = os.getenv('XDG_CURRENT_DESKTOP', '').lower()
            kde_session = os.getenv('KDE_SESSION_VERSION')
            gnome_session = os.getenv('GNOME_DESKTOP_SESSION_ID')
            
            if 'kde' in desktop_env or kde_session:
                logger.info("Detected KDE desktop environment")
                return DesktopEnvironment.KDE
            elif 'gnome' in desktop_env or gnome_session:
                logger.info("Detected GNOME desktop environment")
                return DesktopEnvironment.GNOME
            elif 'xfce' in desktop_env:
                logger.info("Detected XFCE desktop environment")
                return DesktopEnvironment.XFCE
            
            logger.warning("Unknown desktop environment")
            return DesktopEnvironment.UNKNOWN
            
        except Exception as e:
            logger.error(f"Desktop environment detection failed: {e}")
            return DesktopEnvironment.UNKNOWN
    
    def _get_desktop_paths(self) -> Dict[str, Path]:
        """Get desktop integration paths"""
        home = Path.home()
        
        paths = {
            "desktop_files": home / ".local/share/applications",
            "icons": home / ".local/share/icons",
            "autostart": home / ".config/autostart",
            "mime_types": home / ".local/share/mime/packages",
            "menu": home / ".config/menus/applications-merged"
        }
        
        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def _load_notification_config(self) -> NotificationConfig:
        """Load notification configuration"""
        config_file = Path.home() / ".config/ml-shader-predictor/notifications.json"
        
        default_config = NotificationConfig(
            enabled=True,
            urgency=NotificationUrgency.NORMAL,
            timeout_ms=5000,
            show_progress=True,
            sound_enabled=False
        )
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    return NotificationConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load notification config: {e}")
        
        return default_config
    
    def _load_tray_config(self) -> SystemTrayConfig:
        """Load system tray configuration"""
        config_file = Path.home() / ".config/ml-shader-predictor/tray.json"
        
        default_config = SystemTrayConfig(
            enabled=True,
            show_status=True,
            show_progress=True,
            menu_items=[
                "status",
                "separator",
                "start_service",
                "stop_service", 
                "separator",
                "configure",
                "monitor",
                "separator",
                "quit"
            ]
        )
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    return SystemTrayConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load tray config: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize desktop integration components"""
        try:
            # Initialize notification handler
            self.notification_handler = DesktopNotificationHandler(
                self.notification_config,
                self.desktop_env
            )
            
            # Initialize system tray (if desktop supports it)
            if self.tray_config.enabled and self._supports_system_tray():
                self.system_tray = SystemTrayIntegration(
                    self.tray_config,
                    self.desktop_env,
                    self.install_dir
                )
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    def _supports_system_tray(self) -> bool:
        """Check if desktop environment supports system tray"""
        return self.desktop_env in [DesktopEnvironment.KDE, DesktopEnvironment.GNOME]
    
    def install_desktop_integration(self) -> bool:
        """Install desktop integration files"""
        try:
            success = True
            
            # Install desktop entry
            success &= self._install_desktop_entry()
            
            # Install icons
            success &= self._install_icons()
            
            # Install MIME type associations
            success &= self._install_mime_types()
            
            # Install application menu entry
            success &= self._install_menu_entry()
            
            # Register autostart (optional)
            if self._should_enable_autostart():
                success &= self._install_autostart()
            
            # Update desktop database
            success &= self._update_desktop_database()
            
            logger.info(f"Desktop integration installed: {'SUCCESS' if success else 'PARTIAL'}")
            return success
            
        except Exception as e:
            logger.error(f"Desktop integration installation failed: {e}")
            return False
    
    def _install_desktop_entry(self) -> bool:
        """Install main application desktop entry"""
        try:
            desktop_file = self.desktop_paths["desktop_files"] / "ml-shader-predictor.desktop"
            
            # Read the desktop entry template
            template_file = self.install_dir / "ml-shader-predictor.desktop"
            if not template_file.exists():
                logger.error("Desktop entry template not found")
                return False
            
            with open(template_file, 'r') as f:
                desktop_content = f.read()
            
            # Update paths in desktop entry
            desktop_content = desktop_content.replace(
                "/usr/local/bin/ml-shader-predictor",
                str(self.install_dir / "bin" / "ml-shader-predictor")
            )
            
            # Write desktop entry
            with open(desktop_file, 'w') as f:
                f.write(desktop_content)
            
            desktop_file.chmod(0o755)
            logger.info("Installed desktop entry")
            return True
            
        except Exception as e:
            logger.error(f"Desktop entry installation failed: {e}")
            return False
    
    def _install_icons(self) -> bool:
        """Install application icons"""
        try:
            # Icon sizes to install
            icon_sizes = ["16x16", "32x32", "48x48", "64x64", "128x128", "256x256"]
            
            for size in icon_sizes:
                size_dir = self.desktop_paths["icons"] / "hicolor" / size / "apps"
                size_dir.mkdir(parents=True, exist_ok=True)
                
                # Create simple icon (in production, use proper icon files)
                icon_file = size_dir / "ml-shader-predictor.png"
                if not icon_file.exists():
                    # Create placeholder icon using ImageMagick if available
                    try:
                        subprocess.run([
                            'convert', '-size', size.replace('x', 'x'),
                            'xc:blue', '-fill', 'white',
                            '-gravity', 'center', '-annotate', '0', 'ML',
                            str(icon_file)
                        ], check=True, capture_output=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # Fallback: create empty file (desktop will use default icon)
                        icon_file.touch()
            
            logger.info("Installed application icons")
            return True
            
        except Exception as e:
            logger.error(f"Icon installation failed: {e}")
            return False
    
    def _install_mime_types(self) -> bool:
        """Install MIME type associations"""
        try:
            mime_file = self.desktop_paths["mime_types"] / "ml-shader-predictor.xml"
            
            mime_content = '''<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
    <mime-type type="application/x-steam-shader">
        <comment>Steam Shader Cache File</comment>
        <comment xml:lang="en">Steam Shader Cache File</comment>
        <glob pattern="*.foz"/>
        <glob pattern="*.vkpipelinecache"/>
    </mime-type>
    <mime-type type="application/x-vulkan-pipeline">
        <comment>Vulkan Pipeline Cache</comment>
        <comment xml:lang="en">Vulkan Pipeline Cache</comment>
        <glob pattern="*.vkpipelinecache"/>
    </mime-type>
    <mime-type type="application/x-spirv-shader">
        <comment>SPIR-V Shader Binary</comment>
        <comment xml:lang="en">SPIR-V Shader Binary</comment>
        <glob pattern="*.spv"/>
    </mime-type>
</mime-info>'''
            
            with open(mime_file, 'w') as f:
                f.write(mime_content)
            
            # Update MIME database
            try:
                subprocess.run(['update-mime-database', str(self.desktop_paths["mime_types"].parent)],
                             check=True, capture_output=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to update MIME database")
            
            logger.info("Installed MIME type associations")
            return True
            
        except Exception as e:
            logger.error(f"MIME type installation failed: {e}")
            return False
    
    def _install_menu_entry(self) -> bool:
        """Install application menu entry"""
        try:
            if self.desktop_env == DesktopEnvironment.KDE:
                return self._install_kde_menu_entry()
            elif self.desktop_env == DesktopEnvironment.GNOME:
                return self._install_gnome_menu_entry()
            
            logger.info("Menu entry installation not needed for this desktop")
            return True
            
        except Exception as e:
            logger.error(f"Menu entry installation failed: {e}")
            return False
    
    def _install_kde_menu_entry(self) -> bool:
        """Install KDE-specific menu entry"""
        try:
            menu_file = self.desktop_paths["menu"] / "ml-shader-predictor.menu"
            
            menu_content = '''<!DOCTYPE Menu PUBLIC "-//freedesktop//DTD Menu 1.0//EN"
    "http://www.freedesktop.org/standards/menu-spec/1.0/menu.dtd">
<Menu>
    <Name>Applications</Name>
    <Menu>
        <Name>System</Name>
        <Directory>System.directory</Directory>
        <Include>
            <Filename>ml-shader-predictor.desktop</Filename>
        </Include>
    </Menu>
</Menu>'''
            
            with open(menu_file, 'w') as f:
                f.write(menu_content)
            
            logger.info("Installed KDE menu entry")
            return True
            
        except Exception as e:
            logger.error(f"KDE menu entry installation failed: {e}")
            return False
    
    def _install_gnome_menu_entry(self) -> bool:
        """Install GNOME-specific menu entry"""
        # GNOME uses desktop files directly, no additional menu file needed
        logger.info("GNOME menu entry handled by desktop file")
        return True
    
    def _should_enable_autostart(self) -> bool:
        """Check if autostart should be enabled"""
        # Check if user wants autostart (could be a config option)
        return False  # Default to false for now
    
    def _install_autostart(self) -> bool:
        """Install autostart entry"""
        try:
            autostart_file = self.desktop_paths["autostart"] / "ml-shader-predictor.desktop"
            
            autostart_content = f'''[Desktop Entry]
Type=Application
Name=ML Shader Predictor Service
Comment=Start ML Shader Predictor background service
Exec={self.install_dir}/bin/ml-shader-predictor --service --background
Icon=ml-shader-predictor
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
StartupNotify=false
'''
            
            with open(autostart_file, 'w') as f:
                f.write(autostart_content)
            
            logger.info("Installed autostart entry")
            return True
            
        except Exception as e:
            logger.error(f"Autostart installation failed: {e}")
            return False
    
    def _update_desktop_database(self) -> bool:
        """Update desktop database"""
        try:
            subprocess.run(['update-desktop-database', str(self.desktop_paths["desktop_files"])],
                         check=True, capture_output=True)
            logger.info("Updated desktop database")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Failed to update desktop database")
            return False
    
    def show_notification(self, title: str, message: str, 
                         urgency: Optional[NotificationUrgency] = None) -> bool:
        """Show desktop notification"""
        if self.notification_handler:
            return self.notification_handler.show_notification(title, message, urgency)
        return False
    
    def show_progress_notification(self, title: str, progress: float, 
                                 message: Optional[str] = None) -> bool:
        """Show progress notification"""
        if self.notification_handler:
            return self.notification_handler.show_progress_notification(title, progress, message)
        return False
    
    def uninstall_desktop_integration(self) -> bool:
        """Uninstall desktop integration"""
        try:
            success = True
            
            # Remove desktop files
            desktop_files = [
                self.desktop_paths["desktop_files"] / "ml-shader-predictor.desktop",
                self.desktop_paths["autostart"] / "ml-shader-predictor.desktop"
            ]
            
            for file in desktop_files:
                if file.exists():
                    try:
                        file.unlink()
                        logger.info(f"Removed {file}")
                    except Exception as e:
                        logger.error(f"Failed to remove {file}: {e}")
                        success = False
            
            # Remove MIME types
            mime_file = self.desktop_paths["mime_types"] / "ml-shader-predictor.xml"
            if mime_file.exists():
                mime_file.unlink()
                subprocess.run(['update-mime-database', str(mime_file.parent.parent)],
                             capture_output=True)
            
            # Remove menu entries
            menu_file = self.desktop_paths["menu"] / "ml-shader-predictor.menu"
            if menu_file.exists():
                menu_file.unlink()
            
            # Update desktop database
            self._update_desktop_database()
            
            return success
            
        except Exception as e:
            logger.error(f"Desktop integration uninstall failed: {e}")
            return False


class DesktopNotificationHandler:
    """Desktop notification handler"""
    
    def __init__(self, config: NotificationConfig, desktop_env: DesktopEnvironment):
        self.config = config
        self.desktop_env = desktop_env
        self.notification_cmd = self._detect_notification_command()
    
    def _detect_notification_command(self) -> Optional[List[str]]:
        """Detect available notification command"""
        commands = [
            ['notify-send'],  # Standard freedesktop
            ['kdialog', '--passivepopup'],  # KDE
            ['zenity', '--notification'],  # GNOME fallback
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd + ['--version'], capture_output=True, check=True, timeout=2)
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None
    
    def show_notification(self, title: str, message: str,
                         urgency: Optional[NotificationUrgency] = None) -> bool:
        """Show desktop notification"""
        if not self.config.enabled or not self.notification_cmd:
            return False
        
        try:
            urgency_level = urgency or self.config.urgency
            timeout = self.config.timeout_ms
            
            if self.notification_cmd[0] == 'notify-send':
                cmd = self.notification_cmd + [
                    '--urgency', urgency_level.value,
                    '--expire-time', str(timeout),
                    '--app-name', 'ML Shader Predictor',
                    '--icon', 'ml-shader-predictor',
                    title, message
                ]
            elif self.notification_cmd[0] == 'kdialog':
                cmd = self.notification_cmd + [message, str(timeout // 1000)]
            else:
                cmd = self.notification_cmd + [f'--text={title}: {message}']
            
            subprocess.run(cmd, capture_output=True, timeout=5)
            return True
            
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return False
    
    def show_progress_notification(self, title: str, progress: float,
                                 message: Optional[str] = None) -> bool:
        """Show progress notification"""
        if not self.config.show_progress:
            return False
        
        progress_text = f"{title}: {progress:.1%}"
        if message:
            progress_text += f" - {message}"
        
        return self.show_notification("ML Shader Predictor", progress_text)


class SystemTrayIntegration:
    """System tray integration"""
    
    def __init__(self, config: SystemTrayConfig, desktop_env: DesktopEnvironment,
                 install_dir: Path):
        self.config = config
        self.desktop_env = desktop_env
        self.install_dir = install_dir
        self.active = False
    
    def start(self):
        """Start system tray integration"""
        self.active = True
        logger.info("System tray integration started")
    
    def stop(self):
        """Stop system tray integration"""
        self.active = False
        logger.info("System tray integration stopped")


def main():
    """Main entry point with command line argument handling"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Shader Predictor Desktop Integration")
    parser.add_argument("--install", action="store_true", help="Install desktop integration")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall desktop integration")
    parser.add_argument("--test", action="store_true", help="Test desktop integration")
    
    args = parser.parse_args()
    
    install_dir = Path(__file__).parent.parent
    integration = DesktopIntegration(install_dir)
    
    if args.install:
        print("🖥️  Installing Desktop Integration")
        if integration.install_desktop_integration():
            print("✅ Desktop integration installed successfully")
            sys.exit(0)
        else:
            print("❌ Desktop integration installation failed")
            sys.exit(1)
    
    elif args.uninstall:
        print("🗑️  Uninstalling Desktop Integration")
        if integration.uninstall_desktop_integration():
            print("✅ Desktop integration uninstalled successfully")
            sys.exit(0)
        else:
            print("❌ Desktop integration uninstall failed")
            sys.exit(1)
    
    elif args.test:
        print("🖥️  Desktop Integration Test")
        print("=" * 40)
        
        print(f"Desktop environment: {integration.desktop_env.value}")
        print(f"Notification support: {integration.notification_handler is not None}")
        print(f"System tray support: {integration.system_tray is not None}")
        
        # Test notification
        if integration.notification_handler:
            success = integration.show_notification(
                "ML Shader Predictor", 
                "Desktop integration test notification"
            )
            print(f"Test notification: {'SUCCESS' if success else 'FAILED'}")
        
        print("\n🚀 Test completed!")
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()