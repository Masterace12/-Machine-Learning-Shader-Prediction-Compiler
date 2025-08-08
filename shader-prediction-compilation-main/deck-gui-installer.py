#!/usr/bin/env python3
"""
Steam Deck GUI Installer - Touch-friendly installation interface
Works on both LCD and OLED models
"""

import os
import sys
import subprocess
import threading
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

try:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk, GLib, Pango
except ImportError:
    print("Installing required GUI libraries...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "PyGObject"])
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk, GLib, Pango

class SteamDeckInstaller(Gtk.Window):
    def __init__(self):
        super().__init__(title="Shader Optimizer Installer for Steam Deck")
        self.set_default_size(800, 600)
        self.set_position(Gtk.WindowPosition.CENTER)
        
        # Make touch-friendly
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-button-images", True)
        
        # Detect Steam Deck
        self.is_steam_deck = self.detect_steam_deck()
        self.deck_model = self.get_deck_model()
        
        # Installation paths
        self.install_dir = Path.home() / ".local" / "share" / "shader-predict-compile"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="shader-install-"))
        
        # Setup UI
        self.setup_ui()
        
    def detect_steam_deck(self):
        """Detect if running on Steam Deck"""
        try:
            with open("/sys/devices/virtual/dmi/id/product_name", "r") as f:
                return "Jupiter" in f.read()
        except:
            return False
    
    def get_deck_model(self):
        """Detect LCD vs OLED model"""
        if not self.is_steam_deck:
            return "Unknown"
        
        try:
            with open("/sys/class/drm/card0-eDP-1/device/revision", "r") as f:
                revision = f.read().strip()
                if revision in ["0x00", "0x01", "0x02"]:
                    return "LCD"
                elif revision in ["0x03", "0x04", "0x05"]:
                    return "OLED"
        except:
            pass
        
        return "LCD/OLED"
    
    def setup_ui(self):
        """Create the GUI interface"""
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        main_box.set_margin_top(30)
        main_box.set_margin_bottom(30)
        main_box.set_margin_left(40)
        main_box.set_margin_right(40)
        self.add(main_box)
        
        # Header with icon
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        header_box.set_halign(Gtk.Align.CENTER)
        main_box.pack_start(header_box, False, False, 0)
        
        # Title
        title_label = Gtk.Label()
        title_label.set_markup("<span size='xx-large' weight='bold'>🎮 Shader Optimizer for Steam Deck</span>")
        header_box.pack_start(title_label, False, False, 0)
        
        # Steam Deck detection info
        if self.is_steam_deck:
            deck_info = Gtk.Label()
            deck_info.set_markup(f"<span size='large' foreground='#22c55e'>✓ Steam Deck {self.deck_model} Detected!</span>")
            main_box.pack_start(deck_info, False, False, 0)
        else:
            warning_label = Gtk.Label()
            warning_label.set_markup("<span foreground='#f59e0b'>⚠ Not running on Steam Deck - Installation will continue</span>")
            main_box.pack_start(warning_label, False, False, 0)
        
        # Description
        desc_label = Gtk.Label()
        desc_label.set_markup(
            "<span size='large'>Optimize game performance by improving shader compilation</span>\n"
            "<span>• Reduces stuttering in games</span>\n"
            "<span>• Faster loading times</span>\n"
            "<span>• Works with all Proton games</span>"
        )
        desc_label.set_line_wrap(True)
        desc_label.set_justify(Gtk.Justification.CENTER)
        main_box.pack_start(desc_label, False, False, 0)
        
        # Progress section
        self.progress_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.pack_start(self.progress_box, True, True, 0)
        
        # Progress bar
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_show_text(True)
        self.progress_box.pack_start(self.progress_bar, False, False, 0)
        
        # Status text
        self.status_label = Gtk.Label("Ready to install")
        self.status_label.set_line_wrap(True)
        self.progress_box.pack_start(self.status_label, False, False, 0)
        
        # Log view (hidden by default)
        self.log_expander = Gtk.Expander(label="Show Details")
        self.progress_box.pack_start(self.log_expander, True, True, 0)
        
        # Scrolled window for log
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(200)
        self.log_expander.add(scrolled)
        
        # Text view for logs
        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_monospace(True)
        self.log_buffer = self.log_view.get_buffer()
        scrolled.add(self.log_view)
        
        # Button box
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        button_box.set_halign(Gtk.Align.CENTER)
        main_box.pack_start(button_box, False, False, 0)
        
        # Install button (large and touch-friendly)
        self.install_button = Gtk.Button(label="Install Now")
        self.install_button.set_size_request(200, 60)
        self.install_button.get_style_context().add_class("suggested-action")
        
        # Make button text larger
        label = self.install_button.get_child()
        label.modify_font(Pango.FontDescription("16"))
        
        self.install_button.connect("clicked", self.on_install_clicked)
        button_box.pack_start(self.install_button, False, False, 0)
        
        # Cancel button
        self.cancel_button = Gtk.Button(label="Cancel")
        self.cancel_button.set_size_request(150, 60)
        self.cancel_button.set_sensitive(False)
        
        label = self.cancel_button.get_child()
        label.modify_font(Pango.FontDescription("14"))
        
        self.cancel_button.connect("clicked", self.on_cancel_clicked)
        button_box.pack_start(self.cancel_button, False, False, 0)
        
    def log(self, message, level="info"):
        """Add message to log"""
        GLib.idle_add(self._log_thread_safe, message, level)
    
    def _log_thread_safe(self, message, level):
        """Thread-safe logging"""
        # Add to log view
        end_iter = self.log_buffer.get_end_iter()
        
        if level == "error":
            self.log_buffer.insert_markup(end_iter, f"<span foreground='red'>[ERROR] {message}</span>\n", -1)
        elif level == "success":
            self.log_buffer.insert_markup(end_iter, f"<span foreground='green'>[✓] {message}</span>\n", -1)
        elif level == "warning":
            self.log_buffer.insert_markup(end_iter, f"<span foreground='orange'>[!] {message}</span>\n", -1)
        else:
            self.log_buffer.insert(end_iter, f"[INFO] {message}\n")
        
        # Auto-scroll
        self.log_view.scroll_to_iter(end_iter, 0.0, False, 0.0, 0.0)
        
        # Update status
        if level in ["error", "success"]:
            self.status_label.set_text(message)
    
    def update_progress(self, fraction, text=""):
        """Update progress bar"""
        GLib.idle_add(self._update_progress_thread_safe, fraction, text)
    
    def _update_progress_thread_safe(self, fraction, text):
        """Thread-safe progress update"""
        self.progress_bar.set_fraction(fraction)
        if text:
            self.progress_bar.set_text(text)
        else:
            self.progress_bar.set_text(f"{int(fraction * 100)}%")
    
    def on_install_clicked(self, button):
        """Start installation"""
        self.install_button.set_sensitive(False)
        self.cancel_button.set_sensitive(True)
        self.log_expander.set_expanded(True)
        
        # Start installation in separate thread
        self.install_thread = threading.Thread(target=self.run_installation)
        self.install_thread.daemon = True
        self.install_thread.start()
    
    def on_cancel_clicked(self, button):
        """Cancel installation"""
        self.log("Installation cancelled by user", "warning")
        self.cleanup()
        Gtk.main_quit()
    
    def download_from_github(self):
        """Download from GitHub"""
        self.log("Downloading from GitHub...")
        self.update_progress(0.1, "Connecting to GitHub...")
        
        url = "https://github.com/Masterace12/shader-prediction-compilation/archive/refs/heads/main.zip"
        zip_path = self.temp_dir / "shader-optimizer.zip"
        
        try:
            # Download with progress
            def download_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded / total_size, 1.0)
                    self.update_progress(0.1 + (percent * 0.3), f"Downloading... {percent*100:.0f}%")
            
            urllib.request.urlretrieve(url, zip_path, reporthook=download_hook)
            
            self.log("Download complete", "success")
            
            # Extract
            self.update_progress(0.5, "Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            # Find extracted directory
            extracted = list(self.temp_dir.glob("shader-prediction-compilation-*"))
            if extracted:
                return extracted[0]
            
            raise Exception("Could not find extracted files")
            
        except Exception as e:
            self.log(f"Download failed: {str(e)}", "error")
            raise
    
    def fix_permissions(self, source_dir):
        """Fix file permissions and line endings"""
        self.log("Fixing file permissions...")
        self.update_progress(0.6, "Fixing permissions...")
        
        # Fix script permissions
        for script in source_dir.rglob("*.sh"):
            os.chmod(script, 0o755)
        
        for script in source_dir.rglob("*.py"):
            os.chmod(script, 0o755)
        
        # Fix specific files
        for name in ["install", "install-manual"]:
            file_path = source_dir / "shader-predict-compile" / name
            if file_path.exists():
                os.chmod(file_path, 0o755)
        
        self.log("Permissions fixed", "success")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.log("Installing dependencies...")
        self.update_progress(0.7, "Installing dependencies...")
        
        try:
            # Upgrade pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install required packages
            packages = ["PyGObject", "psutil", "numpy"]
            for i, package in enumerate(packages):
                self.update_progress(0.7 + (i * 0.1 / len(packages)), f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", "--user", package], 
                             check=True, capture_output=True)
            
            self.log("Dependencies installed", "success")
            
        except subprocess.CalledProcessError as e:
            self.log(f"Dependency installation failed: {e.stderr.decode()}", "error")
            raise
    
    def copy_files(self, source_dir):
        """Copy files to installation directory"""
        self.log(f"Installing to {self.install_dir}...")
        self.update_progress(0.8, "Copying files...")
        
        # Create installation directory
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        # Find source files
        shader_dir = source_dir / "shader-predict-compile"
        if not shader_dir.exists():
            shader_dir = source_dir
        
        # Copy all files
        if shader_dir.exists():
            shutil.copytree(shader_dir, self.install_dir, dirs_exist_ok=True)
        
        self.log("Files copied", "success")
    
    def create_shortcuts(self):
        """Create desktop and Steam shortcuts"""
        self.log("Creating shortcuts...")
        self.update_progress(0.9, "Creating shortcuts...")
        
        # Create launcher script
        launcher_path = self.install_dir / "launch-deck.sh"
        launcher_content = f"""#!/bin/bash
cd "{self.install_dir}"
export PYTHONPATH="{self.install_dir}/src:$PYTHONPATH"
exec python3 ui/main_window.py "$@"
"""
        launcher_path.write_text(launcher_content)
        os.chmod(launcher_path, 0o755)
        
        # Desktop entry
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        
        desktop_file = desktop_dir / "shader-optimizer.desktop"
        desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Shader Optimizer
Comment=Optimize shader compilation for Steam Deck
Icon={self.install_dir}/icon.png
Exec={launcher_path}
Terminal=false
Categories=Game;Utility;
StartupNotify=true
"""
        desktop_file.write_text(desktop_content)
        
        self.log("Shortcuts created", "success")
    
    def run_installation(self):
        """Main installation process"""
        try:
            # Download
            source_dir = self.download_from_github()
            
            # Fix permissions
            self.fix_permissions(source_dir)
            
            # Install dependencies
            self.install_dependencies()
            
            # Copy files
            self.copy_files(source_dir)
            
            # Create shortcuts
            self.create_shortcuts()
            
            # Complete
            self.update_progress(1.0, "Installation complete!")
            self.log("Installation completed successfully!", "success")
            
            # Show completion dialog
            GLib.idle_add(self.show_completion_dialog)
            
        except Exception as e:
            self.log(f"Installation failed: {str(e)}", "error")
            GLib.idle_add(self.show_error_dialog, str(e))
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def show_completion_dialog(self):
        """Show installation complete dialog"""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Installation Complete!"
        )
        dialog.format_secondary_text(
            "Shader Optimizer has been installed successfully.\n\n"
            "You can find it in:\n"
            "• Gaming Mode: Non-Steam Games\n"
            "• Desktop Mode: Applications → Games"
        )
        dialog.run()
        dialog.destroy()
        self.install_button.set_label("Close")
        self.install_button.set_sensitive(True)
        self.install_button.connect("clicked", lambda x: Gtk.main_quit())
    
    def show_error_dialog(self, error_message):
        """Show error dialog"""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Installation Failed"
        )
        dialog.format_secondary_text(f"Error: {error_message}")
        dialog.run()
        dialog.destroy()
        self.install_button.set_sensitive(True)
        self.cancel_button.set_sensitive(False)

def main():
    """Main entry point"""
    # Set up GTK
    win = SteamDeckInstaller()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    
    # Hide log by default
    win.log_expander.set_expanded(False)
    
    Gtk.main()

if __name__ == "__main__":
    main()