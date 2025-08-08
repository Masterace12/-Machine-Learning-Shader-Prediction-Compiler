#!/usr/bin/env python3
"""
ML Shader Predictor GUI Application

A unified GUI application for managing the ML Shader Prediction Compiler system
on Steam Deck and desktop Linux environments. Provides both desktop and gaming
mode compatible interfaces.

Features:
- Steam Deck optimized UI with controller support
- Shader cache management and visualization
- Real-time performance monitoring
- Game-specific optimization profiles
- Thermal management and power monitoring
- Automatic and manual shader prediction controls
"""

import sys
import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Check if we're in a GUI environment
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI not available - running in CLI mode")

# Import our integration modules
try:
    from gaming_mode_integration import GamingModeIntegration, SteamDeckMode
    from desktop_integration import DesktopIntegration
except ImportError as e:
    print(f"Warning: Integration modules not available: {e}")
    GamingModeIntegration = None
    DesktopIntegration = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UIMode(Enum):
    """UI mode selection"""
    AUTO = "auto"
    DESKTOP = "desktop"
    GAMING = "gaming"
    CLI = "cli"


@dataclass
class AppConfig:
    """Application configuration"""
    ui_mode: UIMode
    window_width: int
    window_height: int
    theme: str
    controller_enabled: bool
    auto_minimize: bool
    show_notifications: bool
    update_interval_ms: int


class MLShaderPredictorGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.config = self._load_config()
        self.running = False
        
        # Integration components
        self.gaming_mode = None
        self.desktop_integration = None
        
        # GUI components
        self.root = None
        self.notebook = None
        self.status_bar = None
        
        # Data
        self.shader_cache_data = {}
        self.performance_data = {}
        self.game_profiles = {}
        
        # Initialize components
        self._initialize_integrations()
        
        if GUI_AVAILABLE:
            self._initialize_gui()
    
    def _load_config(self) -> AppConfig:
        """Load application configuration"""
        config_file = Path.home() / ".config/ml-shader-predictor/gui.json"
        
        default_config = AppConfig(
            ui_mode=UIMode.AUTO,
            window_width=1024,
            window_height=768,
            theme="steam_deck",
            controller_enabled=True,
            auto_minimize=False,
            show_notifications=True,
            update_interval_ms=1000
        )
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    return AppConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load GUI config: {e}")
        
        return default_config
    
    def _save_config(self):
        """Save application configuration"""
        config_file = Path.home() / ".config/ml-shader-predictor/gui.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save GUI config: {e}")
    
    def _initialize_integrations(self):
        """Initialize platform integrations"""
        try:
            # Initialize Gaming Mode integration
            if GamingModeIntegration:
                self.gaming_mode = GamingModeIntegration()
                logger.info("Gaming Mode integration initialized")
            
            # Initialize Desktop integration
            if DesktopIntegration:
                install_dir = Path(__file__).parent.parent
                self.desktop_integration = DesktopIntegration(install_dir)
                logger.info("Desktop integration initialized")
                
        except Exception as e:
            logger.error(f"Integration initialization failed: {e}")
    
    def _initialize_gui(self):
        """Initialize GUI components"""
        if not GUI_AVAILABLE:
            return
        
        try:
            # Create main window
            self.root = tk.Tk()
            self.root.title("ML Shader Predictor")
            self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
            
            # Apply Steam Deck theme if in gaming mode
            if self._is_gaming_mode():
                self._apply_gaming_mode_theme()
            else:
                self._apply_desktop_theme()
            
            # Create main interface
            self._create_main_interface()
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Start update loop
            self._start_update_loop()
            
            logger.info("GUI initialized successfully")
            
        except Exception as e:
            logger.error(f"GUI initialization failed: {e}")
            raise
    
    def _is_gaming_mode(self) -> bool:
        """Check if we're in gaming mode"""
        if self.gaming_mode:
            return self.gaming_mode.current_mode == SteamDeckMode.GAMING
        return False
    
    def _apply_gaming_mode_theme(self):
        """Apply Steam Deck gaming mode theme"""
        try:
            # Configure for Steam Deck screen and controller input
            self.root.configure(bg='#1e2328')
            
            # Configure ttk styles for dark theme
            style = ttk.Style()
            style.theme_use('clam')
            
            # Dark theme colors
            style.configure('TNotebook', background='#1e2328', borderwidth=0)
            style.configure('TNotebook.Tab', background='#2a2e35', foreground='white',
                          padding=[20, 10], focuscolor='none')
            style.map('TNotebook.Tab', background=[('selected', '#66c0f4')])
            
            style.configure('TFrame', background='#1e2328')
            style.configure('TLabel', background='#1e2328', foreground='white', font=('Arial', 12))
            style.configure('TButton', background='#66c0f4', foreground='white', 
                          font=('Arial', 12), padding=[10, 5])
            style.map('TButton', background=[('active', '#4c9fcf')])
            
            logger.info("Applied gaming mode theme")
            
        except Exception as e:
            logger.error(f"Gaming mode theme application failed: {e}")
    
    def _apply_desktop_theme(self):
        """Apply desktop theme"""
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Use a modern-looking theme
            logger.info("Applied desktop theme")
            
        except Exception as e:
            logger.error(f"Desktop theme application failed: {e}")
    
    def _create_main_interface(self):
        """Create main interface"""
        try:
            # Create main notebook for tabs
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create tabs
            self._create_overview_tab()
            self._create_shader_cache_tab()
            self._create_performance_tab()
            self._create_games_tab()
            self._create_settings_tab()
            
            # Create status bar
            self._create_status_bar()
            
            logger.info("Main interface created")
            
        except Exception as e:
            logger.error(f"Main interface creation failed: {e}")
    
    def _create_overview_tab(self):
        """Create overview tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Overview")
        
        # System status section
        status_frame = ttk.LabelFrame(tab, text="System Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.system_status_label = ttk.Label(status_frame, text="Initializing...")
        self.system_status_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Quick stats section
        stats_frame = ttk.LabelFrame(tab, text="Quick Stats")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(stats_grid, text="Cached Shaders:").grid(row=0, column=0, sticky=tk.W)
        self.cached_shaders_label = ttk.Label(stats_grid, text="0")
        self.cached_shaders_label.grid(row=0, column=1, sticky=tk.W, padx=20)
        
        ttk.Label(stats_grid, text="Active Games:").grid(row=1, column=0, sticky=tk.W)
        self.active_games_label = ttk.Label(stats_grid, text="0")
        self.active_games_label.grid(row=1, column=1, sticky=tk.W, padx=20)
        
        ttk.Label(stats_grid, text="CPU Temperature:").grid(row=2, column=0, sticky=tk.W)
        self.cpu_temp_label = ttk.Label(stats_grid, text="--°C")
        self.cpu_temp_label.grid(row=2, column=1, sticky=tk.W, padx=20)
        
        # Control buttons section
        controls_frame = ttk.LabelFrame(tab, text="Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Start Service", 
                  command=self._start_service).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Service", 
                  command=self._stop_service).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh", 
                  command=self._refresh_overview).pack(side=tk.LEFT, padx=5)
    
    def _create_shader_cache_tab(self):
        """Create shader cache management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Shader Cache")
        
        # Cache statistics
        stats_frame = ttk.LabelFrame(tab, text="Cache Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create treeview for cache data
        cache_frame = ttk.LabelFrame(tab, text="Cache Contents")
        cache_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview with scrollbar
        tree_frame = ttk.Frame(cache_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cache_tree = ttk.Treeview(tree_frame, columns=('Game', 'Shaders', 'Size', 'Status'))
        self.cache_tree.heading('#0', text='ID')
        self.cache_tree.heading('Game', text='Game')
        self.cache_tree.heading('Shaders', text='Shader Count')
        self.cache_tree.heading('Size', text='Cache Size')
        self.cache_tree.heading('Status', text='Status')
        
        cache_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.cache_tree.yview)
        self.cache_tree.configure(yscroll=cache_scrollbar.set)
        
        self.cache_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cache_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Cache control buttons
        cache_controls = ttk.Frame(cache_frame)
        cache_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(cache_controls, text="Refresh Cache", 
                  command=self._refresh_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(cache_controls, text="Clear Cache", 
                  command=self._clear_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(cache_controls, text="Optimize Cache", 
                  command=self._optimize_cache).pack(side=tk.LEFT, padx=5)
    
    def _create_performance_tab(self):
        """Create performance monitoring tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Performance")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(tab, text="Real-time Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # GPU metrics
        ttk.Label(metrics_grid, text="GPU Temperature:").grid(row=0, column=0, sticky=tk.W)
        self.gpu_temp_label = ttk.Label(metrics_grid, text="--°C")
        self.gpu_temp_label.grid(row=0, column=1, sticky=tk.W, padx=20)
        
        ttk.Label(metrics_grid, text="GPU Usage:").grid(row=1, column=0, sticky=tk.W)
        self.gpu_usage_label = ttk.Label(metrics_grid, text="--%")
        self.gpu_usage_label.grid(row=1, column=1, sticky=tk.W, padx=20)
        
        ttk.Label(metrics_grid, text="Memory Usage:").grid(row=2, column=0, sticky=tk.W)
        self.memory_usage_label = ttk.Label(metrics_grid, text="-- MB")
        self.memory_usage_label.grid(row=2, column=1, sticky=tk.W, padx=20)
        
        # Thermal management
        thermal_frame = ttk.LabelFrame(tab, text="Thermal Management")
        thermal_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.thermal_mode_var = tk.StringVar(value="Auto")
        thermal_options = ["Auto", "Performance", "Balanced", "Power Save"]
        
        ttk.Label(thermal_frame, text="Thermal Mode:").pack(anchor=tk.W, padx=10)
        thermal_combo = ttk.Combobox(thermal_frame, textvariable=self.thermal_mode_var, 
                                   values=thermal_options, state="readonly")
        thermal_combo.pack(anchor=tk.W, padx=10, pady=5)
        thermal_combo.bind('<<ComboboxSelected>>', self._on_thermal_mode_changed)
    
    def _create_games_tab(self):
        """Create games management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Games")
        
        # Game list
        games_frame = ttk.LabelFrame(tab, text="Detected Games")
        games_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Games treeview
        games_tree_frame = ttk.Frame(games_frame)
        games_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.games_tree = ttk.Treeview(games_tree_frame, columns=('Name', 'Status', 'Shaders', 'Profile'))
        self.games_tree.heading('#0', text='App ID')
        self.games_tree.heading('Name', text='Game Name')
        self.games_tree.heading('Status', text='Status')
        self.games_tree.heading('Shaders', text='Shader Count')
        self.games_tree.heading('Profile', text='Optimization Profile')
        
        games_scrollbar = ttk.Scrollbar(games_tree_frame, orient=tk.VERTICAL, command=self.games_tree.yview)
        self.games_tree.configure(yscroll=games_scrollbar.set)
        
        self.games_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        games_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Game controls
        game_controls = ttk.Frame(games_frame)
        game_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(game_controls, text="Scan for Games", 
                  command=self._scan_games).pack(side=tk.LEFT, padx=5)
        ttk.Button(game_controls, text="Optimize Selected", 
                  command=self._optimize_selected_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(game_controls, text="Create Profile", 
                  command=self._create_game_profile).pack(side=tk.LEFT, padx=5)
    
    def _create_settings_tab(self):
        """Create settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Settings")
        
        # General settings
        general_frame = ttk.LabelFrame(tab, text="General Settings")
        general_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Auto-start service
        self.autostart_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Start service automatically", 
                       variable=self.autostart_var).pack(anchor=tk.W, padx=10, pady=2)
        
        # Show notifications
        self.notifications_var = tk.BooleanVar(value=self.config.show_notifications)
        ttk.Checkbutton(general_frame, text="Show desktop notifications", 
                       variable=self.notifications_var).pack(anchor=tk.W, padx=10, pady=2)
        
        # Enable controller support
        self.controller_var = tk.BooleanVar(value=self.config.controller_enabled)
        ttk.Checkbutton(general_frame, text="Enable controller support", 
                       variable=self.controller_var).pack(anchor=tk.W, padx=10, pady=2)
        
        # Update interval
        ttk.Label(general_frame, text="Update Interval (ms):").pack(anchor=tk.W, padx=10, pady=(10, 2))
        self.update_interval_var = tk.IntVar(value=self.config.update_interval_ms)
        update_scale = ttk.Scale(general_frame, from_=500, to=5000, orient=tk.HORIZONTAL,
                               variable=self.update_interval_var)
        update_scale.pack(fill=tk.X, padx=10, pady=2)
        
        # Buttons
        settings_buttons = ttk.Frame(general_frame)
        settings_buttons.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(settings_buttons, text="Save Settings", 
                  command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_buttons, text="Reset to Defaults", 
                  command=self._reset_settings).pack(side=tk.LEFT, padx=5)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_event_handlers(self):
        """Set up GUI event handlers"""
        if self.root:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _start_update_loop(self):
        """Start the GUI update loop"""
        if self.root:
            self._update_gui()
            self.root.after(self.config.update_interval_ms, self._start_update_loop)
    
    def _update_gui(self):
        """Update GUI with latest data"""
        try:
            # Update system status
            self._update_system_status()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Update cache data
            self._update_cache_data()
            
            # Update games list
            self._update_games_list()
            
        except Exception as e:
            logger.error(f"GUI update failed: {e}")
    
    def _update_system_status(self):
        """Update system status display"""
        try:
            status_text = "System: Online"
            
            if self.gaming_mode:
                running_game = self.gaming_mode.get_running_steam_game()
                if running_game:
                    status_text += f" | Game: {running_game.app_name}"
            
            if hasattr(self, 'system_status_label'):
                self.system_status_label.config(text=status_text)
                
        except Exception as e:
            logger.error(f"System status update failed: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics display"""
        try:
            # Update GPU temperature and usage
            if hasattr(self, 'gpu_temp_label'):
                self.gpu_temp_label.config(text="75°C")  # Placeholder
            
            if hasattr(self, 'gpu_usage_label'):
                self.gpu_usage_label.config(text="65%")  # Placeholder
                
            if hasattr(self, 'memory_usage_label'):
                self.memory_usage_label.config(text="1024 MB")  # Placeholder
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def _update_cache_data(self):
        """Update shader cache data display"""
        # Placeholder implementation
        pass
    
    def _update_games_list(self):
        """Update games list display"""
        # Placeholder implementation
        pass
    
    # Event handlers
    def _start_service(self):
        """Start shader prediction service"""
        self._update_status("Starting shader prediction service...")
        # Implementation would start the service
    
    def _stop_service(self):
        """Stop shader prediction service"""
        self._update_status("Stopping shader prediction service...")
        # Implementation would stop the service
    
    def _refresh_overview(self):
        """Refresh overview data"""
        self._update_status("Refreshing overview data...")
        self._update_gui()
    
    def _refresh_cache(self):
        """Refresh shader cache data"""
        self._update_status("Refreshing shader cache data...")
    
    def _clear_cache(self):
        """Clear shader cache"""
        if messagebox.askyesno("Clear Cache", "Are you sure you want to clear the shader cache?"):
            self._update_status("Clearing shader cache...")
    
    def _optimize_cache(self):
        """Optimize shader cache"""
        self._update_status("Optimizing shader cache...")
    
    def _scan_games(self):
        """Scan for installed games"""
        self._update_status("Scanning for games...")
    
    def _optimize_selected_game(self):
        """Optimize selected game"""
        selection = self.games_tree.selection()
        if selection:
            self._update_status("Optimizing selected game...")
    
    def _create_game_profile(self):
        """Create optimization profile for game"""
        self._update_status("Creating game profile...")
    
    def _on_thermal_mode_changed(self, event):
        """Handle thermal mode change"""
        mode = self.thermal_mode_var.get()
        self._update_status(f"Thermal mode changed to: {mode}")
    
    def _save_settings(self):
        """Save application settings"""
        try:
            self.config.show_notifications = self.notifications_var.get()
            self.config.controller_enabled = self.controller_var.get()
            self.config.update_interval_ms = self.update_interval_var.get()
            
            self._save_config()
            self._update_status("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Settings save failed: {e}")
            self._update_status("Failed to save settings")
    
    def _reset_settings(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            self.config = AppConfig(
                ui_mode=UIMode.AUTO,
                window_width=1024,
                window_height=768,
                theme="steam_deck",
                controller_enabled=True,
                auto_minimize=False,
                show_notifications=True,
                update_interval_ms=1000
            )
            self._update_status("Settings reset to defaults")
    
    def _update_status(self, message: str):
        """Update status bar message"""
        if self.status_bar:
            self.status_bar.config(text=message)
        logger.info(f"Status: {message}")
    
    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit ML Shader Predictor?"):
            self.running = False
            if self.root:
                self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        if not GUI_AVAILABLE:
            logger.error("GUI not available - cannot start graphical interface")
            return 1
        
        try:
            self.running = True
            logger.info("Starting ML Shader Predictor GUI")
            
            if self.root:
                self.root.mainloop()
            
            return 0
            
        except Exception as e:
            logger.error(f"GUI application failed: {e}")
            return 1
        finally:
            self.running = False


def main():
    """Main entry point"""
    app = MLShaderPredictorGUI()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())