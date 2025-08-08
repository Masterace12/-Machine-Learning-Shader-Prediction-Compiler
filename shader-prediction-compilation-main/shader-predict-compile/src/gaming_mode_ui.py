#!/usr/bin/env python3

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject, Gdk, Pango
import threading
import time
import json
from pathlib import Path
import logging
from typing import Dict, Optional

# Import our components
from background_service import BackgroundService
from steam_deck_compat import SteamDeckCompatibility
from realtime_monitor import GamingModeMonitor
from gaming_mode_power import GamingModePowerManager
from advanced_cache_manager import AdvancedCacheManager
from settings_manager import get_settings_manager

class GamingModeUI(Gtk.Window):
    """
    Gaming Mode optimized UI for Shader Predictive Compiler
    - Controller friendly navigation
    - Large text and buttons
    - Gaming Mode specific features
    """
    
    def __init__(self):
        super().__init__(title="Shader Predictive Compiler")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('gaming_mode_ui')
        
        # Initialize components
        self.compat = SteamDeckCompatibility()
        self.monitor = GamingModeMonitor()
        self.power_manager = GamingModePowerManager()
        self.cache_manager = AdvancedCacheManager()
        self.service = None
        
        # UI State
        self.current_page = 0
        self.pages = []
        
        # Setup window for Gaming Mode
        self.setup_gaming_mode_window()
        self.create_ui()
        self.start_monitoring()
        
    def setup_gaming_mode_window(self):
        """Configure window for Gaming Mode display"""
        # Set window properties for Steam Deck screen
        self.set_default_size(1280, 800)  # Steam Deck resolution
        self.set_position(Gtk.WindowPosition.CENTER)
        
        # Make it controller friendly
        self.set_decorated(True)
        self.set_resizable(True)
        
        # Enable keyboard navigation
        self.set_focus_on_map(True)
        
        # Style for Gaming Mode
        css_provider = Gtk.CssProvider()
        css_data = """
        window {
            background-color: #1e2328;
            color: #ffffff;
        }
        
        button {
            min-height: 60px;
            font-size: 16px;
            font-weight: bold;
            margin: 5px;
            border-radius: 8px;
            background: linear-gradient(to bottom, #4a90e2, #357abd);
            color: white;
            border: 2px solid #357abd;
        }
        
        button:hover {
            background: linear-gradient(to bottom, #5ba0f2, #4a90e2);
            border-color: #4a90e2;
        }
        
        button:active, button:focus {
            background: linear-gradient(to bottom, #357abd, #2868a0);
            border-color: #66c2ff;
        }
        
        label {
            font-size: 14px;
            color: #ffffff;
        }
        
        .title-label {
            font-size: 24px;
            font-weight: bold;
            color: #66c2ff;
            margin: 10px;
        }
        
        .status-good {
            color: #4caf50;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ff9800;
            font-weight: bold;
        }
        
        .status-error {
            color: #f44336;
            font-weight: bold;
        }
        
        progressbar {
            min-height: 20px;
        }
        
        .navigation-button {
            min-width: 120px;
            min-height: 50px;
            font-size: 14px;
        }
        """
        
        css_provider.load_from_data(css_data.encode('utf-8'))
        screen = Gdk.Screen.get_default()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen, css_provider, 
                                            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    
    def create_ui(self):
        """Create the main UI with Gaming Mode navigation"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_left(20)
        main_box.set_margin_right(20)
        main_box.set_margin_top(20)
        main_box.set_margin_bottom(20)
        
        # Title
        title_label = Gtk.Label("🎮 Shader Predictive Compiler")
        title_label.get_style_context().add_class("title-label")
        main_box.pack_start(title_label, False, False, 0)
        
        # Create notebook for different pages
        self.notebook = Gtk.Notebook()
        self.notebook.set_show_tabs(False)  # Hide tabs for controller navigation
        main_box.pack_start(self.notebook, True, True, 0)
        
        # Create pages
        self.create_status_page()
        self.create_settings_page()
        self.create_games_page()
        self.create_performance_page()
        
        # Navigation buttons
        nav_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        nav_box.set_halign(Gtk.Align.CENTER)
        
        self.nav_buttons = []
        nav_labels = ["📊 Status", "⚙️ Settings", "🎮 Games", "📈 Performance"]
        
        for i, label in enumerate(nav_labels):
            button = Gtk.Button(label=label)
            button.get_style_context().add_class("navigation-button")
            button.connect("clicked", self.on_nav_button_clicked, i)
            nav_box.pack_start(button, False, False, 0)
            self.nav_buttons.append(button)
        
        main_box.pack_start(nav_box, False, False, 10)
        
        # Close button
        close_button = Gtk.Button(label="❌ Close")
        close_button.connect("clicked", lambda w: self.destroy())
        close_button.set_halign(Gtk.Align.CENTER)
        main_box.pack_start(close_button, False, False, 0)
        
        self.add(main_box)
        
        # Update navigation
        self.update_navigation()
    
    def create_status_page(self):
        """Create status overview page"""
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        page_box.set_margin_left(10)
        page_box.set_margin_right(10)
        
        # System status
        status_frame = Gtk.Frame(label="🖥️ System Status")
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        status_box.set_margin_left(15)
        status_box.set_margin_right(15)
        status_box.set_margin_top(10)
        status_box.set_margin_bottom(15)
        
        self.system_status_labels = {}
        status_items = [
            ("Steam Deck Model", "model"),
            ("Gaming Mode", "gaming_mode"),
            ("Battery Level", "battery"),
            ("CPU Temperature", "cpu_temp"),
            ("GPU Temperature", "gpu_temp"),
            ("Power Profile", "power_profile")
        ]
        
        for label_text, key in status_items:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            
            label = Gtk.Label(label_text + ":")
            label.set_size_request(200, -1)
            label.set_halign(Gtk.Align.START)
            
            value_label = Gtk.Label("Checking...")
            value_label.set_halign(Gtk.Align.START)
            self.system_status_labels[key] = value_label
            
            row.pack_start(label, False, False, 0)
            row.pack_start(value_label, True, True, 0)
            status_box.pack_start(row, False, False, 0)
        
        status_frame.add(status_box)
        page_box.pack_start(status_frame, False, False, 0)
        
        # Service status
        service_frame = Gtk.Frame(label="🔧 Service Status")
        service_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        service_box.set_margin_left(15)
        service_box.set_margin_right(15)
        service_box.set_margin_top(10)
        service_box.set_margin_bottom(15)
        
        # Service toggle
        service_toggle_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        self.service_status_label = Gtk.Label("Service: Stopped")
        service_toggle_box.pack_start(self.service_status_label, True, True, 0)
        
        self.service_toggle_button = Gtk.Button(label="▶️ Start Service")
        self.service_toggle_button.connect("clicked", self.on_service_toggle)
        service_toggle_box.pack_start(self.service_toggle_button, False, False, 0)
        
        service_box.pack_start(service_toggle_box, False, False, 0)
        
        # Compilation stats
        self.compilation_stats_label = Gtk.Label("No compilation statistics available")
        service_box.pack_start(self.compilation_stats_label, False, False, 0)
        
        service_frame.add(service_box)
        page_box.pack_start(service_frame, False, False, 0)
        
        # Quick actions
        actions_frame = Gtk.Frame(label="⚡ Quick Actions")
        actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        actions_box.set_margin_left(15)
        actions_box.set_margin_right(15)
        actions_box.set_margin_top(10)
        actions_box.set_margin_bottom(15)
        
        optimize_button = Gtk.Button(label="🚀 Optimize Cache")
        optimize_button.connect("clicked", self.on_optimize_cache)
        actions_box.pack_start(optimize_button, True, True, 0)
        
        analyze_button = Gtk.Button(label="🔍 Analyze Games")
        analyze_button.connect("clicked", self.on_analyze_games)
        actions_box.pack_start(analyze_button, True, True, 0)
        
        actions_frame.add(actions_box)
        page_box.pack_start(actions_frame, False, False, 0)
        
        self.notebook.append_page(page_box, Gtk.Label("Status"))
    
    def create_settings_page(self):
        """Create settings configuration page"""
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        page_box.set_margin_left(10)
        page_box.set_margin_right(10)
        
        # Power management settings
        power_frame = Gtk.Frame(label="⚡ Power Management")
        power_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        power_box.set_margin_left(15)
        power_box.set_margin_right(15)
        power_box.set_margin_top(10)
        power_box.set_margin_bottom(15)
        
        # Adaptive power management toggle
        adaptive_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        adaptive_label = Gtk.Label("Adaptive Power Management:")
        self.adaptive_power_switch = Gtk.Switch()
        self.adaptive_power_switch.set_active(True)
        adaptive_box.pack_start(adaptive_label, True, True, 0)
        adaptive_box.pack_start(self.adaptive_power_switch, False, False, 0)
        power_box.pack_start(adaptive_box, False, False, 0)
        
        # Battery awareness toggle
        battery_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        battery_label = Gtk.Label("Battery Aware Compilation:")
        self.battery_aware_switch = Gtk.Switch()
        self.battery_aware_switch.set_active(True)
        battery_box.pack_start(battery_label, True, True, 0)
        battery_box.pack_start(self.battery_aware_switch, False, False, 0)
        power_box.pack_start(battery_box, False, False, 0)
        
        power_frame.add(power_box)
        page_box.pack_start(power_frame, False, False, 0)
        
        # Compilation settings
        compilation_frame = Gtk.Frame(label="🔧 Compilation Settings")
        compilation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        compilation_box.set_margin_left(15)
        compilation_box.set_margin_right(15)
        compilation_box.set_margin_top(10)
        compilation_box.set_margin_bottom(15)
        
        # Max threads
        threads_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        threads_label = Gtk.Label("Max Compilation Threads:")
        self.threads_adjustment = Gtk.Adjustment(value=4, lower=1, upper=8, step_increment=1)
        self.threads_spin = Gtk.SpinButton(adjustment=self.threads_adjustment)
        threads_box.pack_start(threads_label, True, True, 0)
        threads_box.pack_start(self.threads_spin, False, False, 0)
        compilation_box.pack_start(threads_box, False, False, 0)
        
        # Gaming mode pause
        gaming_pause_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        gaming_pause_label = Gtk.Label("Pause During Gaming:")
        self.gaming_pause_switch = Gtk.Switch()
        self.gaming_pause_switch.set_active(True)
        gaming_pause_box.pack_start(gaming_pause_label, True, True, 0)
        gaming_pause_box.pack_start(self.gaming_pause_switch, False, False, 0)
        compilation_box.pack_start(gaming_pause_box, False, False, 0)
        
        compilation_frame.add(compilation_box)
        page_box.pack_start(compilation_frame, False, False, 0)
        
        # Save settings button
        save_button = Gtk.Button(label="💾 Save Settings")
        save_button.connect("clicked", self.on_save_settings)
        save_button.set_halign(Gtk.Align.CENTER)
        page_box.pack_start(save_button, False, False, 0)
        
        self.notebook.append_page(page_box, Gtk.Label("Settings"))
    
    def create_games_page(self):
        """Create games management page"""
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        page_box.set_margin_left(10)
        page_box.set_margin_right(10)
        
        # Game list
        games_frame = Gtk.Frame(label="🎮 Detected Games")
        
        # Create scrollable list
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_size_request(-1, 300)
        
        self.games_listbox = Gtk.ListBox()
        self.games_listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        scrolled_window.add(self.games_listbox)
        
        games_frame.add(scrolled_window)
        page_box.pack_start(games_frame, True, True, 0)
        
        # Game actions
        actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        refresh_button = Gtk.Button(label="🔄 Refresh Games")
        refresh_button.connect("clicked", self.on_refresh_games)
        actions_box.pack_start(refresh_button, True, True, 0)
        
        compile_button = Gtk.Button(label="⚙️ Compile Selected")
        compile_button.connect("clicked", self.on_compile_selected_game)
        actions_box.pack_start(compile_button, True, True, 0)
        
        page_box.pack_start(actions_box, False, False, 0)
        
        self.notebook.append_page(page_box, Gtk.Label("Games"))
    
    def create_performance_page(self):
        """Create performance monitoring page"""
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        page_box.set_margin_left(10)
        page_box.set_margin_right(10)
        
        # Real-time metrics
        metrics_frame = Gtk.Frame(label="📊 Real-time Metrics")
        metrics_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        metrics_box.set_margin_left(15)
        metrics_box.set_margin_right(15)
        metrics_box.set_margin_top(10)
        metrics_box.set_margin_bottom(15)
        
        self.metrics_labels = {}
        metrics_items = [
            ("CPU Usage", "cpu_usage"),
            ("GPU Usage", "gpu_usage"),
            ("Memory Usage", "memory_usage"),
            ("Power Draw", "power_draw"),
            ("Fan Speed", "fan_speed")
        ]
        
        for label_text, key in metrics_items:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            
            label = Gtk.Label(label_text + ":")
            label.set_size_request(150, -1)
            label.set_halign(Gtk.Align.START)
            
            # Progress bar for visual representation
            progress = Gtk.ProgressBar()
            progress.set_size_request(200, -1)
            progress.set_show_text(True)
            
            self.metrics_labels[key] = progress
            
            row.pack_start(label, False, False, 0)
            row.pack_start(progress, True, True, 0)
            metrics_box.pack_start(row, False, False, 0)
        
        metrics_frame.add(metrics_box)
        page_box.pack_start(metrics_frame, False, False, 0)
        
        # Cache statistics
        cache_frame = Gtk.Frame(label="💾 Cache Statistics")
        cache_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        cache_box.set_margin_left(15)
        cache_box.set_margin_right(15)
        cache_box.set_margin_top(10)
        cache_box.set_margin_bottom(15)
        
        self.cache_stats_label = Gtk.Label("Loading cache statistics...")
        cache_box.pack_start(self.cache_stats_label, False, False, 0)
        
        # Cache actions
        cache_actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        cleanup_button = Gtk.Button(label="🧹 Cleanup Cache")
        cleanup_button.connect("clicked", self.on_cleanup_cache)
        cache_actions_box.pack_start(cleanup_button, True, True, 0)
        
        export_button = Gtk.Button(label="📤 Export Report")
        export_button.connect("clicked", self.on_export_report)
        cache_actions_box.pack_start(export_button, True, True, 0)
        
        cache_box.pack_start(cache_actions_box, False, False, 0)
        cache_frame.add(cache_box)
        page_box.pack_start(cache_frame, False, False, 0)
        
        self.notebook.append_page(page_box, Gtk.Label("Performance"))
    
    def start_monitoring(self):
        """Start background monitoring for UI updates"""
        def update_loop():
            while True:
                try:
                    GObject.idle_add(self.update_status)
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    self.logger.error(f"Update loop error: {e}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def update_status(self):
        """Update UI with current status"""
        try:
            # Update system status
            thermal_data = self.compat.get_enhanced_thermal_data()
            battery_data = self.compat.get_enhanced_battery_data()
            
            # Update status labels
            self.system_status_labels["model"].set_text(f"{self.compat.model} Steam Deck")
            
            gaming_mode = "🎮 Active" if self.compat.detect_gaming_mode() else "🖥️ Desktop Mode"
            self.system_status_labels["gaming_mode"].set_text(gaming_mode)
            
            if battery_data['battery_present']:
                battery_text = f"{battery_data['capacity']}%"
                if battery_data['charging']:
                    battery_text += " (Charging)"
                elif battery_data['ac_connected']:
                    battery_text += " (Docked)"
                self.system_status_labels["battery"].set_text(battery_text)
            else:
                self.system_status_labels["battery"].set_text("Docked")
            
            # Temperature with color coding
            cpu_temp = thermal_data['cpu_temp']
            cpu_temp_text = f"{cpu_temp:.1f}°C"
            if cpu_temp > 80:
                cpu_temp_text = f"🔥 {cpu_temp_text}"
                self.system_status_labels["cpu_temp"].get_style_context().add_class("status-error")
            elif cpu_temp > 70:
                cpu_temp_text = f"⚠️ {cpu_temp_text}"
                self.system_status_labels["cpu_temp"].get_style_context().add_class("status-warning")
            else:
                cpu_temp_text = f"✅ {cpu_temp_text}"
                self.system_status_labels["cpu_temp"].get_style_context().add_class("status-good")
            self.system_status_labels["cpu_temp"].set_text(cpu_temp_text)
            
            gpu_temp = thermal_data['gpu_temp']
            gpu_temp_text = f"{gpu_temp:.1f}°C"
            if gpu_temp > 85:
                gpu_temp_text = f"🔥 {gpu_temp_text}"
            elif gpu_temp > 75:
                gpu_temp_text = f"⚠️ {gpu_temp_text}"
            else:
                gpu_temp_text = f"✅ {gpu_temp_text}"
            self.system_status_labels["gpu_temp"].set_text(gpu_temp_text)
            
            # Power profile
            if hasattr(self.power_manager, 'current_profile'):
                profile_name = self.power_manager.current_profile.value.replace('_', ' ').title()
                self.system_status_labels["power_profile"].set_text(profile_name)
            else:
                self.system_status_labels["power_profile"].set_text("Not Available")
            
            # Update performance metrics if on performance page
            current_page = self.notebook.get_current_page()
            if current_page == 3:  # Performance page
                self.update_performance_metrics(thermal_data, battery_data)
            
        except Exception as e:
            self.logger.error(f"Status update error: {e}")
        
        return True  # Continue updating
    
    def update_performance_metrics(self, thermal_data: Dict, battery_data: Dict):
        """Update performance metrics display"""
        try:
            # Update progress bars
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            self.metrics_labels["cpu_usage"].set_fraction(cpu_percent / 100.0)
            self.metrics_labels["cpu_usage"].set_text(f"{cpu_percent:.1f}%")
            
            gpu_percent = thermal_data.get('gpu_utilization', 0)
            self.metrics_labels["gpu_usage"].set_fraction(gpu_percent / 100.0)
            self.metrics_labels["gpu_usage"].set_text(f"{gpu_percent:.1f}%")
            
            memory_percent = psutil.virtual_memory().percent
            self.metrics_labels["memory_usage"].set_fraction(memory_percent / 100.0)
            self.metrics_labels["memory_usage"].set_text(f"{memory_percent:.1f}%")
            
            power_draw = battery_data.get('discharge_rate_w', 0)
            power_fraction = min(power_draw / 30.0, 1.0)  # Assume 30W max
            self.metrics_labels["power_draw"].set_fraction(power_fraction)
            self.metrics_labels["power_draw"].set_text(f"{power_draw:.1f}W")
            
            fan_rpm = thermal_data.get('fan_rpm', 0)
            fan_fraction = min(fan_rpm / 5000.0, 1.0)  # Assume 5000 RPM max
            self.metrics_labels["fan_speed"].set_fraction(fan_fraction)
            self.metrics_labels["fan_speed"].set_text(f"{fan_rpm} RPM")
            
            # Update cache stats
            cache_report = self.cache_manager.get_cache_report()
            stats = cache_report.get('statistics', {})
            
            cache_text = f"""📦 Total Entries: {stats.get('total_entries', 0):,}
💾 Cache Size: {stats.get('total_size_mb', 0):.1f} MB
🎯 Hit Rate: {stats.get('hit_rate', 0):.1f}
⏱️ Time Saved: {stats.get('compilation_time_saved_hours', 0):.1f} hours
🎮 Games Cached: {stats.get('games_cached', 0)}"""
            
            self.cache_stats_label.set_text(cache_text)
            
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")
    
    def on_nav_button_clicked(self, button, page_index):
        """Handle navigation button clicks"""
        self.current_page = page_index
        self.notebook.set_current_page(page_index)
        self.update_navigation()
    
    def update_navigation(self):
        """Update navigation button states"""
        for i, button in enumerate(self.nav_buttons):
            if i == self.current_page:
                button.get_style_context().add_class("suggested-action")
            else:
                button.get_style_context().remove_class("suggested-action")
    
    # Event handlers
    def on_service_toggle(self, button):
        """Toggle background service"""
        # Implementation for service start/stop
        button.set_label("🔄 Working...")
        button.set_sensitive(False)
        
        def toggle_service():
            try:
                if self.service is None:
                    # Start service
                    self.service = BackgroundService()
                    self.service.start()
                    GObject.idle_add(lambda: button.set_label("⏹️ Stop Service"))
                    GObject.idle_add(lambda: self.service_status_label.set_text("Service: Running"))
                else:
                    # Stop service
                    self.service.stop()
                    self.service = None
                    GObject.idle_add(lambda: button.set_label("▶️ Start Service"))
                    GObject.idle_add(lambda: self.service_status_label.set_text("Service: Stopped"))
            except Exception as e:
                self.logger.error(f"Service toggle error: {e}")
            finally:
                GObject.idle_add(lambda: button.set_sensitive(True))
        
        threading.Thread(target=toggle_service, daemon=True).start()
    
    def on_optimize_cache(self, button):
        """Optimize shader cache"""
        button.set_label("🔄 Optimizing...")
        button.set_sensitive(False)
        
        def optimize():
            try:
                results = self.cache_manager.optimize_cache_structure()
                message = f"Optimized {results['files_optimized']} files, saved {results['size_reduction_mb']:.1f} MB"
                GObject.idle_add(self.show_message, "Cache Optimization", message)
            except Exception as e:
                GObject.idle_add(self.show_message, "Error", f"Cache optimization failed: {e}")
            finally:
                GObject.idle_add(lambda: button.set_label("🚀 Optimize Cache"))
                GObject.idle_add(lambda: button.set_sensitive(True))
        
        threading.Thread(target=optimize, daemon=True).start()
    
    def on_analyze_games(self, button):
        """Analyze installed games"""
        button.set_label("🔄 Analyzing...")
        button.set_sensitive(False)
        
        def analyze():
            try:
                # Trigger game analysis
                # This would integrate with the existing game analysis logic
                GObject.idle_add(self.show_message, "Game Analysis", "Game analysis started in background")
            except Exception as e:
                GObject.idle_add(self.show_message, "Error", f"Game analysis failed: {e}")
            finally:
                GObject.idle_add(lambda: button.set_label("🔍 Analyze Games"))
                GObject.idle_add(lambda: button.set_sensitive(True))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def on_save_settings(self, button):
        """Save current settings"""
        try:
            settings = {
                'adaptive_power': self.adaptive_power_switch.get_active(),
                'battery_aware': self.battery_aware_switch.get_active(),
                'max_threads': int(self.threads_spin.get_value()),
                'pause_during_gaming': self.gaming_pause_switch.get_active()
            }
            
            settings_file = Path.home() / '.config/shader-predict-compile/gaming_mode_settings.json'
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.show_message("Settings", "Settings saved successfully!")
            
        except Exception as e:
            self.show_message("Error", f"Failed to save settings: {e}")
    
    def on_refresh_games(self, button):
        """Refresh games list"""
        # Clear current list
        for child in self.games_listbox.get_children():
            self.games_listbox.remove(child)
        
        # Add sample games (this would be populated from actual game detection)
        sample_games = [
            ("Cyberpunk 2077", "Ready for compilation"),
            ("Elden Ring", "Shaders pre-compiled"),
            ("Horizon Zero Dawn", "Analysis needed"),
            ("Death Stranding", "Optimized")
        ]
        
        for game_name, status in sample_games:
            row = Gtk.ListBoxRow()
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            box.set_margin_left(10)
            box.set_margin_right(10)
            box.set_margin_top(5)
            box.set_margin_bottom(5)
            
            game_label = Gtk.Label(game_name)
            game_label.set_halign(Gtk.Align.START)
            
            status_label = Gtk.Label(status)
            status_label.set_halign(Gtk.Align.END)
            
            box.pack_start(game_label, True, True, 0)
            box.pack_start(status_label, False, False, 0)
            
            row.add(box)
            self.games_listbox.add(row)
        
        self.games_listbox.show_all()
    
    def on_compile_selected_game(self, button):
        """Compile shaders for selected game"""
        selected_row = self.games_listbox.get_selected_row()
        if selected_row:
            self.show_message("Compilation", "Shader compilation started for selected game")
        else:
            self.show_message("Selection", "Please select a game first")
    
    def on_cleanup_cache(self, button):
        """Cleanup old cache files"""
        button.set_label("🔄 Cleaning...")
        button.set_sensitive(False)
        
        def cleanup():
            try:
                results = self.cache_manager.intelligent_cleanup()
                message = f"Cleaned {results['entries_removed']} entries, freed {results['size_freed_mb']:.1f} MB"
                GObject.idle_add(self.show_message, "Cache Cleanup", message)
            except Exception as e:
                GObject.idle_add(self.show_message, "Error", f"Cache cleanup failed: {e}")
            finally:
                GObject.idle_add(lambda: button.set_label("🧹 Cleanup Cache"))
                GObject.idle_add(lambda: button.set_sensitive(True))
        
        threading.Thread(target=cleanup, daemon=True).start()
    
    def on_export_report(self, button):
        """Export performance report"""
        try:
            report_file = Path.home() / 'shader_compiler_report.json'
            cache_report = self.cache_manager.get_cache_report()
            
            with open(report_file, 'w') as f:
                json.dump(cache_report, f, indent=2)
            
            self.show_message("Export", f"Report exported to {report_file}")
            
        except Exception as e:
            self.show_message("Error", f"Export failed: {e}")
    
    def show_message(self, title: str, message: str):
        """Show message dialog"""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text=title
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

def main():
    """Main entry point for Gaming Mode UI"""
    app = GamingModeUI()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    
    # Start with status page
    app.on_refresh_games(None)  # Populate games list
    
    Gtk.main()

if __name__ == '__main__':
    main()