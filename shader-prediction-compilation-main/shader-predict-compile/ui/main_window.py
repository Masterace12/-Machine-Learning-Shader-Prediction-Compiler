#!/usr/bin/env python3

import sys
import os
import logging
from pathlib import Path

# Set up logging
log_dir = Path.home() / '.cache' / 'shader-predict-compile'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk, GLib, Gdk
except ImportError as e:
    logger.error(f"Failed to import GTK: {e}")
    print("Error: GTK bindings not found. Please install python3-gi package.")
    print("On Steam Deck, run: sudo pacman -S python-gobject")
    sys.exit(1)

import threading
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from shader_analyzer import ShaderAnalyzer
    from heuristic_engine import HeuristicEngine
    from fossilize_integration import FossilizeIntegration
    from settings_manager import get_settings_manager
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    print(f"Error: Could not import required modules: {e}")
    print("Make sure you're running from the correct directory or that the installation is complete.")
    sys.exit(1)

class ShaderPredictCompileWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Shader Predictive Compiler for Steam Deck")
        self.set_default_size(800, 600)
        self.set_border_width(10)
        
        # Initialize components
        self.analyzer = ShaderAnalyzer()
        self.engine = HeuristicEngine()
        self.fossilize = FossilizeIntegration()
        self.settings_manager = get_settings_manager()
        
        # State
        self.is_analyzing = False
        self.is_compiling = False
        self.selected_game = None
        
        # Create UI
        self.setup_ui()
        
        # Load saved settings
        self.load_settings()
        
        # Check compatibility on startup
        self.check_compatibility()
        
    def setup_ui(self):
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(main_box)
        
        # Header with title and status
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.pack_start(header_box, False, False, 0)
        
        title_label = Gtk.Label()
        title_label.set_markup("<b><big>Shader Predictive Compiler</big></b>")
        header_box.pack_start(title_label, False, False, 0)
        
        self.status_label = Gtk.Label("Ready")
        self.status_label.set_halign(Gtk.Align.END)
        header_box.pack_end(self.status_label, False, False, 0)
        
        # Compatibility check box
        self.compat_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.compat_box.set_margin_top(10)
        main_box.pack_start(self.compat_box, False, False, 0)
        
        # Main content notebook
        notebook = Gtk.Notebook()
        main_box.pack_start(notebook, True, True, 0)
        
        # Library Analysis Tab
        analysis_tab = self.create_analysis_tab()
        notebook.append_page(analysis_tab, Gtk.Label("Library Analysis"))
        
        # Shader Compilation Tab
        compile_tab = self.create_compile_tab()
        notebook.append_page(compile_tab, Gtk.Label("Shader Compilation"))
        
        # Settings Tab
        settings_tab = self.create_settings_tab()
        notebook.append_page(settings_tab, Gtk.Label("Settings"))
        
        # Progress bar at bottom
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_show_text(True)
        main_box.pack_end(self.progress_bar, False, False, 0)
        
    def create_analysis_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10)
        
        # Steam library path
        path_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.pack_start(path_box, False, False, 0)
        
        path_label = Gtk.Label("Steam Library Path:")
        path_box.pack_start(path_label, False, False, 0)
        
        self.path_entry = Gtk.Entry()
        self.path_entry.set_text("Auto-detect all Steam libraries")
        self.path_entry.set_editable(False)
        self.path_entry.set_hexpand(True)
        path_box.pack_start(self.path_entry, True, True, 0)
        
        browse_button = Gtk.Button("Browse")
        browse_button.connect("clicked", self.on_browse_clicked)
        path_box.pack_start(browse_button, False, False, 0)
        
        # Auto-detect button
        self.auto_detect_button = Gtk.Button("Auto-Detect Steam Libraries")
        self.auto_detect_button.connect("clicked", self.on_auto_detect_clicked)
        self.auto_detect_button.set_size_request(-1, 40)
        box.pack_start(self.auto_detect_button, False, False, 5)
        
        # Analyze button
        self.analyze_button = Gtk.Button("Analyze All Steam Games")
        self.analyze_button.connect("clicked", self.on_analyze_clicked)
        self.analyze_button.set_size_request(-1, 40)
        box.pack_start(self.analyze_button, False, False, 5)
        
        # Steam App Management button
        add_steam_app_button = Gtk.Button("Save as Steam App")
        add_steam_app_button.connect("clicked", self.on_add_steam_app_clicked)
        add_steam_app_button.set_size_request(-1, 35)
        box.pack_start(add_steam_app_button, False, False, 5)
        
        # Results area
        results_frame = Gtk.Frame(label="Analysis Results")
        box.pack_start(results_frame, True, True, 0)
        
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        results_frame.add(scrolled)
        
        self.results_view = Gtk.TextView()
        self.results_view.set_editable(False)
        self.results_view.set_monospace(True)
        scrolled.add(self.results_view)
        
        return box
    
    def create_compile_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10)
        
        # Game selection
        game_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.pack_start(game_box, False, False, 0)
        
        game_label = Gtk.Label("Select Game:")
        game_box.pack_start(game_label, False, False, 0)
        
        self.game_combo = Gtk.ComboBoxText()
        self.game_combo.set_hexpand(True)
        game_box.pack_start(self.game_combo, True, True, 0)
        
        refresh_button = Gtk.Button("Refresh")
        refresh_button.connect("clicked", self.on_refresh_games)
        game_box.pack_start(refresh_button, False, False, 0)
        
        # Compilation options
        options_frame = Gtk.Frame(label="Compilation Options")
        box.pack_start(options_frame, False, False, 0)
        
        options_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        options_box.set_margin_left(10)
        options_box.set_margin_right(10)
        options_box.set_margin_top(10)
        options_box.set_margin_bottom(10)
        options_frame.add(options_box)
        
        self.priority_check = Gtk.CheckButton("Prioritize common shaders")
        self.priority_check.set_active(True)
        options_box.pack_start(self.priority_check, False, False, 0)
        
        self.optimize_check = Gtk.CheckButton("Steam Deck optimizations")
        self.optimize_check.set_active(True)
        options_box.pack_start(self.optimize_check, False, False, 0)
        
        self.background_check = Gtk.CheckButton("Continue in background")
        self.background_check.set_active(False)
        options_box.pack_start(self.background_check, False, False, 0)
        
        # Thread count
        thread_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        options_box.pack_start(thread_box, False, False, 5)
        
        thread_label = Gtk.Label("Compilation threads:")
        thread_box.pack_start(thread_label, False, False, 0)
        
        self.thread_spin = Gtk.SpinButton()
        self.thread_spin.set_adjustment(Gtk.Adjustment(4, 1, 8, 1, 0, 0))
        self.thread_spin.set_value(4)  # Steam Deck default
        thread_box.pack_start(self.thread_spin, False, False, 0)
        
        # Compile button
        self.compile_button = Gtk.Button("Start Compilation")
        self.compile_button.connect("clicked", self.on_compile_clicked)
        self.compile_button.set_size_request(-1, 40)
        box.pack_start(self.compile_button, False, False, 10)
        
        # Compilation log
        log_frame = Gtk.Frame(label="Compilation Log")
        box.pack_start(log_frame, True, True, 0)
        
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        log_frame.add(scrolled)
        
        self.compile_log = Gtk.TextView()
        self.compile_log.set_editable(False)
        self.compile_log.set_monospace(True)
        scrolled.add(self.compile_log)
        
        return box
    
    def create_settings_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10)
        box.set_margin_left(20)
        box.set_margin_right(20)
        
        # Cache settings
        cache_frame = Gtk.Frame(label="Cache Settings")
        box.pack_start(cache_frame, False, False, 0)
        
        cache_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        cache_box.set_margin_left(10)
        cache_box.set_margin_right(10)
        cache_box.set_margin_top(10)
        cache_box.set_margin_bottom(10)
        cache_frame.add(cache_box)
        
        # Cache location
        cache_path_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        cache_box.pack_start(cache_path_box, False, False, 0)
        
        cache_label = Gtk.Label("Cache location:")
        cache_path_box.pack_start(cache_label, False, False, 0)
        
        self.cache_label_path = Gtk.Label(str(Path.home() / '.cache/shader-predict-compile'))
        cache_path_box.pack_start(self.cache_label_path, True, True, 0)
        
        # Cache cleanup
        cleanup_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        cache_box.pack_start(cleanup_box, False, False, 0)
        
        cleanup_label = Gtk.Label("Clean caches older than:")
        cleanup_box.pack_start(cleanup_label, False, False, 0)
        
        self.cleanup_spin = Gtk.SpinButton()
        self.cleanup_spin.set_adjustment(Gtk.Adjustment(30, 7, 365, 1, 0, 0))
        cleanup_box.pack_start(self.cleanup_spin, False, False, 0)
        
        cleanup_days = Gtk.Label("days")
        cleanup_box.pack_start(cleanup_days, False, False, 0)
        
        cleanup_button = Gtk.Button("Clean Now")
        cleanup_button.connect("clicked", self.on_cleanup_clicked)
        cleanup_box.pack_end(cleanup_button, False, False, 0)
        
        # Performance settings
        perf_frame = Gtk.Frame(label="Performance Settings")
        box.pack_start(perf_frame, False, False, 0)
        
        perf_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        perf_box.set_margin_left(10)
        perf_box.set_margin_right(10)
        perf_box.set_margin_top(10)
        perf_box.set_margin_bottom(10)
        perf_frame.add(perf_box)
        
        self.auto_start = Gtk.CheckButton("Start with system (systemd service)")
        perf_box.pack_start(self.auto_start, False, False, 0)
        
        self.low_power = Gtk.CheckButton("Low power mode (reduce CPU usage)")
        perf_box.pack_start(self.low_power, False, False, 0)
        
        # Memory limit
        mem_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        perf_box.pack_start(mem_box, False, False, 0)
        
        mem_label = Gtk.Label("Memory limit (MB):")
        mem_box.pack_start(mem_label, False, False, 0)
        
        self.mem_spin = Gtk.SpinButton()
        self.mem_spin.set_adjustment(Gtk.Adjustment(2048, 512, 8192, 256, 0, 0))
        mem_box.pack_start(self.mem_spin, False, False, 0)
        
        # Save settings button
        save_button = Gtk.Button("Save Settings")
        save_button.connect("clicked", self.on_save_settings)
        box.pack_start(save_button, False, False, 20)
        
        # Restore to defaults section
        restore_frame = Gtk.Frame(label="Restore to Defaults")
        box.pack_start(restore_frame, False, False, 0)
        
        restore_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        restore_box.set_margin_left(10)
        restore_box.set_margin_right(10)
        restore_box.set_margin_top(10)
        restore_box.set_margin_bottom(10)
        restore_frame.add(restore_box)
        
        # Restore description
        restore_desc = Gtk.Label()
        restore_desc.set_markup("<i>Reset all settings to default values. Creates a backup before resetting.</i>")
        restore_desc.set_line_wrap(True)
        restore_box.pack_start(restore_desc, False, False, 0)
        
        # Restore options
        self.restore_include_cache = Gtk.CheckButton("Also clear shader cache")
        restore_box.pack_start(self.restore_include_cache, False, False, 0)
        
        self.restore_include_logs = Gtk.CheckButton("Also clear log files")
        restore_box.pack_start(self.restore_include_logs, False, False, 0)
        
        # Restore buttons
        restore_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        restore_box.pack_start(restore_button_box, False, False, 5)
        
        restore_button = Gtk.Button("Restore to Defaults")
        restore_button.get_style_context().add_class("suggested-action")
        restore_button.connect("clicked", self.on_restore_defaults_clicked)
        restore_button_box.pack_start(restore_button, False, False, 0)
        
        # Quick settings reset (no cache/logs)
        quick_reset_button = Gtk.Button("Quick Reset Settings Only")
        quick_reset_button.connect("clicked", self.on_quick_reset_clicked)
        restore_button_box.pack_start(quick_reset_button, False, False, 0)
        
        # Uninstall section
        uninstall_frame = Gtk.Frame(label="Uninstall")
        box.pack_end(uninstall_frame, False, False, 0)
        
        uninstall_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        uninstall_box.set_margin_left(10)
        uninstall_box.set_margin_right(10)
        uninstall_box.set_margin_top(10)
        uninstall_box.set_margin_bottom(10)
        uninstall_frame.add(uninstall_box)
        
        uninstall_label = Gtk.Label("Remove all data and uninstall the application")
        uninstall_box.pack_start(uninstall_label, False, False, 0)
        
        uninstall_button = Gtk.Button("Uninstall")
        uninstall_button.get_style_context().add_class("destructive-action")
        uninstall_button.connect("clicked", self.on_uninstall_clicked)
        uninstall_box.pack_start(uninstall_button, False, False, 0)
        
        return box
    
    def check_compatibility(self):
        """Check and display compatibility status"""
        checks = self.fossilize.check_compatibility()
        
        for check, status in checks.items():
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            
            icon = "✓" if status else "✗"
            color = "green" if status else "red"
            
            label = Gtk.Label()
            label.set_markup(f'<span color="{color}">{icon}</span> {check.replace("_", " ").title()}')
            box.pack_start(label, False, False, 0)
            
            self.compat_box.pack_start(box, False, False, 0)
            
        self.compat_box.show_all()
    
    def on_browse_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            "Select Steam Library", self,
            Gtk.FileChooserAction.SELECT_FOLDER,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        )
        
        if dialog.run() == Gtk.ResponseType.OK:
            self.path_entry.set_text(dialog.get_filename())
            self.path_entry.set_editable(True)
            
        dialog.destroy()
    
    def on_auto_detect_clicked(self, button):
        """Auto-detect all Steam libraries"""
        if self.is_analyzing:
            return
            
        self.is_analyzing = True
        self.status_label.set_text("Auto-detecting Steam libraries...")
        self.progress_bar.pulse()
        
        def detect_thread():
            try:
                from steam_deck_compat import SteamDeckCompatibility
                compat = SteamDeckCompatibility()
                detected_libraries = compat.auto_detect_steam_libraries()
                GLib.idle_add(self.display_detected_libraries, detected_libraries)
            except Exception as e:
                GLib.idle_add(self.display_error, f"Auto-detection failed: {e}")
            finally:
                GLib.idle_add(self.analysis_complete)
        
        thread = threading.Thread(target=detect_thread)
        thread.daemon = True
        thread.start()
    
    def display_detected_libraries(self, libraries):
        """Display detected Steam libraries"""
        buffer = self.results_view.get_buffer()
        buffer.set_text("")
        iter_end = buffer.get_end_iter()
        
        if not libraries:
            buffer.insert(iter_end, "❌ No Steam libraries found\n\n")
            buffer.insert(iter_end, "This could mean:\n")
            buffer.insert(iter_end, "• Steam is not installed\n")
            buffer.insert(iter_end, "• No games are installed\n")
            buffer.insert(iter_end, "• Games are installed in non-standard locations\n")
            return
        
        buffer.insert(iter_end, f"🎮 Found {len(libraries)} Steam Libraries\n")
        buffer.insert(iter_end, "=" * 50 + "\n\n")
        
        total_games = 0
        for i, library in enumerate(libraries, 1):
            buffer.insert(iter_end, f"📁 Library {i}: {library['type'].title()}\n")
            buffer.insert(iter_end, f"   Path: {library['path']}\n")
            buffer.insert(iter_end, f"   Games: {library['games_count']}\n")
            buffer.insert(iter_end, f"   Accessible: {'✅' if library['accessible'] else '❌'}\n")
            
            if library.get('device'):
                buffer.insert(iter_end, f"   Device: {library['device']}\n")
            if library.get('mount_point'):
                buffer.insert(iter_end, f"   Mount Point: {library['mount_point']}\n")
                
            buffer.insert(iter_end, "\n")
            total_games += library['games_count']
        
        buffer.insert(iter_end, f"📊 Total Games: {total_games}\n")
        
        # Update path entry to show auto-detection mode
        self.path_entry.set_text(f"Auto-detected {len(libraries)} Steam libraries")
        self.path_entry.set_editable(False)
    
    def analysis_complete(self):
        """Called when analysis is complete"""
        self.status_label.set_text("Analysis complete")
        
        # Re-enable UI elements that might have been disabled during analysis
        self.auto_detect_button.set_sensitive(True)
        self.analyze_button.set_sensitive(True)
    
    def display_error(self, error_message):
        """Display error message in results view"""
        buffer = self.results_view.get_buffer()
        buffer.set_text(f"❌ Error: {error_message}\n\nPlease check the logs for more details.")
        self.status_label.set_text("Error occurred")
    
    def on_analyze_clicked(self, button):
        if self.is_analyzing:
            return
            
        self.is_analyzing = True
        
        self.status_label.set_text("Analyzing all Steam games...")
        self.progress_bar.pulse()
        
        def analyze_thread():
            try:
                from steam_deck_compat import SteamDeckCompatibility
                compat = SteamDeckCompatibility()
                
                # Check if using auto-detection or manual path
                path_text = self.path_entry.get_text()
                if "Auto-detect" in path_text or not self.path_entry.get_editable():
                    # Use auto-detection to analyze all games
                    all_games = compat.get_all_steam_games()
                    results = self.analyze_all_games(all_games)
                else:
                    # Use manual path analysis
                    steam_path = Path(path_text)
                    results = self.engine.analyze_steam_library(steam_path)
                
                GLib.idle_add(self.display_analysis_results, results)
            except Exception as e:
                GLib.idle_add(self.show_error, f"Analysis error: {e}")
            finally:
                self.is_analyzing = False
                
        thread = threading.Thread(target=analyze_thread)
        thread.daemon = True
        thread.start()
    
    def analyze_all_games(self, all_games):
        """Analyze all detected games and compile comprehensive results"""
        results = {
            'total_games': len(all_games),
            'games_with_shaders': 0,
            'total_shaders': 0,
            'priority_games': [],
            'engine_breakdown': {},
            'library_breakdown': {},
            'size_stats': {
                'total_size_gb': 0,
                'average_size_mb': 0,
                'largest_games': []
            }
        }
        
        # Analyze each game
        games_with_shaders = []
        total_size_mb = 0
        
        for game in all_games:
            total_size_mb += game['size_mb']
            
            # Count engines
            engine = game.get('engine_detected', 'Unknown')
            results['engine_breakdown'][engine] = results['engine_breakdown'].get(engine, 0) + 1
            
            # Count library types
            lib_type = game.get('library_type', 'unknown')
            results['library_breakdown'][lib_type] = results['library_breakdown'].get(lib_type, 0) + 1
            
            # Check for shaders
            if game.get('has_shaders', False):
                results['games_with_shaders'] += 1
                games_with_shaders.append(game)
                
                # Priority based on recent activity and size
                if game.get('last_modified') and game['size_mb'] > 1000:  # > 1GB and recently modified
                    results['priority_games'].append({
                        'name': game['name'],
                        'size_mb': game['size_mb'],
                        'engine': engine,
                        'last_modified': game['last_modified'].strftime('%Y-%m-%d') if game['last_modified'] else 'Unknown'
                    })
        
        # Calculate size statistics
        results['size_stats']['total_size_gb'] = round(total_size_mb / 1024, 1)
        if all_games:
            results['size_stats']['average_size_mb'] = round(total_size_mb / len(all_games), 1)
            
        # Find largest games
        sorted_games = sorted(all_games, key=lambda x: x['size_mb'], reverse=True)
        results['size_stats']['largest_games'] = [
            {'name': game['name'], 'size_mb': game['size_mb']}
            for game in sorted_games[:5]
        ]
        
        # Sort priority games by size
        results['priority_games'] = sorted(results['priority_games'], key=lambda x: x['size_mb'], reverse=True)[:10]
        
        # Estimate shader count (rough approximation)
        results['total_shaders'] = results['games_with_shaders'] * 150  # Rough estimate
        
        return results
    
    def on_add_steam_app_clicked(self, button):
        """Add Shader Predictive Compiler as a Steam app"""
        try:
            from gaming_mode_integration import GamingModeIntegration
            gaming_integration = GamingModeIntegration()
            
            # Check status first
            status = gaming_integration.get_gaming_mode_status()
            
            if not status['steam_found']:
                self.show_error("❌ Steam installation not found!\n\nPlease make sure Steam is installed and try again.")
                return
            
            if status['non_steam_game_added']:
                self.show_message("✅ Shader Predictive Compiler is already in your Steam library!\n\nYou can find it at:\n• Library → Non-Steam → Shader Predictive Compiler\n\nIf you don't see it, try restarting Steam.")
                return
            
            # Show progress
            button.set_label("Adding to Steam...")
            button.set_sensitive(False)
            
            def add_to_steam():
                try:
                    # First create the gaming mode launcher
                    launcher_created = gaming_integration.create_gaming_mode_ui_launcher()
                    if not launcher_created:
                        GLib.idle_add(self.show_error, "❌ Failed to create Gaming Mode launcher script.")
                        return
                    
                    # Add the shader-predict-compile app itself
                    success = gaming_integration.add_as_non_steam_game()
                    
                    if success:
                        # Check if manual instructions were created  
                        instructions_file = Path(__file__).parent.parent / 'STEAM_INSTRUCTIONS.txt'
                        if instructions_file.exists():
                            message = "⚠️ Partial success - Manual steps required!\n\nA desktop entry has been created, but automatic Steam integration had issues.\n\nPlease check the STEAM_INSTRUCTIONS.txt file in the application folder for manual setup steps.\n\nKey steps:\n1. Open Steam in Desktop Mode\n2. Games → Add Non-Steam Game\n3. Browse to the launcher script\n4. Add and restart Steam"
                        else:
                            message = "✅ Successfully added to Steam library!\n\nYou can now launch it from Gaming Mode:\n• Library → Non-Steam → Shader Predictive Compiler\n\nNote: You may need to restart Steam to see the changes."
                        
                        GLib.idle_add(self.show_message, message)
                    else:
                        GLib.idle_add(self.show_error, "❌ Failed to add to Steam library.\n\nPossible solutions:\n• Make sure Steam is running\n• Try restarting Steam\n• Check the STEAM_INSTRUCTIONS.txt file for manual setup")
                        
                except Exception as e:
                    GLib.idle_add(self.show_error, f"Error during Steam integration: {e}")
                finally:
                    GLib.idle_add(self.reset_steam_button, button)
            
            # Run in thread to avoid blocking UI
            import threading
            thread = threading.Thread(target=add_to_steam)
            thread.daemon = True
            thread.start()
                
        except Exception as e:
            self.show_error(f"Error initializing Steam integration: {e}")
            self.reset_steam_button(button)
    
    def reset_steam_button(self, button):
        """Reset the Steam button to original state"""
        button.set_label("Save as Steam App")
        button.set_sensitive(True)
    
    
    def show_message(self, message):
        """Show an info message dialog"""
        dialog = Gtk.MessageDialog(
            self, Gtk.DialogFlags.MODAL,
            Gtk.MessageType.INFO, Gtk.ButtonsType.OK,
            message
        )
        dialog.run()
        dialog.destroy()
        
        # Pulse progress bar
        def pulse():
            if self.is_analyzing:
                self.progress_bar.pulse()
                return True
            else:
                self.progress_bar.set_fraction(0)
                return False
                
        GLib.timeout_add(100, pulse)
    
    def display_analysis_results(self, results):
        buffer = self.results_view.get_buffer()
        buffer.set_text("")
        
        text = f"Steam Library Analysis Results\n"
        text += f"{'='*40}\n\n"
        text += f"Total games found: {results.get('total_games', 0)}\n"
        
        # Handle both possible key names for shader count
        shader_count = results.get('total_shaders_estimated') or results.get('total_shaders', 0)
        text += f"Estimated total shaders: {shader_count:,}\n\n"
        
        if results.get('recommendations'):
            text += "Recommendations:\n"
            for rec in results['recommendations']:
                text += f"  • {rec}\n"
                
        buffer.set_text(text)
        self.status_label.set_text("Analysis complete")
        
        # Populate game combo
        self.refresh_game_list()
    
    def on_refresh_games(self, button):
        self.refresh_game_list()
    
    def refresh_game_list(self):
        self.game_combo.remove_all()
        
        for game_id in sorted(self.engine.game_profiles.keys()):
            profile = self.engine.game_profiles[game_id]
            self.game_combo.append_text(f"{game_id} (~{profile.shader_count} shaders)")
            
        if len(self.engine.game_profiles) > 0:
            self.game_combo.set_active(0)
    
    def on_compile_clicked(self, button):
        if self.is_compiling:
            return
            
        selected = self.game_combo.get_active_text()
        if not selected:
            self.show_error("Please select a game first")
            return
            
        game_id = selected.split(" ")[0]
        self.selected_game = game_id
        
        self.is_compiling = True
        self.compile_button.set_label("Compiling...")
        self.compile_button.set_sensitive(False)
        
        # Clear log
        self.compile_log.get_buffer().set_text("")
        
        def compile_thread():
            try:
                # Get priorities
                priorities = self.engine.predict_shader_priorities(game_id)
                
                # Compile shaders
                results = self.fossilize.compile_shaders(
                    game_id, priorities, 
                    progress_callback=self.on_compile_progress
                )
                
                # Integrate with Steam
                cache_dir = Path.home() / '.cache/shader-predict-compile' / game_id
                self.fossilize.integrate_with_steam(game_id, cache_dir)
                
                GLib.idle_add(self.on_compile_complete, results)
                
            except Exception as e:
                GLib.idle_add(self.show_error, f"Compilation error: {e}")
            finally:
                self.is_compiling = False
                GLib.idle_add(self.reset_compile_button)
                
        thread = threading.Thread(target=compile_thread)
        thread.daemon = True
        thread.start()
    
    def on_compile_progress(self, progress):
        def update():
            buffer = self.compile_log.get_buffer()
            end_iter = buffer.get_end_iter()
            
            status = "✓" if progress['success'] else "✗"
            text = f"{status} {progress['shader']} - {progress['time']:.2f}s\n"
            buffer.insert(end_iter, text)
            
            # Auto-scroll
            self.compile_log.scroll_to_iter(end_iter, 0.0, False, 0.0, 0.0)
            
            # Update progress bar
            fraction = progress['total_compiled'] / 100  # Estimate
            self.progress_bar.set_fraction(min(fraction, 1.0))
            self.progress_bar.set_text(f"Compiled: {progress['total_compiled']}")
            
        GLib.idle_add(update)
    
    def on_compile_complete(self, results):
        buffer = self.compile_log.get_buffer()
        end_iter = buffer.get_end_iter()
        
        summary = f"\n{'='*40}\n"
        summary += f"Compilation Complete!\n"
        summary += f"Compiled: {results['compiled']}\n"
        summary += f"Failed: {results['failed']}\n"
        summary += f"Cached: {results['cached']}\n"
        summary += f"Total time: {results['total_time']:.1f}s\n"
        
        buffer.insert(end_iter, summary)
        
        self.status_label.set_text("Compilation complete")
        self.progress_bar.set_fraction(1.0)
        self.progress_bar.set_text("Done")
    
    def reset_compile_button(self):
        self.compile_button.set_label("Start Compilation")
        self.compile_button.set_sensitive(True)
    
    def on_cleanup_clicked(self, button):
        days = int(self.cleanup_spin.get_value())
        cleaned = self.fossilize.cleanup_old_caches(days)
        
        dialog = Gtk.MessageDialog(
            self, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK,
            f"Cleaned {cleaned} old cache directories"
        )
        dialog.run()
        dialog.destroy()
    
    def on_save_settings(self, button):
        """Save all UI settings using the settings manager"""
        settings = {
            'auto_start': self.auto_start.get_active(),
            'low_power': self.low_power.get_active(),
            'memory_limit': int(self.mem_spin.get_value()),
            'cleanup_days': int(self.cleanup_spin.get_value()),
            # Compilation options
            'continue_background': self.background_check.get_active(),
            'prioritize_common': self.priority_check.get_active(),
            'steam_deck_optimize': self.optimize_check.get_active(),
            'thread_count': int(self.thread_spin.get_value())
        }
        
        # Use the settings manager for robust saving
        if self.settings_manager.update(settings) and self.settings_manager.save_settings():
            self.show_info("Settings saved successfully")
        else:
            self.show_error("Failed to save settings. Please check the logs.")
    
    def load_settings(self):
        """Load settings using the settings manager and apply to UI controls"""
        try:
            # Settings manager handles all the complexity of loading, validation, and recovery
            settings = self.settings_manager.get_all_settings()
            
            # Apply loaded settings to UI controls
            self.auto_start.set_active(settings.get('auto_start', True))
            self.low_power.set_active(settings.get('low_power', True))
            self.mem_spin.set_value(settings.get('memory_limit', 2048))
            self.cleanup_spin.set_value(settings.get('cleanup_days', 30))
            
            # Compilation options
            self.background_check.set_active(settings.get('continue_background', False))
            self.priority_check.set_active(settings.get('prioritize_common', True))
            self.optimize_check.set_active(settings.get('steam_deck_optimize', True))
            self.thread_spin.set_value(settings.get('thread_count', 4))
                
        except Exception as e:
            # If loading fails, just use defaults
            print(f"Warning: Could not load settings: {e}")
            # Set UI to default values
            self.auto_start.set_active(True)
            self.low_power.set_active(True)
            self.mem_spin.set_value(2048)
            self.cleanup_spin.set_value(30)
            self.background_check.set_active(False)
            self.priority_check.set_active(True)
            self.optimize_check.set_active(True)
            self.thread_spin.set_value(4)
    
    def on_uninstall_clicked(self, button):
        dialog = Gtk.MessageDialog(
            self, 0, Gtk.MessageType.WARNING,
            Gtk.ButtonsType.YES_NO,
            "Are you sure you want to uninstall?"
        )
        dialog.format_secondary_text(
            "This will remove all cached shaders and application data."
        )
        
        if dialog.run() == Gtk.ResponseType.YES:
            self.perform_uninstall()
            
        dialog.destroy()
    
    def perform_uninstall(self):
        # This would be handled by the install script
        self.show_info("Please run 'sudo ./install.sh --uninstall' to remove the application")
    
    def on_restore_defaults_clicked(self, button):
        """Handle full restore to defaults with options"""
        include_cache = self.restore_include_cache.get_active()
        include_logs = self.restore_include_logs.get_active()
        
        # Confirmation dialog
        dialog = Gtk.MessageDialog(
            self, 0, Gtk.MessageType.WARNING,
            Gtk.ButtonsType.YES_NO,
            "Restore all settings to defaults?"
        )
        
        secondary_text = "This will reset all settings to default values and create a backup."
        if include_cache:
            secondary_text += "\n• Shader cache will be cleared"
        if include_logs:
            secondary_text += "\n• Log files will be cleared"
        secondary_text += "\n\nThis action cannot be undone (except from backup)."
        
        dialog.format_secondary_text(secondary_text)
        
        if dialog.run() == Gtk.ResponseType.YES:
            self.perform_restore_defaults(include_cache, include_logs)
            
        dialog.destroy()
    
    def on_quick_reset_clicked(self, button):
        """Handle quick settings reset (no cache or logs)"""
        dialog = Gtk.MessageDialog(
            self, 0, Gtk.MessageType.QUESTION,
            Gtk.ButtonsType.YES_NO,
            "Reset settings to defaults?"
        )
        dialog.format_secondary_text(
            "This will only reset application settings to defaults.\n"
            "Cache and logs will not be affected.\n"
            "A backup will be created."
        )
        
        if dialog.run() == Gtk.ResponseType.YES:
            self.perform_restore_defaults(include_cache=False, include_logs=False)
            
        dialog.destroy()
    
    def perform_restore_defaults(self, include_cache: bool, include_logs: bool):
        """Perform the actual restore to defaults operation"""
        try:
            # Show progress
            self.status_label.set_text("Restoring to defaults...")
            self.progress_bar.set_fraction(0.1)
            self.progress_bar.set_text("Creating backup...")
            
            # Let the UI update
            while Gtk.events_pending():
                Gtk.main_iteration()
            
            # Perform restore
            results = self.settings_manager.restore_to_defaults(
                include_cache=include_cache, 
                include_logs=include_logs
            )
            
            self.progress_bar.set_fraction(0.5)
            self.progress_bar.set_text("Resetting settings...")
            
            while Gtk.events_pending():
                Gtk.main_iteration()
            
            # Update UI with new settings
            self.load_settings()
            
            self.progress_bar.set_fraction(1.0)
            self.progress_bar.set_text("Restore completed")
            
            # Show results
            success_count = sum(1 for result in results.values() if result)
            total_count = len(results)
            
            result_message = f"Restore completed: {success_count}/{total_count} operations successful"
            
            if results['backup_created']:
                result_message += "\n✓ Settings backup created"
            if results['settings_reset']:
                result_message += "\n✓ Settings reset to defaults"
            if results['enhanced_settings_reset']:
                result_message += "\n✓ Enhanced settings reset"
            if include_cache and results['cache_cleared']:
                result_message += "\n✓ Cache cleared"
            if include_logs and results['logs_cleared']:
                result_message += "\n✓ Logs cleared"
            
            # Show any warnings
            warnings = []
            if include_cache and not results['cache_cleared']:
                warnings.append("• Could not clear all cache files")
            if include_logs and not results['logs_cleared']:
                warnings.append("• Could not clear all log files")
            
            if warnings:
                result_message += "\n\nWarnings:\n" + "\n".join(warnings)
            
            self.show_info(result_message)
            
            # Reset progress bar
            self.progress_bar.set_fraction(0.0)
            self.progress_bar.set_text("")
            self.status_label.set_text("Ready")
            
        except Exception as e:
            self.show_error(f"Error during restore: {str(e)}")
            self.progress_bar.set_fraction(0.0)
            self.progress_bar.set_text("")
            self.status_label.set_text("Error")
    
    def show_error(self, message):
        dialog = Gtk.MessageDialog(
            self, 0, Gtk.MessageType.ERROR,
            Gtk.ButtonsType.OK, message
        )
        dialog.run()
        dialog.destroy()
    
    def show_info(self, message):
        dialog = Gtk.MessageDialog(
            self, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, message
        )
        dialog.run()
        dialog.destroy()


def main():
    win = ShaderPredictCompileWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()