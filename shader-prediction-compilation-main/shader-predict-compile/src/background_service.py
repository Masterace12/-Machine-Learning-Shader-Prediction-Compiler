#!/usr/bin/env python3

import os
import sys
import time
import json
import signal
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Set up logging
log_dir = Path.home() / '.cache' / 'shader-predict-compile'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Create a dummy psutil for basic operations
    class DummyPsutil:
        @staticmethod
        def cpu_count():
            return 4  # Assume 4 cores for Steam Deck
        @staticmethod
        def cpu_percent(interval=None):
            return 25.0  # Assume moderate CPU usage
        @staticmethod
        def virtual_memory():
            class Memory:
                total = 16 * 1024 * 1024 * 1024  # 16GB
                available = 8 * 1024 * 1024 * 1024  # 8GB available
                percent = 50.0
            return Memory()
        @staticmethod
        def process_iter(*args, **kwargs):
            return []  # No processes
        class Process:
            def cpu_affinity(self, *args):
                pass
    psutil = DummyPsutil()

# Import with error handling
try:
    from shader_analyzer import ShaderAnalyzer
    from heuristic_engine import HeuristicEngine
    from fossilize_integration import FossilizeIntegration
    from steam_deck_compat import SteamDeckCompatibility
    from realtime_monitor import GamingModeMonitor
    from radv_optimizer import RADVOptimizer
    from gaming_mode_power import GamingModePowerManager
    from advanced_cache_manager import AdvancedCacheManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Some features may be unavailable. Creating placeholder classes...")
    
    # Create placeholder classes for missing modules
    class ShaderAnalyzer:
        def __init__(self): pass
        def analyze_game(self, *args, **kwargs): return {}
    
    class HeuristicEngine:
        def __init__(self): pass
        def get_predictions(self, *args, **kwargs): return []
    
    class FossilizeIntegration:
        def __init__(self): pass
        def compile_shaders(self, *args, **kwargs): return True
    
    class SteamDeckCompatibility:
        def __init__(self): pass
        def check_system(self): return True
    
    class GamingModeMonitor:
        def __init__(self): pass
        def start(self): pass
        def stop(self): pass
    
    class RADVOptimizer:
        def __init__(self): pass
        def optimize(self): pass
    
    class GamingModePowerManager:
        def __init__(self): pass
        def apply_profile(self, *args): pass
    
    class AdvancedCacheManager:
        def __init__(self): pass
        def clean_cache(self): pass

@dataclass
class ServiceConfig:
    max_cpu_usage: float = 50.0  # Maximum CPU usage percentage
    max_memory_mb: int = 2048    # Maximum memory usage in MB
    check_interval: int = 300    # Check interval in seconds (5 minutes)
    auto_compile_threshold: int = 5  # Auto-compile if game has >5 launches
    background_priority: bool = True  # Run with lower priority
    thermal_throttling: bool = True   # Enable thermal throttling
    battery_aware: bool = True        # Reduce activity on battery

class BackgroundService:
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.running = False
        self.paused = False
        
        # Initialize core components
        self.analyzer = ShaderAnalyzer()
        self.engine = HeuristicEngine()
        self.fossilize = FossilizeIntegration()
        self.compat = SteamDeckCompatibility()
        
        # Initialize new enhanced components
        self.monitor = GamingModeMonitor(sample_rate_hz=60)
        self.radv_optimizer = RADVOptimizer()
        self.power_manager = GamingModePowerManager()
        self.cache_manager = AdvancedCacheManager()
        
        # State tracking
        self.game_launch_counts = {}
        self.last_analysis_time = {}
        self.compilation_queue = []
        self.current_compilation = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("Background service initialized")
        
    def _load_config(self, config_path: Optional[Path]) -> ServiceConfig:
        """Load service configuration"""
        if config_path is None:
            config_path = Path.home() / '.config/shader-predict-compile/service.json'
            
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                return ServiceConfig(**config_data)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                
        return ServiceConfig()
    
    def _setup_logging(self):
        """Setup logging for the service"""
        log_dir = Path.home() / '.cache/shader-predict-compile/logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'service.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('shader_predict_service')
        
        # Rotate logs if they get too large
        log_file = log_dir / 'service.log'
        if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
            backup_file = log_dir / f'service.log.{int(time.time())}'
            log_file.rename(backup_file)
            
            # Keep only last 5 log files
            log_files = sorted(log_dir.glob('service.log.*'))
            for old_log in log_files[:-5]:
                old_log.unlink()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def start(self):
        """Start the background service"""
        if self.running:
            self.logger.warning("Service already running")
            return
            
        self.running = True
        self.logger.info("Starting shader predictive compilation service")
        
        # Apply runtime optimizations
        optimizations = self.compat.apply_runtime_optimizations()
        self.logger.info(f"Applied optimizations: {optimizations}")
        
        # Initialize enhanced systems
        self._initialize_enhanced_systems()
        
        # Set process priority if configured
        if self.config.background_priority:
            try:
                os.nice(10)  # Lower priority
                self.logger.info("Set background priority")
            except:
                self.logger.warning("Could not set background priority")
                
        # Start monitoring threads
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        compile_thread = threading.Thread(target=self._compilation_loop, daemon=True)
        compile_thread.start()
        
        # Main service loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        except Exception as e:
            self.logger.error(f"Service error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the background service"""
        if not self.running:
            return
            
        self.logger.info("Stopping shader predictive compilation service")
        self.running = False
        
        # Stop enhanced systems
        try:
            self.monitor.stop_monitoring()
            self.power_manager.stop_adaptive_power_management()
            self.cache_manager.stop_automatic_management()
        except Exception as e:
            self.logger.warning(f"Error stopping enhanced systems: {e}")
        
        # Wait for current compilation to finish
        if self.current_compilation:
            self.logger.info("Waiting for current compilation to finish...")
            # Give it up to 30 seconds to finish
            for _ in range(30):
                if not self.current_compilation:
                    break
                time.sleep(1)
    
    def _main_loop(self):
        """Main service loop"""
        while self.running:
            try:
                # Check system conditions with enhanced monitoring
                if self._should_pause_enhanced():
                    if not self.paused:
                        self.logger.info("Pausing service due to system conditions")
                        self.paused = True
                    time.sleep(30)  # Check again in 30 seconds
                    continue
                    
                if self.paused:
                    self.logger.info("Resuming service")
                    self.paused = False
                    
                # Monitor Steam games
                self._monitor_steam_activity()
                
                # Process compilation queue
                self._process_compilation_queue()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep until next check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _should_pause(self) -> bool:
        """Check if service should be paused"""
        reasons = []
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.config.max_cpu_usage:
            reasons.append(f"High CPU usage: {cpu_percent:.1f}%")
            
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_mb = (memory.total - memory.available) / (1024 * 1024)
        if memory_mb > self.config.max_memory_mb:
            reasons.append(f"High memory usage: {memory_mb:.0f}MB")
            
        # Check thermal state
        if self.config.thermal_throttling and not self.compat._check_thermal_state():
            reasons.append("High temperature detected")
            
        # Check battery state
        if self.config.battery_aware and not self.compat._check_battery_state():
            reasons.append("Low battery level")
            
        # Check if Steam is running a game
        if self._is_game_running():
            reasons.append("Game is currently running")
            
        if reasons:
            self.logger.debug(f"Pausing due to: {', '.join(reasons)}")
            return True
            
        return False
    
    def _is_game_running(self) -> bool:
        """Check if a Steam game is currently running"""
        try:
            # Look for Steam game processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = proc.info['name'].lower()
                    cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                    
                    # Skip Steam client itself
                    if 'steam' in name and 'steamclient' not in name:
                        continue
                        
                    # Look for common game indicators
                    game_indicators = [
                        'reaper', 'gameoverlayrenderer', 'steaminput',
                        '.exe', 'game', 'launcher'
                    ]
                    
                    if any(indicator in cmdline for indicator in game_indicators):
                        # Check if it's using significant CPU (likely a running game)
                        try:
                            if proc.cpu_percent() > 5:
                                return True
                        except:
                            pass
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.debug(f"Error checking for running games: {e}")
            
        return False
    
    def _monitor_loop(self):
        """Monitor system resources and adjust behavior"""
        while self.running:
            try:
                # Log system stats periodically
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                self.logger.debug(f"System: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
                
                # Adjust compilation parameters based on system load
                if cpu_percent > 80:
                    # Reduce parallel compilation
                    self.fossilize.steam_deck_optimizations['max_parallel_compiles'] = 2
                elif cpu_percent < 30:
                    # Can increase parallel compilation
                    optimal = self.compat.optimizations['max_parallel_compiles']
                    self.fossilize.steam_deck_optimizations['max_parallel_compiles'] = optimal
                    
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)
    
    def _monitor_steam_activity(self):
        """Monitor Steam game launches and update statistics using auto-detection"""
        try:
            # Use auto-detection to find all Steam libraries
            detected_libraries = self.compat.auto_detect_steam_libraries()
            
            if not detected_libraries:
                self.logger.warning("No Steam libraries found with auto-detection")
                return
            
            self.logger.debug(f"Found {len(detected_libraries)} Steam libraries")
            
            for library in detected_libraries:
                if not library.get('accessible', False):
                    self.logger.debug(f"Skipping inaccessible library: {library['path']}")
                    continue
                    
                steamapps_path = Path(library['path'])
                self.logger.debug(f"Scanning Steam library: {steamapps_path} ({library['games_count']} games)")
                
                # Check for recent game launches and update statistics
                self._scan_for_new_games(steamapps_path)
                self._update_game_statistics(steamapps_path)
                
        except Exception as e:
            self.logger.error(f"Error monitoring Steam activity: {e}")
    
    def get_all_detected_games(self):
        """Get all games detected across all Steam libraries with comprehensive info"""
        try:
            return self.compat.get_all_steam_games()
        except Exception as e:
            self.logger.error(f"Error getting all detected games: {e}")
            return []
    
    def get_service_status(self):
        """Get comprehensive service status including auto-detection results"""
        try:
            detected_libraries = self.compat.auto_detect_steam_libraries()
            all_games = self.get_all_detected_games()
            
            return {
                'running': True,
                'paused': self.paused,
                'steam_libraries_detected': len(detected_libraries),
                'total_games_detected': len(all_games),
                'gaming_mode_active': self.compat.detect_gaming_mode(),
                'libraries': [
                    {
                        'path': lib['path'],
                        'type': lib['type'], 
                        'games_count': lib['games_count'],
                        'accessible': lib['accessible']
                    }
                    for lib in detected_libraries
                ],
                'recent_games': all_games[:10] if all_games else []
            }
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {
                'running': False,
                'error': str(e)
            }
    
    def _scan_for_new_games(self, steamapps_path: Path):
        """Scan for newly installed games"""
        common_path = steamapps_path / 'common'
        if not common_path.exists():
            self.logger.debug(f"steamapps/common path does not exist: {common_path}")
            return
            
        games_found = 0
        new_games = 0
        
        for game_dir in common_path.iterdir():
            if game_dir.is_dir():
                games_found += 1
                game_name = game_dir.name
                
                # Check if this is a new game
                if game_name not in self.game_launch_counts:
                    new_games += 1
                    self.logger.info(f"New game detected: {game_name}")
                    self.game_launch_counts[game_name] = 0
                    
                    # Get game size for analysis prioritization
                    try:
                        game_size = sum(f.stat().st_size for f in game_dir.rglob('*') if f.is_file())
                        game_size_mb = game_size / (1024 * 1024)
                        
                        # Higher priority for larger games (likely to have more shaders)
                        priority = min(int(game_size_mb / 1000), 10)  # Cap at priority 10
                        
                        self.logger.info(f"Game {game_name}: {game_size_mb:.1f} MB, priority {priority}")
                    except Exception as e:
                        self.logger.debug(f"Could not calculate size for {game_name}: {e}")
                        priority = 1
                    
                    # Add to compilation queue for analysis
                    self.compilation_queue.append({
                        'type': 'analyze',
                        'game_id': game_name,
                        'game_path': game_dir,
                        'priority': priority,
                        'discovered_time': time.time()
                    })
        
        if new_games > 0:
            self.logger.info(f"Found {new_games} new games out of {games_found} total games")
        else:
            self.logger.debug(f"No new games found (scanned {games_found} existing games)")
    
    def _update_game_statistics(self, steamapps_path: Path):
        """Update game launch statistics"""
        try:
            # This is a simplified approach - in reality, you'd monitor Steam logs
            # or use Steam API if available
            
            # For now, we'll just check file modification times as a heuristic
            common_path = steamapps_path / 'common'
            if not common_path.exists():
                return
                
            for game_dir in common_path.iterdir():
                if game_dir.is_dir():
                    game_id = game_dir.name
                    
                    # Check if game was recently accessed
                    recent_threshold = datetime.now() - timedelta(hours=1)
                    
                    try:
                        last_modified = datetime.fromtimestamp(game_dir.stat().st_mtime)
                        if last_modified > recent_threshold:
                            # Game was likely launched recently
                            if game_id not in self.game_launch_counts:
                                self.game_launch_counts[game_id] = 0
                                
                            self.game_launch_counts[game_id] += 1
                            self.logger.info(f"Game launch detected: {game_id} (count: {self.game_launch_counts[game_id]})")
                            
                            # Queue for compilation if it meets threshold
                            if (self.game_launch_counts[game_id] >= self.config.auto_compile_threshold and
                                game_id not in [item['game_id'] for item in self.compilation_queue]):
                                
                                self.compilation_queue.append({
                                    'type': 'compile',
                                    'game_id': game_id,
                                    'game_path': game_dir,
                                    'priority': self.game_launch_counts[game_id]
                                })
                                
                    except OSError:
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error updating game statistics: {e}")
    
    def _compilation_loop(self):
        """Background compilation loop"""
        while self.running:
            try:
                if self.paused or not self.compilation_queue:
                    time.sleep(10)
                    continue
                    
                # Sort queue by priority
                self.compilation_queue.sort(key=lambda x: x['priority'], reverse=True)
                
                # Process next item
                task = self.compilation_queue.pop(0)
                self.current_compilation = task
                
                self.logger.info(f"Processing {task['type']} for {task['game_id']}")
                
                if task['type'] == 'analyze':
                    self._analyze_game(task)
                elif task['type'] == 'compile':
                    self._compile_game_shaders(task)
                    
                self.current_compilation = None
                
            except Exception as e:
                self.logger.error(f"Error in compilation loop: {e}")
                self.current_compilation = None
                time.sleep(30)
    
    def _analyze_game(self, task: Dict):
        """Analyze a game for shader patterns"""
        try:
            game_path = task['game_path']
            game_id = task['game_id']
            
            # Run analysis
            results = self.analyzer.analyze_game_directory(game_path)
            
            # Update engine with results
            profile = self.engine._profile_game(game_path, {})
            if profile:
                self.engine.game_profiles[game_id] = profile
                
            self.last_analysis_time[game_id] = datetime.now()
            
            self.logger.info(f"Analysis complete for {game_id}: {results['total_shaders']} shaders found")
            
        except Exception as e:
            self.logger.error(f"Error analyzing {task['game_id']}: {e}")
    
    def _compile_game_shaders(self, task: Dict):
        """Compile shaders for a game with adaptive strategy"""
        try:
            game_id = task['game_id']
            
            # Get adaptive compilation strategy
            strategy = self._adaptive_compilation_strategy()
            
            # Get shader priorities
            priorities = self.engine.predict_shader_priorities(game_id)
            if not priorities:
                self.logger.warning(f"No shader priorities found for {game_id}")
                return
            
            # Limit compilation based on strategy
            max_shaders = 10 if strategy['background_only'] else 25
            limited_priorities = priorities[:max_shaders]
            
            # Apply priority multiplier
            for priority in limited_priorities:
                priority['priority'] = int(priority['priority'] * strategy['priority_multiplier'])
            
            self.logger.info(f"Starting adaptive compilation for {game_id} "
                           f"({len(limited_priorities)} shaders, strategy: {strategy})")
            
            # Compile with progress callback and strategy
            def progress_callback(progress):
                self.logger.debug(f"Compiled {progress['shader']}: {progress['success']}")
                
                # Check if we should pause mid-compilation
                if self._should_pause_enhanced():
                    return False  # Signal to stop compilation
                return True
            
            # Apply compilation strategy to fossilize
            self.fossilize.steam_deck_optimizations.update({
                'max_parallel_compiles': strategy['max_parallel'],
                'memory_limit_mb': strategy['memory_limit_mb']
            })
            
            results = self.fossilize.compile_shaders(
                game_id, limited_priorities, progress_callback
            )
            
            self.logger.info(f"Adaptive compilation complete for {game_id}: "
                           f"{results['compiled']} compiled, {results['failed']} failed")
            
            # Check if we should force shader recompilation for known issues
            if results['failed'] > results['compiled'] * 0.5:  # > 50% failure rate
                self.logger.warning(f"High failure rate for {game_id}, forcing recompilation")
                self.radv_optimizer.force_shader_recompilation(
                    game_id, ["high_failure_rate", "compilation_errors"]
                )
            
            # Integrate with Steam
            cache_dir = Path.home() / '.cache/shader-predict-compile' / game_id
            self.fossilize.integrate_with_steam(game_id, cache_dir)
            
        except Exception as e:
            self.logger.error(f"Error compiling shaders for {task['game_id']}: {e}")
    
    def _process_compilation_queue(self):
        """Process the compilation queue"""
        if not self.compilation_queue:
            return
            
        self.logger.debug(f"Compilation queue size: {len(self.compilation_queue)}")
    
    def _cleanup_old_data(self):
        """Clean up old data and caches"""
        try:
            # Clean up old shader caches (older than 30 days)
            cleaned = self.fossilize.cleanup_old_caches(30)
            if cleaned > 0:
                self.logger.info(f"Cleaned up {cleaned} old shader caches")
                
            # Clean up old analysis data
            cutoff_time = datetime.now() - timedelta(days=7)
            
            to_remove = []
            for game_id, analysis_time in self.last_analysis_time.items():
                if analysis_time < cutoff_time:
                    to_remove.append(game_id)
                    
            for game_id in to_remove:
                del self.last_analysis_time[game_id]
                if game_id in self.engine.game_profiles:
                    del self.engine.game_profiles[game_id]
                    
            if to_remove:
                self.logger.info(f"Cleaned up analysis data for {len(to_remove)} games")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _initialize_enhanced_systems(self):
        """Initialize all enhanced systems for 2025 features"""
        try:
            # Apply RADV optimizations
            steam_deck_model = self.compat.model
            self.radv_optimizer.apply_global_radv_optimizations(steam_deck_model)
            
            # Apply known game fixes
            fixes_applied = self.radv_optimizer.apply_known_game_fixes()
            self.logger.info(f"Applied RADV fixes for {fixes_applied} problematic games")
            
            # Start real-time monitoring with gaming mode detection
            def monitoring_callback(metrics):
                # Adapt compilation based on system load
                if metrics.gpu_utilization > 80 or metrics.cpu_percent > 90:
                    self.paused = True
                    self.logger.debug("High system load detected - pausing compilation")
                elif self.paused and metrics.gpu_utilization < 50 and metrics.cpu_percent < 60:
                    self.paused = False
                    self.logger.debug("System load normalized - resuming compilation")
            
            self.monitor.add_callback(monitoring_callback)
            self.monitor.start_gaming_mode_monitoring()
            
            # Start adaptive power management
            def power_callback(profile, state):
                # Update shader compilation settings based on power profile
                shader_settings = self.power_manager.get_shader_compilation_settings()
                self.config.max_cpu_usage = min(self.config.max_cpu_usage, 
                                               shader_settings['max_threads'] * 25)
                self.logger.info(f"Power profile changed to {profile.value}")
            
            self.power_manager.start_adaptive_power_management(power_callback)
            
            # Initialize advanced cache management
            # Integrate with SteamOS cache system
            integration_results = self.cache_manager.integrate_with_steamos_cache()
            self.logger.info(f"SteamOS cache integration: {integration_results}")
            
            # Start automatic cache management
            self.cache_manager.start_automatic_management()
            
            # Optimize existing cache
            optimization_results = self.cache_manager.optimize_cache_structure()
            self.logger.info(f"Cache optimization: {optimization_results}")
            
            self.logger.info("Enhanced systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced systems: {e}")
    
    def _should_pause_enhanced(self) -> bool:
        """Enhanced pause check with thermal and battery awareness"""
        reasons = []
        
        # Original checks
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.config.max_cpu_usage:
            reasons.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        memory = psutil.virtual_memory()
        memory_mb = (memory.total - memory.available) / (1024 * 1024)
        if memory_mb > self.config.max_memory_mb:
            reasons.append(f"High memory usage: {memory_mb:.0f}MB")
        
        # Enhanced thermal and battery checks
        try:
            thermal_data = self.compat.get_enhanced_thermal_data()
            battery_data = self.compat.get_enhanced_battery_data()
            
            # Thermal checks
            if thermal_data['thermal_throttling_active']:
                reasons.append("Thermal throttling active")
            
            if thermal_data['cpu_temp'] > 85:
                reasons.append(f"High CPU temperature: {thermal_data['cpu_temp']:.1f}°C")
            
            if thermal_data['gpu_temp'] > 90:
                reasons.append(f"High GPU temperature: {thermal_data['gpu_temp']:.1f}°C")
            
            # Battery checks (handheld mode)
            if battery_data['battery_present'] and not battery_data['ac_connected']:
                if battery_data['capacity'] < 15:
                    reasons.append(f"Critical battery level: {battery_data['capacity']}%")
                elif (battery_data['capacity'] < 30 and 
                      battery_data['discharge_rate_w'] > 20):
                    reasons.append(f"High power draw on low battery: {battery_data['discharge_rate_w']:.1f}W")
                
                if battery_data['temperature_c'] > 45:
                    reasons.append(f"High battery temperature: {battery_data['temperature_c']:.1f}°C")
            
            # Gaming mode check with enhanced detection
            if self.monitor.gaming_mode_active:
                current_game = self.monitor.get_current_game_process()
                if current_game:
                    reasons.append("Game is actively running")
            
        except Exception as e:
            self.logger.debug(f"Enhanced pause check error: {e}")
        
        if reasons:
            self.logger.debug(f"Pausing due to: {', '.join(reasons)}")
            return True
        
        return False
    
    def _adaptive_compilation_strategy(self) -> Dict:
        """Determine optimal compilation strategy based on current conditions"""
        strategy = {
            'max_parallel': 4,
            'priority_multiplier': 1.0,
            'memory_limit_mb': 2048,
            'aggressive_caching': False,
            'background_only': False
        }
        
        try:
            # Get current power profile settings
            if hasattr(self.power_manager, 'current_profile'):
                shader_settings = self.power_manager.get_shader_compilation_settings()
                strategy.update({
                    'max_parallel': shader_settings['max_threads'],
                    'memory_limit_mb': shader_settings['memory_limit_mb'],
                    'aggressive_caching': shader_settings['aggressive_caching'],
                    'background_only': not shader_settings['compile_in_background']
                })
            
            # Adjust based on real-time monitoring
            current_metrics = self.monitor.get_current_metrics()
            if current_metrics:
                # Reduce intensity if system is under load
                if current_metrics.cpu_percent > 70:
                    strategy['max_parallel'] = max(1, strategy['max_parallel'] // 2)
                    strategy['priority_multiplier'] *= 0.5
                
                if current_metrics.memory_percent > 80:
                    strategy['memory_limit_mb'] = min(strategy['memory_limit_mb'], 1024)
                
                # Pause if gaming
                if current_metrics.shader_cache_activity > 10:  # High shader activity
                    strategy['background_only'] = True
            
            # Thermal adaptation
            thermal_data = self.compat.get_enhanced_thermal_data()
            if thermal_data['cpu_temp'] > 80 or thermal_data['gpu_temp'] > 85:
                strategy['max_parallel'] = max(1, strategy['max_parallel'] // 2)
                strategy['aggressive_caching'] = True  # Cache more to avoid recompilation
            
        except Exception as e:
            self.logger.debug(f"Strategy adaptation error: {e}")
        
        return strategy
    
    def _analyze_new_game(self, game_dir: Path) -> Dict:
        """Enhanced analysis of newly discovered games"""
        analysis = {
            'size_mb': 0,
            'priority': 1,
            'engine': 'Unknown',
            'has_shaders': False,
            'shader_count_estimate': 0
        }
        
        try:
            # Calculate size (limited scan for performance)
            total_size = 0
            file_count = 0
            shader_files = 0
            
            for file_path in game_dir.rglob('*'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_count += 1
                    
                    # Check for shader files
                    if file_path.suffix.lower() in ['.spv', '.dxbc', '.hlsl', '.glsl', '.cso']:
                        shader_files += 1
                        analysis['has_shaders'] = True
                    
                    # Limit scan for performance
                    if file_count > 2000:
                        break
            
            analysis['size_mb'] = total_size / (1024 * 1024)
            analysis['shader_count_estimate'] = shader_files
            
            # Determine game engine
            if (game_dir / 'UnrealEngine').exists() or list(game_dir.glob('*Unreal*')):
                analysis['engine'] = 'Unreal'
                analysis['priority'] = 8  # Unreal games often have many shaders
            elif (game_dir / 'Unity_Data').exists() or list(game_dir.glob('*Unity*')):
                analysis['engine'] = 'Unity'
                analysis['priority'] = 6  # Unity games have moderate shader complexity
            elif list(game_dir.glob('*Source2*')) or list(game_dir.glob('*source2*')):
                analysis['engine'] = 'Source2'
                analysis['priority'] = 7  # Source2 games benefit from shader compilation
            elif list(game_dir.glob('*.pak')):
                analysis['engine'] = 'Unreal/Generic'
                analysis['priority'] = 7
            elif shader_files > 10:
                analysis['engine'] = 'Custom/Shaders'
                analysis['priority'] = 9  # High priority for games with many shaders
            else:
                # Base priority on size for unknown engines
                analysis['priority'] = min(int(analysis['size_mb'] / 1000), 5)
            
            # Boost priority for games with detected shaders
            if analysis['has_shaders']:
                analysis['priority'] = min(analysis['priority'] + 2, 10)
                
        except Exception as e:
            self.logger.debug(f"Error analyzing game {game_dir.name}: {e}")
            
        return analysis
    
    def get_all_detected_games(self) -> List[Dict]:
        """Get list of all detected games across all Steam libraries"""
        try:
            return self.compat.get_all_steam_games()
        except Exception as e:
            self.logger.error(f"Error getting all detected games: {e}")
            return []
    
    def _verify_removable_library_access(self, library: Dict):
        """Verify that a removable library (SD card, USB) is still accessible"""
        try:
            library_path = Path(library['path'])
            if not library_path.exists():
                self.logger.warning(f"Removable library no longer accessible: {library['path']}")
                # Remove games from this library from compilation queue
                self.compilation_queue = [
                    task for task in self.compilation_queue 
                    if not str(task.get('game_path', '')).startswith(str(library_path))
                ]
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Error verifying removable library access: {e}")
            return False
    
    def get_service_status(self) -> Dict:
        """Get comprehensive service status including detected games"""
        try:
            detected_libraries = self.compat.auto_detect_steam_libraries()
            total_games = sum(lib['games_count'] for lib in detected_libraries)
            
            return {
                'running': self.running,
                'paused': self.paused,
                'steam_libraries_detected': len(detected_libraries),
                'total_games_detected': total_games,
                'games_in_queue': len(self.compilation_queue),
                'current_compilation': self.current_compilation['game_id'] if self.current_compilation else None,
                'gaming_mode_active': self.compat.detect_gaming_mode(),
                'libraries': detected_libraries
            }
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {
                'running': self.running,
                'paused': self.paused,
                'error': str(e)
            }

def main():
    """Main entry point for the background service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shader Predictive Compilation Background Service')
    parser.add_argument('--config', type=Path, help='Path to configuration file')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Create and start service
    service = BackgroundService(args.config)
    
    if args.daemon:
        # Daemonize the process
        if os.fork() == 0:
            os.setsid()
            if os.fork() == 0:
                service.start()
        else:
            sys.exit(0)
    else:
        service.start()

if __name__ == '__main__':
    main()