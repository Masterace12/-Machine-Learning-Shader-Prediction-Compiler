#!/usr/bin/env python3
"""
Steam Deck Optimized Thread Pool Manager
Comprehensive thread management system designed for Steam Deck's resource constraints
"""

import os
import sys
import time
import threading
import weakref
import logging
import psutil
from typing import Dict, List, Optional, Callable, Any, Set, NamedTuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import defaultdict, deque
import signal
import gc

class ThreadPriority(Enum):
    """Thread priority levels for Steam Deck optimization"""
    CRITICAL = 0     # System critical threads (max 1)
    HIGH = 1         # ML prediction threads (max 2)
    NORMAL = 2       # Background compilation (max 2)
    LOW = 3          # Cleanup and monitoring (max 2)
    IDLE = 4         # Non-essential tasks (max 1)

class ResourceState(Enum):
    """System resource state for adaptive thread management"""
    OPTIMAL = "optimal"     # Cool temps, good battery, low load
    NORMAL = "normal"       # Normal operating conditions
    CONSTRAINED = "constrained"  # High temps or low battery
    CRITICAL = "critical"   # Thermal throttling or very low battery

@dataclass
class ThreadMetrics:
    """Thread performance and resource metrics"""
    thread_id: int
    name: str
    priority: ThreadPriority
    cpu_time: float
    memory_mb: float
    created_at: float
    last_active: float
    task_count: int
    avg_task_duration: float
    is_daemon: bool

class ThreadPoolConfig:
    """Thread pool configuration optimized for Steam Deck"""
    
    def __init__(self):
        # Steam Deck OLED specific limits (ultra-conservative for gaming)
        self.max_total_threads = 4  # Ultra-conservative limit for Steam Deck OLED
        self.max_ml_threads = 1     # Single ML inference thread
        self.max_compilation_threads = 1  # Single background shader compilation thread
        self.max_monitoring_threads = 1   # Single thermal/performance monitoring
        self.max_cleanup_threads = 1      # Single cleanup thread
        
        # Thread pool sizes by priority (reduced for Steam Deck OLED)
        self.priority_limits = {
            ThreadPriority.CRITICAL: 1,
            ThreadPriority.HIGH: 1,     # ML prediction - single thread only
            ThreadPriority.NORMAL: 1,   # Background compilation - single thread only
            ThreadPriority.LOW: 1,      # Monitoring - single thread only
            ThreadPriority.IDLE: 0      # No idle threads on Steam Deck
        }
        
        # Timeout settings
        self.thread_timeout = 30.0      # Max thread lifetime
        self.task_timeout = 10.0        # Max task execution time
        self.cleanup_interval = 60.0    # Cleanup interval
        
        # Resource thresholds
        self.memory_limit_mb = 150      # Per-thread memory limit
        self.cpu_threshold = 80.0       # CPU usage threshold
        self.temp_threshold = 85.0      # Temperature threshold
        
        # Adaptive scaling
        self.enable_adaptive_scaling = True
        self.scaling_check_interval = 5.0

class SteamDeckThreadManager:
    """
    Steam Deck optimized thread manager
    
    Features:
    - Hardware-aware thread limits
    - Thermal-responsive scaling
    - Memory pressure handling
    - Gaming-aware prioritization
    - Comprehensive cleanup
    """
    
    def __init__(self, config: Optional[ThreadPoolConfig] = None):
        self.config = config or ThreadPoolConfig()
        self.logger = logging.getLogger(__name__)
        
        # Thread tracking
        self._thread_pools: Dict[ThreadPriority, ThreadPoolExecutor] = {}
        self._active_threads: Dict[int, ThreadMetrics] = {}
        self._thread_registry: Set[threading.Thread] = weakref.WeakSet()
        self._resource_state = ResourceState.NORMAL
        
        # Synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Monitoring
        self._monitoring_thread = None
        self._cleanup_thread = None
        self._last_cleanup = time.time()
        
        # Performance tracking
        self._task_history = deque(maxlen=1000)
        self._resource_history = deque(maxlen=100)
        
        # Gaming state detection
        self._gaming_mode_active = False
        self._last_gaming_check = 0
        
        # Initialize thread pools
        self._initialize_thread_pools()
        
        # Start monitoring
        self._start_monitoring()
        
        # Register signal handlers for cleanup
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info(f"Thread manager initialized with {self.config.max_total_threads} thread limit")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down thread manager...")
        self.shutdown()
    
    def _initialize_thread_pools(self):
        """Initialize thread pools for each priority level"""
        with self._lock:
            for priority in ThreadPriority:
                max_workers = self.config.priority_limits[priority]
                if max_workers > 0:
                    self._thread_pools[priority] = ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix=f"steamdeck_{priority.name.lower()}"
                    )
                    self.logger.debug(f"Created {priority.name} thread pool with {max_workers} workers")
    
    def _start_monitoring(self):
        """Start background monitoring threads"""
        if self.config.enable_adaptive_scaling:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="thread_monitor",
                daemon=True
            )
            self._monitoring_thread.start()
            
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="thread_cleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _monitoring_loop(self):
        """Monitor system resources and adapt thread limits"""
        while not self._shutdown_event.is_set():
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory = psutil.virtual_memory()
                
                # Check thermal state (if available)
                temp = self._get_cpu_temperature()
                
                # Determine resource state
                new_state = self._determine_resource_state(cpu_percent, memory.percent, temp)
                
                if new_state != self._resource_state:
                    self.logger.info(f"Resource state changed: {self._resource_state.value} -> {new_state.value}")
                    self._resource_state = new_state
                    self._adapt_thread_limits()
                
                # Record metrics
                self._resource_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'temperature': temp,
                    'resource_state': new_state.value,
                    'active_threads': len(self._active_threads)
                })
                
                # Check gaming mode
                self._check_gaming_mode()
                
                time.sleep(self.config.scaling_check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.scaling_check_interval * 2)
    
    def _cleanup_loop(self):
        """Periodic cleanup of stale threads and resources"""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Check for stale threads
                self._cleanup_stale_threads()
                
                # Force garbage collection if memory pressure is high
                if self._resource_state in (ResourceState.CONSTRAINED, ResourceState.CRITICAL):
                    gc.collect()
                
                self._last_cleanup = current_time
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(self.config.cleanup_interval * 2)
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature for thermal management"""
        try:
            # Try common thermal sensor paths
            sensor_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/hwmon/hwmon0/temp1_input",
            ]
            
            for path in sensor_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp_raw = int(f.read().strip())
                        return temp_raw / 1000.0 if temp_raw > 1000 else temp_raw
        except Exception:
            pass
        
        return 70.0  # Default safe temperature
    
    def _determine_resource_state(self, cpu_percent: float, memory_percent: float, temp: float) -> ResourceState:
        """Determine current resource state"""
        if temp > 90 or cpu_percent > 95 or memory_percent > 90:
            return ResourceState.CRITICAL
        elif temp > self.config.temp_threshold or cpu_percent > self.config.cpu_threshold or memory_percent > 85:
            return ResourceState.CONSTRAINED
        elif temp < 60 and cpu_percent < 50 and memory_percent < 70:
            return ResourceState.OPTIMAL
        else:
            return ResourceState.NORMAL
    
    def _check_gaming_mode(self):
        """Check if gaming mode is active with enhanced Steam Deck detection"""
        current_time = time.time()
        if current_time - self._last_gaming_check < 5.0:  # Check more frequently
            return
        
        self._last_gaming_check = current_time
        
        try:
            gaming_indicators = 0
            high_cpu_processes = []
            
            # Enhanced gaming detection for Steam Deck
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    name = proc.info['name'].lower()
                    cpu_percent = proc.info['cpu_percent'] or 0
                    memory_percent = proc.info['memory_percent'] or 0
                    
                    # Skip system and ML processes
                    skip_processes = [
                        'python3', 'python', 'steam', 'steamwebhelper', 'systemd',
                        'kworker', 'lightgbm', 'sklearn', 'jupyter', 'chrome', 'firefox'
                    ]
                    
                    if any(skip in name for skip in skip_processes):
                        continue
                    
                    # Check for gaming indicators
                    if cpu_percent > 20:  # Lower threshold for Steam Deck
                        high_cpu_processes.append((name, cpu_percent))
                        
                        # Common gaming process patterns
                        gaming_patterns = [
                            'game', 'unity', 'unreal', 'ue4', 'ue5', 'godot',
                            'steamapps', 'proton', 'wine', 'lutris',
                            '.exe', 'vrmonitor', 'openvr', 'steamvr',
                            'csgo', 'dota', 'tf2', 'portal', 'halflife'
                        ]
                        
                        if any(pattern in name for pattern in gaming_patterns):
                            gaming_indicators += 2  # Strong indicator
                        elif cpu_percent > 40:
                            gaming_indicators += 1  # Moderate indicator
                    
                    # High memory usage could indicate gaming
                    if memory_percent > 10:  # 1.6GB+ on 16GB system
                        gaming_indicators += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Determine gaming state
            previous_state = self._gaming_mode_active
            
            if gaming_indicators >= 2:
                self._gaming_mode_active = True
            elif gaming_indicators == 0:
                self._gaming_mode_active = False
            # Keep previous state if indicators == 1 (ambiguous)
            
            # Log gaming mode changes
            if previous_state != self._gaming_mode_active:
                if self._gaming_mode_active:
                    self.logger.info(f"Gaming mode activated (indicators: {gaming_indicators}, "
                                   f"high CPU processes: {high_cpu_processes[:3]})")
                else:
                    self.logger.info("Gaming mode deactivated - returning to normal thread limits")
                    
        except Exception as e:
            self.logger.warning(f"Gaming mode detection error: {e}")
            # Default to non-gaming mode on error
            self._gaming_mode_active = False
    
    def _adapt_thread_limits(self):
        """Adapt thread limits based on resource state"""
        with self._lock:
            if self._resource_state == ResourceState.CRITICAL:
                # Severely limit threads
                new_limits = {
                    ThreadPriority.CRITICAL: 1,
                    ThreadPriority.HIGH: 1,
                    ThreadPriority.NORMAL: 0,  # Pause compilation
                    ThreadPriority.LOW: 1,
                    ThreadPriority.IDLE: 0
                }
            elif self._resource_state == ResourceState.CONSTRAINED:
                # Severely reduce threads for Steam Deck constraints
                new_limits = {
                    ThreadPriority.CRITICAL: 1,
                    ThreadPriority.HIGH: 1,
                    ThreadPriority.NORMAL: 0,  # Pause compilation under constraint
                    ThreadPriority.LOW: 0,     # Pause monitoring under constraint
                    ThreadPriority.IDLE: 0
                }
            elif self._gaming_mode_active:
                # Gaming mode - absolute minimal background work
                new_limits = {
                    ThreadPriority.CRITICAL: 1,
                    ThreadPriority.HIGH: 0,    # No ML prediction during gaming
                    ThreadPriority.NORMAL: 0,  # No compilation during gaming
                    ThreadPriority.LOW: 0,     # No monitoring during gaming
                    ThreadPriority.IDLE: 0
                }
            else:
                # Normal or optimal state
                new_limits = self.config.priority_limits.copy()
            
            # Apply new limits by recreating thread pools if needed
            for priority, new_limit in new_limits.items():
                current_limit = self._thread_pools[priority]._max_workers if priority in self._thread_pools else 0
                if new_limit != current_limit:
                    self._resize_thread_pool(priority, new_limit)
    
    def _resize_thread_pool(self, priority: ThreadPriority, new_size: int):
        """Resize a thread pool safely"""
        if new_size == 0:
            # Shutdown pool
            if priority in self._thread_pools:
                self._thread_pools[priority].shutdown(wait=False)
                del self._thread_pools[priority]
                self.logger.info(f"Shutdown {priority.name} thread pool")
        else:
            # Create or resize pool
            if priority in self._thread_pools:
                # Can't resize ThreadPoolExecutor, so shutdown and recreate
                self._thread_pools[priority].shutdown(wait=False)
            
            self._thread_pools[priority] = ThreadPoolExecutor(
                max_workers=new_size,
                thread_name_prefix=f"steamdeck_{priority.name.lower()}"
            )
            self.logger.info(f"Resized {priority.name} thread pool to {new_size} workers")
    
    def _cleanup_completed_tasks(self):
        """Clean up completed tasks and stale metrics"""
        current_time = time.time()
        stale_threshold = current_time - self.config.thread_timeout
        
        with self._lock:
            stale_threads = [
                tid for tid, metrics in self._active_threads.items()
                if metrics.last_active < stale_threshold
            ]
            
            for tid in stale_threads:
                del self._active_threads[tid]
    
    def _cleanup_stale_threads(self):
        """Clean up stale thread references"""
        # WeakSet automatically removes dead threads
        pass
    
    @contextmanager
    def thread_context(self, priority: ThreadPriority, name: str = "unnamed"):
        """Context manager for thread resource tracking"""
        thread_id = threading.get_ident()
        start_time = time.time()
        
        # Register thread
        with self._lock:
            self._active_threads[thread_id] = ThreadMetrics(
                thread_id=thread_id,
                name=name,
                priority=priority,
                cpu_time=0.0,
                memory_mb=0.0,
                created_at=start_time,
                last_active=start_time,
                task_count=0,
                avg_task_duration=0.0,
                is_daemon=threading.current_thread().daemon
            )
        
        try:
            yield
        finally:
            # Update metrics and cleanup
            with self._lock:
                if thread_id in self._active_threads:
                    metrics = self._active_threads[thread_id]
                    metrics.last_active = time.time()
                    # Keep recent threads for metrics
                    if time.time() - metrics.created_at > self.config.thread_timeout:
                        del self._active_threads[thread_id]
    
    def submit_task(self, priority: ThreadPriority, func: Callable, *args, task_name: str = "unnamed", **kwargs):
        """Submit a task to the appropriate thread pool"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Thread manager is shutting down")
        
        if priority not in self._thread_pools:
            raise ValueError(f"No thread pool available for priority {priority.name}")
        
        task_start = time.time()
        
        def wrapped_task():
            with self.thread_context(priority, task_name):
                try:
                    result = func(*args, **kwargs)
                    
                    # Record task metrics
                    duration = time.time() - task_start
                    self._task_history.append({
                        'priority': priority.name,
                        'task_name': task_name,
                        'duration': duration,
                        'success': True,
                        'timestamp': task_start
                    })
                    
                    return result
                    
                except Exception as e:
                    # Record failure
                    duration = time.time() - task_start
                    self._task_history.append({
                        'priority': priority.name,
                        'task_name': task_name,
                        'duration': duration,
                        'success': False,
                        'error': str(e),
                        'timestamp': task_start
                    })
                    raise
        
        return self._thread_pools[priority].submit(wrapped_task)
    
    def submit_ml_task(self, func: Callable, *args, **kwargs):
        """Convenience method for ML prediction tasks"""
        return self.submit_task(ThreadPriority.HIGH, func, *args, task_name="ml_prediction", **kwargs)
    
    def submit_compilation_task(self, func: Callable, *args, **kwargs):
        """Convenience method for shader compilation tasks"""
        return self.submit_task(ThreadPriority.NORMAL, func, *args, task_name="shader_compilation", **kwargs)
    
    def submit_monitoring_task(self, func: Callable, *args, **kwargs):
        """Convenience method for monitoring tasks"""
        return self.submit_task(ThreadPriority.LOW, func, *args, task_name="monitoring", **kwargs)
    
    def get_thread_metrics(self) -> Dict[str, Any]:
        """Get comprehensive thread metrics"""
        with self._lock:
            total_threads = sum(len(pool._threads) for pool in self._thread_pools.values())
            
            pool_stats = {}
            for priority, pool in self._thread_pools.items():
                pool_stats[priority.name] = {
                    'max_workers': pool._max_workers,
                    'active_threads': len(pool._threads),
                    'queue_size': pool._work_queue.qsize(),
                }
            
            # Task performance metrics
            recent_tasks = [t for t in self._task_history if time.time() - t['timestamp'] < 300]  # Last 5 minutes
            
            success_rate = (sum(1 for t in recent_tasks if t['success']) / max(1, len(recent_tasks))) * 100
            avg_duration = sum(t['duration'] for t in recent_tasks) / max(1, len(recent_tasks))
            
            return {
                'total_active_threads': total_threads,
                'max_total_threads': self.config.max_total_threads,
                'resource_state': self._resource_state.value,
                'gaming_mode_active': self._gaming_mode_active,
                'pool_statistics': pool_stats,
                'task_metrics': {
                    'recent_task_count': len(recent_tasks),
                    'success_rate_percent': success_rate,
                    'average_duration_ms': avg_duration * 1000,
                    'total_tasks_completed': len(self._task_history)
                },
                'system_metrics': {
                    'last_resource_check': self._resource_history[-1] if self._resource_history else None
                }
            }
    
    def configure_ml_libraries(self):
        """Configure ML libraries for optimal threading on Steam Deck"""
        # Set environment variables for thread limits (critical - must be set before imports)
        # Ultra-conservative threading for Steam Deck OLED to prevent "can't start new thread" errors
        steam_deck_env = {
            'OMP_NUM_THREADS': '1',      # Single OpenMP thread
            'MKL_NUM_THREADS': '1',      # Single Intel MKL thread
            'OPENBLAS_NUM_THREADS': '1', # Single OpenBLAS thread
            'NUMEXPR_NUM_THREADS': '1',  # Single NumExpr thread
            'LIGHTGBM_NUM_THREADS': '1', # Single LightGBM thread
            'OMP_DYNAMIC': 'FALSE',      # Disable dynamic thread adjustment
            'OMP_WAIT_POLICY': 'PASSIVE',# Reduce CPU spinning
            'OMP_NESTED': 'FALSE',       # Disable nested parallelism
            'OMP_THREAD_LIMIT': '4',     # Hard limit on total OpenMP threads
            'MALLOC_ARENA_MAX': '1',     # Minimize memory fragmentation
            'NUMBA_NUM_THREADS': '1',    # Single Numba thread
            'VECLIB_MAXIMUM_THREADS': '1', # Single BLAS thread (macOS/general)
            'BLIS_NUM_THREADS': '1'      # Single BLIS thread
        }
        
        for var, value in steam_deck_env.items():
            os.environ[var] = value
        
        # Configure LightGBM with proper API for 4.6.0+
        try:
            import lightgbm as lgb
            
            # LightGBM 4.6.0+ ONLY supports environment variable threading control
            # All set_option() and set_number_threads() methods have been removed
            
            # Verify LightGBM version and warn about compatibility
            try:
                version_info = lgb.__version__
                major, minor, patch = map(int, version_info.split('.')[:3])
                
                if major >= 4 and minor >= 6:
                    self.logger.info(f"LightGBM {version_info} detected - modern version using environment variables")
                elif major >= 3:
                    self.logger.info(f"LightGBM {version_info} detected - using environment variable threading")
                else:
                    self.logger.warning(f"LightGBM {version_info} is very old - may have compatibility issues")
                    
            except (AttributeError, ValueError) as version_error:
                self.logger.warning(f"Could not determine LightGBM version: {version_error}")
            
            # Environment variables are already set above, just verify
            lgb_threads = os.environ.get('LIGHTGBM_NUM_THREADS', 'not set')
            if lgb_threads != 'not set':
                self.logger.info(f"LightGBM threading configured via LIGHTGBM_NUM_THREADS={lgb_threads}")
            else:
                self.logger.warning("LIGHTGBM_NUM_THREADS not set - using LightGBM defaults")
            
            # Store Steam Deck optimized parameters for model creation
            if not hasattr(lgb, '_steamdeck_optimized_params'):
                lgb._steamdeck_optimized_params = {
                    'num_threads': 2,
                    'force_row_wise': True,
                    'device_type': 'cpu',
                    'max_bin': 255,
                    'histogram_pool_size': 128,
                    'verbose': -1,
                    'min_data_in_leaf': 10,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'lambda_l1': 0.1,
                    'lambda_l2': 0.1,
                    'min_split_gain': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'subsample': 0.8,
                    'subsample_freq': 1,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 5,
                    'min_child_weight': 0.001,
                    'subsample_for_bin': 200000,
                    'enable_sparse': True,
                    'is_unbalance': False,
                    'boost_from_average': True,
                    'num_iterations': 100,
                    'early_stopping_round': 10,
                    'first_metric_only': True,
                    'seed': 42,
                    'deterministic': True,
                }
                self.logger.info("Stored Steam Deck optimized LightGBM parameters")
                
        except ImportError:
            self.logger.debug("LightGBM not available - skipping configuration")
        except Exception as e:
            self.logger.error(f"LightGBM configuration failed: {e}")
            self.logger.debug("Continuing with default LightGBM settings")
        
        # Configure scikit-learn for Steam Deck
        try:
            from sklearn import set_config
            # Check sklearn version for correct parameter names
            try:
                set_config(
                    assume_finite=True,      # Performance optimization
                    working_memory=64,       # Limit memory usage (MB)
                    enable_cython_pairwise_dist=True  # Use optimized pairwise distances
                )
            except TypeError:
                # Fallback for older sklearn versions
                set_config(
                    assume_finite=True,      # Performance optimization
                    working_memory=64        # Limit memory usage (MB)
                )
            self.logger.info("Configured scikit-learn for Steam Deck performance")
        except ImportError:
            pass
        
        # Configure Numba for Steam Deck OLED (single-threaded)
        try:
            import numba
            if hasattr(numba, 'set_num_threads'):
                try:
                    numba.set_num_threads(1)  # Single thread for Steam Deck OLED
                except ValueError as e:
                    # If Numba is built without threading support, skip
                    self.logger.warning(f"Numba threading limitation: {e}")
            
            # Additional Numba optimizations for Steam Deck
            if hasattr(numba, 'config'):
                numba.config.THREADING_LAYER = 'workqueue'  # Better for single thread
                numba.config.NUMBA_NUM_THREADS = 1
            
            self.logger.info("Configured Numba for 1 thread (Steam Deck OLED)")
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Numba not available or configured: {e}")
        except Exception as e:
            self.logger.warning(f"Numba configuration failed: {e}")
        
        # Configure NumPy BLAS threading 
        try:
            import numpy as np
            if hasattr(np, 'show_config'):
                # Log BLAS configuration for debugging
                config_info = np.show_config()
                self.logger.debug("NumPy BLAS configuration verified")
        except ImportError:
            pass
        
        # Configure PyTorch if available (some dependencies might use it)
        try:
            import torch
            torch.set_num_threads(1)  # Single thread for Steam Deck OLED
            torch.set_num_interop_threads(1)
            self.logger.info("Configured PyTorch for single-threaded operation")
        except ImportError:
            self.logger.debug("PyTorch not available")
        except Exception as e:
            self.logger.warning(f"PyTorch configuration failed: {e}")
        
        self.logger.info("ML library threading configured for Steam Deck APU")
    
    def shutdown(self, wait: bool = True, timeout: float = 10.0):
        """Shutdown thread manager gracefully"""
        self.logger.info("Shutting down thread manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Shutdown all thread pools
        with self._lock:
            for priority, pool in self._thread_pools.items():
                try:
                    pool.shutdown(wait=wait)
                    self.logger.debug(f"Shutdown {priority.name} thread pool")
                except Exception as e:
                    self.logger.error(f"Error shutting down {priority.name} pool: {e}")
        
        # Wait for monitoring threads
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=timeout/2)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=timeout/2)
        
        self.logger.info("Thread manager shutdown complete")


# Global thread manager instance
_global_thread_manager = None


def get_thread_manager() -> SteamDeckThreadManager:
    """Get or create global thread manager instance"""
    global _global_thread_manager
    if _global_thread_manager is None:
        _global_thread_manager = SteamDeckThreadManager()
        # Configure ML libraries immediately
        _global_thread_manager.configure_ml_libraries()
    return _global_thread_manager


def configure_threading_environment():
    """Configure system-wide threading environment for Steam Deck"""
    # Get thread manager (also configures ML libraries)
    thread_manager = get_thread_manager()
    
    # Set Python thread limits
    threading.stack_size(2**18)  # 256KB stack size (smaller for more threads)
    
    # Configure garbage collection for lower memory pressure
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    return thread_manager


if __name__ == "__main__":
    # Test thread manager
    logging.basicConfig(level=logging.INFO)
    
    print("🧵 Steam Deck Thread Manager Test")
    print("=" * 40)
    
    # Initialize thread manager
    manager = get_thread_manager()
    
    try:
        # Test task submission
        def test_task(name, duration):
            time.sleep(duration)
            return f"Task {name} completed"
        
        # Submit various priority tasks
        futures = []
        futures.append(manager.submit_ml_task(test_task, "ML_1", 0.1))
        futures.append(manager.submit_compilation_task(test_task, "COMPILE_1", 0.2))
        futures.append(manager.submit_monitoring_task(test_task, "MONITOR_1", 0.1))
        
        # Wait for completion
        for future in as_completed(futures, timeout=5.0):
            result = future.result()
            print(f"✓ {result}")
        
        # Show metrics
        metrics = manager.get_thread_metrics()
        print(f"\nThread Metrics:")
        print(f"  Active threads: {metrics['total_active_threads']}/{metrics['max_total_threads']}")
        print(f"  Resource state: {metrics['resource_state']}")
        print(f"  Gaming mode: {metrics['gaming_mode_active']}")
        print(f"  Task success rate: {metrics['task_metrics']['success_rate_percent']:.1f}%")
        
        print("\n✅ Thread manager test completed successfully")
        
    except Exception as e:
        print(f"❌ Thread manager test failed: {e}")
    
    finally:
        manager.shutdown()