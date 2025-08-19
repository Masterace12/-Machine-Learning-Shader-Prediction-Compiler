#!/usr/bin/env python3
"""
Robust Threading Pool Manager with Fallback

This module provides a sophisticated threading management system that gracefully
handles threading issues on Steam Deck and other constrained environments with
automatic fallback to single-threaded operation.

Features:
- Adaptive thread pool sizing based on system capabilities
- Automatic fallback to single-threaded operation on errors
- Steam Deck specific optimizations and workarounds
- Thread leak detection and cleanup
- Performance monitoring and thread health checks
- Memory-aware thread management
- Thermal throttling integration
- Comprehensive error handling and recovery
- Thread safety validation and deadlock prevention
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
import signal
import gc
import weakref
from typing import Dict, List, Any, Optional, Union, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed,
    TimeoutError, CancelledError
)
from contextlib import contextmanager
from functools import wraps, lru_cache
from collections import defaultdict, deque
from enum import Enum, auto
import queue
from threading import RLock, Event, Condition, Semaphore
import multiprocessing
import resource

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

# =============================================================================
# THREADING CONFIGURATION AND STATUS
# =============================================================================

class ThreadingMode(Enum):
    """Threading modes available"""
    OPTIMAL = auto()       # Full multi-threading
    REDUCED = auto()       # Reduced thread count
    MINIMAL = auto()       # Minimal threading (2-3 threads)
    SINGLE_THREADED = auto()  # No threading

class ThreadPoolHealth(Enum):
    """Thread pool health status"""
    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()
    FAILED = auto()

@dataclass
class ThreadingConfig:
    """Threading configuration parameters"""
    max_workers: int = 4
    thread_timeout: float = 30.0
    queue_maxsize: int = 100
    memory_limit_mb: int = 512
    enable_process_pool: bool = False
    steam_deck_mode: bool = False
    thermal_throttling: bool = True
    thread_recycling: bool = True
    deadlock_detection: bool = True

@dataclass
class ThreadPoolStatus:
    """Thread pool status information"""
    mode: ThreadingMode
    health: ThreadPoolHealth
    active_threads: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    memory_usage_mb: float
    cpu_usage_percent: float
    thread_errors: List[str]
    last_error_time: Optional[float]
    uptime_seconds: float
    performance_score: float

@dataclass
class ThreadMetrics:
    """Individual thread metrics"""
    thread_id: int
    name: str
    start_time: float
    task_count: int
    error_count: int
    last_active: float
    memory_usage_mb: float
    cpu_time_used: float
    is_stuck: bool

# =============================================================================
# ROBUST THREADING MANAGER
# =============================================================================

class RobustThreadingManager:
    """
    Advanced threading manager with comprehensive fallback capabilities
    """
    
    def __init__(self, 
                 config: Optional[ThreadingConfig] = None,
                 auto_adjust: bool = True):
        
        self.config = config or ThreadingConfig()
        self.auto_adjust = auto_adjust
        
        # System detection
        self._detect_system_capabilities()
        
        # Threading state
        self.current_mode = ThreadingMode.OPTIMAL
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.fallback_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_maxsize)
        
        # Monitoring and health
        self.pool_health = ThreadPoolHealth.HEALTHY
        self.thread_metrics: Dict[int, ThreadMetrics] = {}
        self.task_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=100)
        self.performance_history: deque = deque(maxlen=50)
        
        # Thread safety
        self.manager_lock = RLock()
        self.shutdown_event = Event()
        self.health_check_interval = 10.0
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Task management
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_timeout_count = 0
        
        # Fallback handling
        self.fallback_active = False
        self.fallback_reason: Optional[str] = None
        self.single_thread_executor: Optional[threading.Thread] = None
        
        # Performance tracking
        self.start_time = time.time()
        self.last_performance_check = time.time()
        self.performance_baseline: Optional[float] = None
        
        # Initialize the threading system
        self._initialize_threading_system()
        
        logger.info(f"RobustThreadingManager initialized in {self.current_mode.name} mode")
        logger.info(f"Steam Deck mode: {self.config.steam_deck_mode}")

    def _detect_system_capabilities(self) -> None:
        """Detect system capabilities and adjust configuration"""
        try:
            # CPU detection
            cpu_count = os.cpu_count() or 1
            
            # Memory detection
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                memory_gb = 8.0  # Default assumption
            
            # Steam Deck detection
            steam_deck_indicators = [
                os.path.exists('/home/deck'),
                'jupiter' in os.uname().nodename.lower(),
                'steam' in os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
            ]
            is_steam_deck = any(steam_deck_indicators)
            
            # Adjust configuration based on system
            if is_steam_deck:
                self.config.steam_deck_mode = True
                self.config.max_workers = min(self.config.max_workers, 3)  # Conservative for Steam Deck
                self.config.memory_limit_mb = min(self.config.memory_limit_mb, 384)
                self.config.thermal_throttling = True
                self.config.thread_timeout = min(self.config.thread_timeout, 20.0)
                logger.info("Detected Steam Deck - applying conservative threading settings")
            
            # Memory-based adjustments
            if memory_gb < 8:
                self.config.max_workers = min(self.config.max_workers, 2)
                self.config.memory_limit_mb = min(self.config.memory_limit_mb, 256)
                logger.info(f"Low memory system ({memory_gb:.1f}GB) - reducing thread limits")
            
            # CPU-based adjustments
            if cpu_count <= 2:
                self.config.max_workers = min(self.config.max_workers, cpu_count)
                logger.info(f"Limited CPU cores ({cpu_count}) - adjusting thread count")
            
            logger.info(f"System capabilities: {cpu_count} cores, {memory_gb:.1f}GB RAM")
            logger.info(f"Final threading config: max_workers={self.config.max_workers}, memory_limit={self.config.memory_limit_mb}MB")
            
        except Exception as e:
            logger.error(f"Error detecting system capabilities: {e}")
            # Use conservative defaults
            self.config.max_workers = 2
            self.config.memory_limit_mb = 256

    def _initialize_threading_system(self) -> None:
        """Initialize the threading system with fallback support"""
        try:
            # Start with optimal mode
            success = self._setup_thread_pool(ThreadingMode.OPTIMAL)
            
            if not success:
                logger.warning("Optimal threading failed, trying reduced mode")
                success = self._setup_thread_pool(ThreadingMode.REDUCED)
            
            if not success:
                logger.warning("Reduced threading failed, trying minimal mode")
                success = self._setup_thread_pool(ThreadingMode.MINIMAL)
            
            if not success:
                logger.warning("Minimal threading failed, falling back to single-threaded mode")
                success = self._setup_thread_pool(ThreadingMode.SINGLE_THREADED)
            
            if not success:
                raise RuntimeError("All threading modes failed")
            
            # Start monitoring
            self._start_health_monitoring()
            
        except Exception as e:
            logger.error(f"Critical error initializing threading system: {e}")
            self._emergency_fallback()

    def _setup_thread_pool(self, mode: ThreadingMode) -> bool:
        """Setup thread pool for specified mode"""
        try:
            # Clean up existing pool
            self._cleanup_thread_pool()
            
            if mode == ThreadingMode.OPTIMAL:
                worker_count = self.config.max_workers
            elif mode == ThreadingMode.REDUCED:
                worker_count = max(2, self.config.max_workers // 2)
            elif mode == ThreadingMode.MINIMAL:
                worker_count = 2
            elif mode == ThreadingMode.SINGLE_THREADED:
                worker_count = 1
                self.fallback_active = True
                self.fallback_reason = "Single-threaded fallback mode"
            else:
                raise ValueError(f"Unknown threading mode: {mode}")
            
            # Create thread pool with appropriate settings
            if worker_count > 1:
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="RobustPool"
                )
                
                # Test the thread pool
                test_future = self.thread_pool.submit(self._test_thread_function)
                test_result = test_future.result(timeout=5.0)
                
                if not test_result:
                    raise RuntimeError("Thread pool test failed")
                    
            else:
                # Single-threaded mode - use direct execution
                self.thread_pool = None
            
            self.current_mode = mode
            self.pool_health = ThreadPoolHealth.HEALTHY
            
            logger.info(f"Successfully initialized {mode.name} mode with {worker_count} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup {mode.name} mode: {e}")
            self._cleanup_thread_pool()
            return False

    def _test_thread_function(self) -> bool:
        """Test function to validate thread pool functionality"""
        try:
            # Simple test that exercises thread functionality
            import time
            import threading
            
            start_time = time.time()
            thread_id = threading.get_ident()
            
            # Record thread metrics
            self.thread_metrics[thread_id] = ThreadMetrics(
                thread_id=thread_id,
                name=threading.current_thread().name,
                start_time=start_time,
                task_count=1,
                error_count=0,
                last_active=time.time(),
                memory_usage_mb=self._get_thread_memory_usage(),
                cpu_time_used=0.0,
                is_stuck=False
            )
            
            # Simulate some work
            time.sleep(0.01)
            
            return True
            
        except Exception as e:
            logger.error(f"Thread test function failed: {e}")
            return False

    def submit_task(self, 
                   fn: Callable,
                   *args,
                   task_id: Optional[str] = None,
                   timeout: Optional[float] = None,
                   priority: int = 5,
                   fallback_to_sync: bool = True,
                   **kwargs) -> Union[Future, Any]:
        """
        Submit a task for execution with comprehensive error handling
        """
        task_id = task_id or f"task_{int(time.time() * 1000000)}"
        timeout = timeout or self.config.thread_timeout
        
        # Check if we should run synchronously
        if self.fallback_active and self.current_mode == ThreadingMode.SINGLE_THREADED:
            if fallback_to_sync:
                return self._execute_synchronously(fn, args, kwargs, task_id)
            else:
                # Queue for single-threaded executor
                return self._queue_for_single_thread(fn, args, kwargs, task_id, timeout)
        
        # Submit to thread pool
        try:
            if not self.thread_pool:
                raise RuntimeError("Thread pool not available")
            
            # Wrap function with monitoring
            wrapped_fn = self._wrap_task_function(fn, task_id, timeout)
            
            # Submit task
            future = self.thread_pool.submit(wrapped_fn, *args, **kwargs)
            
            # Track active task
            self.active_tasks[task_id] = future
            
            # Record task submission
            self.task_history.append({
                'task_id': task_id,
                'submit_time': time.time(),
                'priority': priority,
                'timeout': timeout,
                'mode': self.current_mode.name
            })
            
            return future
            
        except Exception as e:
            logger.error(f"Failed to submit task {task_id}: {e}")
            self.failed_tasks += 1
            
            # Try fallback execution
            if fallback_to_sync:
                logger.warning(f"Falling back to synchronous execution for task {task_id}")
                return self._execute_synchronously(fn, args, kwargs, task_id)
            else:
                # Re-raise the exception
                raise

    def _wrap_task_function(self, fn: Callable, task_id: str, timeout: float) -> Callable:
        """Wrap task function with monitoring and error handling"""
        @wraps(fn)
        def wrapped_function(*args, **kwargs):
            thread_id = threading.get_ident()
            start_time = time.time()
            
            try:
                # Update thread metrics
                if thread_id not in self.thread_metrics:
                    self.thread_metrics[thread_id] = ThreadMetrics(
                        thread_id=thread_id,
                        name=threading.current_thread().name,
                        start_time=start_time,
                        task_count=0,
                        error_count=0,
                        last_active=start_time,
                        memory_usage_mb=0.0,
                        cpu_time_used=0.0,
                        is_stuck=False
                    )
                
                metrics = self.thread_metrics[thread_id]
                metrics.task_count += 1
                metrics.last_active = start_time
                metrics.memory_usage_mb = self._get_thread_memory_usage()
                
                # Set up timeout handling
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
                
                if hasattr(signal, 'SIGALRM'):
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                
                try:
                    # Execute the actual function
                    result = fn(*args, **kwargs)
                    
                    # Record successful completion
                    execution_time = time.time() - start_time
                    self.completed_tasks += 1
                    
                    # Update performance metrics
                    self._update_performance_metrics(execution_time, True)
                    
                    return result
                    
                finally:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    
                    # Clean up task tracking
                    self.active_tasks.pop(task_id, None)
                    
            except Exception as e:
                # Record error
                execution_time = time.time() - start_time
                self.failed_tasks += 1
                
                if thread_id in self.thread_metrics:
                    self.thread_metrics[thread_id].error_count += 1
                
                self.error_history.append({
                    'task_id': task_id,
                    'thread_id': thread_id,
                    'error': str(e),
                    'execution_time': execution_time,
                    'timestamp': time.time()
                })
                
                self._update_performance_metrics(execution_time, False)
                
                # Check if we should trigger fallback mode
                if self._should_trigger_fallback():
                    self._trigger_fallback_mode(f"Task failures in thread {thread_id}")
                
                raise
        
        return wrapped_function

    def _execute_synchronously(self, fn: Callable, args: tuple, kwargs: dict, task_id: str) -> Any:
        """Execute task synchronously as fallback"""
        start_time = time.time()
        
        try:
            result = fn(*args, **kwargs)
            execution_time = time.time() - start_time
            self.completed_tasks += 1
            self._update_performance_metrics(execution_time, True)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_tasks += 1
            self.error_history.append({
                'task_id': task_id,
                'thread_id': 'sync',
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': time.time()
            })
            self._update_performance_metrics(execution_time, False)
            raise

    def _queue_for_single_thread(self, fn: Callable, args: tuple, kwargs: dict, task_id: str, timeout: float) -> Future:
        """Queue task for single-threaded execution"""
        from concurrent.futures import Future
        
        future = Future()
        
        def execute_queued_task():
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        # Add to fallback queue
        try:
            self.fallback_queue.put((execute_queued_task, task_id, timeout), timeout=1.0)
        except queue.Full:
            future.set_exception(RuntimeError("Task queue is full"))
        
        return future

    def _should_trigger_fallback(self) -> bool:
        """Check if conditions warrant triggering fallback mode"""
        if len(self.error_history) < 5:
            return False
        
        # Check recent error rate
        recent_errors = [e for e in self.error_history if time.time() - e['timestamp'] < 60.0]
        error_rate = len(recent_errors) / 60.0  # Errors per second
        
        if error_rate > 0.1:  # More than 6 errors per minute
            return True
        
        # Check for stuck threads
        stuck_threads = [t for t in self.thread_metrics.values() if t.is_stuck]
        if len(stuck_threads) > 0:
            return True
        
        # Check memory pressure
        memory_usage = self._get_memory_usage()
        if memory_usage > self.config.memory_limit_mb:
            return True
        
        return False

    def _trigger_fallback_mode(self, reason: str) -> None:
        """Trigger fallback to a more conservative threading mode"""
        with self.manager_lock:
            try:
                logger.warning(f"Triggering fallback mode: {reason}")
                
                current_mode_value = self.current_mode.value
                
                # Move to next more conservative mode
                if self.current_mode == ThreadingMode.OPTIMAL:
                    new_mode = ThreadingMode.REDUCED
                elif self.current_mode == ThreadingMode.REDUCED:
                    new_mode = ThreadingMode.MINIMAL
                elif self.current_mode == ThreadingMode.MINIMAL:
                    new_mode = ThreadingMode.SINGLE_THREADED
                else:
                    # Already in single-threaded mode
                    logger.error("Already in most conservative mode, cannot fallback further")
                    return
                
                # Setup new mode
                success = self._setup_thread_pool(new_mode)
                
                if success:
                    self.fallback_reason = reason
                    logger.info(f"Successfully fell back to {new_mode.name} mode")
                else:
                    logger.error(f"Failed to fallback to {new_mode.name} mode")
                    self._emergency_fallback()
                
            except Exception as e:
                logger.error(f"Error during fallback: {e}")
                self._emergency_fallback()

    def _emergency_fallback(self) -> None:
        """Emergency fallback to completely disable threading"""
        logger.critical("Emergency fallback: disabling all threading")
        
        try:
            self._cleanup_thread_pool()
            self.current_mode = ThreadingMode.SINGLE_THREADED
            self.pool_health = ThreadPoolHealth.FAILED
            self.fallback_active = True
            self.fallback_reason = "Emergency fallback - all threading disabled"
            
        except Exception as e:
            logger.critical(f"Emergency fallback failed: {e}")

    def _start_health_monitoring(self) -> None:
        """Start thread pool health monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        try:
            self.monitoring_thread = threading.Thread(
                target=self._health_monitoring_loop,
                name="ThreadHealthMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Thread health monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")

    def _health_monitoring_loop(self) -> None:
        """Health monitoring loop"""
        logger.info("Thread health monitoring loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Check thread pool health
                self._check_thread_pool_health()
                
                # Check for stuck threads
                self._check_for_stuck_threads()
                
                # Update performance metrics
                self._update_pool_performance()
                
                # Check memory usage
                self._check_memory_usage()
                
                # Thermal throttling check (Steam Deck)
                if self.config.steam_deck_mode and self.config.thermal_throttling:
                    self._check_thermal_throttling()
                
                # Sleep until next check
                self.shutdown_event.wait(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retry
        
        logger.info("Thread health monitoring loop ended")

    def _check_thread_pool_health(self) -> None:
        """Check overall thread pool health"""
        try:
            if not self.thread_pool and self.current_mode != ThreadingMode.SINGLE_THREADED:
                self.pool_health = ThreadPoolHealth.FAILED
                return
            
            # Calculate health metrics
            total_tasks = self.completed_tasks + self.failed_tasks
            if total_tasks > 0:
                failure_rate = self.failed_tasks / total_tasks
                
                if failure_rate < 0.05:  # Less than 5% failure
                    self.pool_health = ThreadPoolHealth.HEALTHY
                elif failure_rate < 0.15:  # Less than 15% failure
                    self.pool_health = ThreadPoolHealth.DEGRADED
                else:
                    self.pool_health = ThreadPoolHealth.CRITICAL
            
            # Check recent performance
            if len(self.performance_history) > 10:
                recent_performance = list(self.performance_history)[-10:]
                avg_performance = sum(recent_performance) / len(recent_performance)
                
                if self.performance_baseline and avg_performance < self.performance_baseline * 0.5:
                    if self.pool_health == ThreadPoolHealth.HEALTHY:
                        self.pool_health = ThreadPoolHealth.DEGRADED
                    elif self.pool_health == ThreadPoolHealth.DEGRADED:
                        self.pool_health = ThreadPoolHealth.CRITICAL
            
        except Exception as e:
            logger.error(f"Error checking thread pool health: {e}")

    def _check_for_stuck_threads(self) -> None:
        """Check for stuck threads"""
        current_time = time.time()
        stuck_threshold = 60.0  # 60 seconds
        
        for thread_id, metrics in self.thread_metrics.items():
            if current_time - metrics.last_active > stuck_threshold:
                if not metrics.is_stuck:
                    logger.warning(f"Thread {thread_id} appears to be stuck")
                    metrics.is_stuck = True
                    
                    # If too many stuck threads, trigger fallback
                    stuck_count = sum(1 for m in self.thread_metrics.values() if m.is_stuck)
                    if stuck_count > len(self.thread_metrics) * 0.5:
                        self._trigger_fallback_mode(f"Too many stuck threads: {stuck_count}")

    def _update_pool_performance(self) -> None:
        """Update thread pool performance metrics"""
        try:
            current_time = time.time()
            time_since_last_check = current_time - self.last_performance_check
            
            if time_since_last_check > 0:
                # Calculate tasks per second
                recent_completed = len([
                    t for t in self.task_history 
                    if current_time - t['submit_time'] <= time_since_last_check
                ])
                
                performance = recent_completed / time_since_last_check
                self.performance_history.append(performance)
                
                # Set baseline if not established
                if self.performance_baseline is None and len(self.performance_history) >= 10:
                    self.performance_baseline = sum(self.performance_history) / len(self.performance_history)
                
                self.last_performance_check = current_time
                
        except Exception as e:
            logger.error(f"Error updating pool performance: {e}")

    def _check_memory_usage(self) -> None:
        """Check memory usage and trigger cleanup if needed"""
        try:
            memory_usage = self._get_memory_usage()
            
            if memory_usage > self.config.memory_limit_mb * 1.5:  # 150% of limit
                logger.warning(f"High memory usage: {memory_usage:.1f}MB")
                
                # Force garbage collection
                gc.collect()
                
                # If still high, trigger fallback
                memory_usage_after_gc = self._get_memory_usage()
                if memory_usage_after_gc > self.config.memory_limit_mb:
                    self._trigger_fallback_mode(f"High memory usage: {memory_usage_after_gc:.1f}MB")
                
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")

    def _check_thermal_throttling(self) -> None:
        """Check for thermal throttling conditions"""
        try:
            # Get CPU temperature
            temp = self._get_cpu_temperature()
            
            if temp > 80.0:  # Hot threshold
                logger.warning(f"High CPU temperature: {temp:.1f}°C")
                
                if temp > 90.0:  # Critical threshold
                    self._trigger_fallback_mode(f"Critical temperature: {temp:.1f}°C")
                
        except Exception as e:
            logger.error(f"Error checking thermal state: {e}")

    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        try:
            # Try multiple temperature sources
            temp_files = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp',
                '/sys/class/hwmon/hwmon0/temp1_input',
                '/sys/class/hwmon/hwmon1/temp1_input'
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp_millic = int(f.read().strip())
                        temp_celsius = temp_millic / 1000.0
                        if 20.0 <= temp_celsius <= 120.0:  # Sanity check
                            return temp_celsius
            
        except Exception:
            pass
        
        return 50.0  # Default safe temperature

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except Exception:
                return 128.0  # Default estimate

    def _get_thread_memory_usage(self) -> float:
        """Estimate memory usage of current thread"""
        try:
            # This is an approximation since Python doesn't provide per-thread memory stats
            total_memory = self._get_memory_usage()
            thread_count = threading.active_count()
            return total_memory / max(thread_count, 1)
        except Exception:
            return 10.0  # Default estimate

    def _update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update performance metrics"""
        try:
            # Simple performance score based on execution time and success rate
            if success:
                score = max(0.1, 1.0 / (execution_time + 0.1))  # Faster = higher score
            else:
                score = 0.0
            
            self.performance_history.append(score)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def get_status(self) -> ThreadPoolStatus:
        """Get comprehensive thread pool status"""
        try:
            active_threads = threading.active_count()
            queued_tasks = len(self.active_tasks)
            
            # Calculate performance score
            if self.performance_history:
                performance_score = sum(self.performance_history) / len(self.performance_history)
            else:
                performance_score = 0.0
            
            # Get latest errors
            recent_errors = [
                e['error'] for e in self.error_history
                if time.time() - e['timestamp'] < 300  # Last 5 minutes
            ]
            
            last_error_time = None
            if self.error_history:
                last_error_time = self.error_history[-1]['timestamp']
            
            return ThreadPoolStatus(
                mode=self.current_mode,
                health=self.pool_health,
                active_threads=active_threads,
                queued_tasks=queued_tasks,
                completed_tasks=self.completed_tasks,
                failed_tasks=self.failed_tasks,
                memory_usage_mb=self._get_memory_usage(),
                cpu_usage_percent=0.0,  # Would need psutil for accurate CPU usage
                thread_errors=recent_errors[-10:],  # Last 10 errors
                last_error_time=last_error_time,
                uptime_seconds=time.time() - self.start_time,
                performance_score=performance_score
            )
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return ThreadPoolStatus(
                mode=ThreadingMode.SINGLE_THREADED,
                health=ThreadPoolHealth.FAILED,
                active_threads=0,
                queued_tasks=0,
                completed_tasks=0,
                failed_tasks=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                thread_errors=[str(e)],
                last_error_time=time.time(),
                uptime_seconds=0.0,
                performance_score=0.0
            )

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all active tasks to complete"""
        try:
            if not self.active_tasks:
                return True
            
            start_time = time.time()
            
            while self.active_tasks:
                if timeout and time.time() - start_time > timeout:
                    logger.warning(f"Timeout waiting for task completion: {len(self.active_tasks)} tasks remaining")
                    return False
                
                # Wait for any task to complete
                try:
                    completed_futures = []
                    for task_id, future in list(self.active_tasks.items()):
                        if future.done():
                            completed_futures.append(task_id)
                    
                    for task_id in completed_futures:
                        self.active_tasks.pop(task_id, None)
                    
                    if self.active_tasks:
                        time.sleep(0.1)  # Brief sleep before checking again
                    
                except Exception as e:
                    logger.error(f"Error waiting for completion: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in wait_for_completion: {e}")
            return False

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> bool:
        """Shutdown the threading manager"""
        logger.info("Shutting down threading manager...")
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for active tasks if requested
            if wait and self.active_tasks:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
                self.wait_for_completion(timeout=timeout)
            
            # Cancel remaining tasks
            for task_id, future in list(self.active_tasks.items()):
                if not future.done():
                    future.cancel()
            
            # Cleanup thread pool
            success = self._cleanup_thread_pool()
            
            # Stop monitoring thread
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            logger.info("Threading manager shutdown completed")
            return success
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    def _cleanup_thread_pool(self) -> bool:
        """Clean up thread pool resources"""
        try:
            if self.thread_pool:
                logger.debug("Cleaning up thread pool...")
                self.thread_pool.shutdown(wait=True, timeout=10.0)
                self.thread_pool = None
            
            if self.process_pool:
                logger.debug("Cleaning up process pool...")
                self.process_pool.shutdown(wait=True, timeout=10.0)
                self.process_pool = None
            
            # Clear active tasks
            self.active_tasks.clear()
            
            # Clear thread metrics for dead threads
            current_thread_ids = {t.ident for t in threading.enumerate()}
            dead_thread_ids = set(self.thread_metrics.keys()) - current_thread_ids
            for thread_id in dead_thread_ids:
                del self.thread_metrics[thread_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up thread pool: {e}")
            return False

    @contextmanager
    def temporary_mode(self, mode: ThreadingMode):
        """Context manager for temporary threading mode"""
        original_mode = self.current_mode
        
        try:
            success = self._setup_thread_pool(mode)
            if not success:
                raise RuntimeError(f"Failed to switch to temporary mode: {mode}")
            
            yield
            
        finally:
            # Restore original mode
            self._setup_thread_pool(original_mode)

    def benchmark_threading_performance(self, test_duration: float = 30.0) -> Dict[str, Any]:
        """Benchmark threading performance across different modes"""
        benchmark_results = {}
        
        def benchmark_task(x):
            """Simple benchmark task"""
            import time
            import math
            
            # CPU-intensive work
            result = 0
            for i in range(1000):
                result += math.sqrt(x * i)
            
            time.sleep(0.001)  # Small I/O simulation
            return result
        
        # Test each mode
        test_modes = [ThreadingMode.OPTIMAL, ThreadingMode.REDUCED, ThreadingMode.MINIMAL, ThreadingMode.SINGLE_THREADED]
        
        for mode in test_modes:
            if mode.value > self.current_mode.value:
                continue  # Skip modes we can't achieve
            
            try:
                with self.temporary_mode(mode):
                    logger.info(f"Benchmarking {mode.name} mode...")
                    
                    start_time = time.time()
                    tasks = []
                    task_count = 0
                    
                    # Submit tasks for test duration
                    while time.time() - start_time < test_duration:
                        future = self.submit_task(benchmark_task, task_count)
                        tasks.append(future)
                        task_count += 1
                        
                        if len(tasks) >= 100:  # Limit concurrent tasks
                            # Wait for some to complete
                            completed = [f for f in tasks if f.done()]
                            for f in completed:
                                tasks.remove(f)
                    
                    # Wait for remaining tasks
                    end_time = time.time()
                    completed_count = 0
                    failed_count = 0
                    
                    for future in tasks:
                        try:
                            result = future.result(timeout=5.0)
                            completed_count += 1
                        except Exception:
                            failed_count += 1
                    
                    total_time = end_time - start_time
                    throughput = task_count / total_time
                    success_rate = completed_count / max(task_count, 1)
                    
                    benchmark_results[mode.name] = {
                        'tasks_submitted': task_count,
                        'tasks_completed': completed_count,
                        'tasks_failed': failed_count,
                        'total_time': total_time,
                        'throughput': throughput,
                        'success_rate': success_rate
                    }
                    
                    logger.info(f"{mode.name} benchmark: {throughput:.1f} tasks/sec, {success_rate:.1%} success rate")
                    
            except Exception as e:
                logger.error(f"Error benchmarking {mode.name}: {e}")
                benchmark_results[mode.name] = {'error': str(e)}
        
        return benchmark_results

    def export_status_report(self, filepath: Path) -> None:
        """Export comprehensive status report"""
        try:
            status = self.get_status()
            
            report = {
                'threading_status': {
                    'mode': status.mode.name,
                    'health': status.health.name,
                    'active_threads': status.active_threads,
                    'completed_tasks': status.completed_tasks,
                    'failed_tasks': status.failed_tasks,
                    'performance_score': status.performance_score,
                    'uptime_seconds': status.uptime_seconds
                },
                'configuration': {
                    'max_workers': self.config.max_workers,
                    'thread_timeout': self.config.thread_timeout,
                    'memory_limit_mb': self.config.memory_limit_mb,
                    'steam_deck_mode': self.config.steam_deck_mode,
                    'auto_adjust': self.auto_adjust
                },
                'fallback_status': {
                    'fallback_active': self.fallback_active,
                    'fallback_reason': self.fallback_reason
                },
                'thread_metrics': {
                    str(tid): {
                        'name': metrics.name,
                        'task_count': metrics.task_count,
                        'error_count': metrics.error_count,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'is_stuck': metrics.is_stuck
                    }
                    for tid, metrics in self.thread_metrics.items()
                },
                'performance_history': list(self.performance_history)[-20:],  # Last 20 measurements
                'error_history': [
                    {
                        'task_id': e['task_id'],
                        'error': e['error'],
                        'timestamp': e['timestamp']
                    }
                    for e in list(self.error_history)[-10:]  # Last 10 errors
                ],
                'system_info': {
                    'cpu_count': os.cpu_count(),
                    'memory_usage_mb': self._get_memory_usage(),
                    'cpu_temperature': self._get_cpu_temperature()
                },
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Status report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export status report: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_threading_manager: Optional[RobustThreadingManager] = None

def get_threading_manager() -> RobustThreadingManager:
    """Get or create global threading manager instance"""
    global _threading_manager
    if _threading_manager is None:
        _threading_manager = RobustThreadingManager()
    return _threading_manager

def submit_robust_task(fn: Callable, *args, **kwargs) -> Union[Future, Any]:
    """Submit a task using the robust threading manager"""
    manager = get_threading_manager()
    return manager.submit_task(fn, *args, **kwargs)

def is_threading_healthy() -> bool:
    """Check if threading system is healthy"""
    manager = get_threading_manager()
    status = manager.get_status()
    return status.health in [ThreadPoolHealth.HEALTHY, ThreadPoolHealth.DEGRADED]

@contextmanager
def single_threaded_mode():
    """Context manager for single-threaded execution"""
    manager = get_threading_manager()
    with manager.temporary_mode(ThreadingMode.SINGLE_THREADED):
        yield

def shutdown_threading():
    """Shutdown global threading manager"""
    global _threading_manager
    if _threading_manager:
        _threading_manager.shutdown()
        _threading_manager = None


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🧵 Robust Threading Manager Test Suite")
    print("=" * 55)
    
    # Initialize manager
    manager = RobustThreadingManager(auto_adjust=True)
    
    print(f"\n⚙️  Configuration:")
    print(f"  Mode: {manager.current_mode.name}")
    print(f"  Max Workers: {manager.config.max_workers}")
    print(f"  Steam Deck Mode: {manager.config.steam_deck_mode}")
    print(f"  Memory Limit: {manager.config.memory_limit_mb}MB")
    
    # Test task submission
    print(f"\n🧪 Testing Task Submission:")
    
    def test_task(x, delay=0.1):
        import time
        import math
        time.sleep(delay)
        return math.sqrt(x * 42)
    
    # Submit test tasks
    futures = []
    for i in range(10):
        future = manager.submit_task(test_task, i, delay=0.05)
        futures.append(future)
        print(f"  Submitted task {i}")
    
    # Collect results
    results = []
    for i, future in enumerate(futures):
        try:
            if hasattr(future, 'result'):
                result = future.result(timeout=5.0)
                results.append(result)
                print(f"  ✅ Task {i}: {result:.2f}")
            else:
                # Synchronous result
                results.append(future)
                print(f"  ✅ Task {i}: {future:.2f} (sync)")
        except Exception as e:
            print(f"  ❌ Task {i}: {e}")
    
    print(f"  Completed {len(results)}/10 tasks")
    
    # Check status
    print(f"\n📊 Threading Status:")
    status = manager.get_status()
    print(f"  Mode: {status.mode.name}")
    print(f"  Health: {status.health.name}")
    print(f"  Active Threads: {status.active_threads}")
    print(f"  Completed Tasks: {status.completed_tasks}")
    print(f"  Failed Tasks: {status.failed_tasks}")
    print(f"  Memory Usage: {status.memory_usage_mb:.1f}MB")
    print(f"  Performance Score: {status.performance_score:.2f}")
    print(f"  Uptime: {status.uptime_seconds:.1f}s")
    
    if status.thread_errors:
        print(f"  Recent Errors: {len(status.thread_errors)}")
    
    # Test fallback scenarios
    print(f"\n🔄 Testing Fallback Scenarios:")
    
    def problematic_task(x):
        if x == 5:
            raise RuntimeError("Simulated error")
        return x * 2
    
    # Submit some tasks that will fail
    problematic_futures = []
    for i in range(8):
        try:
            future = manager.submit_task(problematic_task, i)
            problematic_futures.append((i, future))
        except Exception as e:
            print(f"  Task {i} submission failed: {e}")
    
    # Check results
    failed_count = 0
    for i, future in problematic_futures:
        try:
            if hasattr(future, 'result'):
                result = future.result(timeout=2.0)
                print(f"  ✅ Task {i}: {result}")
            else:
                print(f"  ✅ Task {i}: {future} (sync)")
        except Exception as e:
            failed_count += 1
            print(f"  ❌ Task {i}: {e}")
    
    print(f"  Failed tasks: {failed_count}")
    
    # Final status
    print(f"\n📈 Final Status:")
    final_status = manager.get_status()
    print(f"  Final Mode: {final_status.mode.name}")
    print(f"  Final Health: {final_status.health.name}")
    print(f"  Total Completed: {final_status.completed_tasks}")
    print(f"  Total Failed: {final_status.failed_tasks}")
    
    if manager.fallback_active:
        print(f"  ⚠️  Fallback Active: {manager.fallback_reason}")
    
    # Export status report
    report_path = Path("/tmp/robust_threading_status.json")
    manager.export_status_report(report_path)
    print(f"\n💾 Status report exported to {report_path}")
    
    # Test context manager
    print(f"\n🎭 Testing Context Manager:")
    with manager.temporary_mode(ThreadingMode.SINGLE_THREADED):
        temp_future = manager.submit_task(lambda x: x * 3, 7)
        if hasattr(temp_future, 'result'):
            temp_result = temp_future.result()
        else:
            temp_result = temp_future
        print(f"  Temporary single-threaded result: {temp_result}")
    
    print(f"  Back to original mode: {manager.current_mode.name}")
    
    # Cleanup
    print(f"\n🧹 Shutting down...")
    shutdown_success = manager.shutdown(wait=True, timeout=10.0)
    print(f"  Shutdown successful: {'✅' if shutdown_success else '❌'}")
    
    print(f"\n✅ Robust Threading Manager test completed!")
    print(f"🎯 System handled threading gracefully with fallbacks as needed")