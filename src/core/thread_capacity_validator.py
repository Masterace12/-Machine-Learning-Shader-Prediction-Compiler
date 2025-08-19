#!/usr/bin/env python3
"""
Thread Capacity Validation and Pre-Flight Checks for Steam Deck
=============================================================

Provides comprehensive thread capacity validation and system resource
checking before thread pool creation to prevent "can't start new thread" errors.

Steam Deck Resource Management:
- System thread limit detection and validation
- Memory pressure monitoring
- Process resource usage analysis
- Emergency resource recovery
- Gaming mode resource coordination

Usage:
    validator = ThreadCapacityValidator()
    if validator.validate_thread_capacity(requested_threads=4):
        # Safe to create thread pool
        pass
"""

import os
import sys
import time
import psutil
import resource
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Early threading setup MUST be done before any other imports
if 'setup_threading' not in sys.modules:
    import setup_threading
    setup_threading.configure_for_steam_deck()

logger = logging.getLogger(__name__)

class ResourceState(Enum):
    """System resource states."""
    OPTIMAL = "optimal"          # Plenty of resources available
    GOOD = "good"                # Good resource availability  
    LIMITED = "limited"          # Limited resources, caution needed
    CONSTRAINED = "constrained"  # Heavily constrained, minimal threads only
    CRITICAL = "critical"        # Critical resource shortage
    EXHAUSTED = "exhausted"      # Resources exhausted, emergency mode

@dataclass
class ThreadInfo:
    """Information about system threading."""
    total_threads: int = 0
    user_threads: int = 0
    kernel_threads: int = 0
    available_capacity: int = 0
    soft_limit: int = 0
    hard_limit: int = 0
    threads_per_process_avg: float = 0.0

@dataclass
class MemoryInfo:
    """Memory usage information."""
    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    cached_mb: float = 0.0
    swap_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    pressure_percent: float = 0.0

@dataclass  
class ProcessInfo:
    """Process resource information."""
    process_count: int = 0
    steam_processes: int = 0
    python_processes: int = 0
    high_thread_processes: int = 0
    total_cpu_percent: float = 0.0
    total_memory_percent: float = 0.0

@dataclass
class SystemLimits:
    """System resource limits."""
    max_processes: int = 0
    max_threads: int = 0
    max_open_files: int = 0
    max_memory_mb: float = 0.0
    max_stack_size_kb: int = 0
    systemd_tasks_max: int = 0

@dataclass
class ValidationResult:
    """Thread capacity validation result."""
    success: bool = False
    recommended_threads: int = 1
    max_safe_threads: int = 1
    resource_state: ResourceState = ResourceState.CRITICAL
    warnings: List[str] = None
    errors: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.metrics is None:
            self.metrics = {}

class ThreadCapacityValidator:
    """Thread capacity validation and resource monitoring."""
    
    # Safe threading limits for different resource states  
    THREAD_LIMITS = {
        ResourceState.OPTIMAL: {'max': 8, 'recommended': 4},
        ResourceState.GOOD: {'max': 4, 'recommended': 2},
        ResourceState.LIMITED: {'max': 2, 'recommended': 1},
        ResourceState.CONSTRAINED: {'max': 1, 'recommended': 1},
        ResourceState.CRITICAL: {'max': 1, 'recommended': 1},
        ResourceState.EXHAUSTED: {'max': 1, 'recommended': 1}
    }
    
    # Warning thresholds
    WARNING_THRESHOLDS = {
        'thread_usage_percent': 80,
        'memory_usage_percent': 85,
        'cpu_usage_percent': 90,
        'process_count': 300,
        'swap_usage_percent': 50
    }
    
    # Critical thresholds
    CRITICAL_THRESHOLDS = {
        'thread_usage_percent': 95,
        'memory_usage_percent': 95, 
        'cpu_usage_percent': 98,
        'process_count': 500,
        'swap_usage_percent': 80
    }
    
    def __init__(self):
        self.is_steam_deck = setup_threading.is_steam_deck()
        self.last_validation = 0.0
        self.validation_cache_duration = 2.0  # Cache results for 2 seconds
        self.cached_result: Optional[ValidationResult] = None
        
        # System information
        self.system_limits = SystemLimits()
        self.thread_info = ThreadInfo()
        self.memory_info = MemoryInfo()
        self.process_info = ProcessInfo()
        
        logger.info(f"Thread capacity validator initialized - Steam Deck: {self.is_steam_deck}")
        
        # Initialize system limits
        self._detect_system_limits()
    
    def _detect_system_limits(self) -> None:
        """Detect system resource limits."""
        try:
            # Process limits
            nproc_soft, nproc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
            self.system_limits.max_processes = nproc_soft
            
            # File descriptor limits
            nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.system_limits.max_open_files = nofile_soft
            
            # Stack size
            stack_soft, stack_hard = resource.getrlimit(resource.RLIMIT_STACK)
            self.system_limits.max_stack_size_kb = stack_soft // 1024
            
            # Memory limits
            if hasattr(resource, 'RLIMIT_AS'):
                mem_soft, mem_hard = resource.getrlimit(resource.RLIMIT_AS)
                if mem_soft != resource.RLIM_INFINITY:
                    self.system_limits.max_memory_mb = mem_soft // (1024 * 1024)
            
            # Thread limits (estimate based on process limits)
            self.system_limits.max_threads = min(nproc_soft * 2, 100000)
            
            # SystemD TasksMax detection  
            try:
                result = subprocess.run(
                    ['systemctl', '--user', 'show', '--property=TasksMax'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if 'TasksMax=' in output:
                        tasks_max_str = output.split('=')[1]
                        if tasks_max_str.isdigit():
                            self.system_limits.systemd_tasks_max = int(tasks_max_str)
            except Exception as e:
                logger.debug(f"SystemD TasksMax detection failed: {e}")
            
            logger.info(f"System limits detected - Processes: {self.system_limits.max_processes}, "
                       f"Threads: {self.system_limits.max_threads}, "
                       f"Files: {self.system_limits.max_open_files}, "
                       f"SystemD Tasks: {self.system_limits.systemd_tasks_max}")
                       
        except Exception as e:
            logger.error(f"System limits detection failed: {e}")
            # Set conservative defaults
            self.system_limits.max_processes = 1000
            self.system_limits.max_threads = 2000
            self.system_limits.max_open_files = 1024
    
    def _get_thread_info(self) -> ThreadInfo:
        """Get detailed threading information."""
        thread_info = ThreadInfo()
        
        try:
            # Get detailed process/thread information
            result = subprocess.run(
                ['ps', '-eLf'], capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.splitlines()[1:]  # Skip header
                thread_info.total_threads = len(lines)
                
                user_threads = 0
                kernel_threads = 0
                process_thread_counts = {}
                
                for line in lines:
                    try:
                        parts = line.split()
                        if len(parts) >= 8:
                            uid = parts[0]
                            pid = parts[1] 
                            command = parts[7] if len(parts) > 7 else ''
                            
                            # Count user vs kernel threads
                            if command.startswith('[') and command.endswith(']'):
                                kernel_threads += 1
                            else:
                                user_threads += 1
                            
                            # Track threads per process
                            if pid not in process_thread_counts:
                                process_thread_counts[pid] = 0
                            process_thread_counts[pid] += 1
                            
                    except (IndexError, ValueError):
                        continue
                
                thread_info.user_threads = user_threads
                thread_info.kernel_threads = kernel_threads
                
                # Calculate average threads per process
                if process_thread_counts:
                    thread_info.threads_per_process_avg = sum(process_thread_counts.values()) / len(process_thread_counts)
            
            # Get resource limits
            thread_info.soft_limit = self.system_limits.max_processes
            thread_info.hard_limit = self.system_limits.max_threads
            
            # Calculate available capacity
            used_capacity = thread_info.total_threads
            if self.system_limits.systemd_tasks_max > 0:
                available_from_systemd = max(0, self.system_limits.systemd_tasks_max - used_capacity)
                available_from_limit = max(0, thread_info.soft_limit - used_capacity)
                thread_info.available_capacity = min(available_from_systemd, available_from_limit)
            else:
                thread_info.available_capacity = max(0, thread_info.soft_limit - used_capacity)
                
        except Exception as e:
            logger.error(f"Thread info collection failed: {e}")
            # Set conservative defaults
            thread_info.total_threads = 200
            thread_info.available_capacity = max(0, self.system_limits.max_processes - 200)
        
        return thread_info
    
    def _get_memory_info(self) -> MemoryInfo:
        """Get detailed memory information."""
        memory_info = MemoryInfo()
        
        try:
            # Use psutil for accurate memory information
            mem = psutil.virtual_memory()
            memory_info.total_mb = mem.total / (1024 * 1024)
            memory_info.available_mb = mem.available / (1024 * 1024)
            memory_info.used_mb = mem.used / (1024 * 1024)
            memory_info.cached_mb = getattr(mem, 'cached', 0) / (1024 * 1024)
            memory_info.pressure_percent = mem.percent
            
            # Swap information
            swap = psutil.swap_memory()
            memory_info.swap_total_mb = swap.total / (1024 * 1024)
            memory_info.swap_used_mb = swap.used / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Memory info collection failed: {e}")
            # Fallback to /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            value_kb = int(value.strip().split()[0])
                            meminfo[key.strip()] = value_kb
                
                memory_info.total_mb = meminfo.get('MemTotal', 0) / 1024
                memory_info.available_mb = meminfo.get('MemAvailable', 0) / 1024
                memory_info.used_mb = (meminfo.get('MemTotal', 0) - meminfo.get('MemFree', 0)) / 1024
                memory_info.cached_mb = meminfo.get('Cached', 0) / 1024
                
                if memory_info.total_mb > 0:
                    memory_info.pressure_percent = (memory_info.used_mb / memory_info.total_mb) * 100
                    
            except Exception as e2:
                logger.error(f"Fallback memory info failed: {e2}")
        
        return memory_info
    
    def _get_process_info(self) -> ProcessInfo:
        """Get process resource information."""
        process_info = ProcessInfo()
        
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'num_threads', 'cpu_percent', 'memory_percent']))
            process_info.process_count = len(processes)
            
            steam_count = 0
            python_count = 0
            high_thread_count = 0
            total_cpu = 0.0
            total_memory = 0.0
            
            for proc_info in processes:
                try:
                    name = proc_info.info['name'].lower()
                    threads = proc_info.info.get('num_threads', 1)
                    cpu = proc_info.info.get('cpu_percent', 0.0) or 0.0
                    memory = proc_info.info.get('memory_percent', 0.0) or 0.0
                    
                    # Count process types
                    if 'steam' in name:
                        steam_count += 1
                    if 'python' in name:
                        python_count += 1
                    if threads > 10:
                        high_thread_count += 1
                    
                    total_cpu += cpu
                    total_memory += memory
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            process_info.steam_processes = steam_count
            process_info.python_processes = python_count
            process_info.high_thread_processes = high_thread_count
            process_info.total_cpu_percent = min(total_cpu, 800.0)  # Cap at 8 cores * 100%
            process_info.total_memory_percent = min(total_memory, 100.0)
            
        except Exception as e:
            logger.error(f"Process info collection failed: {e}")
            # Set conservative estimates
            process_info.process_count = 150
            process_info.total_cpu_percent = 50.0
            process_info.total_memory_percent = 60.0
        
        return process_info
    
    def _determine_resource_state(self) -> ResourceState:
        """Determine overall resource state from collected metrics."""
        try:
            # Calculate resource usage percentages
            thread_usage = (self.thread_info.total_threads / max(self.system_limits.max_processes, 1)) * 100
            memory_usage = self.memory_info.pressure_percent
            cpu_usage = self.process_info.total_cpu_percent
            process_usage = (self.process_info.process_count / max(self.system_limits.max_processes, 1)) * 100
            
            # Check swap usage
            swap_usage = 0.0
            if self.memory_info.swap_total_mb > 0:
                swap_usage = (self.memory_info.swap_used_mb / self.memory_info.swap_total_mb) * 100
            
            # Determine state based on worst constraint
            max_usage = max(thread_usage, memory_usage, process_usage)
            
            # Emergency conditions
            if (swap_usage > self.CRITICAL_THRESHOLDS['swap_usage_percent'] or
                memory_usage > self.CRITICAL_THRESHOLDS['memory_usage_percent'] or
                thread_usage > self.CRITICAL_THRESHOLDS['thread_usage_percent'] or
                self.thread_info.available_capacity < 5):
                return ResourceState.EXHAUSTED
            
            # Critical conditions
            if (max_usage > 95 or 
                cpu_usage > self.CRITICAL_THRESHOLDS['cpu_usage_percent'] or
                self.thread_info.available_capacity < 10):
                return ResourceState.CRITICAL
            
            # Constrained conditions
            if (max_usage > 85 or
                swap_usage > self.WARNING_THRESHOLDS['swap_usage_percent'] or
                self.thread_info.available_capacity < 20):
                return ResourceState.CONSTRAINED
            
            # Limited conditions
            if (max_usage > 70 or
                cpu_usage > self.WARNING_THRESHOLDS['cpu_usage_percent'] or
                self.thread_info.available_capacity < 50):
                return ResourceState.LIMITED
            
            # Good conditions
            if max_usage > 50 or self.thread_info.available_capacity < 100:
                return ResourceState.GOOD
            
            # Optimal conditions
            return ResourceState.OPTIMAL
            
        except Exception as e:
            logger.error(f"Resource state determination failed: {e}")
            return ResourceState.CRITICAL
    
    def _generate_warnings_and_errors(self, resource_state: ResourceState) -> Tuple[List[str], List[str]]:
        """Generate warnings and errors based on resource state."""
        warnings = []
        errors = []
        
        try:
            # Thread warnings
            thread_usage = (self.thread_info.total_threads / max(self.system_limits.max_processes, 1)) * 100
            if thread_usage > self.WARNING_THRESHOLDS['thread_usage_percent']:
                if thread_usage > self.CRITICAL_THRESHOLDS['thread_usage_percent']:
                    errors.append(f"Critical thread usage: {thread_usage:.1f}% ({self.thread_info.total_threads}/{self.system_limits.max_processes})")
                else:
                    warnings.append(f"High thread usage: {thread_usage:.1f}% ({self.thread_info.total_threads}/{self.system_limits.max_processes})")
            
            # Memory warnings
            if self.memory_info.pressure_percent > self.WARNING_THRESHOLDS['memory_usage_percent']:
                if self.memory_info.pressure_percent > self.CRITICAL_THRESHOLDS['memory_usage_percent']:
                    errors.append(f"Critical memory usage: {self.memory_info.pressure_percent:.1f}% ({self.memory_info.used_mb:.0f}MB/{self.memory_info.total_mb:.0f}MB)")
                else:
                    warnings.append(f"High memory usage: {self.memory_info.pressure_percent:.1f}% ({self.memory_info.used_mb:.0f}MB/{self.memory_info.total_mb:.0f}MB)")
            
            # CPU warnings
            if self.process_info.total_cpu_percent > self.WARNING_THRESHOLDS['cpu_usage_percent']:
                if self.process_info.total_cpu_percent > self.CRITICAL_THRESHOLDS['cpu_usage_percent']:
                    errors.append(f"Critical CPU usage: {self.process_info.total_cpu_percent:.1f}%")
                else:
                    warnings.append(f"High CPU usage: {self.process_info.total_cpu_percent:.1f}%")
            
            # Swap warnings
            if self.memory_info.swap_total_mb > 0:
                swap_usage = (self.memory_info.swap_used_mb / self.memory_info.swap_total_mb) * 100
                if swap_usage > self.WARNING_THRESHOLDS['swap_usage_percent']:
                    if swap_usage > self.CRITICAL_THRESHOLDS['swap_usage_percent']:
                        errors.append(f"Critical swap usage: {swap_usage:.1f}% ({self.memory_info.swap_used_mb:.0f}MB)")
                    else:
                        warnings.append(f"High swap usage: {swap_usage:.1f}% ({self.memory_info.swap_used_mb:.0f}MB)")
            
            # Process count warnings
            if self.process_info.process_count > self.WARNING_THRESHOLDS['process_count']:
                if self.process_info.process_count > self.CRITICAL_THRESHOLDS['process_count']:
                    errors.append(f"Critical process count: {self.process_info.process_count}")
                else:
                    warnings.append(f"High process count: {self.process_info.process_count}")
            
            # Available capacity warnings
            if self.thread_info.available_capacity < 20:
                if self.thread_info.available_capacity < 5:
                    errors.append(f"Critical thread capacity: {self.thread_info.available_capacity} threads available")
                else:
                    warnings.append(f"Low thread capacity: {self.thread_info.available_capacity} threads available")
            
            # Steam Deck specific warnings
            if self.is_steam_deck:
                if self.process_info.steam_processes == 0:
                    warnings.append("Steam client not detected - may not be in gaming mode")
                elif self.process_info.steam_processes > 10:
                    warnings.append(f"Many Steam processes detected ({self.process_info.steam_processes}) - possible resource contention")
                    
        except Exception as e:
            errors.append(f"Warning generation failed: {e}")
        
        return warnings, errors
    
    def validate_thread_capacity(self, requested_threads: int = 4, force_update: bool = False) -> ValidationResult:
        """Validate thread capacity and recommend safe threading parameters."""
        current_time = time.time()
        
        # Use cached result if recent and not forcing update
        if (not force_update and 
            self.cached_result is not None and
            current_time - self.last_validation < self.validation_cache_duration):
            return self.cached_result
        
        try:
            # Collect system metrics
            self.thread_info = self._get_thread_info()
            self.memory_info = self._get_memory_info()
            self.process_info = self._get_process_info()
            
            # Determine resource state
            resource_state = self._determine_resource_state()
            
            # Get threading limits for current state
            limits = self.THREAD_LIMITS.get(resource_state, self.THREAD_LIMITS[ResourceState.CRITICAL])
            max_safe_threads = limits['max']
            recommended_threads = limits['recommended']
            
            # Adjust recommendations based on available capacity
            if self.thread_info.available_capacity < requested_threads:
                max_safe_threads = min(max_safe_threads, max(1, self.thread_info.available_capacity // 2))
                recommended_threads = min(recommended_threads, max(1, self.thread_info.available_capacity // 4))
            
            # Steam Deck specific adjustments
            if self.is_steam_deck:
                # More conservative limits on Steam Deck
                max_safe_threads = min(max_safe_threads, 4)
                recommended_threads = min(recommended_threads, 2)
                
                # Adjust for gaming mode
                if self.process_info.steam_processes > 5:  # Likely gaming
                    max_safe_threads = min(max_safe_threads, 2)
                    recommended_threads = 1
            
            # Generate warnings and errors
            warnings, errors = self._generate_warnings_and_errors(resource_state)
            
            # Determine success
            success = (resource_state not in [ResourceState.CRITICAL, ResourceState.EXHAUSTED] and
                      len(errors) == 0 and
                      requested_threads <= max_safe_threads)
            
            # Collect metrics
            metrics = {
                'resource_state': resource_state.value,
                'thread_info': {
                    'total': self.thread_info.total_threads,
                    'available': self.thread_info.available_capacity,
                    'user': self.thread_info.user_threads,
                    'kernel': self.thread_info.kernel_threads
                },
                'memory_info': {
                    'total_mb': self.memory_info.total_mb,
                    'available_mb': self.memory_info.available_mb,
                    'used_percent': self.memory_info.pressure_percent,
                    'swap_used_mb': self.memory_info.swap_used_mb
                },
                'process_info': {
                    'count': self.process_info.process_count,
                    'steam_processes': self.process_info.steam_processes,
                    'cpu_percent': self.process_info.total_cpu_percent
                },
                'system_limits': {
                    'max_processes': self.system_limits.max_processes,
                    'max_threads': self.system_limits.max_threads,
                    'systemd_tasks_max': self.system_limits.systemd_tasks_max
                }
            }
            
            # Create result
            result = ValidationResult(
                success=success,
                recommended_threads=recommended_threads,
                max_safe_threads=max_safe_threads,
                resource_state=resource_state,
                warnings=warnings,
                errors=errors,
                metrics=metrics
            )
            
            # Cache result
            self.cached_result = result
            self.last_validation = current_time
            
            logger.info(f"Thread capacity validation - State: {resource_state.value}, "
                       f"Recommended: {recommended_threads}, Max safe: {max_safe_threads}, "
                       f"Success: {success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Thread capacity validation failed: {e}")
            # Return emergency result
            return ValidationResult(
                success=False,
                recommended_threads=1,
                max_safe_threads=1,
                resource_state=ResourceState.CRITICAL,
                errors=[f"Validation failed: {e}"]
            )
    
    def get_current_resource_state(self) -> ResourceState:
        """Get current resource state without full validation."""
        if self.cached_result:
            return self.cached_result.resource_state
        
        # Quick check
        result = self.validate_thread_capacity(force_update=True)
        return result.resource_state
    
    def is_safe_for_threading(self, requested_threads: int = 1) -> bool:
        """Quick check if it's safe to create requested threads."""
        result = self.validate_thread_capacity(requested_threads)
        return result.success and requested_threads <= result.max_safe_threads
    
    def get_safe_thread_recommendation(self) -> int:
        """Get safe thread count recommendation."""
        result = self.validate_thread_capacity()
        return result.recommended_threads
    
    def clear_cache(self) -> None:
        """Clear validation cache to force fresh check."""
        self.cached_result = None
        self.last_validation = 0.0

# Global validator instance
_capacity_validator = None

def get_capacity_validator() -> ThreadCapacityValidator:
    """Get the global thread capacity validator instance."""
    global _capacity_validator
    if _capacity_validator is None:
        _capacity_validator = ThreadCapacityValidator()
    return _capacity_validator

# Convenience functions
def validate_thread_capacity(requested_threads: int = 4) -> ValidationResult:
    """Validate thread capacity (main entry point)."""
    validator = get_capacity_validator()
    return validator.validate_thread_capacity(requested_threads)

def is_safe_for_threading(requested_threads: int = 1) -> bool:
    """Check if it's safe to create requested threads."""
    validator = get_capacity_validator()
    return validator.is_safe_for_threading(requested_threads)

def get_safe_thread_count() -> int:
    """Get safe thread count recommendation."""
    validator = get_capacity_validator()
    return validator.get_safe_thread_recommendation()