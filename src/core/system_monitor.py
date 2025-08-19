#!/usr/bin/env python3
"""
System Monitor Wrapper for Steam Deck ML Shader Prediction System

Provides unified interface for system monitoring with proper fallbacks
"""

import os
import time
import logging
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class SystemMemoryInfo:
    """Memory usage information"""
    rss: int = 0  # Resident Set Size in bytes
    vms: int = 0  # Virtual Memory Size in bytes
    percent: float = 0.0  # Memory usage percentage
    available: int = 0  # Available memory in bytes
    total: int = 0  # Total memory in bytes


@dataclass
class SystemCPUInfo:
    """CPU usage information"""
    percent: float = 0.0  # CPU usage percentage
    load_avg: tuple = (0.0, 0.0, 0.0)  # Load averages (1, 5, 15 minutes)
    core_count: int = 1  # Number of CPU cores


class SystemMonitor:
    """
    Unified system monitoring interface with psutil integration and pure Python fallbacks
    
    Handles the psutil API correctly by creating Process instances when needed,
    and provides graceful fallbacks when psutil is not available.
    """
    
    def __init__(self, enable_fallback: bool = True):
        """
        Initialize system monitor
        
        Args:
            enable_fallback: Enable pure Python fallbacks when psutil unavailable
        """
        self.enable_fallback = enable_fallback
        self.logger = logging.getLogger(__name__)
        
        # Try to initialize psutil
        self._psutil_available = False
        self._process = None
        
        try:
            import psutil
            self._psutil = psutil
            # Create process instance for current process
            self._process = psutil.Process()
            self._psutil_available = True
            self.logger.info("System monitor initialized with psutil support")
        except ImportError:
            self.logger.warning("psutil not available, using pure Python fallbacks")
        except Exception as e:
            self.logger.warning(f"Failed to initialize psutil: {e}, using fallbacks")
    
    def get_memory_info(self, pid: Optional[int] = None) -> SystemMemoryInfo:
        """
        Get memory information for a process (current process by default)
        
        Args:
            pid: Process ID (None for current process)
            
        Returns:
            SystemMemoryInfo object with memory details
        """
        if self._psutil_available:
            try:
                if pid is None:
                    # Use the cached process instance for current process
                    proc = self._process
                else:
                    # Create new process instance for specified PID
                    proc = self._psutil.Process(pid)
                
                # Get process memory info (this is the correct API usage)
                proc_memory = proc.memory_info()
                
                # Get system memory info
                sys_memory = self._psutil.virtual_memory()
                
                return SystemMemoryInfo(
                    rss=proc_memory.rss,
                    vms=proc_memory.vms,
                    percent=proc.memory_percent(),
                    available=sys_memory.available,
                    total=sys_memory.total
                )
                
            except Exception as e:
                self.logger.warning(f"psutil memory info failed: {e}, using fallback")
        
        # Pure Python fallback
        if self.enable_fallback:
            return self._get_memory_info_fallback()
        else:
            return SystemMemoryInfo()
    
    def get_cpu_info(self) -> SystemCPUInfo:
        """
        Get CPU usage information
        
        Returns:
            SystemCPUInfo object with CPU details
        """
        if self._psutil_available:
            try:
                cpu_percent = self._psutil.cpu_percent(interval=0.1)
                load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
                core_count = self._psutil.cpu_count()
                
                return SystemCPUInfo(
                    percent=cpu_percent,
                    load_avg=load_avg,
                    core_count=core_count
                )
                
            except Exception as e:
                self.logger.warning(f"psutil CPU info failed: {e}, using fallback")
        
        # Pure Python fallback
        if self.enable_fallback:
            return self._get_cpu_info_fallback()
        else:
            return SystemCPUInfo()
    
    def get_process_list(self) -> list:
        """
        Get list of running processes
        
        Returns:
            List of process information
        """
        if self._psutil_available:
            try:
                processes = []
                for proc in self._psutil.process_iter(['pid', 'name', 'memory_percent']):
                    try:
                        processes.append(proc.info)
                    except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                        continue
                return processes
            except Exception as e:
                self.logger.warning(f"psutil process list failed: {e}")
        
        # Fallback - minimal process info
        return [{'pid': os.getpid(), 'name': 'current', 'memory_percent': 0.0}]
    
    def is_steam_running(self) -> bool:
        """
        Check if Steam is currently running
        
        Returns:
            True if Steam process is detected
        """
        if self._psutil_available:
            try:
                for proc in self._psutil.process_iter(['name']):
                    try:
                        if proc.info['name'] and 'steam' in proc.info['name'].lower():
                            return True
                    except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                        continue
            except Exception as e:
                self.logger.warning(f"Steam detection failed: {e}")
        
        # Fallback - check for Steam directories
        steam_paths = [
            '/home/deck/.steam',
            '/home/deck/.local/share/Steam',
            os.path.expanduser('~/.steam'),
            os.path.expanduser('~/.local/share/Steam')
        ]
        
        return any(os.path.exists(path) for path in steam_paths)
    
    def _get_memory_info_fallback(self) -> SystemMemoryInfo:
        """Pure Python fallback for memory information"""
        try:
            # Try to read from /proc/meminfo
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(':')
                            value = int(parts[1]) * 1024  # Convert from kB to bytes
                            meminfo[key] = value
                
                total = meminfo.get('MemTotal', 0)
                available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
                
                return SystemMemoryInfo(
                    rss=0,  # Can't get process RSS without psutil
                    vms=0,
                    percent=0.0,
                    available=available,
                    total=total
                )
        except Exception as e:
            self.logger.warning(f"Memory fallback failed: {e}")
        
        # Ultimate fallback - return zeros
        return SystemMemoryInfo()
    
    def _get_cpu_info_fallback(self) -> SystemCPUInfo:
        """Pure Python fallback for CPU information"""
        try:
            # Get load average
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
            
            # Get CPU count
            cpu_count = os.cpu_count() or 1
            
            # Estimate CPU usage from load average
            cpu_percent = min(100.0, (load_avg[0] / cpu_count) * 100.0)
            
            return SystemCPUInfo(
                percent=cpu_percent,
                load_avg=load_avg,
                core_count=cpu_count
            )
        except Exception as e:
            self.logger.warning(f"CPU fallback failed: {e}")
        
        # Ultimate fallback
        return SystemCPUInfo(core_count=os.cpu_count() or 1)
    
    @property
    def psutil_available(self) -> bool:
        """Check if psutil is available and working"""
        return self._psutil_available
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system monitor status
        
        Returns:
            Dictionary with system monitor status information
        """
        return {
            'psutil_available': self._psutil_available,
            'fallback_enabled': self.enable_fallback,
            'current_pid': os.getpid(),
            'monitor_working': True
        }


# Global instance for convenience
_global_monitor = None


def get_system_monitor() -> SystemMonitor:
    """Get or create global system monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor


def get_memory_usage() -> SystemMemoryInfo:
    """Convenience function to get memory usage"""
    return get_system_monitor().get_memory_info()


def get_cpu_usage() -> SystemCPUInfo:
    """Convenience function to get CPU usage"""
    return get_system_monitor().get_cpu_info()


def is_steam_running() -> bool:
    """Convenience function to check if Steam is running"""
    return get_system_monitor().is_steam_running()


if __name__ == "__main__":
    # Test the system monitor
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 System Monitor Test")
    print("=" * 40)
    
    monitor = SystemMonitor()
    
    print(f"Psutil available: {monitor.psutil_available}")
    
    # Test memory info
    memory = monitor.get_memory_info()
    print(f"Memory RSS: {memory.rss / (1024*1024):.1f} MB")
    print(f"Memory percent: {memory.percent:.1f}%")
    print(f"Available memory: {memory.available / (1024*1024):.1f} MB")
    
    # Test CPU info  
    cpu = monitor.get_cpu_info()
    print(f"CPU usage: {cpu.percent:.1f}%")
    print(f"CPU cores: {cpu.core_count}")
    print(f"Load average: {cpu.load_avg}")
    
    # Test Steam detection
    steam_running = monitor.is_steam_running()
    print(f"Steam running: {steam_running}")
    
    print("\n✅ System monitor test completed!")