#!/usr/bin/env python3
"""
Performance Monitoring System for Shader Prediction Compiler
Tracks system health, memory usage, and performance metrics
"""

import time
import threading
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# System monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


class HealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"    # > 90% health score
    GOOD = "good"             # 70-90% health score  
    FAIR = "fair"             # 50-70% health score
    POOR = "poor"             # 30-50% health score
    CRITICAL = "critical"     # < 30% health score


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_recv_mb: float
    network_io_sent_mb: float
    process_count: int
    load_average_1m: float
    temperature_c: float = 0.0
    
    def __post_init__(self):
        """Validate metrics"""
        self.cpu_usage_percent = max(0, min(100, self.cpu_usage_percent))
        self.memory_usage_percent = max(0, min(100, self.memory_usage_percent))


@dataclass
class PerformanceAlert:
    """Performance alert/warning"""
    timestamp: float
    level: str  # "warning", "error", "critical"
    category: str  # "cpu", "memory", "disk", "thermal", "process"
    message: str
    value: float
    threshold: float


class PerformanceTracker:
    """Track and analyze performance trends"""
    
    def __init__(self, history_size: int = 300):  # 5 minutes at 1 second intervals
        self.history = deque(maxlen=history_size)
        self.alerts = deque(maxlen=100)
        
        # Alert thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "disk_io_warning": 100.0,  # MB/s
            "disk_io_critical": 200.0,
            "temp_warning": 80.0,
            "temp_critical": 90.0,
            "load_warning": 4.0,
            "load_critical": 8.0
        }
        
        self._lock = threading.Lock()
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add performance metrics and check for alerts"""
        with self._lock:
            self.history.append(metrics)
            self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check metrics against thresholds and generate alerts"""
        timestamp = metrics.timestamp
        
        # CPU usage alerts
        if metrics.cpu_usage_percent > self.thresholds["cpu_critical"]:
            self.alerts.append(PerformanceAlert(
                timestamp=timestamp,
                level="critical",
                category="cpu",
                message=f"CPU usage critically high: {metrics.cpu_usage_percent:.1f}%",
                value=metrics.cpu_usage_percent,
                threshold=self.thresholds["cpu_critical"]
            ))
        elif metrics.cpu_usage_percent > self.thresholds["cpu_warning"]:
            self.alerts.append(PerformanceAlert(
                timestamp=timestamp,
                level="warning", 
                category="cpu",
                message=f"CPU usage high: {metrics.cpu_usage_percent:.1f}%",
                value=metrics.cpu_usage_percent,
                threshold=self.thresholds["cpu_warning"]
            ))
        
        # Memory usage alerts
        if metrics.memory_usage_percent > self.thresholds["memory_critical"]:
            self.alerts.append(PerformanceAlert(
                timestamp=timestamp,
                level="critical",
                category="memory",
                message=f"Memory usage critically high: {metrics.memory_usage_percent:.1f}%",
                value=metrics.memory_usage_percent,
                threshold=self.thresholds["memory_critical"]
            ))
        elif metrics.memory_usage_percent > self.thresholds["memory_warning"]:
            self.alerts.append(PerformanceAlert(
                timestamp=timestamp,
                level="warning",
                category="memory", 
                message=f"Memory usage high: {metrics.memory_usage_percent:.1f}%",
                value=metrics.memory_usage_percent,
                threshold=self.thresholds["memory_warning"]
            ))
        
        # Temperature alerts
        if metrics.temperature_c > self.thresholds["temp_critical"]:
            self.alerts.append(PerformanceAlert(
                timestamp=timestamp,
                level="critical",
                category="thermal",
                message=f"Temperature critically high: {metrics.temperature_c:.1f}°C",
                value=metrics.temperature_c,
                threshold=self.thresholds["temp_critical"]
            ))
        elif metrics.temperature_c > self.thresholds["temp_warning"]:
            self.alerts.append(PerformanceAlert(
                timestamp=timestamp,
                level="warning",
                category="thermal",
                message=f"Temperature high: {metrics.temperature_c:.1f}°C",
                value=metrics.temperature_c,
                threshold=self.thresholds["temp_warning"]
            ))
    
    def get_recent_metrics(self, seconds: int = 60) -> List[PerformanceMetrics]:
        """Get metrics from the last N seconds"""
        cutoff_time = time.time() - seconds
        with self._lock:
            return [m for m in self.history if m.timestamp >= cutoff_time]
    
    def get_recent_alerts(self, seconds: int = 300) -> List[PerformanceAlert]:
        """Get alerts from the last N seconds"""
        cutoff_time = time.time() - seconds
        with self._lock:
            return [a for a in self.alerts if a.timestamp >= cutoff_time]
    
    def calculate_health_score(self) -> float:
        """Calculate system health score (0-100)"""
        recent = self.get_recent_metrics(60)
        if not recent:
            return 50.0  # Unknown health
        
        # Average metrics over recent period
        avg_cpu = sum(m.cpu_usage_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_usage_percent for m in recent) / len(recent)
        avg_temp = sum(m.temperature_c for m in recent) / len(recent) if recent[0].temperature_c > 0 else 70.0
        
        # Calculate component scores (higher is better)
        cpu_score = max(0, 100 - avg_cpu)
        memory_score = max(0, 100 - avg_memory) 
        temp_score = max(0, 100 - (avg_temp - 30) * 2)  # 30°C baseline
        
        # Recent alerts penalty
        recent_alerts = self.get_recent_alerts(300)  # Last 5 minutes
        alert_penalty = len([a for a in recent_alerts if a.level == "critical"]) * 20
        alert_penalty += len([a for a in recent_alerts if a.level == "warning"]) * 5
        
        # Combined health score
        health_score = (cpu_score + memory_score + temp_score) / 3 - alert_penalty
        return max(0, min(100, health_score))
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        score = self.calculate_health_score()
        
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 30:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL


class OptimizedPerformanceMonitor:
    """High-performance system monitoring with minimal overhead"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize performance monitor"""
        self.config_path = config_path or Path.home() / '.config' / 'shader-predict-compile' / 'performance.json'
        
        # Performance tracking
        self.tracker = PerformanceTracker()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        self._monitoring_thread = None
        
        # Callbacks for alerts
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # System information cache
        self._system_info = None
        self._last_io_counters = None
        self._last_io_timestamp = 0
        
        # Mock mode for systems without psutil
        self._mock_mode = not HAS_PSUTIL
        self._mock_cpu = 25.0
        self._mock_memory = 40.0
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        if self._mock_mode:
            self.logger.info("Performance monitor running in mock mode - psutil not available")
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text())
                
                # Update monitoring interval
                self.monitoring_interval = config.get("monitoring_interval", 1.0)
                
                # Update thresholds
                thresholds = config.get("thresholds", {})
                self.tracker.thresholds.update(thresholds)
                
                self.logger.info("Performance monitor configuration loaded")
            except Exception as e:
                self.logger.warning(f"Could not load performance config: {e}")
    
    def _save_config(self):
        """Save current configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "monitoring_interval": self.monitoring_interval,
                "thresholds": self.tracker.thresholds
            }
            
            self.config_path.write_text(json.dumps(config, indent=2))
            self.logger.debug("Performance monitor configuration saved")
        except Exception as e:
            self.logger.error(f"Could not save performance config: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get cached system information"""
        if self._system_info is None:
            if HAS_PSUTIL:
                self._system_info = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "boot_time": psutil.boot_time(),
                }
            else:
                self._system_info = {
                    "cpu_count": 4,
                    "memory_total_gb": 16.0,
                    "boot_time": time.time() - 3600,  # Mock 1 hour uptime
                }
        
        return self._system_info
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        if self._mock_mode:
            return self._collect_mock_metrics()
        
        timestamp = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024**2)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            
            if self._last_io_counters and disk_io:
                time_delta = timestamp - self._last_io_timestamp
                read_delta = disk_io.read_bytes - self._last_io_counters.read_bytes
                write_delta = disk_io.write_bytes - self._last_io_counters.write_bytes
                
                if time_delta > 0:
                    disk_read_mb = (read_delta / (1024**2)) / time_delta
                    disk_write_mb = (write_delta / (1024**2)) / time_delta
            
            if disk_io:
                self._last_io_counters = disk_io
                self._last_io_timestamp = timestamp
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_recv_mb = net_io.bytes_recv / (1024**2) if net_io else 0.0
            net_sent_mb = net_io.bytes_sent / (1024**2) if net_io else 0.0
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
            except (AttributeError, OSError):
                load_avg = cpu_percent / 100.0 * self._get_system_info()["cpu_count"]
            
            return PerformanceMetrics(
                timestamp=timestamp,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_recv_mb=net_recv_mb,
                network_io_sent_mb=net_sent_mb,
                process_count=process_count,
                load_average_1m=load_avg,
                temperature_c=0.0  # Will be updated by thermal manager if available
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return self._collect_mock_metrics()
    
    def _collect_mock_metrics(self) -> PerformanceMetrics:
        """Collect mock metrics for testing"""
        import random
        
        # Simulate realistic metric variations
        self._mock_cpu += random.uniform(-5, 5)
        self._mock_cpu = max(10, min(80, self._mock_cpu))
        
        self._mock_memory += random.uniform(-2, 2)
        self._mock_memory = max(20, min(90, self._mock_memory))
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=self._mock_cpu + random.uniform(-5, 5),
            memory_usage_percent=self._mock_memory + random.uniform(-3, 3),
            memory_available_mb=8192 - (self._mock_memory / 100.0 * 16384),
            disk_io_read_mb=random.uniform(0, 10),
            disk_io_write_mb=random.uniform(0, 5),
            network_io_recv_mb=random.uniform(0, 100),
            network_io_sent_mb=random.uniform(0, 50),
            process_count=random.randint(80, 120),
            load_average_1m=self._mock_cpu / 100.0 * 4 + random.uniform(-0.5, 0.5),
            temperature_c=65 + random.uniform(-10, 15)
        )
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def _handle_alert(self, alert: PerformanceAlert):
        """Handle new performance alert"""
        self.logger.warning(f"Performance alert ({alert.level}): {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            previous_alert_count = 0
            
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = self._collect_metrics()
                    
                    # Add to tracker (this checks thresholds and generates alerts)
                    self.tracker.add_metrics(metrics)
                    
                    # Handle new alerts
                    current_alert_count = len(self.tracker.alerts)
                    if current_alert_count > previous_alert_count:
                        # New alert(s) generated
                        new_alerts = list(self.tracker.alerts)[previous_alert_count:]
                        for alert in new_alerts:
                            self._handle_alert(alert)
                    previous_alert_count = current_alert_count
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    time.sleep(self.monitoring_interval * 2)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2)
        
        # Save configuration
        self._save_config()
        
        self.logger.info("Performance monitoring stopped")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics"""
        if self.tracker.history:
            return self.tracker.history[-1]
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        recent_metrics = self.tracker.get_recent_metrics(60)
        recent_alerts = self.tracker.get_recent_alerts(300)
        health_score = self.tracker.calculate_health_score()
        health_status = self.tracker.get_health_status()
        current_metrics = self.get_current_metrics()
        
        report = {
            "health_score": health_score,
            "health_status": health_status.value,
            "health_description": self._get_health_description(health_status),
            "system_info": self._get_system_info(),
            "recent_metrics": {
                "count": len(recent_metrics),
                "avg_cpu_percent": sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "avg_memory_percent": sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "avg_load": sum(m.load_average_1m for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
            },
            "current_metrics": {
                "cpu_usage_percent": current_metrics.cpu_usage_percent if current_metrics else 0,
                "memory_usage_percent": current_metrics.memory_usage_percent if current_metrics else 0,
                "memory_available_mb": current_metrics.memory_available_mb if current_metrics else 0,
                "process_count": current_metrics.process_count if current_metrics else 0,
                "temperature_c": current_metrics.temperature_c if current_metrics else 0,
            },
            "alerts": {
                "recent_count": len(recent_alerts),
                "critical_count": len([a for a in recent_alerts if a.level == "critical"]),
                "warning_count": len([a for a in recent_alerts if a.level == "warning"]),
                "latest": recent_alerts[-3:] if recent_alerts else []  # Last 3 alerts
            },
            "mock_mode": self._mock_mode,
            "monitoring_active": self.monitoring_active
        }
        
        return report
    
    def _get_health_description(self, status: HealthStatus) -> str:
        """Get human-readable health description"""
        descriptions = {
            HealthStatus.EXCELLENT: "System running optimally",
            HealthStatus.GOOD: "System performing well", 
            HealthStatus.FAIR: "System performance is acceptable",
            HealthStatus.POOR: "System performance is degraded",
            HealthStatus.CRITICAL: "System performance is severely impacted"
        }
        return descriptions.get(status, "Unknown health status")


# Global instance
_performance_monitor = None


def get_performance_monitor() -> OptimizedPerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = OptimizedPerformanceMonitor()
    return _performance_monitor


if __name__ == "__main__":
    # Test performance monitor
    logging.basicConfig(level=logging.INFO)
    
    monitor = get_performance_monitor()
    
    print("📊 Performance Monitor Test")
    print("=" * 30)
    
    # Show initial report
    report = monitor.get_performance_report()
    print(f"Health Score: {report['health_score']:.1f}")
    print(f"Health Status: {report['health_status']}")
    print(f"Mock Mode: {report['mock_mode']}")
    print(f"CPU Count: {report['system_info']['cpu_count']}")
    print(f"Memory Total: {report['system_info']['memory_total_gb']:.1f}GB")
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        print("\nMonitoring for 10 seconds...")
        for i in range(10):
            time.sleep(1)
            report = monitor.get_performance_report()
            current = report['current_metrics']
            print(f"[{i+1:2d}s] CPU: {current['cpu_usage_percent']:.1f}% "
                  f"Mem: {current['memory_usage_percent']:.1f}% "
                  f"Health: {report['health_score']:.1f}")
    
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()
        print("✓ Performance monitor test completed")