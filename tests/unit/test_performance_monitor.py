#!/usr/bin/env python3
"""
Comprehensive tests for performance monitoring system
"""

import pytest
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.monitoring.performance_monitor import (
        OptimizedPerformanceMonitor,
        PerformanceMetrics,
        PerformanceAlert,
        PerformanceTracker,
        HealthStatus,
        get_performance_monitor
    )
    HAS_MONITOR_MODULE = True
except ImportError as e:
    print(f"Performance monitor module not available: {e}")
    HAS_MONITOR_MODULE = False


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
@pytest.mark.unit
@pytest.mark.performance
class TestPerformanceMetrics:
    """Test performance metrics data structure"""
    
    def test_metrics_creation(self):
        """Test performance metrics creation"""
        timestamp = time.time()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage_percent=45.5,
            memory_usage_percent=60.2,
            memory_available_mb=8192.0,
            disk_io_read_mb=15.5,
            disk_io_write_mb=8.2,
            network_io_recv_mb=125.0,
            network_io_sent_mb=42.5,
            process_count=142,
            load_average_1m=1.85,
            temperature_c=72.5
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage_percent == 45.5
        assert metrics.memory_usage_percent == 60.2
        assert metrics.memory_available_mb == 8192.0
        assert metrics.disk_io_read_mb == 15.5
        assert metrics.disk_io_write_mb == 8.2
        assert metrics.network_io_recv_mb == 125.0
        assert metrics.network_io_sent_mb == 42.5
        assert metrics.process_count == 142
        assert metrics.load_average_1m == 1.85
        assert metrics.temperature_c == 72.5
    
    def test_metrics_validation(self):
        """Test metrics validation in __post_init__"""
        # Test clamping of percentage values
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=150.0,  # Too high
            memory_usage_percent=-10.0,  # Too low
            memory_available_mb=4096.0,
            disk_io_read_mb=0.0,
            disk_io_write_mb=0.0,
            network_io_recv_mb=0.0,
            network_io_sent_mb=0.0,
            process_count=100,
            load_average_1m=0.5,
            temperature_c=65.0
        )
        
        # Percentages should be clamped
        assert metrics.cpu_usage_percent == 100.0  # Clamped to max
        assert metrics.memory_usage_percent == 0.0  # Clamped to min
    
    def test_metrics_edge_cases(self):
        """Test metrics with edge case values"""
        metrics = PerformanceMetrics(
            timestamp=0.0,
            cpu_usage_percent=0.0,
            memory_usage_percent=100.0,
            memory_available_mb=0.0,
            disk_io_read_mb=0.0,
            disk_io_write_mb=0.0,
            network_io_recv_mb=0.0,
            network_io_sent_mb=0.0,
            process_count=1,
            load_average_1m=0.0,
            temperature_c=0.0
        )
        
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_percent == 100.0
        assert metrics.memory_available_mb == 0.0
        assert metrics.process_count == 1


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
@pytest.mark.unit
@pytest.mark.performance
class TestPerformanceAlert:
    """Test performance alert data structure"""
    
    def test_alert_creation(self):
        """Test performance alert creation"""
        timestamp = time.time()
        alert = PerformanceAlert(
            timestamp=timestamp,
            level="warning",
            category="cpu",
            message="CPU usage high: 85.5%",
            value=85.5,
            threshold=80.0
        )
        
        assert alert.timestamp == timestamp
        assert alert.level == "warning"
        assert alert.category == "cpu"
        assert alert.message == "CPU usage high: 85.5%"
        assert alert.value == 85.5
        assert alert.threshold == 80.0
    
    def test_alert_levels(self):
        """Test different alert levels"""
        levels = ["warning", "error", "critical"]
        categories = ["cpu", "memory", "disk", "thermal", "process"]
        
        for level in levels:
            for category in categories:
                alert = PerformanceAlert(
                    timestamp=time.time(),
                    level=level,
                    category=category,
                    message=f"{category} {level}",
                    value=100.0,
                    threshold=80.0
                )
                assert alert.level == level
                assert alert.category == category


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
@pytest.mark.unit
@pytest.mark.performance
class TestPerformanceTracker:
    """Test performance tracking functionality"""
    
    @pytest.fixture
    def tracker(self):
        """Create a test performance tracker"""
        return PerformanceTracker(history_size=100)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics"""
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=45.0,
            memory_usage_percent=60.0,
            memory_available_mb=6144.0,
            disk_io_read_mb=5.0,
            disk_io_write_mb=2.0,
            network_io_recv_mb=10.0,
            network_io_sent_mb=5.0,
            process_count=120,
            load_average_1m=1.2,
            temperature_c=70.0
        )
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization"""
        assert len(tracker.history) == 0
        assert len(tracker.alerts) == 0
        assert isinstance(tracker.thresholds, dict)
        assert 'cpu_warning' in tracker.thresholds
        assert 'memory_critical' in tracker.thresholds
        assert tracker._lock is not None
    
    def test_add_metrics(self, tracker, sample_metrics):
        """Test adding metrics to tracker"""
        tracker.add_metrics(sample_metrics)
        
        assert len(tracker.history) == 1
        assert tracker.history[0] is sample_metrics
    
    def test_threshold_checking_cpu(self, tracker):
        """Test CPU threshold checking"""
        # Create metrics that should trigger CPU warning
        warning_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=85.0,  # Above warning threshold (80)
            memory_usage_percent=50.0,
            memory_available_mb=8192.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=1.0,
            network_io_recv_mb=1.0,
            network_io_sent_mb=1.0,
            process_count=100,
            load_average_1m=1.0,
            temperature_c=65.0
        )
        
        tracker.add_metrics(warning_metrics)
        
        # Should have generated a warning alert
        assert len(tracker.alerts) > 0
        alert = tracker.alerts[-1]
        assert alert.level == "warning"
        assert alert.category == "cpu"
        assert alert.value == 85.0
    
    def test_threshold_checking_memory_critical(self, tracker):
        """Test memory critical threshold checking"""
        critical_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=50.0,
            memory_usage_percent=96.0,  # Above critical threshold (95)
            memory_available_mb=1024.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=1.0,
            network_io_recv_mb=1.0,
            network_io_sent_mb=1.0,
            process_count=100,
            load_average_1m=1.0,
            temperature_c=65.0
        )
        
        tracker.add_metrics(critical_metrics)
        
        # Should have generated a critical alert
        assert len(tracker.alerts) > 0
        alert = tracker.alerts[-1]
        assert alert.level == "critical"
        assert alert.category == "memory"
        assert alert.value == 96.0
    
    def test_threshold_checking_temperature(self, tracker):
        """Test temperature threshold checking"""
        hot_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=50.0,
            memory_usage_percent=50.0,
            memory_available_mb=8192.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=1.0,
            network_io_recv_mb=1.0,
            network_io_sent_mb=1.0,
            process_count=100,
            load_average_1m=1.0,
            temperature_c=85.0  # Above warning threshold (80)
        )
        
        tracker.add_metrics(hot_metrics)
        
        # Should have generated a temperature warning
        alerts = [a for a in tracker.alerts if a.category == "thermal"]
        assert len(alerts) > 0
        alert = alerts[-1]
        assert alert.level == "warning"
        assert alert.category == "thermal"
    
    def test_get_recent_metrics(self, tracker):
        """Test getting recent metrics"""
        now = time.time()
        
        # Add metrics with different timestamps
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=now - (4-i) * 10,  # 40s ago to now, 10s intervals
                cpu_usage_percent=50.0,
                memory_usage_percent=60.0,
                memory_available_mb=6144.0,
                disk_io_read_mb=1.0,
                disk_io_write_mb=1.0,
                network_io_recv_mb=1.0,
                network_io_sent_mb=1.0,
                process_count=100,
                load_average_1m=1.0,
                temperature_c=70.0
            )
            tracker.add_metrics(metrics)
        
        # Get metrics from last 30 seconds
        recent = tracker.get_recent_metrics(30)
        
        # Should get metrics from last 30 seconds
        assert len(recent) >= 3  # Last 3 entries should be within 30s
    
    def test_get_recent_alerts(self, tracker):
        """Test getting recent alerts"""
        # Generate some alerts
        for temp in [82, 85, 88]:  # Generate temperature warnings
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage_percent=50.0,
                memory_usage_percent=50.0,
                memory_available_mb=8192.0,
                disk_io_read_mb=1.0,
                disk_io_write_mb=1.0,
                network_io_recv_mb=1.0,
                network_io_sent_mb=1.0,
                process_count=100,
                load_average_1m=1.0,
                temperature_c=temp
            )
            tracker.add_metrics(metrics)
        
        recent_alerts = tracker.get_recent_alerts(60)  # Last minute
        
        assert len(recent_alerts) >= 3
        for alert in recent_alerts:
            assert isinstance(alert, PerformanceAlert)
    
    def test_health_score_calculation(self, tracker):
        """Test health score calculation"""
        # Add some good metrics
        good_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=30.0,   # Good CPU usage
            memory_usage_percent=40.0, # Good memory usage
            memory_available_mb=9216.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=1.0,
            network_io_recv_mb=1.0,
            network_io_sent_mb=1.0,
            process_count=100,
            load_average_1m=1.0,
            temperature_c=60.0  # Good temperature
        )
        
        tracker.add_metrics(good_metrics)
        health_score = tracker.calculate_health_score()
        
        # Should be a good health score
        assert 50.0 <= health_score <= 100.0
        assert isinstance(health_score, float)
    
    def test_health_status_determination(self, tracker):
        """Test health status determination"""
        # Test with good metrics
        good_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=20.0,
            memory_usage_percent=30.0,
            memory_available_mb=10240.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=1.0,
            network_io_recv_mb=1.0,
            network_io_sent_mb=1.0,
            process_count=100,
            load_average_1m=0.5,
            temperature_c=55.0
        )
        
        tracker.add_metrics(good_metrics)
        status = tracker.get_health_status()
        
        assert isinstance(status, HealthStatus)
        assert status in [HealthStatus.EXCELLENT, HealthStatus.GOOD, HealthStatus.FAIR]


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
@pytest.mark.unit
@pytest.mark.performance
class TestOptimizedPerformanceMonitor:
    """Test optimized performance monitor"""
    
    @pytest.fixture
    def monitor(self, clean_temp_dir):
        """Create a test performance monitor"""
        config_path = clean_temp_dir / "perf_test.json"
        monitor = OptimizedPerformanceMonitor(config_path=config_path)
        yield monitor
        if monitor.monitoring_active:
            monitor.stop_monitoring()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.config_path is not None
        assert monitor.tracker is not None
        assert not monitor.monitoring_active
        assert monitor.monitoring_interval > 0
        assert len(monitor.alert_callbacks) == 0
        assert monitor._mock_mode in [True, False]
        assert monitor.logger is not None
    
    def test_system_info_collection(self, monitor):
        """Test system information collection"""
        system_info = monitor._get_system_info()
        
        assert isinstance(system_info, dict)
        assert 'cpu_count' in system_info
        assert 'memory_total_gb' in system_info
        assert 'boot_time' in system_info
        
        assert isinstance(system_info['cpu_count'], int)
        assert system_info['cpu_count'] > 0
        assert isinstance(system_info['memory_total_gb'], float)
        assert system_info['memory_total_gb'] > 0
        assert isinstance(system_info['boot_time'], (int, float))
    
    def test_metrics_collection(self, monitor):
        """Test metrics collection"""
        metrics = monitor._collect_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.timestamp > 0
        assert 0.0 <= metrics.cpu_usage_percent <= 100.0
        assert 0.0 <= metrics.memory_usage_percent <= 100.0
        assert metrics.memory_available_mb >= 0
        assert metrics.disk_io_read_mb >= 0
        assert metrics.disk_io_write_mb >= 0
        assert metrics.process_count > 0
        assert metrics.load_average_1m >= 0
    
    def test_mock_metrics_collection(self, monitor):
        """Test mock metrics collection"""
        if monitor._mock_mode:
            # Should generate realistic mock data
            for _ in range(5):
                metrics = monitor._collect_mock_metrics()
                
                assert isinstance(metrics, PerformanceMetrics)
                assert 0.0 <= metrics.cpu_usage_percent <= 100.0
                assert 0.0 <= metrics.memory_usage_percent <= 100.0
                assert metrics.memory_available_mb > 0
                assert metrics.process_count > 50  # Reasonable process count
                assert metrics.temperature_c > 40  # Reasonable temperature
    
    def test_alert_callback_registration(self, monitor):
        """Test alert callback registration and triggering"""
        callback_calls = []
        
        def test_callback(alert):
            callback_calls.append(alert)
        
        monitor.add_alert_callback(test_callback)
        
        # Manually trigger an alert
        alert = PerformanceAlert(
            timestamp=time.time(),
            level="warning",
            category="test",
            message="Test alert",
            value=90.0,
            threshold=80.0
        )
        
        monitor._handle_alert(alert)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] is alert
    
    def test_monitoring_start_stop(self, monitor):
        """Test monitoring start and stop"""
        assert not monitor.monitoring_active
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor._monitoring_thread is not None
        
        # Let it run briefly
        time.sleep(1.1)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
        
        # Should have collected some metrics
        assert len(monitor.tracker.history) > 0
    
    def test_current_metrics_retrieval(self, monitor):
        """Test current metrics retrieval"""
        # Initially should be None
        current = monitor.get_current_metrics()
        assert current is None
        
        # After starting monitoring, should get metrics
        monitor.start_monitoring()
        time.sleep(1.1)
        
        current = monitor.get_current_metrics()
        assert current is not None
        assert isinstance(current, PerformanceMetrics)
        
        monitor.stop_monitoring()
    
    def test_performance_report_generation(self, monitor):
        """Test comprehensive performance report generation"""
        # Start monitoring to collect some data
        monitor.start_monitoring()
        time.sleep(1.5)
        monitor.stop_monitoring()
        
        report = monitor.get_performance_report()
        
        required_keys = [
            'health_score',
            'health_status',
            'health_description',
            'system_info',
            'recent_metrics',
            'current_metrics',
            'alerts',
            'mock_mode',
            'monitoring_active'
        ]
        
        for key in required_keys:
            assert key in report, f"Report missing required key: {key}"
        
        # Check value types and structure
        assert isinstance(report['health_score'], (int, float))
        assert 0.0 <= report['health_score'] <= 100.0
        assert isinstance(report['health_status'], str)
        assert isinstance(report['health_description'], str)
        assert isinstance(report['system_info'], dict)
        assert isinstance(report['recent_metrics'], dict)
        assert isinstance(report['current_metrics'], dict)
        assert isinstance(report['alerts'], dict)
        assert isinstance(report['mock_mode'], bool)
        assert isinstance(report['monitoring_active'], bool)
        
        # Check nested structures
        assert 'count' in report['recent_metrics']
        assert 'avg_cpu_percent' in report['recent_metrics']
        assert 'recent_count' in report['alerts']
        assert 'critical_count' in report['alerts']
    
    def test_health_description_mapping(self, monitor):
        """Test health description mapping"""
        descriptions = {
            HealthStatus.EXCELLENT: "System running optimally",
            HealthStatus.GOOD: "System performing well",
            HealthStatus.FAIR: "System performance is acceptable", 
            HealthStatus.POOR: "System performance is degraded",
            HealthStatus.CRITICAL: "System performance is severely impacted"
        }
        
        for status, expected_desc in descriptions.items():
            desc = monitor._get_health_description(status)
            assert desc == expected_desc


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceMonitorIntegration:
    """Integration tests for performance monitor"""
    
    def test_full_monitoring_cycle(self, clean_temp_dir):
        """Test complete monitoring cycle with alert generation"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "integration.json")
        
        try:
            alerts_received = []
            
            def alert_handler(alert):
                alerts_received.append(alert)
            
            monitor.add_alert_callback(alert_handler)
            
            # Start monitoring
            monitor.start_monitoring()
            assert monitor.monitoring_active
            
            # Let it run and collect data
            time.sleep(2.1)  # More than 2 monitoring intervals
            
            # Should have collected metrics
            assert len(monitor.tracker.history) > 0
            
            # Get final report
            report = monitor.get_performance_report()
            assert report['health_score'] >= 0.0
            
            monitor.stop_monitoring()
            
            # Should have reasonable data
            assert report['recent_metrics']['count'] > 0
            assert report['system_info']['cpu_count'] > 0
        
        finally:
            if monitor.monitoring_active:
                monitor.stop_monitoring()
    
    def test_concurrent_monitoring_access(self, clean_temp_dir):
        """Test thread safety during monitoring"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "concurrent.json")
        
        try:
            monitor.start_monitoring()
            
            results = []
            errors = []
            
            def access_monitor():
                try:
                    for _ in range(10):
                        report = monitor.get_performance_report()
                        current = monitor.get_current_metrics()
                        results.append((report, current))
                        time.sleep(0.05)
                except Exception as e:
                    errors.append(e)
            
            # Multiple threads accessing simultaneously
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=access_monitor)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=10)
            
            monitor.stop_monitoring()
            
            # Should have no errors
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) > 0
            
            # All results should be valid
            for report, current in results:
                assert isinstance(report, dict)
                assert 'health_score' in report
        
        finally:
            if monitor.monitoring_active:
                monitor.stop_monitoring()
    
    def test_long_running_monitoring(self, clean_temp_dir):
        """Test monitor stability over longer period"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "longrun.json")
        
        try:
            monitor.start_monitoring()
            
            # Run for a longer period
            runtime = 3.0
            start_time = time.time()
            
            while time.time() - start_time < runtime:
                time.sleep(0.5)
                # Check that monitoring is still active
                assert monitor.monitoring_active
                
                # Get report to ensure system is responding
                report = monitor.get_performance_report()
                assert isinstance(report, dict)
            
            monitor.stop_monitoring()
            
            # Should have collected substantial data
            assert len(monitor.tracker.history) >= 3  # At least 3 samples over 3 seconds
            
            final_report = monitor.get_performance_report()
            assert final_report['recent_metrics']['count'] > 0
        
        finally:
            if monitor.monitoring_active:
                monitor.stop_monitoring()


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
@pytest.mark.benchmark
@pytest.mark.performance
class TestPerformanceMonitorBenchmarks:
    """Benchmark tests for performance monitor"""
    
    def test_metrics_collection_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark metrics collection performance"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "benchmark.json")
        
        # Benchmark metrics collection
        result = benchmark(monitor._collect_metrics)
        
        assert isinstance(result, PerformanceMetrics)
        assert result.timestamp > 0
    
    def test_health_calculation_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark health score calculation"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "health_bench.json")
        
        # Add some metrics first
        for _ in range(20):
            metrics = monitor._collect_metrics()
            monitor.tracker.add_metrics(metrics)
        
        # Benchmark health calculation
        result = benchmark(monitor.tracker.calculate_health_score)
        
        assert isinstance(result, (int, float))
        assert 0.0 <= result <= 100.0
    
    def test_report_generation_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark report generation"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "report_bench.json")
        
        # Add metrics and alerts for realistic benchmark
        for i in range(30):
            metrics = monitor._collect_metrics()
            monitor.tracker.add_metrics(metrics)
        
        # Benchmark report generation
        result = benchmark(monitor.get_performance_report)
        
        assert isinstance(result, dict)
        assert 'health_score' in result
    
    def test_monitoring_overhead(self, clean_temp_dir):
        """Test monitoring overhead impact"""
        monitor = OptimizedPerformanceMonitor(config_path=clean_temp_dir / "overhead.json")
        
        try:
            # Measure overhead of monitoring
            start_time = time.time()
            monitor.start_monitoring()
            
            # Let it run
            time.sleep(1.0)
            
            elapsed = time.time() - start_time
            monitor.stop_monitoring()
            
            # Should have minimal overhead
            samples = len(monitor.tracker.history)
            if samples > 0:
                avg_overhead = elapsed / samples
                # Each monitoring cycle should be very fast
                assert avg_overhead < 0.05, f"Too much overhead: {avg_overhead:.3f}s per sample"
        
        finally:
            if monitor.monitoring_active:
                monitor.stop_monitoring()


@pytest.mark.skipif(not HAS_MONITOR_MODULE, reason="Performance monitor module not available")
def test_global_performance_monitor():
    """Test global performance monitor singleton"""
    monitor1 = get_performance_monitor()
    monitor2 = get_performance_monitor()
    
    # Should be the same instance
    assert monitor1 is monitor2
    assert isinstance(monitor1, OptimizedPerformanceMonitor)
    
    # Clean up
    if monitor1.monitoring_active:
        monitor1.stop_monitoring()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])