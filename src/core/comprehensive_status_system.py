#!/usr/bin/env python3
"""
Comprehensive Status System and User Reporting

This module provides advanced logging, status monitoring, and user-friendly
reporting for the entire ML shader prediction system with Steam Deck optimizations.

Features:
- Multi-level logging with context-aware formatting
- Real-time status monitoring and health reporting
- User-friendly status messages and progress indicators
- Steam Deck specific status information and optimizations
- Performance metrics collection and analysis
- System resource monitoring and alerting
- Interactive status dashboard capabilities
- Export capabilities for debugging and support
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
import warnings
from typing import Dict, List, Any, Optional, Union, Callable, TextIO
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, deque
from enum import Enum, auto
import datetime
import socket
import platform

# Import our systems
try:
    from .enhanced_dependency_coordinator import get_coordinator
    from .tiered_fallback_system import get_fallback_system, FallbackTier
    from .robust_threading_manager import get_threading_manager, ThreadingMode
    from .pure_python_fallbacks import PureThermalMonitor, PureSteamDeckDetector
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import all systems: {e}")

# =============================================================================
# STATUS SYSTEM CONFIGURATION
# =============================================================================

class LogLevel(Enum):
    """Extended log levels for comprehensive reporting"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class StatusCategory(Enum):
    """Categories for status reporting"""
    SYSTEM_HEALTH = auto()
    DEPENDENCY_STATUS = auto()
    PERFORMANCE_METRICS = auto()
    FALLBACK_STATUS = auto()
    THREADING_STATUS = auto()
    STEAM_DECK_STATUS = auto()
    USER_RECOMMENDATIONS = auto()

class ReportFormat(Enum):
    """Report output formats"""
    CONSOLE = auto()
    JSON = auto()
    HTML = auto()
    MARKDOWN = auto()

@dataclass
class StatusMessage:
    """Structured status message"""
    timestamp: float
    level: LogLevel
    category: StatusCategory
    component: str
    title: str
    description: str
    details: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    is_user_visible: bool = True
    requires_action: bool = False
    action_suggestion: Optional[str] = None

@dataclass
class SystemSnapshot:
    """Complete system status snapshot"""
    timestamp: float
    overall_health: float
    health_description: str
    active_warnings: List[str]
    critical_issues: List[str]
    recommendations: List[str]
    component_status: Dict[str, Dict[str, Any]]
    performance_summary: Dict[str, float]
    steam_deck_status: Optional[Dict[str, Any]]
    uptime_seconds: float

# =============================================================================
# ENHANCED LOGGING SYSTEM
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored console formatter with context awareness"""
    
    # ANSI color codes
    COLORS = {
        'TRACE': '\033[90m',      # Dark gray
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[37m',       # White
        'SUCCESS': '\033[92m',    # Bright green
        'WARNING': '\033[93m',    # Bright yellow
        'ERROR': '\033[91m',      # Bright red
        'CRITICAL': '\033[95m',   # Bright magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',        # Bold
        'DIM': '\033[2m'          # Dim
    }
    
    ICONS = {
        'TRACE': '🔍',
        'DEBUG': '🐛',
        'INFO': 'ℹ️ ',
        'SUCCESS': '✅',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🚨',
        'STEAM_DECK': '🎮',
        'PERFORMANCE': '⚡',
        'DEPENDENCY': '📦',
        'FALLBACK': '🔄',
        'THREADING': '🧵'
    }
    
    def __init__(self, use_colors: bool = True, steam_deck_mode: bool = False):
        super().__init__()
        self.use_colors = use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        self.steam_deck_mode = steam_deck_mode
        
    def format(self, record):
        # Extract component from logger name
        component = record.name.split('.')[-1] if '.' in record.name else record.name
        
        # Determine appropriate icon
        level_name = record.levelname
        icon = self.ICONS.get(level_name, '📝')
        
        # Add component-specific icons
        if 'steam' in component.lower():
            icon = self.ICONS['STEAM_DECK']
        elif 'performance' in component.lower() or 'threading' in component.lower():
            icon = self.ICONS['PERFORMANCE']
        elif 'dependency' in component.lower():
            icon = self.ICONS['DEPENDENCY']
        elif 'fallback' in component.lower():
            icon = self.ICONS['FALLBACK']
        elif 'thread' in component.lower():
            icon = self.ICONS['THREADING']
        
        # Format timestamp
        timestamp = datetime.datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Build message
        message = record.getMessage()
        
        if self.use_colors:
            color = self.COLORS.get(level_name, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            bold = self.COLORS['BOLD']
            dim = self.COLORS['DIM']
            
            formatted = f"{dim}[{timestamp}]{reset} {icon} {color}{bold}{component}{reset} {message}"
        else:
            formatted = f"[{timestamp}] {icon} {component} {message}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted

class ProgressTracker:
    """Track and display progress for long-running operations"""
    
    def __init__(self, total_steps: int, description: str, show_eta: bool = True):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.show_eta = show_eta
        self.start_time = time.time()
        self.step_times: deque = deque(maxlen=10)  # Track last 10 step times for ETA
        
    def update(self, steps: int = 1, status: str = ""):
        """Update progress"""
        self.current_step = min(self.current_step + steps, self.total_steps)
        current_time = time.time()
        self.step_times.append(current_time)
        
        progress = self.current_step / self.total_steps
        
        # Calculate ETA
        eta_str = ""
        if self.show_eta and len(self.step_times) > 1 and self.current_step < self.total_steps:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_step_time
            eta_str = f" ETA: {eta_seconds:.0f}s"
        
        # Create progress bar
        bar_width = 20
        filled = int(bar_width * progress)
        bar = f"[{'█' * filled}{'░' * (bar_width - filled)}]"
        
        status_msg = f"{self.description} {bar} {progress:.1%} ({self.current_step}/{self.total_steps}){eta_str}"
        if status:
            status_msg += f" - {status}"
        
        # Log progress
        logger = logging.getLogger(__name__)
        logger.info(status_msg)
        
    def complete(self, final_status: str = "Complete"):
        """Mark progress as complete"""
        self.current_step = self.total_steps
        elapsed = time.time() - self.start_time
        logger = logging.getLogger(__name__)
        logger.info(f"{self.description} ✅ {final_status} (took {elapsed:.1f}s)")

# =============================================================================
# COMPREHENSIVE STATUS SYSTEM
# =============================================================================

class ComprehensiveStatusSystem:
    """
    Central status monitoring and reporting system
    """
    
    def __init__(self, 
                 log_level: LogLevel = LogLevel.INFO,
                 console_colors: bool = True,
                 status_update_interval: float = 5.0):
        
        self.log_level = log_level
        self.console_colors = console_colors
        self.status_update_interval = status_update_interval
        
        # System detection
        self.is_steam_deck = PureSteamDeckDetector.is_steam_deck()
        self.thermal_monitor = PureThermalMonitor()
        
        # Status tracking
        self.status_messages: deque = deque(maxlen=1000)
        self.system_snapshots: deque = deque(maxlen=100)
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.user_notifications: List[StatusMessage] = []
        
        # System components
        self.coordinator = None
        self.fallback_system = None
        self.threading_manager = None
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.status_callbacks: List[Callable] = []
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize component connections
        self._connect_to_components()
        
        # Start monitoring
        self.start_monitoring()
        
        logger = logging.getLogger(__name__)
        logger.info("ComprehensiveStatusSystem initialized")
        logger.info(f"Steam Deck mode: {self.is_steam_deck}")

    def _setup_logging(self) -> None:
        """Setup comprehensive logging system"""
        # Add custom log levels
        logging.addLevelName(LogLevel.TRACE.value, 'TRACE')
        logging.addLevelName(LogLevel.SUCCESS.value, 'SUCCESS')
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level.value)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            use_colors=self.console_colors,
            steam_deck_mode=self.is_steam_deck
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for detailed logs
        log_dir = Path('/tmp/ml_shader_prediction_logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"system_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Capture warnings
        logging.captureWarnings(True)
        warnings.filterwarnings('default', category=DeprecationWarning, module='.*')
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging system initialized - Log file: {log_file}")

    def _connect_to_components(self) -> None:
        """Connect to system components"""
        try:
            # Connect to coordinator
            self.coordinator = get_coordinator()
            self.coordinator.add_health_callback(self._handle_health_update)
            self.coordinator.add_optimization_callback(self._handle_optimization_update)
            
            # Connect to fallback system
            self.fallback_system = get_fallback_system()
            
            # Connect to threading manager
            self.threading_manager = get_threading_manager()
            
            logger = logging.getLogger(__name__)
            logger.info("Successfully connected to all system components")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error connecting to components: {e}")

    def _handle_health_update(self, health_report):
        """Handle health updates from coordinator"""
        try:
            # Create status message
            if health_report.health_level.name == 'EXCELLENT':
                level = LogLevel.SUCCESS
                title = "System Health Excellent"
                icon = "✅"
            elif health_report.health_level.name == 'GOOD':
                level = LogLevel.INFO
                title = "System Health Good"
                icon = "👍"
            elif health_report.health_level.name in ['FAIR', 'POOR']:
                level = LogLevel.WARNING
                title = f"System Health {health_report.health_level.name.title()}"
                icon = "⚠️"
            else:
                level = LogLevel.ERROR
                title = "System Health Critical"
                icon = "🚨"
            
            description = f"{icon} Overall health: {health_report.overall_health:.1%}"
            
            self._add_status_message(
                level=level,
                category=StatusCategory.SYSTEM_HEALTH,
                component="health_monitor",
                title=title,
                description=description,
                details={
                    'overall_health': health_report.overall_health,
                    'component_health': health_report.component_health,
                    'critical_issues': health_report.critical_issues,
                    'recommendations': health_report.recommendations
                }
            )
            
            # Add user notifications for critical issues
            for issue in health_report.critical_issues:
                self.user_notifications.append(StatusMessage(
                    timestamp=time.time(),
                    level=LogLevel.ERROR,
                    category=StatusCategory.SYSTEM_HEALTH,
                    component="health_monitor",
                    title="Critical Issue Detected",
                    description=issue,
                    is_user_visible=True,
                    requires_action=True,
                    action_suggestion="Run system diagnostics and apply recommended fixes"
                ))
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error handling health update: {e}")

    def _handle_optimization_update(self, optimization_result):
        """Handle optimization updates"""
        try:
            if optimization_result['optimizations_applied']:
                self._add_status_message(
                    level=LogLevel.SUCCESS,
                    category=StatusCategory.PERFORMANCE_METRICS,
                    component="optimizer",
                    title="System Optimization Applied",
                    description=f"✨ Applied {len(optimization_result['optimizations_applied'])} optimizations",
                    details=optimization_result
                )
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error handling optimization update: {e}")

    def _add_status_message(self, 
                           level: LogLevel,
                           category: StatusCategory,
                           component: str,
                           title: str,
                           description: str,
                           details: Optional[Dict[str, Any]] = None,
                           progress: Optional[float] = None,
                           is_user_visible: bool = True,
                           requires_action: bool = False,
                           action_suggestion: Optional[str] = None) -> None:
        """Add a status message to the system"""
        
        message = StatusMessage(
            timestamp=time.time(),
            level=level,
            category=category,
            component=component,
            title=title,
            description=description,
            details=details,
            progress=progress,
            is_user_visible=is_user_visible,
            requires_action=requires_action,
            action_suggestion=action_suggestion
        )
        
        self.status_messages.append(message)
        
        # Log the message
        logger = logging.getLogger(f"status.{component}")
        
        log_message = f"{title}: {description}"
        if progress is not None:
            log_message += f" ({progress:.1%})"
        
        if level == LogLevel.TRACE:
            logger.log(LogLevel.TRACE.value, log_message)
        elif level == LogLevel.DEBUG:
            logger.debug(log_message)
        elif level == LogLevel.INFO:
            logger.info(log_message)
        elif level == LogLevel.SUCCESS:
            logger.log(LogLevel.SUCCESS.value, log_message)
        elif level == LogLevel.WARNING:
            logger.warning(log_message)
        elif level == LogLevel.ERROR:
            logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            logger.critical(log_message)

    def create_system_snapshot(self) -> SystemSnapshot:
        """Create comprehensive system status snapshot"""
        try:
            current_time = time.time()
            
            # Get health from coordinator
            overall_health = 0.5
            health_description = "Unknown"
            active_warnings = []
            critical_issues = []
            recommendations = []
            
            if self.coordinator:
                try:
                    health_report = self.coordinator.get_comprehensive_health_report()
                    overall_health = health_report.overall_health
                    health_description = health_report.health_level.name.title()
                    active_warnings = health_report.warnings[:5]  # Top 5
                    critical_issues = health_report.critical_issues
                    recommendations = health_report.recommendations[:3]  # Top 3
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error getting health report: {e}")
            
            # Component status
            component_status = {}
            
            # Dependency status
            try:
                if self.coordinator and hasattr(self.coordinator, 'detector'):
                    detection_summary = self.coordinator.detector.get_detection_summary()
                    component_status['dependencies'] = {
                        'success_rate': detection_summary.get('success_rate', 0.0),
                        'total_dependencies': detection_summary.get('total_dependencies', 0),
                        'successful_detections': detection_summary.get('successful_detections', 0)
                    }
            except Exception as e:
                component_status['dependencies'] = {'error': str(e)}
            
            # Fallback status
            try:
                if self.fallback_system:
                    fallback_status = self.fallback_system.get_fallback_status()
                    component_status['fallbacks'] = {
                        'active_components': fallback_status.get('active_components', 0),
                        'average_performance_multiplier': fallback_status.get('average_performance_multiplier', 1.0),
                        'total_memory_usage_mb': fallback_status.get('total_memory_usage_mb', 0)
                    }
            except Exception as e:
                component_status['fallbacks'] = {'error': str(e)}
            
            # Threading status
            try:
                if self.threading_manager:
                    threading_status = self.threading_manager.get_status()
                    component_status['threading'] = {
                        'mode': threading_status.mode.name,
                        'health': threading_status.health.name,
                        'active_threads': threading_status.active_threads,
                        'completed_tasks': threading_status.completed_tasks,
                        'performance_score': threading_status.performance_score
                    }
            except Exception as e:
                component_status['threading'] = {'error': str(e)}
            
            # Performance summary
            performance_summary = {
                'overall_health': overall_health,
                'system_responsiveness': 1.0  # Default
            }
            
            # Add latest performance metrics
            for metric_name, metric_values in self.performance_metrics.items():
                if metric_values:
                    performance_summary[metric_name] = metric_values[-1]
            
            # Steam Deck status
            steam_deck_status = None
            if self.is_steam_deck:
                steam_deck_status = {
                    'cpu_temperature': self.thermal_monitor.get_cpu_temperature(),
                    'thermal_state': self.thermal_monitor.get_thermal_state(),
                    'model': PureSteamDeckDetector.get_steam_deck_model(),
                    'optimizations_active': True  # This would be calculated
                }
            
            snapshot = SystemSnapshot(
                timestamp=current_time,
                overall_health=overall_health,
                health_description=health_description,
                active_warnings=active_warnings,
                critical_issues=critical_issues,
                recommendations=recommendations,
                component_status=component_status,
                performance_summary=performance_summary,
                steam_deck_status=steam_deck_status,
                uptime_seconds=current_time - getattr(self, 'start_time', current_time)
            )
            
            self.system_snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating system snapshot: {e}")
            
            # Return minimal snapshot
            return SystemSnapshot(
                timestamp=time.time(),
                overall_health=0.0,
                health_description="Error",
                active_warnings=[],
                critical_issues=[f"Snapshot creation failed: {e}"],
                recommendations=["System diagnostics required"],
                component_status={'error': str(e)},
                performance_summary={'error': 1.0},
                steam_deck_status=None,
                uptime_seconds=0.0
            )

    def start_monitoring(self) -> bool:
        """Start continuous status monitoring"""
        if self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = True
            self.start_time = time.time()
            
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="StatusMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger = logging.getLogger(__name__)
            logger.info("Status monitoring started")
            return True
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to start status monitoring: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """Stop status monitoring"""
        if not self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10.0)
            
            logger = logging.getLogger(__name__)
            logger.info("Status monitoring stopped")
            return True
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error stopping status monitoring: {e}")
            return False

    def _monitoring_loop(self) -> None:
        """Main status monitoring loop"""
        logger = logging.getLogger(__name__)
        logger.info("Status monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Create system snapshot
                snapshot = self.create_system_snapshot()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for issues requiring user attention
                self._check_for_user_notifications()
                
                # Call status callbacks
                for callback in self.status_callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Error in status callback: {e}")
                
                # Sleep until next update
                time.sleep(self.status_update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retry
        
        logger.info("Status monitoring loop ended")

    def _update_performance_metrics(self) -> None:
        """Update performance metrics collection"""
        try:
            current_time = time.time()
            
            # System responsiveness test
            start_time = time.time()
            # Simple test
            test_data = [i for i in range(100)]
            test_sum = sum(test_data)
            responsiveness = time.time() - start_time
            
            self.performance_metrics['system_responsiveness'].append(1.0 / (responsiveness + 0.001))
            
            # Memory usage
            memory_usage = self._get_memory_usage()
            self.performance_metrics['memory_usage_mb'].append(memory_usage)
            
            # CPU temperature (Steam Deck)
            if self.is_steam_deck:
                cpu_temp = self.thermal_monitor.get_cpu_temperature()
                self.performance_metrics['cpu_temperature'].append(cpu_temp)
            
            # Threading performance
            if self.threading_manager:
                threading_status = self.threading_manager.get_status()
                self.performance_metrics['threading_performance'].append(threading_status.performance_score)
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error updating performance metrics: {e}")

    def _check_for_user_notifications(self) -> None:
        """Check for conditions that require user notification"""
        try:
            snapshot = self.system_snapshots[-1] if self.system_snapshots else None
            if not snapshot:
                return
            
            # Clear old notifications (keep last 10)
            self.user_notifications = self.user_notifications[-10:]
            
            # Check for high temperature on Steam Deck
            if self.is_steam_deck and snapshot.steam_deck_status:
                temp = snapshot.steam_deck_status.get('cpu_temperature', 50.0)
                if temp > 85.0:
                    self.user_notifications.append(StatusMessage(
                        timestamp=time.time(),
                        level=LogLevel.WARNING,
                        category=StatusCategory.STEAM_DECK_STATUS,
                        component="thermal_monitor",
                        title="High Temperature Warning",
                        description=f"🌡️ CPU temperature is {temp:.1f}°C",
                        is_user_visible=True,
                        requires_action=True,
                        action_suggestion="Reduce computational load or improve cooling"
                    ))
            
            # Check for low system health
            if snapshot.overall_health < 0.5:
                self.user_notifications.append(StatusMessage(
                    timestamp=time.time(),
                    level=LogLevel.ERROR,
                    category=StatusCategory.SYSTEM_HEALTH,
                    component="health_monitor",
                    title="Low System Health",
                    description=f"🏥 System health is {snapshot.overall_health:.1%}",
                    is_user_visible=True,
                    requires_action=True,
                    action_suggestion="Run system optimization or check for issues"
                ))
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking for user notifications: {e}")

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
                return 0.0

    def get_user_dashboard(self) -> Dict[str, Any]:
        """Get user-friendly dashboard information"""
        try:
            latest_snapshot = self.system_snapshots[-1] if self.system_snapshots else None
            
            if not latest_snapshot:
                return {
                    'status': 'Unknown',
                    'health': 0.0,
                    'message': 'System status unavailable'
                }
            
            # Determine status emoji and message
            if latest_snapshot.overall_health >= 0.9:
                status_emoji = "✅"
                status_text = "Excellent"
                status_color = "green"
            elif latest_snapshot.overall_health >= 0.7:
                status_emoji = "👍"
                status_text = "Good"
                status_color = "blue"
            elif latest_snapshot.overall_health >= 0.5:
                status_emoji = "⚠️"
                status_text = "Fair"
                status_color = "yellow"
            elif latest_snapshot.overall_health >= 0.3:
                status_emoji = "😟"
                status_text = "Poor"
                status_color = "orange"
            else:
                status_emoji = "🚨"
                status_text = "Critical"
                status_color = "red"
            
            dashboard = {
                'timestamp': latest_snapshot.timestamp,
                'status': {
                    'emoji': status_emoji,
                    'text': status_text,
                    'color': status_color,
                    'health_percentage': f"{latest_snapshot.overall_health:.0%}"
                },
                'overall_health': latest_snapshot.overall_health,
                'uptime': self._format_uptime(latest_snapshot.uptime_seconds),
                'components': {},
                'alerts': [],
                'recommendations': latest_snapshot.recommendations[:3],  # Top 3
                'steam_deck': None
            }
            
            # Component status
            for component, status in latest_snapshot.component_status.items():
                if isinstance(status, dict) and 'error' not in status:
                    if component == 'dependencies':
                        success_rate = status.get('success_rate', 0.0)
                        dashboard['components']['Dependencies'] = {
                            'status': '✅ Good' if success_rate > 0.8 else '⚠️ Issues' if success_rate > 0.5 else '❌ Poor',
                            'details': f"{status.get('successful_detections', 0)}/{status.get('total_dependencies', 0)} available"
                        }
                    elif component == 'threading':
                        health = status.get('health', 'UNKNOWN')
                        dashboard['components']['Threading'] = {
                            'status': '✅ Healthy' if health == 'HEALTHY' else '⚠️ Issues',
                            'details': f"{status.get('mode', 'Unknown')} mode, {status.get('active_threads', 0)} threads"
                        }
                    elif component == 'fallbacks':
                        perf_mult = status.get('average_performance_multiplier', 1.0)
                        dashboard['components']['Performance'] = {
                            'status': '✅ Optimal' if perf_mult > 2.0 else '👍 Good' if perf_mult > 1.5 else '⚠️ Fallback',
                            'details': f"{perf_mult:.1f}x performance"
                        }
            
            # Alerts
            for issue in latest_snapshot.critical_issues:
                dashboard['alerts'].append({
                    'level': 'critical',
                    'message': issue,
                    'icon': '🚨'
                })
            
            for warning in latest_snapshot.active_warnings[:2]:  # Top 2
                dashboard['alerts'].append({
                    'level': 'warning',
                    'message': warning,
                    'icon': '⚠️'
                })
            
            # Steam Deck specific info
            if latest_snapshot.steam_deck_status:
                temp = latest_snapshot.steam_deck_status.get('cpu_temperature', 0)
                thermal_state = latest_snapshot.steam_deck_status.get('thermal_state', 'unknown')
                
                temp_emoji = '🟢' if temp < 70 else '🟡' if temp < 80 else '🔴'
                
                dashboard['steam_deck'] = {
                    'model': latest_snapshot.steam_deck_status.get('model', 'Unknown'),
                    'temperature': {
                        'value': f"{temp:.1f}°C",
                        'status': thermal_state.title(),
                        'emoji': temp_emoji
                    },
                    'optimizations': '✅ Active' if latest_snapshot.steam_deck_status.get('optimizations_active') else '⚪ Inactive'
                }
            
            return dashboard
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating user dashboard: {e}")
            return {
                'status': {'emoji': '❌', 'text': 'Error', 'color': 'red'},
                'message': f'Dashboard error: {e}'
            }

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

    def get_recent_messages(self, 
                          limit: int = 20,
                          level_filter: Optional[LogLevel] = None,
                          category_filter: Optional[StatusCategory] = None) -> List[StatusMessage]:
        """Get recent status messages with optional filtering"""
        messages = list(self.status_messages)
        
        # Apply filters
        if level_filter:
            messages = [m for m in messages if m.level.value >= level_filter.value]
        
        if category_filter:
            messages = [m for m in messages if m.category == category_filter]
        
        # Sort by timestamp (newest first) and limit
        messages.sort(key=lambda m: m.timestamp, reverse=True)
        return messages[:limit]

    def export_status_report(self, 
                           filepath: Path, 
                           format: ReportFormat = ReportFormat.JSON,
                           include_history: bool = True) -> bool:
        """Export comprehensive status report"""
        try:
            logger = logging.getLogger(__name__)
            logger.info(f"Exporting status report to {filepath} in {format.name} format")
            
            # Gather report data
            latest_snapshot = self.system_snapshots[-1] if self.system_snapshots else None
            dashboard = self.get_user_dashboard()
            
            report_data = {
                'export_info': {
                    'timestamp': time.time(),
                    'format': format.name,
                    'system_info': {
                        'hostname': socket.gethostname(),
                        'platform': platform.system(),
                        'python_version': platform.python_version(),
                        'is_steam_deck': self.is_steam_deck
                    }
                },
                'dashboard': dashboard,
                'latest_snapshot': latest_snapshot._asdict() if latest_snapshot else None,
                'user_notifications': [
                    {
                        'timestamp': msg.timestamp,
                        'level': msg.level.name,
                        'category': msg.category.name,
                        'title': msg.title,
                        'description': msg.description,
                        'requires_action': msg.requires_action,
                        'action_suggestion': msg.action_suggestion
                    }
                    for msg in self.user_notifications
                ],
                'recent_messages': [
                    {
                        'timestamp': msg.timestamp,
                        'level': msg.level.name,
                        'category': msg.category.name,
                        'component': msg.component,
                        'title': msg.title,
                        'description': msg.description
                    }
                    for msg in self.get_recent_messages(limit=50)
                ]
            }
            
            if include_history:
                report_data['history'] = {
                    'snapshots': [s._asdict() for s in list(self.system_snapshots)],
                    'performance_metrics': {
                        name: list(values) 
                        for name, values in self.performance_metrics.items()
                    }
                }
            
            # Write report based on format
            if format == ReportFormat.JSON:
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            
            elif format == ReportFormat.MARKDOWN:
                self._export_markdown_report(filepath, report_data)
            
            elif format == ReportFormat.HTML:
                self._export_html_report(filepath, report_data)
            
            logger.info(f"Status report exported successfully to {filepath}")
            return True
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to export status report: {e}")
            return False

    def _export_markdown_report(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Export report in Markdown format"""
        with open(filepath, 'w') as f:
            f.write("# ML Shader Prediction System Status Report\n\n")
            
            # Dashboard summary
            dashboard = data['dashboard']
            f.write(f"## System Status: {dashboard['status']['emoji']} {dashboard['status']['text']}\n\n")
            f.write(f"- **Overall Health**: {dashboard.get('overall_health', 0):.1%}\n")
            f.write(f"- **Uptime**: {dashboard.get('uptime', 'Unknown')}\n")
            
            if dashboard.get('steam_deck'):
                f.write(f"- **Steam Deck**: {dashboard['steam_deck']['model']}\n")
                f.write(f"- **Temperature**: {dashboard['steam_deck']['temperature']['emoji']} {dashboard['steam_deck']['temperature']['value']} ({dashboard['steam_deck']['temperature']['status']})\n")
            
            f.write("\n")
            
            # Components
            if dashboard.get('components'):
                f.write("## Component Status\n\n")
                for name, status in dashboard['components'].items():
                    f.write(f"- **{name}**: {status['status']} - {status['details']}\n")
                f.write("\n")
            
            # Alerts
            if dashboard.get('alerts'):
                f.write("## Alerts\n\n")
                for alert in dashboard['alerts']:
                    f.write(f"- {alert['icon']} **{alert['level'].title()}**: {alert['message']}\n")
                f.write("\n")
            
            # Recommendations
            if dashboard.get('recommendations'):
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(dashboard['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            # Export timestamp
            export_time = datetime.datetime.fromtimestamp(data['export_info']['timestamp'])
            f.write(f"\n---\n*Report generated on {export_time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    def _export_html_report(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Export report in HTML format"""
        dashboard = data['dashboard']
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML Shader Prediction System Status</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
        .status-header {{ text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
        .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .status-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #667eea; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .alert.critical {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .alert.warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .health-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}
        .health-fill {{ height: 100%; background: linear-gradient(90deg, #f44336 0%, #ff9800 50%, #4caf50 100%); transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="status-header">
            <h1>{dashboard['status']['emoji']} System Status: {dashboard['status']['text']}</h1>
            <p>Overall Health: {dashboard.get('overall_health', 0):.0%} | Uptime: {dashboard.get('uptime', 'Unknown')}</p>
        </div>
        
        <div class="health-bar">
            <div class="health-fill" style="width: {dashboard.get('overall_health', 0):.0%}"></div>
        </div>
        
        <div class="status-grid">
        """
        
        # Add component cards
        for name, status in dashboard.get('components', {}).items():
            html_content += f"""
            <div class="status-card">
                <h3>{name}</h3>
                <p><strong>Status:</strong> {status['status']}</p>
                <p>{status['details']}</p>
            </div>
            """
        
        # Steam Deck info
        if dashboard.get('steam_deck'):
            steam = dashboard['steam_deck']
            html_content += f"""
            <div class="status-card">
                <h3>🎮 Steam Deck</h3>
                <p><strong>Model:</strong> {steam['model']}</p>
                <p><strong>Temperature:</strong> {steam['temperature']['emoji']} {steam['temperature']['value']} ({steam['temperature']['status']})</p>
                <p><strong>Optimizations:</strong> {steam['optimizations']}</p>
            </div>
            """
        
        html_content += """
        </div>
        """
        
        # Alerts
        if dashboard.get('alerts'):
            html_content += "<h2>Alerts</h2>"
            for alert in dashboard['alerts']:
                html_content += f'<div class="alert {alert["level"]}">{alert["icon"]} <strong>{alert["level"].title()}:</strong> {alert["message"]}</div>'
        
        # Recommendations
        if dashboard.get('recommendations'):
            html_content += "<h2>Recommendations</h2><ul>"
            for rec in dashboard['recommendations']:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
        
        # Footer
        export_time = datetime.datetime.fromtimestamp(data['export_info']['timestamp'])
        html_content += f"""
        <hr>
        <p><em>Report generated on {export_time.strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)

    def add_status_callback(self, callback: Callable) -> None:
        """Add status update callback"""
        self.status_callbacks.append(callback)

    def log_success(self, component: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a success message"""
        self._add_status_message(
            level=LogLevel.SUCCESS,
            category=StatusCategory.SYSTEM_HEALTH,
            component=component,
            title="Success",
            description=message,
            details=details
        )

    def log_progress(self, component: str, message: str, progress: float, details: Optional[Dict[str, Any]] = None):
        """Log a progress update"""
        self._add_status_message(
            level=LogLevel.INFO,
            category=StatusCategory.PERFORMANCE_METRICS,
            component=component,
            title="Progress Update",
            description=message,
            details=details,
            progress=progress
        )

    @contextmanager
    def progress_context(self, total_steps: int, description: str, component: str = "system"):
        """Context manager for progress tracking"""
        tracker = ProgressTracker(total_steps, description)
        
        def update_progress(steps: int = 1, status: str = ""):
            tracker.update(steps, status)
            self.log_progress(component, f"{description} - {status}", tracker.current_step / tracker.total_steps)
        
        tracker.update_progress = update_progress
        
        try:
            yield tracker
            tracker.complete()
            self.log_success(component, f"{description} completed successfully")
        except Exception as e:
            self._add_status_message(
                level=LogLevel.ERROR,
                category=StatusCategory.SYSTEM_HEALTH,
                component=component,
                title="Operation Failed",
                description=f"{description} failed: {e}"
            )
            raise


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_status_system: Optional[ComprehensiveStatusSystem] = None

def get_status_system() -> ComprehensiveStatusSystem:
    """Get or create global status system instance"""
    global _status_system
    if _status_system is None:
        _status_system = ComprehensiveStatusSystem()
    return _status_system

def log_success(component: str, message: str):
    """Quick success logging"""
    system = get_status_system()
    system.log_success(component, message)

def log_progress(component: str, message: str, progress: float):
    """Quick progress logging"""
    system = get_status_system()
    system.log_progress(component, message, progress)

def get_user_status() -> Dict[str, Any]:
    """Get user-friendly status summary"""
    system = get_status_system()
    return system.get_user_dashboard()

def export_status_report(filepath: str, format: str = "json") -> bool:
    """Export status report"""
    system = get_status_system()
    format_enum = ReportFormat.JSON if format.lower() == "json" else ReportFormat.MARKDOWN if format.lower() == "markdown" else ReportFormat.HTML
    return system.export_status_report(Path(filepath), format_enum)

@contextmanager
def progress_tracker(total_steps: int, description: str, component: str = "system"):
    """Context manager for progress tracking"""
    system = get_status_system()
    with system.progress_context(total_steps, description, component) as tracker:
        yield tracker


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n📊 Comprehensive Status System Test Suite")
    print("=" * 55)
    
    # Initialize status system
    status_system = ComprehensiveStatusSystem(
        log_level=LogLevel.INFO,
        console_colors=True,
        status_update_interval=2.0
    )
    
    print("\n🚀 System Initialization:")
    log_success("test", "Status system initialized successfully")
    
    # Test progress tracking
    print("\n📈 Testing Progress Tracking:")
    with progress_tracker(5, "Testing system components", "test") as tracker:
        for i in range(5):
            time.sleep(0.5)
            tracker.update_progress(1, f"Testing component {i+1}")
    
    # Test various log levels
    print("\n🏷️  Testing Log Levels:")
    logger = logging.getLogger("test.component")
    
    logger.log(LogLevel.TRACE.value, "This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.log(LogLevel.SUCCESS.value, "This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test user dashboard
    print("\n📱 User Dashboard:")
    dashboard = status_system.get_user_dashboard()
    print(f"Status: {dashboard['status']['emoji']} {dashboard['status']['text']} ({dashboard['status']['health_percentage']})")
    print(f"Uptime: {dashboard.get('uptime', 'Unknown')}")
    
    if dashboard.get('components'):
        print("Components:")
        for name, comp_status in dashboard['components'].items():
            print(f"  {name}: {comp_status['status']} - {comp_status['details']}")
    
    if dashboard.get('alerts'):
        print("Alerts:")
        for alert in dashboard['alerts']:
            print(f"  {alert['icon']} {alert['message']}")
    
    if dashboard.get('recommendations'):
        print("Recommendations:")
        for i, rec in enumerate(dashboard['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    if dashboard.get('steam_deck'):
        print("Steam Deck:")
        steam = dashboard['steam_deck']
        print(f"  Model: {steam['model']}")
        print(f"  Temperature: {steam['temperature']['emoji']} {steam['temperature']['value']} ({steam['temperature']['status']})")
        print(f"  Optimizations: {steam['optimizations']}")
    
    # Test recent messages
    print("\n📝 Recent Messages:")
    recent_messages = status_system.get_recent_messages(limit=5)
    for msg in recent_messages:
        timestamp = datetime.datetime.fromtimestamp(msg.timestamp).strftime('%H:%M:%S')
        print(f"  [{timestamp}] {msg.level.name} {msg.component}: {msg.description}")
    
    # Test export functionality
    print("\n💾 Testing Report Export:")
    
    # JSON export
    json_path = Path("/tmp/status_report.json")
    json_success = status_system.export_status_report(json_path, ReportFormat.JSON)
    print(f"JSON Export: {'✅' if json_success else '❌'} -> {json_path}")
    
    # Markdown export
    md_path = Path("/tmp/status_report.md")
    md_success = status_system.export_status_report(md_path, ReportFormat.MARKDOWN)
    print(f"Markdown Export: {'✅' if md_success else '❌'} -> {md_path}")
    
    # HTML export
    html_path = Path("/tmp/status_report.html")
    html_success = status_system.export_status_report(html_path, ReportFormat.HTML)
    print(f"HTML Export: {'✅' if html_success else '❌'} -> {html_path}")
    
    # Let monitoring run briefly
    print(f"\n⏱️  Monitoring for 5 seconds...")
    time.sleep(5)
    
    # Final status
    print(f"\n📊 Final Status:")
    final_dashboard = status_system.get_user_dashboard()
    print(f"  System Health: {final_dashboard['status']['emoji']} {final_dashboard['status']['text']} ({final_dashboard['status']['health_percentage']})")
    print(f"  Uptime: {final_dashboard.get('uptime', 'Unknown')}")
    print(f"  Messages Logged: {len(status_system.status_messages)}")
    print(f"  Snapshots Created: {len(status_system.system_snapshots)}")
    
    # Cleanup
    print(f"\n🧹 Stopping monitoring...")
    stop_success = status_system.stop_monitoring()
    print(f"Monitoring stopped: {'✅' if stop_success else '❌'}")
    
    print(f"\n✅ Comprehensive Status System test completed!")
    print(f"🎯 System provided detailed logging, monitoring, and reporting capabilities")