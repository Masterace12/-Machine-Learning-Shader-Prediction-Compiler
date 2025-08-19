#!/usr/bin/env python3
"""
Steam Deck Thread Diagnostics and Debugging Utilities
Comprehensive tools for diagnosing and fixing threading issues
"""

import os
import sys
import time
import threading
import traceback
import logging
import psutil
import gc
from typing import Dict, List, Optional, Any, Set, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
import weakref
import signal

class ThreadIssueType(Enum):
    """Types of threading issues that can be detected"""
    THREAD_LEAK = "thread_leak"
    DEADLOCK = "deadlock"
    HIGH_CPU = "high_cpu"
    MEMORY_LEAK = "memory_leak"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LIBRARY_CONFLICT = "library_conflict"
    STACK_OVERFLOW = "stack_overflow"

@dataclass
class ThreadIssue:
    """Details about a detected threading issue"""
    issue_type: ThreadIssueType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_threads: List[int] = field(default_factory=list)
    suggested_fix: str = ""
    detected_at: float = field(default_factory=time.time)
    stack_traces: Dict[int, List[str]] = field(default_factory=dict)

@dataclass
class ThreadInfo:
    """Comprehensive thread information"""
    thread_id: int
    name: str
    is_daemon: bool
    is_alive: bool
    cpu_time: float
    memory_mb: float
    created_at: float
    stack_trace: List[str]
    library_origin: Optional[str] = None

class SteamDeckThreadDiagnostics:
    """
    Comprehensive thread diagnostics system for Steam Deck
    
    Features:
    - Real-time thread monitoring
    - Issue detection and analysis
    - Performance bottleneck identification
    - Resource usage tracking
    - Threading library conflict detection
    - Automated fix suggestions
    """
    
    def __init__(self, enable_continuous_monitoring: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 2.0
        self._monitoring_thread = None
        
        # Issue tracking
        self.detected_issues: List[ThreadIssue] = []
        self.thread_history: deque = deque(maxlen=1000)
        self.resource_history: deque = deque(maxlen=500)
        
        # Thread tracking
        self._tracked_threads: Dict[int, ThreadInfo] = {}
        self._thread_start_times: Dict[int, float] = {}
        self._thread_libraries: Dict[int, str] = {}
        
        # System limits and thresholds (Steam Deck OLED optimized)
        self.max_threads_warning = 6   # Warn at 6 threads for Steam Deck
        self.max_threads_critical = 8  # Critical at 8 threads for Steam Deck
        self.cpu_time_threshold = 5.0  # 5 seconds
        self.memory_threshold_mb = 100
        
        # Library detection patterns
        self.library_patterns = {
            'lightgbm': ['lightgbm', 'lgb'],
            'sklearn': ['sklearn', 'joblib'],
            'numba': ['numba', 'llvmlite'],
            'numpy': ['numpy', 'openblas', 'mkl'],
            'threading': ['threading', 'concurrent.futures'],
            'asyncio': ['asyncio', 'uvloop'],
            'multiprocessing': ['multiprocessing', 'spawn'],
        }
        
        # Steam Deck specific detection
        self.is_steam_deck = os.path.exists("/home/deck")
        self.cpu_cores = os.cpu_count() or 8
        
        if enable_continuous_monitoring:
            self.start_monitoring()
        
        self.logger.info("Thread diagnostics system initialized")
    
    def start_monitoring(self):
        """Start continuous thread monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self._collect_thread_data()
                    self._analyze_threading_issues()
                    self._check_resource_limits()
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    time.sleep(self.monitoring_interval * 2)
        
        self._monitoring_thread = threading.Thread(
            target=monitoring_loop,
            name="thread_diagnostics",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Thread monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Thread monitoring stopped")
    
    def _collect_thread_data(self):
        """Collect comprehensive thread data"""
        current_time = time.time()
        active_threads = threading.enumerate()
        
        # Get system process info
        try:
            process = psutil.Process()
            process_threads = process.threads()
            thread_count = len(process_threads)
        except Exception:
            thread_count = len(active_threads)
        
        # Collect thread information
        thread_data = {}
        for thread in active_threads:
            try:
                thread_id = thread.ident or 0
                
                # Get stack trace
                frame = sys._current_frames().get(thread_id)
                stack_trace = []
                if frame:
                    stack_trace = traceback.format_stack(frame)
                
                # Determine library origin
                library_origin = self._identify_thread_library(stack_trace, thread.name)
                
                # Get CPU time (approximation)
                cpu_time = 0.0
                memory_mb = 0.0
                try:
                    # Try to get thread-specific metrics
                    for proc_thread in process_threads:
                        if proc_thread.id == thread_id:
                            cpu_time = proc_thread.user_time + proc_thread.system_time
                            break
                except Exception:
                    pass
                
                thread_info = ThreadInfo(
                    thread_id=thread_id,
                    name=thread.name or f"Thread-{thread_id}",
                    is_daemon=thread.daemon,
                    is_alive=thread.is_alive(),
                    cpu_time=cpu_time,
                    memory_mb=memory_mb,
                    created_at=self._thread_start_times.get(thread_id, current_time),
                    stack_trace=stack_trace,
                    library_origin=library_origin
                )
                
                thread_data[thread_id] = thread_info
                
                # Track new threads
                if thread_id not in self._tracked_threads:
                    self._thread_start_times[thread_id] = current_time
                    self.logger.debug(f"New thread detected: {thread.name} ({thread_id})")
                
            except Exception as e:
                self.logger.warning(f"Error collecting data for thread {thread.name}: {e}")
        
        # Update tracked threads
        self._tracked_threads = thread_data
        
        # Record historical data
        self.thread_history.append({
            'timestamp': current_time,
            'thread_count': thread_count,
            'active_threads': len(active_threads),
            'thread_data': thread_data
        })
        
        # Record resource data
        try:
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            self.resource_history.append({
                'timestamp': current_time,
                'memory_rss_mb': memory_info.rss / (1024 * 1024),
                'memory_vms_mb': memory_info.vms / (1024 * 1024),
                'cpu_percent': cpu_percent,
                'thread_count': thread_count
            })
        except Exception as e:
            self.logger.warning(f"Error collecting resource data: {e}")
    
    def _identify_thread_library(self, stack_trace: List[str], thread_name: str) -> Optional[str]:
        """Identify which library likely created this thread"""
        combined_text = " ".join(stack_trace + [thread_name]).lower()
        
        for library, patterns in self.library_patterns.items():
            for pattern in patterns:
                if pattern in combined_text:
                    return library
        
        return None
    
    def _analyze_threading_issues(self):
        """Analyze current threading state for issues"""
        if not self.thread_history:
            return
        
        current_data = self.thread_history[-1]
        current_time = time.time()
        
        # Check for thread leaks
        thread_count = current_data['thread_count']
        if thread_count > self.max_threads_critical:
            self._report_issue(ThreadIssue(
                issue_type=ThreadIssueType.THREAD_LEAK,
                severity="critical",
                description=f"Critical thread count: {thread_count} threads (limit: {self.max_threads_critical})",
                suggested_fix="Review thread creation and ensure proper cleanup. Check for unbounded thread pools."
            ))
        elif thread_count > self.max_threads_warning:
            self._report_issue(ThreadIssue(
                issue_type=ThreadIssueType.THREAD_LEAK,
                severity="high",
                description=f"High thread count: {thread_count} threads (warning: {self.max_threads_warning})",
                suggested_fix="Monitor thread creation patterns and implement thread pooling."
            ))
        
        # Check for long-running threads
        for thread_id, thread_info in self._tracked_threads.items():
            thread_age = current_time - thread_info.created_at
            
            if thread_info.cpu_time > self.cpu_time_threshold:
                self._report_issue(ThreadIssue(
                    issue_type=ThreadIssueType.HIGH_CPU,
                    severity="medium",
                    description=f"Thread {thread_info.name} has high CPU time: {thread_info.cpu_time:.2f}s",
                    affected_threads=[thread_id],
                    stack_traces={thread_id: thread_info.stack_trace},
                    suggested_fix="Review thread workload and consider optimization or break-up into smaller tasks."
                ))
        
        # Check for library conflicts
        self._check_library_conflicts()
    
    def _check_library_conflicts(self):
        """Check for known library threading conflicts"""
        library_threads = defaultdict(list)
        
        for thread_id, thread_info in self._tracked_threads.items():
            if thread_info.library_origin:
                library_threads[thread_info.library_origin].append(thread_id)
        
        # Check for excessive ML library threading
        ml_libraries = ['lightgbm', 'sklearn', 'numba', 'numpy']
        total_ml_threads = sum(len(library_threads[lib]) for lib in ml_libraries if lib in library_threads)
        
        if total_ml_threads > 2:  # Steam Deck OLED shouldn't have more than 2 ML threads
            self._report_issue(ThreadIssue(
                issue_type=ThreadIssueType.LIBRARY_CONFLICT,
                severity="high",
                description=f"Excessive ML library threads: {total_ml_threads} (recommended: ≤2 for Steam Deck OLED)",
                suggested_fix="Configure ML libraries with proper thread limits (OMP_NUM_THREADS=1, LIGHTGBM_NUM_THREADS=1, etc.)"
            ))
        
        # Check for specific library conflicts
        if 'lightgbm' in library_threads and 'sklearn' in library_threads:
            lgb_count = len(library_threads['lightgbm'])
            sklearn_count = len(library_threads['sklearn'])
            if lgb_count + sklearn_count > 2:  # Steam Deck OLED: max 2 total ML threads
                self._report_issue(ThreadIssue(
                    issue_type=ThreadIssueType.LIBRARY_CONFLICT,
                    severity="high",
                    description=f"LightGBM ({lgb_count}) + sklearn ({sklearn_count}) thread conflict detected (Steam Deck max: 2)",
                    suggested_fix="Set LIGHTGBM_NUM_THREADS=1 and sklearn n_jobs=1 for Steam Deck OLED"
                ))
    
    def _check_resource_limits(self):
        """Check for resource limit violations"""
        if not self.resource_history:
            return
        
        recent_data = list(self.resource_history)[-10:]  # Last 10 samples
        
        # Check memory growth
        if len(recent_data) >= 5:
            memory_trend = [d['memory_rss_mb'] for d in recent_data]
            memory_growth = memory_trend[-1] - memory_trend[0]
            
            if memory_growth > 100:  # 100MB growth in recent samples
                self._report_issue(ThreadIssue(
                    issue_type=ThreadIssueType.MEMORY_LEAK,
                    severity="high",
                    description=f"Memory growth detected: +{memory_growth:.1f}MB in recent monitoring",
                    suggested_fix="Check for memory leaks in thread operations and implement proper cleanup"
                ))
        
        # Check current memory usage (Steam Deck has 16GB shared with GPU)
        current_memory = recent_data[-1]['memory_rss_mb']
        if current_memory > 500:  # 500MB threshold for Steam Deck (more conservative)
            self._report_issue(ThreadIssue(
                issue_type=ThreadIssueType.RESOURCE_EXHAUSTION,
                severity="critical",
                description=f"High memory usage: {current_memory:.1f}MB (Steam Deck threshold: 500MB)",
                suggested_fix="Reduce thread count or implement memory-efficient algorithms for Steam Deck"
            ))
    
    def _report_issue(self, issue: ThreadIssue):
        """Report a detected issue (avoid duplicates)"""
        # Check if similar issue already reported recently
        recent_threshold = time.time() - 300  # 5 minutes
        for existing_issue in self.detected_issues:
            if (existing_issue.issue_type == issue.issue_type and
                existing_issue.detected_at > recent_threshold):
                return  # Don't report duplicate
        
        self.detected_issues.append(issue)
        
        # Log based on severity
        log_msg = f"Thread issue detected: {issue.description}"
        if issue.severity == "critical":
            self.logger.critical(log_msg)
        elif issue.severity == "high":
            self.logger.error(log_msg)
        elif issue.severity == "medium":
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
        
        if issue.suggested_fix:
            self.logger.info(f"Suggested fix: {issue.suggested_fix}")
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        if not self.thread_history:
            return {"error": "No data collected yet"}
        
        current_data = self.thread_history[-1]
        current_time = time.time()
        
        # Basic statistics
        thread_count = current_data['thread_count']
        active_threads = current_data['active_threads']
        
        # Library breakdown
        library_breakdown = defaultdict(int)
        for thread_info in self._tracked_threads.values():
            lib = thread_info.library_origin or "unknown"
            library_breakdown[lib] += 1
        
        # Recent issues
        recent_issues = [
            issue for issue in self.detected_issues
            if current_time - issue.detected_at < 3600  # Last hour
        ]
        
        # Performance metrics
        if self.resource_history:
            recent_memory = [d['memory_rss_mb'] for d in self.resource_history]
            avg_memory = sum(recent_memory) / len(recent_memory)
            max_memory = max(recent_memory)
        else:
            avg_memory = max_memory = 0
        
        # Thread age analysis
        thread_ages = []
        for thread_info in self._tracked_threads.values():
            age = current_time - thread_info.created_at
            thread_ages.append(age)
        
        avg_thread_age = sum(thread_ages) / len(thread_ages) if thread_ages else 0
        
        return {
            'timestamp': current_time,
            'system_info': {
                'is_steam_deck': self.is_steam_deck,
                'cpu_cores': self.cpu_cores,
                'total_threads': thread_count,
                'active_threads': active_threads,
            },
            'threading_analysis': {
                'library_breakdown': dict(library_breakdown),
                'average_thread_age_seconds': avg_thread_age,
                'threads_by_daemon_status': {
                    'daemon': sum(1 for t in self._tracked_threads.values() if t.is_daemon),
                    'non_daemon': sum(1 for t in self._tracked_threads.values() if not t.is_daemon)
                }
            },
            'performance_metrics': {
                'average_memory_mb': avg_memory,
                'peak_memory_mb': max_memory,
                'monitoring_duration_minutes': (current_time - self.thread_history[0]['timestamp']) / 60 if self.thread_history else 0
            },
            'issues': {
                'total_detected': len(self.detected_issues),
                'recent_issues': len(recent_issues),
                'by_severity': {
                    'critical': len([i for i in recent_issues if i.severity == 'critical']),
                    'high': len([i for i in recent_issues if i.severity == 'high']),
                    'medium': len([i for i in recent_issues if i.severity == 'medium']),
                    'low': len([i for i in recent_issues if i.severity == 'low'])
                },
                'recent_issue_details': [
                    {
                        'type': issue.issue_type.value,
                        'severity': issue.severity,
                        'description': issue.description,
                        'suggested_fix': issue.suggested_fix
                    }
                    for issue in recent_issues[-5:]  # Last 5 issues
                ]
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if not self.thread_history:
            return ["Start monitoring to collect data for analysis"]
        
        current_data = self.thread_history[-1]
        thread_count = current_data['thread_count']
        
        # Thread count recommendations for Steam Deck OLED
        if thread_count > self.max_threads_critical:
            recommendations.append("CRITICAL: Steam Deck OLED thread limit exceeded - system instability likely")
            recommendations.append("Immediately set OMP_NUM_THREADS=1, LIGHTGBM_NUM_THREADS=1")
            recommendations.append("Stop all non-essential background processes")
            recommendations.append("Restart application with proper environment variables")
        elif thread_count > self.max_threads_warning:
            recommendations.append("Steam Deck OLED thread warning: Reduce threads to prevent 'can't start new thread' errors")
            recommendations.append("Set all ML library thread limits to 1 (OMP_NUM_THREADS=1, etc.)")
            recommendations.append("Enable gaming mode detection to reduce threads during gaming")
        
        # Library-specific recommendations
        library_threads = defaultdict(int)
        for thread_info in self._tracked_threads.values():
            if thread_info.library_origin:
                library_threads[thread_info.library_origin] += 1
        
        if library_threads.get('lightgbm', 0) > 1:
            recommendations.append("Configure LightGBM with LIGHTGBM_NUM_THREADS=1 for Steam Deck OLED")
        
        if library_threads.get('sklearn', 0) > 1:
            recommendations.append("Configure scikit-learn with n_jobs=1 for Steam Deck OLED")
        
        if sum(library_threads.get(lib, 0) for lib in ['numpy', 'numba']) > 1:
            recommendations.append("Set OMP_NUM_THREADS=1 and NUMBA_NUM_THREADS=1 for Steam Deck OLED")
        
        # Steam Deck OLED specific recommendations
        if self.is_steam_deck:
            recommendations.append("Steam Deck OLED detected: Keep total threads ≤6 for optimal performance")
            recommendations.append("Set all environment variables: export OMP_NUM_THREADS=1 LIGHTGBM_NUM_THREADS=1 MKL_NUM_THREADS=1")
            recommendations.append("Enable thermal-aware thread scaling for battery life")
            recommendations.append("Consider running in gaming mode during gameplay to minimize background threads")
        
        return recommendations
    
    def apply_fixes(self, auto_fix: bool = False) -> Dict[str, Any]:
        """Apply automatic fixes for detected issues"""
        fixes_applied = []
        fixes_failed = []
        
        if not auto_fix:
            return {
                "message": "Auto-fix disabled. Use apply_fixes(auto_fix=True) to enable.",
                "available_fixes": len([i for i in self.detected_issues if i.suggested_fix])
            }
        
        for issue in self.detected_issues:
            if not issue.suggested_fix:
                continue
            
            try:
                fix_applied = False
                
                # Apply specific fixes based on issue type
                if issue.issue_type == ThreadIssueType.LIBRARY_CONFLICT:
                    if "OMP_NUM_THREADS" in issue.suggested_fix:
                        os.environ['OMP_NUM_THREADS'] = '2'
                        os.environ['MKL_NUM_THREADS'] = '2'
                        os.environ['OPENBLAS_NUM_THREADS'] = '2'
                        fix_applied = True
                
                elif issue.issue_type == ThreadIssueType.MEMORY_LEAK:
                    # Force garbage collection
                    gc.collect()
                    fix_applied = True
                
                if fix_applied:
                    fixes_applied.append({
                        'issue_type': issue.issue_type.value,
                        'fix_description': issue.suggested_fix
                    })
                
            except Exception as e:
                fixes_failed.append({
                    'issue_type': issue.issue_type.value,
                    'error': str(e)
                })
        
        return {
            'fixes_applied': fixes_applied,
            'fixes_failed': fixes_failed,
            'total_attempted': len(fixes_applied) + len(fixes_failed)
        }


# Global diagnostics instance
_global_diagnostics = None


def get_thread_diagnostics() -> SteamDeckThreadDiagnostics:
    """Get or create global thread diagnostics instance"""
    global _global_diagnostics
    if _global_diagnostics is None:
        _global_diagnostics = SteamDeckThreadDiagnostics()
    return _global_diagnostics


def diagnose_threading_issues() -> Dict[str, Any]:
    """Quick threading issue diagnosis"""
    diagnostics = get_thread_diagnostics()
    return diagnostics.get_diagnostic_report()


def fix_threading_issues(auto_fix: bool = False) -> Dict[str, Any]:
    """Attempt to fix detected threading issues"""
    diagnostics = get_thread_diagnostics()
    return diagnostics.apply_fixes(auto_fix=auto_fix)


if __name__ == "__main__":
    # Test thread diagnostics
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Steam Deck Thread Diagnostics Test")
    print("=" * 50)
    
    # Initialize diagnostics
    diagnostics = SteamDeckThreadDiagnostics()
    
    try:
        # Collect some data
        print("Collecting thread data...")
        time.sleep(3)
        
        # Generate report
        report = diagnostics.get_diagnostic_report()
        
        print(f"\nDiagnostic Report:")
        print(f"  Total threads: {report['system_info']['total_threads']}")
        print(f"  Active threads: {report['system_info']['active_threads']}")
        print(f"  Library breakdown: {report['threading_analysis']['library_breakdown']}")
        print(f"  Recent issues: {report['issues']['recent_issues']}")
        
        if report['issues']['recent_issue_details']:
            print(f"\nRecent Issues:")
            for issue in report['issues']['recent_issue_details']:
                print(f"  • {issue['severity'].upper()}: {issue['description']}")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("\n✅ Thread diagnostics test completed")
        
    except Exception as e:
        print(f"❌ Thread diagnostics test failed: {e}")
    
    finally:
        diagnostics.stop_monitoring()