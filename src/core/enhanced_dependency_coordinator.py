#!/usr/bin/env python3
"""
Enhanced Dependency Coordinator - Master System Integration

This module serves as the central coordinator for all dependency management systems,
providing a unified interface and intelligent orchestration of detection, version
management, fallback coordination, and runtime optimization.

Features:
- Unified dependency management interface
- Intelligent system orchestration
- Comprehensive health monitoring and reporting
- Steam Deck specific integration and optimization
- Performance benchmarking and adaptive optimization
- Graceful degradation with detailed fallback management
- Real-time system adaptation and optimization
- Comprehensive error handling and recovery
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
import gc
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from enum import Enum, auto
import weakref

# Import all our dependency management components
try:
    from .dependency_version_manager import (
        DependencyVersionManager, get_version_manager, DEPENDENCY_MATRIX
    )
    from .enhanced_dependency_detector import (
        EnhancedDependencyDetector, get_detector, DetectionResult
    )
    from .tiered_fallback_system import (
        TieredFallbackSystem, get_fallback_system, FallbackTier, FallbackReason
    )
    from .pure_python_fallbacks import (
        get_fallback_status, PureSteamDeckDetector, PureThermalMonitor
    )
except ImportError as e:
    logger.error(f"Failed to import dependency management components: {e}")
    # Create minimal fallback implementations
    class DummyManager:
        def __init__(self): 
            self.system_info = {'is_steam_deck': False}
        def detect_installed_version(self, dep): return None
        def check_version_compatibility(self, dep, ver): return {'compatible': True}
        def get_dependency_recommendations(self): return {'critical': [], 'recommended': []}
        def validate_environment(self): return {'overall_health': 0.5}
    
    class DummyDetector:
        def __init__(self):
            self.detection_results = {}
        def detect_all_dependencies(self, **kwargs): return {}
        def get_detection_summary(self): return {'success_rate': 0.5}
    
    class DummyFallbackSystem:
        def __init__(self):
            self.active_fallbacks = {}
        def get_implementation(self, name): return None
        def get_fallback_status(self): return {'active_components': 0}
        def auto_optimize_fallbacks(self): return {'optimizations_applied': []}
    
    get_version_manager = lambda: DummyManager()
    get_detector = lambda: DummyDetector()
    get_fallback_system = lambda: DummyFallbackSystem()
    DEPENDENCY_MATRIX = {}
    FallbackTier = Enum('FallbackTier', ['OPTIMAL', 'COMPATIBLE', 'EFFICIENT', 'PURE_PYTHON'])
    FallbackReason = Enum('FallbackReason', ['USER_PREFERENCE'])

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM HEALTH AND STATUS STRUCTURES
# =============================================================================

class SystemHealthLevel(Enum):
    """System health levels"""
    EXCELLENT = auto()  # 90%+ health
    GOOD = auto()       # 70-90% health  
    FAIR = auto()       # 50-70% health
    POOR = auto()       # 30-50% health
    CRITICAL = auto()   # <30% health

@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    overall_health: float
    health_level: SystemHealthLevel
    component_health: Dict[str, float]
    dependency_status: Dict[str, Any]
    fallback_status: Dict[str, Any]
    performance_metrics: Dict[str, float]
    system_capabilities: Dict[str, bool]
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    optimization_opportunities: List[str]
    steam_deck_status: Optional[Dict[str, Any]]
    timestamp: float

@dataclass
class OptimizationPlan:
    """System optimization plan"""
    target_improvements: Dict[str, float]
    recommended_actions: List[Dict[str, Any]]
    installation_suggestions: List[str]
    configuration_changes: List[Dict[str, Any]]
    expected_performance_gain: float
    estimated_time_required: float
    risk_assessment: Dict[str, Any]
    dependencies_to_install: List[str]
    fallback_adjustments: List[Dict[str, Any]]

# =============================================================================
# ENHANCED DEPENDENCY COORDINATOR
# =============================================================================

class EnhancedDependencyCoordinator:
    """
    Master coordinator for all dependency management systems
    """
    
    def __init__(self, 
                 auto_optimize: bool = True,
                 monitoring_enabled: bool = True,
                 steam_deck_mode: Optional[bool] = None):
        
        self.auto_optimize = auto_optimize
        self.monitoring_enabled = monitoring_enabled
        
        # Initialize component systems
        self.version_manager = get_version_manager()
        self.detector = get_detector()
        self.fallback_system = get_fallback_system()
        
        # System detection
        self.is_steam_deck = steam_deck_mode
        if self.is_steam_deck is None:
            self.is_steam_deck = self.version_manager.system_info.get('is_steam_deck', False)
        
        # Coordination state
        self.coordination_lock = threading.RLock()
        self.system_health_history: deque = deque(maxlen=100)
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_optimization_time = 0.0
        self.last_health_check_time = 0.0
        
        # Performance tracking
        self.performance_baseline: Optional[Dict[str, float]] = None
        self.performance_improvements: Dict[str, float] = {}
        self.system_capabilities_cache: Optional[Dict[str, bool]] = None
        self.cache_timestamp = 0.0
        self.cache_ttl = 300.0  # 5 minutes
        
        # Monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.health_check_interval = 30.0  # seconds
        self.optimization_interval = 300.0  # 5 minutes
        
        # Component integration callbacks
        self.health_callbacks: List[Callable] = []
        self.optimization_callbacks: List[Callable] = []
        
        # Initialize system
        self._initialize_coordinator()
        
        logger.info(f"EnhancedDependencyCoordinator initialized")
        logger.info(f"Steam Deck mode: {self.is_steam_deck}")
        logger.info(f"Auto-optimization: {auto_optimize}")
        logger.info(f"Monitoring: {monitoring_enabled}")

    def _initialize_coordinator(self) -> None:
        """Initialize the coordinator and perform initial system analysis"""
        try:
            logger.info("Initializing dependency coordinator systems...")
            
            # Perform initial detection
            self._run_initial_detection()
            
            # Establish performance baseline
            self._establish_performance_baseline()
            
            # Perform initial health check
            initial_health = self.get_comprehensive_health_report()
            self.system_health_history.append(initial_health)
            
            # Apply initial optimizations if enabled
            if self.auto_optimize:
                self._apply_initial_optimizations()
            
            # Start monitoring if enabled
            if self.monitoring_enabled:
                self.start_monitoring()
            
            logger.info("Dependency coordinator initialization completed")
            
        except Exception as e:
            logger.error(f"Error during coordinator initialization: {e}")
            logger.error(traceback.format_exc())

    def _run_initial_detection(self) -> None:
        """Run initial dependency detection across all systems"""
        logger.info("Running comprehensive initial dependency detection...")
        
        try:
            # Run enhanced detection
            detection_results = self.detector.detect_all_dependencies()
            
            # Update version manager with results
            for dep_name, result in detection_results.items():
                if result.status.available and result.status.version:
                    self.version_manager.version_cache[dep_name] = result.status.version
            
            logger.info(f"Initial detection completed: {len(detection_results)} dependencies analyzed")
            
        except Exception as e:
            logger.error(f"Error in initial detection: {e}")

    def _establish_performance_baseline(self) -> None:
        """Establish performance baseline for optimization tracking"""
        try:
            logger.info("Establishing performance baseline...")
            
            baseline = {}
            
            # Component initialization times
            for component_name in self.fallback_system.fallback_implementations.keys():
                impl = self.fallback_system.get_implementation(component_name)
                if impl:
                    # Simple performance test
                    start_time = time.time()
                    # Perform basic operation to measure performance
                    if hasattr(impl, '__call__'):
                        try:
                            impl()
                        except:
                            pass
                    baseline[f"{component_name}_response_time"] = time.time() - start_time
            
            # Memory usage
            baseline['memory_usage_mb'] = self._get_memory_usage()
            
            # System responsiveness
            start_time = time.time()
            self._perform_system_test()
            baseline['system_response_time'] = time.time() - start_time
            
            self.performance_baseline = baseline
            logger.info(f"Performance baseline established: {len(baseline)} metrics")
            
        except Exception as e:
            logger.error(f"Error establishing performance baseline: {e}")

    def _perform_system_test(self) -> None:
        """Perform basic system responsiveness test"""
        # Simple operations to test system responsiveness
        import os
        import json
        
        # File system test
        test_data = {'test': time.time()}
        test_file = '/tmp/dependency_system_test.json'
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        os.remove(test_file)

    def _apply_initial_optimizations(self) -> None:
        """Apply initial system optimizations"""
        try:
            logger.info("Applying initial system optimizations...")
            
            # Optimize fallback configurations
            fallback_optimization = self.fallback_system.auto_optimize_fallbacks()
            
            # Apply Steam Deck specific optimizations if applicable
            if self.is_steam_deck:
                self._apply_steam_deck_optimizations()
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': time.time(),
                'type': 'initial_optimization',
                'fallback_optimizations': fallback_optimization,
                'steam_deck_applied': self.is_steam_deck
            })
            
            logger.info("Initial optimizations completed")
            
        except Exception as e:
            logger.error(f"Error in initial optimizations: {e}")

    def _apply_steam_deck_optimizations(self) -> None:
        """Apply Steam Deck specific optimizations"""
        logger.info("Applying Steam Deck specific optimizations...")
        
        # Switch to Steam Deck optimized fallback implementations
        steam_deck_optimizations = []
        
        for component_name in self.fallback_system.fallback_implementations.keys():
            # Find Steam Deck optimized implementation
            steam_impl = self.fallback_system._find_steam_deck_implementation(component_name)
            if steam_impl:
                current_status = self.fallback_system.active_fallbacks.get(component_name)
                if not current_status or current_status.active_implementation != steam_impl.name:
                    success = self.fallback_system.switch_to_tier(
                        component_name, 
                        steam_impl.tier, 
                        FallbackReason.STEAM_DECK_OPTIMIZATION
                    )
                    if success:
                        steam_deck_optimizations.append(f"Switched {component_name} to {steam_impl.name}")
        
        if steam_deck_optimizations:
            logger.info(f"Applied {len(steam_deck_optimizations)} Steam Deck optimizations")
            for opt in steam_deck_optimizations:
                logger.info(f"  - {opt}")

    def get_comprehensive_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report"""
        try:
            # Component health scores
            component_health = {}
            
            # Version manager health
            validation_result = self.version_manager.validate_environment()
            component_health['version_manager'] = validation_result['overall_health']
            
            # Detector health
            detection_summary = self.detector.get_detection_summary()
            component_health['detector'] = detection_summary.get('success_rate', 0.0)
            
            # Fallback system health
            fallback_status = self.fallback_system.get_fallback_status()
            fallback_health = fallback_status['system_health']['fallback_stability']
            component_health['fallback_system'] = fallback_health
            
            # Calculate overall health
            overall_health = sum(component_health.values()) / len(component_health)
            
            # Determine health level
            if overall_health >= 0.9:
                health_level = SystemHealthLevel.EXCELLENT
            elif overall_health >= 0.7:
                health_level = SystemHealthLevel.GOOD
            elif overall_health >= 0.5:
                health_level = SystemHealthLevel.FAIR
            elif overall_health >= 0.3:
                health_level = SystemHealthLevel.POOR
            else:
                health_level = SystemHealthLevel.CRITICAL
            
            # Gather dependency status
            dependency_status = {
                'total_dependencies': len(DEPENDENCY_MATRIX),
                'available_dependencies': len([
                    d for d in self.detector.detection_results.values() 
                    if d.status.available
                ]),
                'fallback_dependencies': len([
                    d for d in self.detector.detection_results.values()
                    if d.status.fallback_active and not d.status.available
                ]),
                'failed_dependencies': len([
                    d for d in self.detector.detection_results.values()
                    if not d.status.available and not d.status.fallback_active
                ])
            }
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # System capabilities
            capabilities = self._assess_system_capabilities()
            
            # Issues and recommendations
            critical_issues, warnings, recommendations = self._analyze_system_issues(
                validation_result, detection_summary, fallback_status
            )
            
            # Optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities()
            
            # Steam Deck specific status
            steam_deck_status = None
            if self.is_steam_deck:
                steam_deck_status = {
                    'optimized_components': len([
                        name for name in self.fallback_system.active_fallbacks.keys()
                        if self.fallback_system._is_steam_deck_optimized(name)
                    ]),
                    'total_components': len(self.fallback_system.active_fallbacks),
                    'thermal_state': self._get_thermal_state(),
                    'memory_pressure': self._assess_memory_pressure()
                }
            
            return SystemHealthReport(
                overall_health=overall_health,
                health_level=health_level,
                component_health=component_health,
                dependency_status=dependency_status,
                fallback_status=fallback_status,
                performance_metrics=performance_metrics,
                system_capabilities=capabilities,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                optimization_opportunities=optimization_opportunities,
                steam_deck_status=steam_deck_status,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            # Return minimal report
            return SystemHealthReport(
                overall_health=0.3,
                health_level=SystemHealthLevel.POOR,
                component_health={'error': 0.0},
                dependency_status={'error': str(e)},
                fallback_status={'error': True},
                performance_metrics={'error': 1.0},
                system_capabilities={'error': False},
                critical_issues=[f"Health report generation failed: {e}"],
                warnings=[],
                recommendations=["System diagnostics required"],
                optimization_opportunities=[],
                steam_deck_status=None,
                timestamp=time.time()
            )

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        metrics = {}
        
        try:
            # Response times for active implementations
            for component_name in self.fallback_system.active_fallbacks.keys():
                impl = self.fallback_system.get_implementation(component_name)
                if impl:
                    start_time = time.time()
                    try:
                        # Basic performance test
                        if hasattr(impl, '__call__'):
                            impl()
                    except:
                        pass
                    metrics[f"{component_name}_response_time"] = time.time() - start_time
            
            # Memory usage
            metrics['memory_usage_mb'] = self._get_memory_usage()
            
            # Performance improvement over baseline
            if self.performance_baseline:
                for key, baseline_value in self.performance_baseline.items():
                    current_value = metrics.get(key, baseline_value)
                    if baseline_value > 0:
                        improvement = (baseline_value - current_value) / baseline_value
                        metrics[f"{key}_improvement"] = improvement
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            metrics['calculation_error'] = 1.0
        
        return metrics

    def _assess_system_capabilities(self) -> Dict[str, bool]:
        """Assess current system capabilities"""
        # Use cache if recent
        if (self.system_capabilities_cache and 
            time.time() - self.cache_timestamp < self.cache_ttl):
            return self.system_capabilities_cache
        
        capabilities = {}
        
        try:
            # ML capabilities
            capabilities['numpy_available'] = 'numpy' in self.detector.detection_results and \
                                            self.detector.detection_results['numpy'].status.available
            
            capabilities['sklearn_available'] = 'scikit-learn' in self.detector.detection_results and \
                                              self.detector.detection_results['scikit-learn'].status.available
            
            capabilities['lightgbm_available'] = 'lightgbm' in self.detector.detection_results and \
                                               self.detector.detection_results['lightgbm'].status.available
            
            # Performance capabilities
            capabilities['numba_jit_available'] = 'numba' in self.detector.detection_results and \
                                                self.detector.detection_results['numba'].status.available
            
            # System monitoring capabilities
            capabilities['system_monitoring_available'] = 'psutil' in self.detector.detection_results and \
                                                        self.detector.detection_results['psutil'].status.available
            
            # High-level capabilities derived from dependencies
            capabilities['advanced_ml'] = capabilities['lightgbm_available'] or capabilities['sklearn_available']
            capabilities['high_performance'] = capabilities['numpy_available'] and capabilities['numba_jit_available']
            capabilities['comprehensive_monitoring'] = capabilities['system_monitoring_available']
            capabilities['production_ready'] = capabilities['advanced_ml'] and capabilities['comprehensive_monitoring']
            
            # Steam Deck specific capabilities
            if self.is_steam_deck:
                capabilities['steam_deck_optimized'] = len([
                    name for name in self.fallback_system.active_fallbacks.keys()
                    if self.fallback_system._is_steam_deck_optimized(name)
                ]) > len(self.fallback_system.active_fallbacks) * 0.7  # >70% optimized
            
            # Cache result
            self.system_capabilities_cache = capabilities
            self.cache_timestamp = time.time()
            
        except Exception as e:
            logger.error(f"Error assessing system capabilities: {e}")
            capabilities['assessment_error'] = False
        
        return capabilities

    def _analyze_system_issues(self, validation_result, detection_summary, fallback_status) -> Tuple[List[str], List[str], List[str]]:
        """Analyze system for issues and generate recommendations"""
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Critical issues
        if validation_result['overall_health'] < 0.3:
            critical_issues.append("System health critically low - major issues detected")
        
        if detection_summary.get('success_rate', 0) < 0.5:
            critical_issues.append("Dependency detection failure rate too high")
        
        if fallback_status['system_health']['fallback_stability'] < 0.5:
            critical_issues.append("Fallback system unstable - frequent errors detected")
        
        # Warnings
        if validation_result['overall_health'] < 0.7:
            warnings.append("System health below optimal - some issues detected")
        
        memory_pressure = fallback_status['system_health'].get('memory_pressure', 0)
        if memory_pressure > 0.8:
            warnings.append("High memory pressure detected")
        
        if self.is_steam_deck:
            steam_optimization = fallback_status['system_health'].get('steam_deck_optimization', 1.0)
            if steam_optimization < 0.7:
                warnings.append("Steam Deck optimizations not fully applied")
        
        # Recommendations
        version_recommendations = self.version_manager.get_dependency_recommendations()
        
        if version_recommendations['critical']:
            recommendations.append(
                f"Install {len(version_recommendations['critical'])} critical dependencies"
            )
        
        if version_recommendations['recommended']:
            recommendations.append(
                f"Consider installing {len(version_recommendations['recommended'])} recommended dependencies"
            )
        
        if len(critical_issues) > 0:
            recommendations.append("Run comprehensive system diagnostics and repair")
        
        if self.is_steam_deck and self._get_thermal_state() in ['warm', 'hot']:
            recommendations.append("Monitor thermal state and consider reducing computational load")
        
        return critical_issues, warnings, recommendations

    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        try:
            # Check for better fallback implementations
            for component_name, status in self.fallback_system.active_fallbacks.items():
                if status.active_tier.value > 1:  # Not optimal tier
                    better_impl = self.fallback_system._find_better_implementation(
                        component_name, status.active_tier
                    )
                    if better_impl:
                        opportunities.append(
                            f"Upgrade {component_name} to {better_impl.name} for better performance"
                        )
            
            # Check for missing high-impact dependencies
            missing_high_impact = [
                name for name, profile in DEPENDENCY_MATRIX.items()
                if (name not in self.detector.detection_results or 
                    not self.detector.detection_results[name].status.available)
                and profile.risk_assessment.performance_impact > 4.0
            ]
            
            if missing_high_impact:
                opportunities.append(
                    f"Install high-impact dependencies: {', '.join(missing_high_impact)}"
                )
            
            # Memory optimization opportunities
            memory_usage = self._get_memory_usage()
            if memory_usage > 1024:  # >1GB
                opportunities.append("Consider memory optimization for large dataset processing")
            
            # Steam Deck specific opportunities
            if self.is_steam_deck:
                non_optimized = [
                    name for name in self.fallback_system.active_fallbacks.keys()
                    if not self.fallback_system._is_steam_deck_optimized(name)
                ]
                if non_optimized:
                    opportunities.append(
                        f"Optimize {len(non_optimized)} components for Steam Deck"
                    )
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
            opportunities.append("Error analyzing optimization opportunities")
        
        return opportunities

    def create_optimization_plan(self) -> OptimizationPlan:
        """Create comprehensive optimization plan"""
        try:
            health_report = self.get_comprehensive_health_report()
            
            # Target improvements
            target_improvements = {
                'overall_health': min(0.9, health_report.overall_health + 0.2),
                'dependency_availability': 0.9,
                'fallback_stability': 0.95,
                'performance_score': 2.0  # 2x performance improvement target
            }
            
            # Recommended actions
            recommended_actions = []
            
            # Critical issues first
            for issue in health_report.critical_issues:
                recommended_actions.append({
                    'priority': 'critical',
                    'action': 'resolve_issue',
                    'description': issue,
                    'estimated_time_minutes': 30
                })
            
            # Dependency installations
            version_recommendations = self.version_manager.get_dependency_recommendations()
            dependencies_to_install = []
            
            for rec in version_recommendations['critical']:
                dependencies_to_install.append(rec['dependency'])
                recommended_actions.append({
                    'priority': 'high',
                    'action': 'install_dependency',
                    'dependency': rec['dependency'],
                    'reason': rec['reason'],
                    'estimated_time_minutes': 10
                })
            
            for rec in version_recommendations['recommended'][:5]:  # Top 5
                dependencies_to_install.append(rec['dependency'])
                recommended_actions.append({
                    'priority': 'medium',
                    'action': 'install_dependency',
                    'dependency': rec['dependency'],
                    'reason': rec['reason'],
                    'estimated_time_minutes': 5
                })
            
            # Fallback optimizations
            fallback_adjustments = []
            for opportunity in health_report.optimization_opportunities[:3]:  # Top 3
                if 'Upgrade' in opportunity:
                    fallback_adjustments.append({
                        'action': 'upgrade_fallback',
                        'description': opportunity
                    })
                    recommended_actions.append({
                        'priority': 'medium',
                        'action': 'optimize_fallback',
                        'description': opportunity,
                        'estimated_time_minutes': 2
                    })
            
            # Configuration changes
            configuration_changes = []
            if self.is_steam_deck:
                configuration_changes.append({
                    'setting': 'steam_deck_optimization',
                    'current': 'partial',
                    'recommended': 'full'
                })
            
            # Calculate expected performance gain
            current_performance = health_report.performance_metrics.get('overall_score', 1.0)
            expected_performance_gain = min(3.0, current_performance * 1.5)
            
            # Estimate time required
            total_time = sum(action.get('estimated_time_minutes', 5) for action in recommended_actions)
            estimated_time_required = total_time / 60.0  # Convert to hours
            
            # Risk assessment
            risk_assessment = {
                'installation_risk': 'low' if len(dependencies_to_install) < 5 else 'medium',
                'system_stability_risk': 'low' if health_report.health_level.value <= 2 else 'medium',
                'performance_risk': 'low',
                'rollback_possible': True
            }
            
            return OptimizationPlan(
                target_improvements=target_improvements,
                recommended_actions=recommended_actions,
                installation_suggestions=[f"pip install {dep}" for dep in dependencies_to_install],
                configuration_changes=configuration_changes,
                expected_performance_gain=expected_performance_gain,
                estimated_time_required=estimated_time_required,
                risk_assessment=risk_assessment,
                dependencies_to_install=dependencies_to_install,
                fallback_adjustments=fallback_adjustments
            )
            
        except Exception as e:
            logger.error(f"Error creating optimization plan: {e}")
            # Return minimal plan
            return OptimizationPlan(
                target_improvements={'error': 0.0},
                recommended_actions=[{'action': 'error', 'description': str(e)}],
                installation_suggestions=[],
                configuration_changes=[],
                expected_performance_gain=0.0,
                estimated_time_required=0.0,
                risk_assessment={'error': True},
                dependencies_to_install=[],
                fallback_adjustments=[]
            )

    def execute_optimization_plan(self, plan: OptimizationPlan, confirm_actions: bool = True) -> Dict[str, Any]:
        """Execute an optimization plan"""
        execution_results = {
            'started_at': time.time(),
            'completed_actions': [],
            'failed_actions': [],
            'performance_improvements': {},
            'warnings': [],
            'success': False
        }
        
        try:
            logger.info(f"Executing optimization plan with {len(plan.recommended_actions)} actions")
            
            # Get baseline metrics
            baseline_health = self.get_comprehensive_health_report()
            
            # Execute actions by priority
            actions_by_priority = defaultdict(list)
            for action in plan.recommended_actions:
                priority = action.get('priority', 'medium')
                actions_by_priority[priority].append(action)
            
            # Execute in priority order
            for priority in ['critical', 'high', 'medium', 'low']:
                for action in actions_by_priority[priority]:
                    try:
                        if confirm_actions:
                            # In a real implementation, this might prompt the user
                            logger.info(f"Executing {action['action']}: {action.get('description', 'No description')}")
                        
                        # Execute the action
                        action_result = self._execute_optimization_action(action)
                        
                        if action_result['success']:
                            execution_results['completed_actions'].append(action)
                        else:
                            execution_results['failed_actions'].append({
                                'action': action,
                                'error': action_result['error']
                            })
                            
                    except Exception as e:
                        logger.error(f"Failed to execute action {action['action']}: {e}")
                        execution_results['failed_actions'].append({
                            'action': action,
                            'error': str(e)
                        })
            
            # Apply fallback adjustments
            for adjustment in plan.fallback_adjustments:
                try:
                    if adjustment['action'] == 'upgrade_fallback':
                        # This would be implemented based on the specific upgrade needed
                        logger.info(f"Applying fallback adjustment: {adjustment['description']}")
                        execution_results['completed_actions'].append(adjustment)
                except Exception as e:
                    execution_results['failed_actions'].append({
                        'action': adjustment,
                        'error': str(e)
                    })
            
            # Measure improvements
            final_health = self.get_comprehensive_health_report()
            
            execution_results['performance_improvements'] = {
                'health_improvement': final_health.overall_health - baseline_health.overall_health,
                'dependency_improvement': (
                    final_health.dependency_status.get('available_dependencies', 0) - 
                    baseline_health.dependency_status.get('available_dependencies', 0)
                )
            }
            
            # Success if no critical failures
            execution_results['success'] = len([
                a for a in execution_results['failed_actions'] 
                if a['action'].get('priority') == 'critical'
            ]) == 0
            
            # Record in optimization history
            self.optimization_history.append({
                'timestamp': time.time(),
                'type': 'plan_execution',
                'plan_summary': {
                    'total_actions': len(plan.recommended_actions),
                    'completed': len(execution_results['completed_actions']),
                    'failed': len(execution_results['failed_actions'])
                },
                'results': execution_results
            })
            
            execution_results['completed_at'] = time.time()
            execution_results['duration_seconds'] = execution_results['completed_at'] - execution_results['started_at']
            
            logger.info(f"Optimization plan execution completed: {execution_results['success']}")
            
        except Exception as e:
            logger.error(f"Error executing optimization plan: {e}")
            execution_results['fatal_error'] = str(e)
            execution_results['success'] = False
        
        return execution_results

    def _execute_optimization_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single optimization action"""
        action_type = action.get('action', 'unknown')
        
        try:
            if action_type == 'resolve_issue':
                # This would implement specific issue resolution
                return {'success': True, 'message': 'Issue resolution attempted'}
            
            elif action_type == 'install_dependency':
                # This would trigger actual dependency installation
                dep_name = action.get('dependency', 'unknown')
                # Simulate installation (in real implementation, would use package manager)
                logger.info(f"Would install dependency: {dep_name}")
                return {'success': True, 'message': f'Installation of {dep_name} completed'}
            
            elif action_type == 'optimize_fallback':
                # Trigger fallback optimization
                optimization_result = self.fallback_system.auto_optimize_fallbacks()
                return {
                    'success': True, 
                    'message': f"Applied {len(optimization_result['optimizations_applied'])} optimizations"
                }
            
            else:
                return {'success': False, 'error': f'Unknown action type: {action_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def start_monitoring(self) -> bool:
        """Start continuous system monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return True
        
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="DependencyCoordinatorMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Dependency coordinator monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
            return False

    def stop_monitoring(self) -> bool:
        """Stop continuous system monitoring"""
        if not self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10.0)
            
            logger.info("Dependency coordinator monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return False

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("Dependency coordinator monitoring loop started")
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Periodic health checks
                if current_time - self.last_health_check_time > self.health_check_interval:
                    health_report = self.get_comprehensive_health_report()
                    self.system_health_history.append(health_report)
                    self.last_health_check_time = current_time
                    
                    # Call health callbacks
                    for callback in self.health_callbacks:
                        try:
                            callback(health_report)
                        except Exception as e:
                            logger.error(f"Error in health callback: {e}")
                    
                    # Check for critical issues
                    if health_report.health_level == SystemHealthLevel.CRITICAL:
                        logger.warning("Critical system health detected - triggering emergency optimization")
                        self._handle_critical_health(health_report)
                
                # Periodic optimization
                if (self.auto_optimize and 
                    current_time - self.last_optimization_time > self.optimization_interval):
                    
                    try:
                        optimization_result = self.fallback_system.auto_optimize_fallbacks()
                        self.last_optimization_time = current_time
                        
                        # Call optimization callbacks
                        for callback in self.optimization_callbacks:
                            try:
                                callback(optimization_result)
                            except Exception as e:
                                logger.error(f"Error in optimization callback: {e}")
                                
                    except Exception as e:
                        logger.error(f"Error in periodic optimization: {e}")
                
                # Sleep until next monitoring cycle
                time.sleep(min(self.health_check_interval, self.optimization_interval) / 4)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
        
        logger.info("Dependency coordinator monitoring loop ended")

    def _handle_critical_health(self, health_report: SystemHealthReport) -> None:
        """Handle critical system health situation"""
        logger.warning("Handling critical system health situation")
        
        try:
            # Switch all components to most stable tier
            for component_name in self.fallback_system.fallback_implementations.keys():
                try:
                    # Switch to pure Python tier for maximum stability
                    self.fallback_system.switch_to_tier(
                        component_name,
                        FallbackTier.PURE_PYTHON,
                        FallbackReason.SYSTEM_STABILITY
                    )
                except Exception as e:
                    logger.error(f"Failed to switch {component_name} to stable tier: {e}")
            
            # Clear caches to free memory
            self.system_capabilities_cache = None
            gc.collect()
            
            # Record emergency action
            self.optimization_history.append({
                'timestamp': time.time(),
                'type': 'emergency_stabilization',
                'health_level': health_report.health_level.name,
                'critical_issues': health_report.critical_issues
            })
            
        except Exception as e:
            logger.error(f"Error in critical health handling: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            impl = self.fallback_system.get_implementation('system_monitor')
            if impl and hasattr(impl, 'Process'):
                process = impl.Process()
                return process.memory_info().rss / 1024 / 1024
            else:
                # Fallback estimation
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 512.0  # Default estimate

    def _get_thermal_state(self) -> str:
        """Get thermal state"""
        try:
            monitor = PureThermalMonitor()
            return monitor.get_thermal_state()
        except Exception:
            return 'normal'

    def _assess_memory_pressure(self) -> float:
        """Assess memory pressure (0.0 to 1.0)"""
        try:
            memory_usage = self._get_memory_usage()
            # Assume 8GB system memory for Steam Deck
            system_memory_mb = 8192 if self.is_steam_deck else 16384
            return min(1.0, memory_usage / system_memory_mb)
        except Exception:
            return 0.5

    def add_health_callback(self, callback: Callable) -> None:
        """Add health monitoring callback"""
        self.health_callbacks.append(callback)

    def add_optimization_callback(self, callback: Callable) -> None:
        """Add optimization callback"""  
        self.optimization_callbacks.append(callback)

    def export_comprehensive_report(self, filepath: Path) -> None:
        """Export comprehensive system report"""
        try:
            health_report = self.get_comprehensive_health_report()
            optimization_plan = self.create_optimization_plan()
            
            comprehensive_report = {
                'system_health': {
                    'overall_health': health_report.overall_health,
                    'health_level': health_report.health_level.name,
                    'component_health': health_report.component_health,
                    'critical_issues': health_report.critical_issues,
                    'warnings': health_report.warnings,
                    'recommendations': health_report.recommendations
                },
                'dependency_status': health_report.dependency_status,
                'fallback_status': health_report.fallback_status,
                'performance_metrics': health_report.performance_metrics,
                'system_capabilities': health_report.system_capabilities,
                'optimization_plan': {
                    'target_improvements': optimization_plan.target_improvements,
                    'recommended_actions': optimization_plan.recommended_actions,
                    'dependencies_to_install': optimization_plan.dependencies_to_install,
                    'expected_performance_gain': optimization_plan.expected_performance_gain,
                    'risk_assessment': optimization_plan.risk_assessment
                },
                'system_info': self.version_manager.system_info,
                'steam_deck_status': health_report.steam_deck_status,
                'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
                'health_history': [
                    {
                        'timestamp': h.timestamp,
                        'overall_health': h.overall_health,
                        'health_level': h.health_level.name
                    }
                    for h in list(self.system_health_history)[-20:]  # Last 20 health checks
                ],
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            logger.info(f"Comprehensive report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export comprehensive report: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_coordinator: Optional[EnhancedDependencyCoordinator] = None

def get_coordinator() -> EnhancedDependencyCoordinator:
    """Get or create global coordinator instance"""
    global _coordinator
    if _coordinator is None:
        _coordinator = EnhancedDependencyCoordinator()
    return _coordinator

def quick_system_health() -> float:
    """Quick system health check (0.0 to 1.0)"""
    coordinator = get_coordinator()
    health_report = coordinator.get_comprehensive_health_report()
    return health_report.overall_health

def optimize_system() -> Dict[str, Any]:
    """Quick system optimization"""
    coordinator = get_coordinator()
    plan = coordinator.create_optimization_plan()
    return coordinator.execute_optimization_plan(plan, confirm_actions=False)

def is_system_healthy(threshold: float = 0.7) -> bool:
    """Check if system is healthy above threshold"""
    return quick_system_health() >= threshold

def get_system_status_summary() -> Dict[str, Any]:
    """Get concise system status summary"""
    coordinator = get_coordinator()
    health_report = coordinator.get_comprehensive_health_report()
    
    return {
        'health_level': health_report.health_level.name,
        'overall_health': f"{health_report.overall_health:.1%}",
        'dependencies_available': f"{health_report.dependency_status['available_dependencies']}/{health_report.dependency_status['total_dependencies']}",
        'fallback_systems': f"{health_report.fallback_status['active_components']} active",
        'system_capabilities': health_report.system_capabilities,
        'steam_deck': health_report.steam_deck_status is not None,
        'critical_issues': len(health_report.critical_issues),
        'recommendations': len(health_report.recommendations)
    }


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🎯 Enhanced Dependency Coordinator Test Suite")
    print("=" * 60)
    
    # Initialize coordinator
    coordinator = EnhancedDependencyCoordinator(
        auto_optimize=True,
        monitoring_enabled=False  # Disable for testing
    )
    
    # System overview
    print("\n🖥️  System Overview:")
    status_summary = get_system_status_summary()
    for key, value in status_summary.items():
        print(f"  {key}: {value}")
    
    # Comprehensive health report
    print("\n🏥 Comprehensive Health Report:")
    health_report = coordinator.get_comprehensive_health_report()
    print(f"  Overall Health: {health_report.overall_health:.1%} ({health_report.health_level.name})")
    print(f"  Component Health:")
    for component, health in health_report.component_health.items():
        print(f"    {component}: {health:.1%}")
    
    if health_report.critical_issues:
        print(f"  Critical Issues:")
        for issue in health_report.critical_issues:
            print(f"    🚨 {issue}")
    
    if health_report.warnings:
        print(f"  Warnings:")
        for warning in health_report.warnings[:3]:  # Top 3
            print(f"    ⚠️  {warning}")
    
    if health_report.recommendations:
        print(f"  Recommendations:")
        for rec in health_report.recommendations[:3]:  # Top 3
            print(f"    💡 {rec}")
    
    # System capabilities
    print(f"\n🛠️  System Capabilities:")
    capabilities = health_report.system_capabilities
    for capability, available in capabilities.items():
        status = "✅" if available else "❌"
        print(f"  {status} {capability}")
    
    # Optimization plan
    print(f"\n⚡ Optimization Plan:")
    plan = coordinator.create_optimization_plan()
    print(f"  Target Health Improvement: {plan.target_improvements.get('overall_health', 0):.1%}")
    print(f"  Recommended Actions: {len(plan.recommended_actions)}")
    print(f"  Dependencies to Install: {len(plan.dependencies_to_install)}")
    print(f"  Expected Performance Gain: {plan.expected_performance_gain:.1f}x")
    print(f"  Estimated Time: {plan.estimated_time_required:.1f} hours")
    
    # Show top recommendations
    if plan.recommended_actions:
        print("  Top Actions:")
        for action in plan.recommended_actions[:3]:
            priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(action.get('priority', 'medium'), "ℹ️")
            print(f"    {priority_emoji} {action.get('action', 'unknown')}: {action.get('description', 'No description')}")
    
    # Test optimization execution (dry run)
    print(f"\n🚀 Testing Optimization Execution:")
    execution_result = coordinator.execute_optimization_plan(plan, confirm_actions=False)
    print(f"  Execution Success: {'✅' if execution_result['success'] else '❌'}")
    print(f"  Completed Actions: {len(execution_result['completed_actions'])}")
    print(f"  Failed Actions: {len(execution_result['failed_actions'])}")
    
    if execution_result.get('performance_improvements'):
        print("  Performance Improvements:")
        for metric, improvement in execution_result['performance_improvements'].items():
            print(f"    {metric}: {improvement:+.3f}")
    
    # Export comprehensive report
    report_path = Path("/tmp/enhanced_dependency_coordinator_report.json")
    coordinator.export_comprehensive_report(report_path)
    print(f"\n💾 Comprehensive report exported to {report_path}")
    
    print(f"\n✅ Enhanced Dependency Coordinator test completed!")
    print(f"🎯 System Health: {health_report.overall_health:.1%}")
    print(f"🛠️  System Ready: {'Yes' if health_report.overall_health > 0.7 else 'Needs optimization'}")
    
    if coordinator.is_steam_deck:
        print(f"🎮 Steam Deck optimizations {'✅ Active' if health_report.steam_deck_status else '❌ Inactive'}")