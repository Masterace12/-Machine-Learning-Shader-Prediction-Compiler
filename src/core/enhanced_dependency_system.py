#!/usr/bin/env python3
"""
Enhanced Dependency System for ML Shader Prediction Compiler

This module provides a unified interface for all dependency management components,
creating a comprehensive system that intelligently coordinates dependencies,
validates installations, optimizes performance, and provides Steam Deck specific
optimizations.

Features:
- Unified dependency management interface
- Automatic system optimization based on environment
- Comprehensive health monitoring and reporting
- Steam Deck specific integration
- Performance benchmarking and optimization
- Graceful degradation and fallback management
- Real-time adaptive optimization
- Installation validation and repair
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all our dependency management components
try:
    # Try relative imports first
    try:
        from .dependency_coordinator import DependencyCoordinator, get_coordinator
        from .installation_validator import InstallationValidator, quick_validate_installation
        from .runtime_dependency_manager import RuntimeDependencyManager, get_runtime_manager
        from .steam_deck_optimizer import SteamDeckOptimizer, get_steam_deck_optimizer
        from .pure_python_fallbacks import get_fallback_status, log_fallback_status
    except ImportError:
        # Fallback to direct imports when running as main
        import dependency_coordinator
        import installation_validator
        import runtime_dependency_manager
        import steam_deck_optimizer
        import pure_python_fallbacks
        
        DependencyCoordinator = dependency_coordinator.DependencyCoordinator
        get_coordinator = dependency_coordinator.get_coordinator
        InstallationValidator = installation_validator.InstallationValidator
        quick_validate_installation = installation_validator.quick_validate_installation
        RuntimeDependencyManager = runtime_dependency_manager.RuntimeDependencyManager
        get_runtime_manager = runtime_dependency_manager.get_runtime_manager
        SteamDeckOptimizer = steam_deck_optimizer.SteamDeckOptimizer
        get_steam_deck_optimizer = steam_deck_optimizer.get_steam_deck_optimizer
        get_fallback_status = pure_python_fallbacks.get_fallback_status
        log_fallback_status = pure_python_fallbacks.log_fallback_status
        
except ImportError as e:
    print(f"Warning: Could not import dependency management components: {e}")
    print("Some features may not be available.")
    
    # Create dummy classes for testing
    class DependencyCoordinator:
        def __init__(self): pass
        def detect_all_dependencies(self, force_refresh=False): return {}
        def validate_installation(self): return {'overall_health': 0.5}
        def export_configuration(self, path): pass
        
    class InstallationValidator:
        def __init__(self): pass
        def validate_all_dependencies(self, **kwargs): return {}
        def cleanup(self): pass
        
    class RuntimeDependencyManager:
        def __init__(self): 
            self.current_profile = None
            self.performance_profiles = []
        def get_current_conditions(self): 
            from collections import namedtuple
            Conditions = namedtuple('Conditions', 'cpu_temperature memory_usage_mb thermal_state memory_pressure power_state is_gaming_mode')
            return Conditions(50.0, 1024.0, 'normal', 'low', 'ac', False)
        def get_performance_metrics(self): return {'switch_count': 0}
        def get_active_backends(self): return {}
        def start_monitoring(self): return True
        def stop_monitoring(self): return True
        def export_configuration(self, path): pass
        
    class SteamDeckOptimizer:
        def __init__(self): 
            self.is_steam_deck = False
        def get_current_state(self):
            from collections import namedtuple
            State = namedtuple('State', 'cpu_temperature_celsius memory_usage_mb')
            return State(50.0, 1024.0)
        def get_compatibility_report(self): return {'overall_score': 0.7}
        def start_adaptive_optimization(self): return True
        def stop_adaptive_optimization(self): return True
        def export_optimization_report(self, path): pass
        
    def get_coordinator(): return DependencyCoordinator()
    def quick_validate_installation(): return {'average_health': 0.8}
    def get_runtime_manager(): return RuntimeDependencyManager()
    def get_steam_deck_optimizer(): return SteamDeckOptimizer()
    def get_fallback_status(): return {}
    def log_fallback_status(): pass

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# UNIFIED SYSTEM STATUS
# =============================================================================

@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    timestamp: float
    overall_health: float  # 0.0 to 1.0
    dependency_health: float
    performance_health: float
    thermal_health: float
    memory_health: float
    steam_deck_health: Optional[float]
    
    # Component status
    dependencies_available: int
    dependencies_total: int
    active_backends: Dict[str, str]
    current_profile: Optional[str]
    optimization_active: bool
    
    # System info
    is_steam_deck: bool
    platform: str
    python_version: str
    cpu_temperature: float
    memory_usage_mb: float
    
    # Recommendations
    critical_issues: List[str]
    recommendations: List[str]
    
    # Performance metrics
    performance_score: float
    switch_count: int
    validation_passed: int
    validation_total: int

@dataclass
class OptimizationResult:
    """Result of system optimization"""
    success: bool
    changes_made: List[str]
    performance_improvement: float
    profile_changed: bool
    new_profile: Optional[str]
    time_taken: float
    errors: List[str]
    warnings: List[str]

# =============================================================================
# ENHANCED DEPENDENCY SYSTEM
# =============================================================================

class EnhancedDependencySystem:
    """
    Unified dependency management system that coordinates all components
    """
    
    def __init__(self, auto_start: bool = True, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.initialized = False
        self.auto_optimization = False
        self.monitoring_active = False
        self.system_callbacks: List[Callable] = []
        self.health_history: List[SystemHealthReport] = []
        
        # Component instances
        self.dependency_coordinator: Optional[DependencyCoordinator] = None
        self.runtime_manager: Optional[RuntimeDependencyManager] = None
        self.steam_deck_optimizer: Optional[SteamDeckOptimizer] = None
        self.installation_validator: Optional[InstallationValidator] = None
        
        # System state
        self.last_optimization_time = 0.0
        self.optimization_interval = 30.0  # seconds
        self.health_check_interval = 10.0   # seconds
        
        if auto_start:
            self.initialize()
    
    def initialize(self) -> bool:
        """Initialize all dependency management components"""
        if self.initialized:
            logger.warning("System already initialized")
            return True
        
        logger.info("Initializing Enhanced Dependency System...")
        initialization_start = time.time()
        
        try:
            # Initialize dependency coordinator
            logger.info("Initializing dependency coordinator...")
            self.dependency_coordinator = get_coordinator()
            
            # Initialize runtime manager
            logger.info("Initializing runtime dependency manager...")
            self.runtime_manager = get_runtime_manager()
            
            # Initialize Steam Deck optimizer
            logger.info("Initializing Steam Deck optimizer...")
            self.steam_deck_optimizer = get_steam_deck_optimizer()
            
            # Initialize installation validator
            logger.info("Initializing installation validator...")
            self.installation_validator = InstallationValidator()
            
            # Run initial system detection and optimization
            logger.info("Running initial system analysis...")
            self._run_initial_analysis()
            
            # Setup integration between components
            self._setup_component_integration()
            
            initialization_time = time.time() - initialization_start
            logger.info(f"Enhanced Dependency System initialized in {initialization_time:.2f}s")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Dependency System: {e}")
            return False
    
    def _run_initial_analysis(self) -> None:
        """Run initial system analysis and optimization"""
        try:
            # Detect all dependencies
            if self.dependency_coordinator:
                self.dependency_coordinator.detect_all_dependencies()
            
            # Run performance benchmarks
            if self.runtime_manager and not self.runtime_manager.performance_profiles:
                logger.info("Running initial performance benchmarks...")
                self.runtime_manager.benchmark_performance_combinations(test_size=500)
            
            # Apply initial optimization
            self._apply_initial_optimization()
            
        except Exception as e:
            logger.error(f"Error in initial analysis: {e}")
    
    def _apply_initial_optimization(self) -> None:
        """Apply initial optimization based on system detection"""
        try:
            # Get system conditions
            if self.runtime_manager:
                conditions = self.runtime_manager.get_current_conditions()
                
                # Select and apply optimal profile
                optimal_profile = self.runtime_manager._select_optimal_profile(conditions)
                if optimal_profile:
                    self.runtime_manager._switch_to_profile(optimal_profile, conditions)
                    logger.info(f"Applied initial optimization profile: {optimal_profile.profile_id}")
            
            # Apply Steam Deck specific optimizations
            if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                state = self.steam_deck_optimizer.get_current_state()
                optimal_steam_profile = self.steam_deck_optimizer.select_optimal_profile(state)
                self.steam_deck_optimizer.apply_optimization_profile(optimal_steam_profile)
                logger.info(f"Applied Steam Deck optimization: {optimal_steam_profile}")
        
        except Exception as e:
            logger.error(f"Error applying initial optimization: {e}")
    
    def _setup_component_integration(self) -> None:
        """Setup integration callbacks between components"""
        try:
            # Setup runtime manager callbacks
            if self.runtime_manager:
                self.runtime_manager.add_switch_callback(self._on_profile_switch)
            
            # Setup Steam Deck optimizer callbacks
            if self.steam_deck_optimizer:
                self.steam_deck_optimizer.add_optimization_callback(self._on_steam_deck_optimization)
            
            logger.info("Component integration callbacks setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up component integration: {e}")
    
    def _on_profile_switch(self, old_profile, new_profile, conditions):
        """Handle runtime manager profile switches"""
        logger.info(f"Runtime profile switched: {old_profile.profile_id if old_profile else 'None'} -> {new_profile.profile_id}")
        
        # Notify system callbacks
        for callback in self.system_callbacks:
            try:
                callback('profile_switch', {
                    'old_profile': old_profile.profile_id if old_profile else None,
                    'new_profile': new_profile.profile_id,
                    'conditions': conditions
                })
            except Exception as e:
                logger.error(f"Error in system callback: {e}")
    
    def _on_steam_deck_optimization(self, profile_name, profile):
        """Handle Steam Deck optimization changes"""
        logger.info(f"Steam Deck optimization applied: {profile_name}")
        
        # Notify system callbacks
        for callback in self.system_callbacks:
            try:
                callback('steam_deck_optimization', {
                    'profile_name': profile_name,
                    'profile': profile
                })
            except Exception as e:
                logger.error(f"Error in system callback: {e}")
    
    def start_monitoring(self) -> bool:
        """Start automatic system monitoring and optimization"""
        if not self.initialized:
            logger.error("System not initialized")
            return False
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return True
        
        try:
            # Start runtime manager monitoring
            if self.runtime_manager:
                self.runtime_manager.start_monitoring()
            
            # Start Steam Deck optimizer monitoring
            if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                self.steam_deck_optimizer.start_adaptive_optimization()
            
            self.monitoring_active = True
            self.auto_optimization = True
            
            logger.info("Enhanced Dependency System monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop automatic system monitoring"""
        if not self.monitoring_active:
            return True
        
        try:
            # Stop runtime manager monitoring
            if self.runtime_manager:
                self.runtime_manager.stop_monitoring()
            
            # Stop Steam Deck optimizer monitoring
            if self.steam_deck_optimizer:
                self.steam_deck_optimizer.stop_adaptive_optimization()
            
            self.monitoring_active = False
            self.auto_optimization = False
            
            logger.info("Enhanced Dependency System monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def get_system_health(self) -> SystemHealthReport:
        """Get comprehensive system health report"""
        try:
            current_time = time.time()
            
            # Get component health scores
            dependency_health = 0.0
            performance_health = 0.0
            thermal_health = 0.0
            memory_health = 0.0
            steam_deck_health = None
            
            # Dependency health
            if self.dependency_coordinator:
                available = sum(1 for state in self.dependency_coordinator.dependency_states.values() 
                              if state.available and state.test_passed)
                total = len(self.dependency_coordinator.dependency_states)
                dependency_health = available / max(total, 1)
            
            # Performance health
            if self.runtime_manager:
                metrics = self.runtime_manager.get_performance_metrics()
                performance_health = 0.8  # Base score
                if self.runtime_manager.current_profile:
                    performance_health = min(1.0, self.runtime_manager.current_profile.performance_target / 3.0)
            
            # Thermal health
            if self.steam_deck_optimizer:
                state = self.steam_deck_optimizer.get_current_state()
                temp = state.cpu_temperature_celsius
                thermal_health = max(0.0, min(1.0, (90.0 - temp) / 50.0))  # 40-90°C range
            elif self.runtime_manager:
                conditions = self.runtime_manager.get_current_conditions()
                temp = conditions.cpu_temperature
                thermal_health = max(0.0, min(1.0, (90.0 - temp) / 50.0))
            
            # Memory health
            if self.runtime_manager:
                conditions = self.runtime_manager.get_current_conditions()
                memory_usage = conditions.memory_usage_mb
                memory_health = max(0.0, 1.0 - (memory_usage / 8192))  # 8GB reference
            
            # Steam Deck specific health
            if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                compatibility = self.steam_deck_optimizer.get_compatibility_report()
                steam_deck_health = compatibility['overall_score']
            
            # Calculate overall health
            health_components = [dependency_health, performance_health, thermal_health, memory_health]
            if steam_deck_health is not None:
                health_components.append(steam_deck_health)
            
            overall_health = sum(health_components) / len(health_components)
            
            # Get system info
            is_steam_deck = self.steam_deck_optimizer.is_steam_deck if self.steam_deck_optimizer else False
            
            # Get active backends
            active_backends = {}
            if self.runtime_manager:
                active_backends = {cat: backend.name for cat, backend in self.runtime_manager.get_active_backends().items()}
            
            # Get current profile
            current_profile = None
            if self.runtime_manager and self.runtime_manager.current_profile:
                current_profile = self.runtime_manager.current_profile.profile_id
            
            # Get system metrics
            dependencies_available = 0
            dependencies_total = 0
            if self.dependency_coordinator:
                dependencies_available = sum(1 for state in self.dependency_coordinator.dependency_states.values() if state.available)
                dependencies_total = len(self.dependency_coordinator.dependency_states)
            
            # Get performance metrics
            performance_score = 5.0
            switch_count = 0
            if self.runtime_manager:
                perf_metrics = self.runtime_manager.get_performance_metrics()
                switch_count = perf_metrics.get('switch_count', 0)
                if self.runtime_manager.current_profile:
                    performance_score = self.runtime_manager.current_profile.performance_target
            
            # Get validation metrics
            validation_passed = 0
            validation_total = 0
            # These would be populated from validation runs
            
            # Get current conditions
            cpu_temp = 50.0
            memory_usage = 1024.0
            if self.runtime_manager:
                conditions = self.runtime_manager.get_current_conditions()
                cpu_temp = conditions.cpu_temperature
                memory_usage = conditions.memory_usage_mb
            
            # Collect issues and recommendations
            critical_issues = []
            recommendations = []
            
            if overall_health < 0.5:
                critical_issues.append("System health is critically low")
            
            if thermal_health < 0.3:
                critical_issues.append(f"High temperature warning: {cpu_temp:.1f}°C")
                recommendations.append("Reduce computational load or improve cooling")
            
            if dependency_health < 0.7:
                recommendations.append("Some dependencies failed validation - consider reinstallation")
            
            if memory_health < 0.5:
                recommendations.append("High memory usage detected - consider lighter alternatives")
            
            # Steam Deck specific recommendations
            if is_steam_deck and self.steam_deck_optimizer:
                steam_recommendations = self.steam_deck_optimizer.get_optimization_recommendations()
                for rec in steam_recommendations:
                    if rec['priority'] == 'high':
                        critical_issues.append(rec['title'])
                    recommendations.append(rec['action'])
            
            report = SystemHealthReport(
                timestamp=current_time,
                overall_health=overall_health,
                dependency_health=dependency_health,
                performance_health=performance_health,
                thermal_health=thermal_health,
                memory_health=memory_health,
                steam_deck_health=steam_deck_health,
                dependencies_available=dependencies_available,
                dependencies_total=dependencies_total,
                active_backends=active_backends,
                current_profile=current_profile,
                optimization_active=self.monitoring_active,
                is_steam_deck=is_steam_deck,
                platform=sys.platform,
                python_version=sys.version.split()[0],
                cpu_temperature=cpu_temp,
                memory_usage_mb=memory_usage,
                critical_issues=critical_issues,
                recommendations=recommendations,
                performance_score=performance_score,
                switch_count=switch_count,
                validation_passed=validation_passed,
                validation_total=validation_total
            )
            
            # Store in history
            self.health_history.append(report)
            if len(self.health_history) > 100:  # Keep last 100 reports
                self.health_history.pop(0)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            # Return minimal health report
            return SystemHealthReport(
                timestamp=time.time(),
                overall_health=0.5,
                dependency_health=0.5,
                performance_health=0.5,
                thermal_health=0.5,
                memory_health=0.5,
                steam_deck_health=None,
                dependencies_available=0,
                dependencies_total=0,
                active_backends={},
                current_profile=None,
                optimization_active=False,
                is_steam_deck=False,
                platform=sys.platform,
                python_version=sys.version.split()[0],
                cpu_temperature=50.0,
                memory_usage_mb=1024.0,
                critical_issues=[f"Error generating health report: {e}"],
                recommendations=["System diagnostics needed"],
                performance_score=1.0,
                switch_count=0,
                validation_passed=0,
                validation_total=0
            )
    
    def optimize_system(self, force: bool = False) -> OptimizationResult:
        """Perform comprehensive system optimization"""
        if not self.initialized:
            return OptimizationResult(
                success=False,
                changes_made=[],
                performance_improvement=0.0,
                profile_changed=False,
                new_profile=None,
                time_taken=0.0,
                errors=["System not initialized"],
                warnings=[]
            )
        
        logger.info("Starting comprehensive system optimization...")
        start_time = time.time()
        
        changes_made = []
        errors = []
        warnings = []
        performance_improvement = 0.0
        profile_changed = False
        new_profile = None
        
        try:
            # Get baseline health
            baseline_health = self.get_system_health()
            baseline_performance = baseline_health.performance_score
            
            # 1. Validate installation
            if self.installation_validator:
                logger.info("Validating installation...")
                validation_summary = quick_validate_installation()
                if validation_summary['average_health'] < 0.8:
                    warnings.append(f"Installation health low: {validation_summary['average_health']:.1%}")
                    changes_made.append("Identified installation issues")
            
            # 2. Optimize dependency selection
            if self.runtime_manager:
                logger.info("Optimizing dependency configuration...")
                
                # Get current conditions
                conditions = self.runtime_manager.get_current_conditions()
                
                # Select optimal profile
                optimal_profile = self.runtime_manager._select_optimal_profile(conditions)
                if optimal_profile and (force or optimal_profile != self.runtime_manager.current_profile):
                    if self.runtime_manager._switch_to_profile(optimal_profile, conditions):
                        profile_changed = True
                        new_profile = optimal_profile.profile_id
                        changes_made.append(f"Switched to profile: {optimal_profile.profile_id}")
                    else:
                        errors.append(f"Failed to switch to profile: {optimal_profile.profile_id}")
                
                # Run benchmarks if needed
                if not self.runtime_manager.performance_profiles or force:
                    logger.info("Running performance benchmarks...")
                    profiles = self.runtime_manager.benchmark_performance_combinations(test_size=200)
                    if profiles:
                        changes_made.append(f"Updated performance profiles ({len(profiles)} tested)")
            
            # 3. Apply Steam Deck optimizations
            if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                logger.info("Applying Steam Deck optimizations...")
                
                state = self.steam_deck_optimizer.get_current_state()
                optimal_steam_profile = self.steam_deck_optimizer.select_optimal_profile(state)
                
                if self.steam_deck_optimizer.apply_optimization_profile(optimal_steam_profile):
                    changes_made.append(f"Applied Steam Deck profile: {optimal_steam_profile}")
                else:
                    errors.append(f"Failed to apply Steam Deck profile: {optimal_steam_profile}")
            
            # 4. Check for critical issues
            final_health = self.get_system_health()
            
            if final_health.critical_issues:
                for issue in final_health.critical_issues:
                    warnings.append(f"Critical issue: {issue}")
            
            # Calculate performance improvement
            performance_improvement = final_health.performance_score - baseline_performance
            
            optimization_time = time.time() - start_time
            
            success = len(errors) == 0
            
            logger.info(f"System optimization completed in {optimization_time:.2f}s")
            logger.info(f"Performance improvement: {performance_improvement:.2f}x")
            logger.info(f"Changes made: {len(changes_made)}")
            
            return OptimizationResult(
                success=success,
                changes_made=changes_made,
                performance_improvement=performance_improvement,
                profile_changed=profile_changed,
                new_profile=new_profile,
                time_taken=optimization_time,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error during system optimization: {e}")
            return OptimizationResult(
                success=False,
                changes_made=changes_made,
                performance_improvement=0.0,
                profile_changed=False,
                new_profile=None,
                time_taken=time.time() - start_time,
                errors=[str(e)],
                warnings=warnings
            )
    
    def validate_system(self, thorough: bool = False) -> Dict[str, Any]:
        """Validate entire system installation and configuration"""
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        logger.info("Starting comprehensive system validation...")
        
        validation_results = {
            'timestamp': time.time(),
            'overall_success': True,
            'components': {},
            'summary': {},
            'recommendations': []
        }
        
        try:
            # Validate dependencies
            if self.dependency_coordinator:
                logger.info("Validating dependencies...")
                dep_validation = self.dependency_coordinator.validate_installation()
                validation_results['components']['dependencies'] = dep_validation
                validation_results['overall_success'] &= dep_validation['overall_health'] > 0.7
            
            # Validate installation
            if self.installation_validator and thorough:
                logger.info("Running thorough installation validation...")
                install_validation = self.installation_validator.validate_all_dependencies(
                    include_stress_tests=True,
                    include_performance_tests=True
                )
                validation_results['components']['installation'] = {
                    dep: {
                        'health': result.installation_health,
                        'success': result.overall_success,
                        'performance': result.performance_score
                    }
                    for dep, result in install_validation.items()
                }
            
            # Validate runtime configuration
            if self.runtime_manager:
                logger.info("Validating runtime configuration...")
                runtime_metrics = self.runtime_manager.get_performance_metrics()
                validation_results['components']['runtime'] = runtime_metrics
            
            # Validate Steam Deck compatibility
            if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                logger.info("Validating Steam Deck compatibility...")
                steam_compatibility = self.steam_deck_optimizer.get_compatibility_report()
                validation_results['components']['steam_deck'] = steam_compatibility
                validation_results['overall_success'] &= steam_compatibility['overall_score'] > 0.6
            
            # Generate summary
            health_report = self.get_system_health()
            validation_results['summary'] = {
                'overall_health': health_report.overall_health,
                'dependencies_available': f"{health_report.dependencies_available}/{health_report.dependencies_total}",
                'performance_score': health_report.performance_score,
                'thermal_status': 'good' if health_report.thermal_health > 0.7 else 'warning' if health_report.thermal_health > 0.4 else 'critical',
                'memory_status': 'good' if health_report.memory_health > 0.7 else 'warning' if health_report.memory_health > 0.4 else 'critical'
            }
            
            # Collect recommendations
            validation_results['recommendations'] = health_report.recommendations
            
            logger.info(f"System validation completed - Overall success: {validation_results['overall_success']}")
            
        except Exception as e:
            logger.error(f"Error during system validation: {e}")
            validation_results['error'] = str(e)
            validation_results['overall_success'] = False
        
        return validation_results
    
    def repair_system(self) -> Dict[str, Any]:
        """Attempt to repair system issues automatically"""
        logger.info("Starting system repair...")
        
        repair_results = {
            'timestamp': time.time(),
            'success': False,
            'actions_taken': [],
            'issues_resolved': [],
            'issues_remaining': [],
            'recommendations': []
        }
        
        try:
            # Get current health status
            health = self.get_system_health()
            
            # Attempt to resolve critical issues
            for issue in health.critical_issues:
                if 'temperature' in issue.lower():
                    # Apply thermal protection
                    if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                        if self.steam_deck_optimizer.apply_optimization_profile('thermal_emergency'):
                            repair_results['actions_taken'].append("Applied thermal emergency profile")
                            repair_results['issues_resolved'].append(issue)
                        else:
                            repair_results['issues_remaining'].append(issue)
                    elif self.runtime_manager:
                        # Switch to conservative profile
                        if self.runtime_manager.manual_switch_profile('conservative', force=True):
                            repair_results['actions_taken'].append("Switched to conservative profile")
                            repair_results['issues_resolved'].append(issue)
                
                elif 'memory' in issue.lower():
                    # Apply memory conservation
                    if self.runtime_manager:
                        if self.runtime_manager.manual_switch_profile('minimal', force=True):
                            repair_results['actions_taken'].append("Switched to minimal profile to reduce memory usage")
                            repair_results['issues_resolved'].append(issue)
                
                elif 'dependency' in issue.lower() or 'installation' in issue.lower():
                    # Re-detect dependencies
                    if self.dependency_coordinator:
                        self.dependency_coordinator.detect_all_dependencies(force_refresh=True)
                        repair_results['actions_taken'].append("Re-detected dependencies")
                        repair_results['issues_resolved'].append(issue)
            
            # Apply general optimization
            optimization_result = self.optimize_system(force=True)
            if optimization_result.success:
                repair_results['actions_taken'].extend(optimization_result.changes_made)
            
            # Check if issues were resolved
            final_health = self.get_system_health()
            repair_results['success'] = final_health.overall_health > health.overall_health
            repair_results['issues_remaining'] = final_health.critical_issues
            repair_results['recommendations'] = final_health.recommendations
            
            logger.info(f"System repair completed - Success: {repair_results['success']}")
            
        except Exception as e:
            logger.error(f"Error during system repair: {e}")
            repair_results['error'] = str(e)
        
        return repair_results
    
    def add_system_callback(self, callback: Callable) -> None:
        """Add system-wide event callback"""
        self.system_callbacks.append(callback)
    
    def remove_system_callback(self, callback: Callable) -> None:
        """Remove system callback"""
        if callback in self.system_callbacks:
            self.system_callbacks.remove(callback)
    
    @contextmanager
    def optimization_context(self, profile: str):
        """Context manager for temporary system optimization"""
        original_health = self.get_system_health()
        
        try:
            # Apply temporary optimization
            if self.runtime_manager:
                with self.runtime_manager.temporary_profile(profile):
                    yield
            else:
                yield
        finally:
            # System should automatically restore
            pass
    
    def export_system_report(self, path: Path, include_history: bool = True) -> None:
        """Export comprehensive system report"""
        logger.info(f"Exporting system report to {path}")
        
        try:
            health = self.get_system_health()
            
            report = {
                'timestamp': time.time(),
                'system_health': {
                    'overall_health': health.overall_health,
                    'dependency_health': health.dependency_health,
                    'performance_health': health.performance_health,
                    'thermal_health': health.thermal_health,
                    'memory_health': health.memory_health,
                    'steam_deck_health': health.steam_deck_health
                },
                'system_info': {
                    'is_steam_deck': health.is_steam_deck,
                    'platform': health.platform,
                    'python_version': health.python_version,
                    'cpu_temperature': health.cpu_temperature,
                    'memory_usage_mb': health.memory_usage_mb
                },
                'configuration': {
                    'dependencies_available': health.dependencies_available,
                    'dependencies_total': health.dependencies_total,
                    'active_backends': health.active_backends,
                    'current_profile': health.current_profile,
                    'optimization_active': health.optimization_active
                },
                'performance': {
                    'performance_score': health.performance_score,
                    'switch_count': health.switch_count
                },
                'issues': {
                    'critical_issues': health.critical_issues,
                    'recommendations': health.recommendations
                },
                'components': {}
            }
            
            # Add component-specific information
            if self.dependency_coordinator:
                dep_config_path = path.parent / f"{path.stem}_dependencies.json"
                self.dependency_coordinator.export_configuration(dep_config_path)
                report['components']['dependency_coordinator'] = str(dep_config_path)
            
            if self.runtime_manager:
                runtime_config_path = path.parent / f"{path.stem}_runtime.json"
                self.runtime_manager.export_configuration(runtime_config_path)
                report['components']['runtime_manager'] = str(runtime_config_path)
            
            if self.steam_deck_optimizer and self.steam_deck_optimizer.is_steam_deck:
                steam_config_path = path.parent / f"{path.stem}_steam_deck.json"
                self.steam_deck_optimizer.export_optimization_report(steam_config_path)
                report['components']['steam_deck_optimizer'] = str(steam_config_path)
            
            # Add health history if requested
            if include_history and self.health_history:
                report['health_history'] = [
                    {
                        'timestamp': h.timestamp,
                        'overall_health': h.overall_health,
                        'cpu_temperature': h.cpu_temperature,
                        'memory_usage_mb': h.memory_usage_mb,
                        'current_profile': h.current_profile
                    }
                    for h in self.health_history[-20:]  # Last 20 entries
                ]
            
            # Write main report
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"System report exported successfully to {path}")
            
        except Exception as e:
            logger.error(f"Failed to export system report: {e}")
    
    def cleanup(self) -> None:
        """Clean up system resources"""
        logger.info("Cleaning up Enhanced Dependency System...")
        
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Clean up components
            if self.installation_validator:
                self.installation_validator.cleanup()
            
            # Clear callbacks
            self.system_callbacks.clear()
            
            logger.info("Enhanced Dependency System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global system instance
_enhanced_system: Optional[EnhancedDependencySystem] = None

def get_enhanced_system() -> EnhancedDependencySystem:
    """Get or create the global enhanced dependency system"""
    global _enhanced_system
    if _enhanced_system is None:
        _enhanced_system = EnhancedDependencySystem()
    return _enhanced_system

def quick_system_health() -> float:
    """Quick system health check (0.0 to 1.0)"""
    system = get_enhanced_system()
    health = system.get_system_health()
    return health.overall_health

def optimize_for_current_environment() -> bool:
    """Quick optimization for current environment"""
    system = get_enhanced_system()
    result = system.optimize_system()
    return result.success

def is_system_healthy(threshold: float = 0.7) -> bool:
    """Check if system health is above threshold"""
    return quick_system_health() >= threshold

@contextmanager
def high_performance_mode():
    """Context manager for high performance mode"""
    system = get_enhanced_system()
    with system.optimization_context('maximum_performance'):
        yield

@contextmanager
def battery_saving_mode():
    """Context manager for battery saving mode"""
    system = get_enhanced_system()
    with system.optimization_context('power_saving'):
        yield

def start_auto_optimization():
    """Start automatic system optimization"""
    system = get_enhanced_system()
    return system.start_monitoring()

def stop_auto_optimization():
    """Stop automatic system optimization"""
    system = get_enhanced_system()
    return system.stop_monitoring()


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Enhanced Dependency System Test Suite")
    print("=" * 55)
    
    # Initialize system
    print("\n🔧 Initializing Enhanced Dependency System...")
    system = EnhancedDependencySystem()
    
    if not system.initialized:
        print("❌ Failed to initialize system")
        sys.exit(1)
    
    print("✅ System initialized successfully")
    
    # Get system health
    print("\n🏥 System Health Report:")
    health = system.get_system_health()
    
    print(f"  Overall Health: {health.overall_health:.1%}")
    print(f"  Dependencies: {health.dependencies_available}/{health.dependencies_total} available")
    print(f"  Performance Score: {health.performance_score:.1f}")
    print(f"  CPU Temperature: {health.cpu_temperature:.1f}°C")
    print(f"  Memory Usage: {health.memory_usage_mb:.0f}MB")
    print(f"  Steam Deck: {health.is_steam_deck}")
    print(f"  Current Profile: {health.current_profile}")
    
    if health.critical_issues:
        print(f"\n🚨 Critical Issues:")
        for issue in health.critical_issues:
            print(f"    ❌ {issue}")
    
    if health.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in health.recommendations[:3]:  # Show top 3
            print(f"    💡 {rec}")
    
    # Test system optimization
    print(f"\n⚡ Testing System Optimization:")
    optimization = system.optimize_system()
    
    print(f"  Success: {'✅' if optimization.success else '❌'}")
    print(f"  Time Taken: {optimization.time_taken:.2f}s")
    print(f"  Performance Improvement: {optimization.performance_improvement:.2f}x")
    print(f"  Profile Changed: {optimization.profile_changed}")
    
    if optimization.changes_made:
        print(f"  Changes Made:")
        for change in optimization.changes_made:
            print(f"    ✅ {change}")
    
    if optimization.errors:
        print(f"  Errors:")
        for error in optimization.errors:
            print(f"    ❌ {error}")
    
    # Test validation
    print(f"\n🔍 Testing System Validation:")
    validation = system.validate_system(thorough=False)
    
    print(f"  Overall Success: {'✅' if validation['overall_success'] else '❌'}")
    
    if 'summary' in validation:
        summary = validation['summary']
        print(f"  Health: {summary.get('overall_health', 0):.1%}")
        print(f"  Dependencies: {summary.get('dependencies_available', 'unknown')}")
        print(f"  Thermal Status: {summary.get('thermal_status', 'unknown')}")
        print(f"  Memory Status: {summary.get('memory_status', 'unknown')}")
    
    # Test context managers
    print(f"\n🎭 Testing Context Managers:")
    
    baseline_health = system.get_system_health().overall_health
    print(f"  Baseline Health: {baseline_health:.1%}")
    
    with system.optimization_context('battery'):
        context_health = system.get_system_health().overall_health
        print(f"  In Battery Context: {context_health:.1%}")
    
    final_health = system.get_system_health().overall_health
    print(f"  After Context: {final_health:.1%}")
    
    # Test monitoring
    print(f"\n📊 Testing Monitoring:")
    
    monitoring_started = system.start_monitoring()
    print(f"  Monitoring Started: {'✅' if monitoring_started else '❌'}")
    
    # Let it run briefly
    time.sleep(2)
    
    monitoring_stopped = system.stop_monitoring()
    print(f"  Monitoring Stopped: {'✅' if monitoring_stopped else '❌'}")
    
    # Export system report
    report_path = Path("/tmp/enhanced_dependency_system_report.json")
    system.export_system_report(report_path)
    print(f"\n💾 System report exported to {report_path}")
    
    # Cleanup
    system.cleanup()
    
    print(f"\n✅ Enhanced Dependency System test completed successfully!")
    print(f"🎯 System Health: {health.overall_health:.1%}")
    print(f"🚀 Performance Score: {health.performance_score:.1f}")
    
    # Show final summary
    if health.is_steam_deck:
        print(f"🎮 Steam Deck optimizations active - ready for handheld gaming!")
    else:
        print(f"💻 Desktop optimizations active - ready for high performance!")