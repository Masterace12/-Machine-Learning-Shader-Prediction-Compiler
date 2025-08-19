#!/usr/bin/env python3
"""
Steam Deck Threading Startup Configuration
Initialize all threading optimizations at system startup
"""

import os
import sys
import logging
import signal
import atexit
from typing import Optional, Dict, Any
from pathlib import Path

# Import all threading components
try:
    from .threading_config import configure_threading_for_steam_deck, ThreadingConfig
    from .thread_pool_manager import get_thread_manager
    from .thread_diagnostics import get_thread_diagnostics
    HAS_THREADING_COMPONENTS = True
except ImportError as e:
    HAS_THREADING_COMPONENTS = False
    print(f"Warning: Threading components not available: {e}")

class SteamDeckThreadingStartup:
    """
    Startup manager for Steam Deck threading optimizations
    
    Handles:
    - Early environment configuration
    - Thread manager initialization
    - Diagnostics setup
    - Cleanup registration
    - Error recovery
    """
    
    def __init__(self, config: Optional[ThreadingConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.initialized = False
        
        # Component references
        self.threading_configurator = None
        self.thread_manager = None
        self.diagnostics = None
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, cleaning up threading systems...")
        self.cleanup()
    
    def initialize(self) -> bool:
        """Initialize all threading components"""
        if self.initialized:
            return True
        
        if not HAS_THREADING_COMPONENTS:
            self.logger.error("Threading components not available - cannot initialize")
            return False
        
        try:
            self.logger.info("Initializing Steam Deck threading optimizations...")
            
            # Step 1: Configure environment and libraries
            self.threading_configurator = configure_threading_for_steam_deck(self.config)
            self.logger.info("Threading configuration completed")
            
            # Step 2: Initialize thread manager
            self.thread_manager = get_thread_manager()
            self.logger.info("Thread pool manager initialized")
            
            # Step 3: Start diagnostics
            self.diagnostics = get_thread_diagnostics()
            self.logger.info("Thread diagnostics started")
            
            # Step 4: Validate initialization
            if not self._validate_initialization():
                self.logger.error("Threading initialization validation failed")
                return False
            
            self.initialized = True
            self.logger.info("✅ Steam Deck threading optimizations initialized successfully")
            
            # Log configuration summary
            self._log_configuration_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Threading initialization failed: {e}")
            self.cleanup()
            return False
    
    def _validate_initialization(self) -> bool:
        """Validate that all components initialized correctly"""
        try:
            # Check environment variables
            required_env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']
            for var in required_env_vars:
                if var not in os.environ:
                    self.logger.warning(f"Environment variable {var} not set")
                    return False
            
            # Check thread manager
            if not self.thread_manager:
                self.logger.error("Thread manager not initialized")
                return False
            
            # Test thread manager functionality
            try:
                metrics = self.thread_manager.get_thread_metrics()
                if not isinstance(metrics, dict):
                    self.logger.error("Thread manager metrics invalid")
                    return False
            except Exception as e:
                self.logger.error(f"Thread manager test failed: {e}")
                return False
            
            # Check diagnostics
            if not self.diagnostics:
                self.logger.error("Thread diagnostics not initialized")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
    
    def _log_configuration_summary(self):
        """Log summary of current threading configuration"""
        try:
            if self.threading_configurator:
                status = self.threading_configurator.get_configuration_status()
                self.logger.info(f"Threading Configuration Summary:")
                self.logger.info(f"  Steam Deck: {status['is_steam_deck']}")
                self.logger.info(f"  CPU cores: {status['cpu_count']}")
                self.logger.info(f"  Configured libraries: {', '.join(status['configured_libraries'])}")
                self.logger.info(f"  Max threads: {status['config']['max_threads']}")
                self.logger.info(f"  ML threads: {status['config']['ml_threads']}")
            
            if self.thread_manager:
                metrics = self.thread_manager.get_thread_metrics()
                self.logger.info(f"Thread Pool Status:")
                self.logger.info(f"  Active threads: {metrics['total_active_threads']}")
                self.logger.info(f"  Resource state: {metrics['resource_state']}")
                self.logger.info(f"  Gaming mode: {metrics['gaming_mode_active']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log configuration summary: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of threading system"""
        status = {
            'initialized': self.initialized,
            'components_available': HAS_THREADING_COMPONENTS,
            'components': {
                'threading_configurator': self.threading_configurator is not None,
                'thread_manager': self.thread_manager is not None,
                'diagnostics': self.diagnostics is not None
            }
        }
        
        if self.initialized:
            try:
                # Get detailed status from components
                if self.threading_configurator:
                    status['threading_config'] = self.threading_configurator.get_configuration_status()
                
                if self.thread_manager:
                    status['thread_metrics'] = self.thread_manager.get_thread_metrics()
                
                if self.diagnostics:
                    status['diagnostic_report'] = self.diagnostics.get_diagnostic_report()
                
            except Exception as e:
                status['status_error'] = str(e)
        
        return status
    
    def apply_runtime_optimizations(self):
        """Apply runtime optimizations based on current system state"""
        if not self.initialized:
            self.logger.warning("Cannot apply runtime optimizations - system not initialized")
            return
        
        try:
            # Get current system state
            if self.diagnostics:
                report = self.diagnostics.get_diagnostic_report()
                issues = report.get('issues', {})
                
                # Apply fixes for detected issues
                if issues.get('by_severity', {}).get('critical', 0) > 0:
                    self.logger.warning("Critical threading issues detected - applying emergency fixes")
                    fixes = self.diagnostics.apply_fixes(auto_fix=True)
                    self.logger.info(f"Applied {len(fixes.get('fixes_applied', []))} emergency fixes")
            
            # Update thermal state if available
            if self.thread_manager and hasattr(self.thread_manager, '_resource_state'):
                current_state = self.thread_manager._resource_state
                if self.threading_configurator:
                    self.threading_configurator.update_thermal_state(current_state.value)
            
        except Exception as e:
            self.logger.error(f"Runtime optimization failed: {e}")
    
    def cleanup(self):
        """Cleanup all threading components"""
        if not self.initialized:
            return
        
        try:
            self.logger.info("Cleaning up threading components...")
            
            # Stop diagnostics first
            if self.diagnostics:
                try:
                    self.diagnostics.stop_monitoring()
                    self.logger.debug("Thread diagnostics stopped")
                except Exception as e:
                    self.logger.warning(f"Error stopping diagnostics: {e}")
            
            # Shutdown thread manager
            if self.thread_manager:
                try:
                    self.thread_manager.shutdown(wait=True, timeout=10.0)
                    self.logger.debug("Thread manager shutdown completed")
                except Exception as e:
                    self.logger.warning(f"Error shutting down thread manager: {e}")
            
            # Cleanup configurator
            if self.threading_configurator:
                try:
                    self.threading_configurator.cleanup()
                    self.logger.debug("Threading configurator cleanup completed")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up configurator: {e}")
            
            self.initialized = False
            self.logger.info("✅ Threading component cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# Global startup manager
_startup_manager = None


def initialize_steam_deck_threading(config: Optional[ThreadingConfig] = None) -> bool:
    """Initialize Steam Deck threading optimizations (main entry point)"""
    global _startup_manager
    
    if _startup_manager is None:
        _startup_manager = SteamDeckThreadingStartup(config)
    
    return _startup_manager.initialize()


def get_threading_status() -> Dict[str, Any]:
    """Get current threading system status"""
    if _startup_manager:
        return _startup_manager.get_status()
    else:
        return {
            'initialized': False,
            'error': 'Threading system not initialized'
        }


def apply_threading_optimizations():
    """Apply runtime threading optimizations"""
    if _startup_manager:
        _startup_manager.apply_runtime_optimizations()


def cleanup_threading():
    """Cleanup threading system"""
    global _startup_manager
    if _startup_manager:
        _startup_manager.cleanup()
        _startup_manager = None


def diagnose_threading_problems() -> Dict[str, Any]:
    """Quick diagnosis of threading problems"""
    if not HAS_THREADING_COMPONENTS:
        return {
            'error': 'Threading components not available',
            'suggestion': 'Ensure all threading modules are properly installed'
        }
    
    try:
        # Initialize if needed
        if not _startup_manager or not _startup_manager.initialized:
            success = initialize_steam_deck_threading()
            if not success:
                return {
                    'error': 'Failed to initialize threading system',
                    'suggestion': 'Check logs for detailed error messages'
                }
        
        # Get diagnostic report
        status = get_threading_status()
        
        if 'diagnostic_report' in status:
            report = status['diagnostic_report']
            issues = report.get('issues', {})
            
            return {
                'total_threads': report.get('system_info', {}).get('total_threads', 0),
                'critical_issues': issues.get('by_severity', {}).get('critical', 0),
                'high_issues': issues.get('by_severity', {}).get('high', 0),
                'recommendations': report.get('recommendations', []),
                'status': 'healthy' if issues.get('by_severity', {}).get('critical', 0) == 0 else 'issues_detected'
            }
        else:
            return {
                'error': 'Diagnostic report not available',
                'status': 'unknown'
            }
            
    except Exception as e:
        return {
            'error': f'Diagnosis failed: {e}',
            'suggestion': 'Manual investigation required'
        }


if __name__ == "__main__":
    # Test startup system
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Steam Deck Threading Startup Test")
    print("=" * 40)
    
    # Initialize
    success = initialize_steam_deck_threading()
    
    if success:
        print("✅ Threading system initialized successfully")
        
        # Show status
        status = get_threading_status()
        print(f"Active threads: {status.get('thread_metrics', {}).get('total_active_threads', 'unknown')}")
        print(f"Resource state: {status.get('thread_metrics', {}).get('resource_state', 'unknown')}")
        
        # Test optimizations
        apply_threading_optimizations()
        print("✅ Runtime optimizations applied")
        
        # Test diagnosis
        diagnosis = diagnose_threading_problems()
        print(f"System health: {diagnosis.get('status', 'unknown')}")
        
    else:
        print("❌ Threading system initialization failed")
    
    # Cleanup
    cleanup_threading()
    print("✅ Cleanup completed")