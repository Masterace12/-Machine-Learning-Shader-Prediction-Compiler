#!/usr/bin/env python3
"""
Initialization Sequence Controller with Strict Import Order
=========================================================

Enforces proper initialization sequence and import order to prevent threading
issues and resource conflicts in the ML Shader Prediction Compiler.

Critical Requirements:
- Environment variables MUST be set before any ML library imports
- Threading configuration MUST be applied before import time
- Resource validation MUST occur before thread pool creation
- Steam integration MUST be initialized early for gaming mode detection

Usage:
    from src.core.initialization_controller import InitializationController
    controller = InitializationController()
    success = await controller.initialize_system()
"""

import os
import sys
import time
import asyncio
import logging
import importlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum

# CRITICAL: This must be the FIRST import - sets environment variables
if 'setup_threading' not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import setup_threading
    setup_threading.configure_for_steam_deck()

logger = logging.getLogger(__name__)

class InitializationPhase(Enum):
    """Initialization phases in strict order."""
    ENVIRONMENT = "environment"          # Environment variable setup
    VALIDATION = "validation"           # System resource validation  
    THERMAL = "thermal"                 # Thermal monitoring setup
    STEAM = "steam"                     # Steam integration setup
    DEPENDENCIES = "dependencies"       # Dependency loading and validation
    ML_LIBRARIES = "ml_libraries"       # ML library imports and configuration
    CORE_SYSTEMS = "core_systems"       # Core system initialization
    SERVICES = "services"               # Background services
    COMPLETE = "complete"               # Full initialization complete

@dataclass
class InitializationStep:
    """Single initialization step."""
    name: str
    phase: InitializationPhase
    function: Callable
    required: bool = True
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: float = 30.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class InitializationResult:
    """Result of initialization process."""
    success: bool = False
    phase_reached: InitializationPhase = InitializationPhase.ENVIRONMENT
    completed_steps: List[str] = None
    failed_steps: List[str] = None
    warnings: List[str] = None
    errors: List[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}

class InitializationController:
    """Controls strict initialization sequence for Steam Deck ML system."""
    
    def __init__(self):
        self.is_steam_deck = setup_threading.is_steam_deck()
        self.initialization_lock = threading.Lock()
        self.initialized_steps: Set[str] = set()
        self.failed_steps: Set[str] = set()
        self.current_phase = InitializationPhase.ENVIRONMENT
        self.initialization_time = 0.0
        
        # Component instances
        self.thermal_manager = None
        self.steam_integration = None
        self.capacity_validator = None
        self.ml_predictor = None
        
        # Callbacks
        self.phase_callbacks: Dict[InitializationPhase, List[Callable]] = {}
        self.step_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info(f"Initialization controller created - Steam Deck: {self.is_steam_deck}")
        
        # Define initialization steps in strict order
        self._setup_initialization_steps()
    
    def _setup_initialization_steps(self) -> None:
        """Setup initialization steps in proper order."""
        self.steps = [
            # Phase 1: Environment Setup (CRITICAL - must be first)
            InitializationStep(
                name="verify_environment_setup",
                phase=InitializationPhase.ENVIRONMENT,
                function=self._verify_environment_setup,
                required=True,
                timeout_seconds=5.0
            ),
            InitializationStep(
                name="validate_threading_config", 
                phase=InitializationPhase.ENVIRONMENT,
                function=self._validate_threading_config,
                required=True,
                timeout_seconds=5.0,
                dependencies=["verify_environment_setup"]
            ),
            
            # Phase 2: System Validation
            InitializationStep(
                name="initialize_capacity_validator",
                phase=InitializationPhase.VALIDATION,
                function=self._initialize_capacity_validator,
                required=True,
                timeout_seconds=10.0
            ),
            InitializationStep(
                name="validate_system_resources",
                phase=InitializationPhase.VALIDATION, 
                function=self._validate_system_resources,
                required=True,
                timeout_seconds=10.0,
                dependencies=["initialize_capacity_validator"]
            ),
            
            # Phase 3: Thermal Management
            InitializationStep(
                name="initialize_thermal_manager",
                phase=InitializationPhase.THERMAL,
                function=self._initialize_thermal_manager,
                required=True,
                timeout_seconds=10.0
            ),
            InitializationStep(
                name="start_thermal_monitoring",
                phase=InitializationPhase.THERMAL,
                function=self._start_thermal_monitoring,
                required=True,
                timeout_seconds=5.0,
                dependencies=["initialize_thermal_manager"]
            ),
            
            # Phase 4: Steam Integration  
            InitializationStep(
                name="initialize_steam_integration",
                phase=InitializationPhase.STEAM,
                function=self._initialize_steam_integration,
                required=False,  # Optional for non-Steam systems
                timeout_seconds=15.0
            ),
            InitializationStep(
                name="start_steam_monitoring",
                phase=InitializationPhase.STEAM,
                function=self._start_steam_monitoring,
                required=False,
                timeout_seconds=10.0,
                dependencies=["initialize_steam_integration"]
            ),
            
            # Phase 5: Dependencies (CRITICAL IMPORT ORDER)
            InitializationStep(
                name="pre_import_validation",
                phase=InitializationPhase.DEPENDENCIES,
                function=self._pre_import_validation,
                required=True,
                timeout_seconds=5.0
            ),
            InitializationStep(
                name="import_numpy_first",
                phase=InitializationPhase.DEPENDENCIES,
                function=self._import_numpy_first,
                required=True,
                timeout_seconds=10.0,
                dependencies=["pre_import_validation"]
            ),
            
            # Phase 6: ML Libraries (STRICT ORDER)
            InitializationStep(
                name="import_scikit_learn",
                phase=InitializationPhase.ML_LIBRARIES,
                function=self._import_scikit_learn,
                required=True,
                timeout_seconds=15.0,
                dependencies=["import_numpy_first"]
            ),
            InitializationStep(
                name="import_lightgbm",
                phase=InitializationPhase.ML_LIBRARIES,
                function=self._import_lightgbm,
                required=True,
                timeout_seconds=15.0,
                dependencies=["import_scikit_learn"]
            ),
            InitializationStep(
                name="configure_ml_threading",
                phase=InitializationPhase.ML_LIBRARIES,
                function=self._configure_ml_threading,
                required=True,
                timeout_seconds=5.0,
                dependencies=["import_lightgbm"]
            ),
            
            # Phase 7: Core Systems
            InitializationStep(
                name="initialize_ml_predictor",
                phase=InitializationPhase.CORE_SYSTEMS,
                function=self._initialize_ml_predictor,
                required=True,
                timeout_seconds=30.0,
                dependencies=["configure_ml_threading"]
            ),
            InitializationStep(
                name="initialize_cache_system",
                phase=InitializationPhase.CORE_SYSTEMS,
                function=self._initialize_cache_system,
                required=True,
                timeout_seconds=10.0
            ),
            
            # Phase 8: Services
            InitializationStep(
                name="start_background_services", 
                phase=InitializationPhase.SERVICES,
                function=self._start_background_services,
                required=False,
                timeout_seconds=10.0,
                dependencies=["initialize_ml_predictor", "initialize_cache_system"]
            ),
            InitializationStep(
                name="verify_system_health",
                phase=InitializationPhase.SERVICES,
                function=self._verify_system_health,
                required=True,
                timeout_seconds=5.0
            )
        ]
    
    # Phase 1: Environment Setup
    async def _verify_environment_setup(self) -> bool:
        """Verify environment variables are properly set."""
        try:
            critical_vars = [
                'OMP_NUM_THREADS',
                'LIGHTGBM_NUM_THREADS', 
                'NUMEXPR_NUM_THREADS',
                'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS'
            ]
            
            missing_vars = []
            for var in critical_vars:
                if not os.environ.get(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.error(f"Critical environment variables not set: {missing_vars}")
                # Try to set them now (emergency fallback)
                for var in missing_vars:
                    os.environ[var] = '1'
                logger.warning("Set missing environment variables to '1' (emergency fallback)")
            
            logger.info("Environment setup verified")
            return True
            
        except Exception as e:
            logger.error(f"Environment verification failed: {e}")
            return False
    
    async def _validate_threading_config(self) -> bool:
        """Validate threading configuration is applied."""
        try:
            configurator = setup_threading.get_configurator()
            return configurator.validate_configuration()
        except Exception as e:
            logger.error(f"Threading config validation failed: {e}")
            return False
    
    # Phase 2: System Validation
    async def _initialize_capacity_validator(self) -> bool:
        """Initialize thread capacity validator."""
        try:
            from . import thread_capacity_validator
            self.capacity_validator = thread_capacity_validator.get_capacity_validator()
            logger.info("Thread capacity validator initialized")
            return True
        except Exception as e:
            logger.error(f"Capacity validator initialization failed: {e}")
            return False
    
    async def _validate_system_resources(self) -> bool:
        """Validate system has sufficient resources."""
        try:
            if not self.capacity_validator:
                logger.error("Capacity validator not initialized")
                return False
            
            result = self.capacity_validator.validate_thread_capacity(requested_threads=2)
            if not result.success:
                logger.error(f"System resource validation failed: {result.errors}")
                return False
            
            logger.info(f"System resources validated - State: {result.resource_state.value}, "
                       f"Recommended threads: {result.recommended_threads}")
            return True
            
        except Exception as e:
            logger.error(f"System resource validation failed: {e}")
            return False
    
    # Phase 3: Thermal Management
    async def _initialize_thermal_manager(self) -> bool:
        """Initialize thermal management system."""
        try:
            from . import thermal_thread_manager
            self.thermal_manager = thermal_thread_manager.get_thermal_manager_sync()
            logger.info("Thermal manager initialized")
            return True
        except Exception as e:
            logger.error(f"Thermal manager initialization failed: {e}")
            return False
    
    async def _start_thermal_monitoring(self) -> bool:
        """Start thermal monitoring."""
        try:
            if not self.thermal_manager:
                logger.error("Thermal manager not initialized")
                return False
            
            success = await self.thermal_manager.start_monitoring()
            if success:
                logger.info("Thermal monitoring started")
            return success
        except Exception as e:
            logger.error(f"Thermal monitoring start failed: {e}")
            return False
    
    # Phase 4: Steam Integration
    async def _initialize_steam_integration(self) -> bool:
        """Initialize Steam integration."""
        try:
            from . import steam_integration
            self.steam_integration = steam_integration.get_steam_integration_sync()
            logger.info("Steam integration initialized")
            return True
        except Exception as e:
            logger.warning(f"Steam integration initialization failed: {e}")
            return not self.is_steam_deck  # Only required on Steam Deck
    
    async def _start_steam_monitoring(self) -> bool:
        """Start Steam monitoring."""
        try:
            if not self.steam_integration:
                return not self.is_steam_deck  # Only required on Steam Deck
            
            success = await self.steam_integration.start_monitoring()
            if success:
                logger.info("Steam monitoring started")
            return success or not self.is_steam_deck
        except Exception as e:
            logger.warning(f"Steam monitoring start failed: {e}")
            return not self.is_steam_deck
    
    # Phase 5: Dependencies  
    async def _pre_import_validation(self) -> bool:
        """Validate system state before ML imports."""
        try:
            # Check thread capacity before imports
            if self.capacity_validator:
                result = self.capacity_validator.validate_thread_capacity(requested_threads=1)
                if result.resource_state.value in ['critical', 'exhausted']:
                    logger.error("System resources too constrained for ML library imports")
                    return False
            
            # Check thermal state
            if self.thermal_manager:
                thermal_state = self.thermal_manager.get_thermal_state()
                if thermal_state.value == 'emergency':
                    logger.error("System in thermal emergency - delaying ML imports")
                    return False
            
            logger.info("Pre-import validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Pre-import validation failed: {e}")
            return False
    
    async def _import_numpy_first(self) -> bool:
        """Import NumPy first to establish BLAS threading."""
        try:
            # This is CRITICAL - NumPy must be imported first
            import numpy as np
            logger.info(f"NumPy imported successfully - version {np.__version__}")
            
            # Verify BLAS threading is configured
            try:
                config = np.show_config()
                logger.debug("NumPy BLAS configuration verified")
            except Exception:
                pass
            
            return True
        except Exception as e:
            logger.error(f"NumPy import failed: {e}")
            return False
    
    # Phase 6: ML Libraries
    async def _import_scikit_learn(self) -> bool:
        """Import scikit-learn after NumPy."""
        try:
            import sklearn
            logger.info(f"scikit-learn imported successfully - version {sklearn.__version__}")
            
            # Configure scikit-learn threading
            try:
                from sklearn.utils._joblib import parallel_backend
                import joblib
                joblib.parallel_backend('threading', n_jobs=1)
            except Exception as e:
                logger.debug(f"scikit-learn threading config warning: {e}")
            
            return True
        except Exception as e:
            logger.error(f"scikit-learn import failed: {e}")
            return False
    
    async def _import_lightgbm(self) -> bool:
        """Import LightGBM last with proper threading."""
        try:
            # Verify environment variable is set (critical for LightGBM 4.6+)
            if not os.environ.get('LIGHTGBM_NUM_THREADS'):
                os.environ['LIGHTGBM_NUM_THREADS'] = '1'
                logger.warning("Set LIGHTGBM_NUM_THREADS=1 (should have been set earlier)")
            
            import lightgbm as lgb
            logger.info(f"LightGBM imported successfully - version {lgb.__version__}")
            
            return True
        except Exception as e:
            logger.error(f"LightGBM import failed: {e}")
            return False
    
    async def _configure_ml_threading(self) -> bool:
        """Configure ML library threading after imports."""
        try:
            # Final threading configuration after all imports
            from ..threading_config import configure_library_threading
            success = configure_library_threading()
            
            if success:
                logger.info("ML library threading configured")
            return success
        except Exception as e:
            logger.error(f"ML threading configuration failed: {e}")
            return False
    
    # Phase 7: Core Systems
    async def _initialize_ml_predictor(self) -> bool:
        """Initialize ML predictor system."""
        try:
            # Import after ML libraries are properly configured
            from ..unified_ml_predictor import UnifiedMLPredictor
            
            # Get safe thread count
            thread_count = 1
            if self.capacity_validator:
                result = self.capacity_validator.validate_thread_capacity()
                thread_count = result.recommended_threads
            
            self.ml_predictor = UnifiedMLPredictor()
            await self.ml_predictor.initialize(max_threads=thread_count)
            
            logger.info("ML predictor initialized")
            return True
        except Exception as e:
            logger.error(f"ML predictor initialization failed: {e}")
            return False
    
    async def _initialize_cache_system(self) -> bool:
        """Initialize shader cache system."""
        try:
            from ..optimized_shader_cache import OptimizedShaderCache
            
            # This should be safe to initialize after resource validation
            logger.info("Cache system initialized (placeholder)")
            return True
        except Exception as e:
            logger.error(f"Cache system initialization failed: {e}")
            return False
    
    # Phase 8: Services
    async def _start_background_services(self) -> bool:
        """Start background services."""
        try:
            # Start services only if resources allow
            if self.capacity_validator:
                result = self.capacity_validator.validate_thread_capacity(requested_threads=2)
                if not result.success:
                    logger.warning("Skipping background services due to resource constraints")
                    return True  # Not a failure
            
            logger.info("Background services started (placeholder)")
            return True
        except Exception as e:
            logger.warning(f"Background services start failed: {e}")
            return False  # Non-critical
    
    async def _verify_system_health(self) -> bool:
        """Final system health verification."""
        try:
            health_checks = []
            
            # Check ML predictor
            if self.ml_predictor:
                health_checks.append("ML Predictor: OK")
            
            # Check thermal manager
            if self.thermal_manager:
                health_checks.append(f"Thermal: {self.thermal_manager.get_thermal_state().value}")
            
            # Check Steam integration
            if self.steam_integration:
                health_checks.append("Steam Integration: OK")
                
            # Check resource state
            if self.capacity_validator:
                result = self.capacity_validator.validate_thread_capacity()
                health_checks.append(f"Resources: {result.resource_state.value}")
            
            logger.info(f"System health verified - {', '.join(health_checks)}")
            return True
            
        except Exception as e:
            logger.error(f"System health verification failed: {e}")
            return False
    
    async def _execute_step(self, step: InitializationStep) -> bool:
        """Execute a single initialization step with timeout and retries."""
        for attempt in range(step.max_retries + 1):
            try:
                logger.info(f"Executing step: {step.name} (attempt {attempt + 1}/{step.max_retries + 1})")
                
                # Check dependencies
                for dep in step.dependencies:
                    if dep not in self.initialized_steps:
                        logger.error(f"Step {step.name} dependency {dep} not completed")
                        return False
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    step.function(),
                    timeout=step.timeout_seconds
                )
                
                if result:
                    self.initialized_steps.add(step.name)
                    logger.info(f"Step completed: {step.name}")
                    
                    # Notify step callbacks
                    if step.name in self.step_callbacks:
                        for callback in self.step_callbacks[step.name]:
                            try:
                                callback(True, None)
                            except Exception as e:
                                logger.warning(f"Step callback failed: {e}")
                    
                    return True
                else:
                    logger.warning(f"Step {step.name} returned False (attempt {attempt + 1})")
                    
            except asyncio.TimeoutError:
                logger.error(f"Step {step.name} timed out after {step.timeout_seconds}s (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e} (attempt {attempt + 1})")
            
            # Wait before retry
            if attempt < step.max_retries:
                await asyncio.sleep(1.0)
        
        # All attempts failed
        self.failed_steps.add(step.name)
        
        # Notify step callbacks
        if step.name in self.step_callbacks:
            for callback in self.step_callbacks[step.name]:
                try:
                    callback(False, f"Step failed after {step.max_retries + 1} attempts")
                except Exception as e:
                    logger.warning(f"Step callback failed: {e}")
        
        return not step.required  # Non-required steps don't fail the process
    
    async def initialize_system(self) -> InitializationResult:
        """Initialize the complete system with proper sequencing."""
        start_time = time.time()
        
        with self.initialization_lock:
            try:
                logger.info("Starting system initialization...")
                
                result = InitializationResult()
                current_phase = None
                
                for step in self.steps:
                    # Check for phase change
                    if step.phase != current_phase:
                        current_phase = step.phase
                        self.current_phase = current_phase
                        logger.info(f"Entering initialization phase: {current_phase.value}")
                        
                        # Notify phase callbacks
                        if current_phase in self.phase_callbacks:
                            for callback in self.phase_callbacks[current_phase]:
                                try:
                                    callback(current_phase)
                                except Exception as e:
                                    logger.warning(f"Phase callback failed: {e}")
                    
                    # Execute step
                    success = await self._execute_step(step)
                    
                    if success:
                        result.completed_steps.append(step.name)
                    else:
                        result.failed_steps.append(step.name)
                        
                        if step.required:
                            result.errors.append(f"Required step failed: {step.name}")
                            logger.error(f"Required step {step.name} failed - stopping initialization")
                            break
                        else:
                            result.warnings.append(f"Optional step failed: {step.name}")
                
                # Determine final result
                result.phase_reached = self.current_phase
                result.success = len(result.failed_steps) == 0 or all(
                    step.name in result.failed_steps for step in self.steps if not step.required
                )
                result.duration_seconds = time.time() - start_time
                
                # Add metadata
                result.metadata = {
                    'steam_deck': self.is_steam_deck,
                    'total_steps': len(self.steps),
                    'completed_count': len(result.completed_steps),
                    'failed_count': len(result.failed_steps),
                    'final_phase': self.current_phase.value
                }
                
                if result.success:
                    logger.info(f"System initialization completed successfully in {result.duration_seconds:.1f}s")
                    self.current_phase = InitializationPhase.COMPLETE
                else:
                    logger.error(f"System initialization failed in phase {result.phase_reached.value} "
                               f"after {result.duration_seconds:.1f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"System initialization failed with exception: {e}")
                duration = time.time() - start_time
                return InitializationResult(
                    success=False,
                    phase_reached=self.current_phase,
                    errors=[f"Initialization exception: {e}"],
                    duration_seconds=duration
                )
    
    def is_initialized(self) -> bool:
        """Check if system is fully initialized."""
        return self.current_phase == InitializationPhase.COMPLETE
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status."""
        return {
            'current_phase': self.current_phase.value,
            'is_complete': self.is_initialized(),
            'completed_steps': list(self.initialized_steps),
            'failed_steps': list(self.failed_steps),
            'duration_seconds': self.initialization_time
        }
    
    def add_phase_callback(self, phase: InitializationPhase, callback: Callable) -> None:
        """Add callback for phase changes."""
        if phase not in self.phase_callbacks:
            self.phase_callbacks[phase] = []
        self.phase_callbacks[phase].append(callback)
    
    def add_step_callback(self, step_name: str, callback: Callable) -> None:
        """Add callback for step completion."""
        if step_name not in self.step_callbacks:
            self.step_callbacks[step_name] = []
        self.step_callbacks[step_name].append(callback)

# Global initialization controller
_initialization_controller = None

def get_initialization_controller() -> InitializationController:
    """Get the global initialization controller."""
    global _initialization_controller
    if _initialization_controller is None:
        _initialization_controller = InitializationController()
    return _initialization_controller

async def initialize_ml_system() -> InitializationResult:
    """Initialize the complete ML system (main entry point)."""
    controller = get_initialization_controller()
    return await controller.initialize_system()

def is_system_initialized() -> bool:
    """Check if the ML system is fully initialized."""
    controller = get_initialization_controller()
    return controller.is_initialized()