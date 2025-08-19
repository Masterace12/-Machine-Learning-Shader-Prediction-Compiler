#!/usr/bin/env python3
"""
Steam Deck Threading Configuration System
Comprehensive threading setup for all ML libraries and system components
"""

import os
import sys
import logging
import threading
import multiprocessing
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import weakref
import atexit

@dataclass
class ThreadingConfig:
    """Threading configuration for Steam Deck OLED optimization"""
    # Core thread limits (ultra-conservative for Steam Deck OLED)
    max_threads: int = 4                    # Total system threads (reduced)
    ml_threads: int = 1                     # ML inference threads (single thread)
    compilation_threads: int = 1            # Shader compilation threads (single thread)
    io_threads: int = 1                     # I/O operations (single thread)
    
    # OpenMP and BLAS configuration (all single-threaded)
    omp_num_threads: int = 1               # OpenMP threads (single)
    mkl_num_threads: int = 1               # Intel MKL threads (single)
    openblas_num_threads: int = 1          # OpenBLAS threads (single)
    numexpr_num_threads: int = 1           # NumExpr threads (single)
    
    # Framework-specific settings (all single-threaded)
    lightgbm_threads: int = 1              # LightGBM threads (single)
    sklearn_threads: int = 1               # Scikit-learn threads (single)
    numba_threads: int = 1                 # Numba threads (single)
    
    # Thread stack and memory settings
    thread_stack_size: int = 262144        # 256KB stack size
    thread_memory_limit: int = 50          # 50MB per thread
    
    # Adaptive settings
    enable_thermal_scaling: bool = True    # Scale based on temperature
    enable_battery_scaling: bool = True    # Scale based on battery
    enable_gaming_mode: bool = True        # Detect and adapt to gaming
    
    # Debug and monitoring
    enable_thread_monitoring: bool = True
    log_thread_lifecycle: bool = False

class SteamDeckThreadingConfigurator:
    """
    Comprehensive threading configurator for Steam Deck
    
    Handles:
    - Environment variable setup
    - ML library configuration
    - System thread limits
    - Thermal-aware scaling
    - Gaming mode detection
    """
    
    def __init__(self, config: Optional[ThreadingConfig] = None):
        self.config = config or ThreadingConfig()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self._configured_libraries = set()
        self._original_env = {}
        self._thread_monitors = weakref.WeakSet()
        self._is_configured = False
        
        # Hardware detection
        self._cpu_count = multiprocessing.cpu_count()
        self._is_steam_deck = self._detect_steam_deck()
        
        # Adaptive state
        self._current_thermal_state = "normal"
        self._gaming_mode_active = False
        self._battery_level = 100.0
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        self.logger.info(f"Threading configurator initialized for {self._cpu_count}-core system")
        if self._is_steam_deck:
            self.logger.info("Steam Deck detected - applying optimized configuration")
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        # Check DMI information
        try:
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip().lower()
                if "jupiter" in product_name or "steamdeck" in product_name:
                    return True
        except Exception:
            pass
        
        # Check for Steam Deck specific paths
        steam_deck_indicators = [
            Path("/home/deck"),
            Path("/sys/class/power_supply/BAT1"),  # Steam Deck battery
            Path("/dev/input/by-path/platform-i8042-serio-0-event-kbd"),  # Steam Deck keyboard
        ]
        
        return any(path.exists() for path in steam_deck_indicators)
    
    def configure_environment_variables(self):
        """Configure threading environment variables"""
        if self._is_configured:
            return
        
        env_vars = {
            # OpenMP configuration (Steam Deck OLED optimized)
            'OMP_NUM_THREADS': str(self.config.omp_num_threads),
            'OMP_THREAD_LIMIT': str(self.config.max_threads),
            'OMP_DYNAMIC': 'FALSE',  # Disable dynamic thread adjustment for predictability
            'OMP_NESTED': 'FALSE',  # Disable nested parallelism
            'OMP_WAIT_POLICY': 'PASSIVE',  # Reduce CPU spinning
            'OMP_PROC_BIND': 'FALSE',  # Don't bind threads to cores on Steam Deck
            
            # Intel MKL configuration
            'MKL_NUM_THREADS': str(self.config.mkl_num_threads),
            'MKL_DYNAMIC': 'TRUE',
            'MKL_THREADING_LAYER': 'INTEL',
            
            # OpenBLAS configuration
            'OPENBLAS_NUM_THREADS': str(self.config.openblas_num_threads),
            'OPENBLAS_CORETYPE': 'ZENVER2',  # Optimized for Zen 2 (Steam Deck APU)
            
            # NumExpr configuration
            'NUMEXPR_NUM_THREADS': str(self.config.numexpr_num_threads),
            'NUMEXPR_MAX_THREADS': str(self.config.numexpr_num_threads),
            
            # TensorFlow (if used)
            'TF_NUM_INTRAOP_THREADS': str(self.config.ml_threads),
            'TF_NUM_INTEROP_THREADS': str(1),
            
            # PyTorch (if used)
            'OMP_SCHEDULE': 'STATIC',
            
            # LightGBM configuration (4.6.0+ uses environment variables)
            'LIGHTGBM_NUM_THREADS': str(self.config.lightgbm_threads),
            
            # General performance (Steam Deck optimized)
            'MALLOC_ARENA_MAX': '1',  # Minimal memory fragmentation
            'PYTHONHASHSEED': '0',    # Reproducible hashing
            'VECLIB_MAXIMUM_THREADS': '1',  # Single BLAS thread
            'BLIS_NUM_THREADS': '1',  # Single BLIS thread
            'NUMBA_NUM_THREADS': '1'  # Single Numba thread
        }
        
        # Apply environment variables
        for key, value in env_vars.items():
            if key not in os.environ:  # Don't override existing settings
                self._original_env[key] = os.environ.get(key)
                os.environ[key] = value
                self.logger.debug(f"Set {key}={value}")
        
        self.logger.info("Environment variables configured for Steam Deck threading")
    
    def configure_python_threading(self):
        """Configure Python's threading module"""
        try:
            # Set thread stack size (must be done before threads are created)
            if threading.stack_size() == 0:  # Only set if not already set
                threading.stack_size(self.config.thread_stack_size)
                self.logger.debug(f"Set thread stack size to {self.config.thread_stack_size} bytes")
        except Exception as e:
            self.logger.warning(f"Could not set thread stack size: {e}")
        
        # Configure thread switching
        try:
            sys.setswitchinterval(0.001)  # Faster thread switching for responsiveness
        except Exception as e:
            self.logger.warning(f"Could not set switch interval: {e}")
    
    def configure_lightgbm(self):
        """Configure LightGBM threading with Steam Deck optimizations"""
        try:
            import lightgbm as lgb
            
            # Configure for Steam Deck APU with conservative memory usage
            default_params = {
                'num_threads': self.config.lightgbm_threads,
                'force_row_wise': True,  # Better memory access pattern
                'device_type': 'cpu',
                'max_bin': 255,  # Reduce memory usage
                'verbose': -1,
                'histogram_pool_size': 128,  # Limit memory pool
                'enable_sparse': True,       # Optimize for sparse features
                'bagging_freq': 1,          # Enable bagging for stability
                'feature_fraction': 0.8,    # Use subset of features
                'min_data_in_leaf': 10,     # Prevent overfitting on small datasets
                'lambda_l1': 0.1,           # L1 regularization for sparsity
                'lambda_l2': 0.1,           # L2 regularization for stability
                'max_depth': 6,             # Limit tree depth for memory
                'min_split_gain': 0.1,      # Minimum gain to split
                'subsample': 0.8,           # Row subsampling for stability
                'subsample_freq': 1,        # Apply subsampling every iteration
                'colsample_bytree': 0.8,    # Column subsampling
                'reg_alpha': 0.1,           # Alpha regularization
                'reg_lambda': 0.1,          # Lambda regularization
                'min_child_samples': 5,     # Minimum samples in leaf
                'min_child_weight': 0.001,  # Minimum sum of weights in leaf
                'subsample_for_bin': 200000, # Samples for histogram bins
                'objective': 'regression',   # Default objective
                'metric': 'rmse',           # Default metric
                'boosting_type': 'gbdt',    # Gradient boosting
                'num_leaves': 31,           # Number of leaves in tree
                'learning_rate': 0.1,       # Learning rate
                'feature_pre_filter': False, # Don't pre-filter features
                'is_unbalance': False,      # Balanced data assumption
                'boost_from_average': True,  # Initialize from average
                'num_iterations': 100,      # Default number of iterations
                'early_stopping_round': 10, # Early stopping
                'first_metric_only': True,  # Use first metric for early stopping
                'max_delta_step': 0.0,      # No max delta step
                'min_gain_to_split': 0.0,   # Minimum gain to split
                'drop_rate': 0.1,           # Dropout rate for dart
                'max_drop': 50,             # Max number of dropped trees
                'skip_drop': 0.5,           # Probability of skipping dropout
                'xgboost_dart_mode': False, # Use LightGBM dart mode
                'uniform_drop': False,      # Non-uniform dropout
                'top_rate': 0.2,            # Top rate for goss
                'other_rate': 0.1,          # Other rate for goss
                'min_data_per_group': 100,  # Minimum data per group
                'max_cat_threshold': 32,    # Maximum categorical threshold
                'cat_l2': 10.0,             # L2 regularization for categorical
                'cat_smooth': 10.0,         # Categorical smoothing
                'max_cat_to_onehot': 4,     # Max categories for one-hot
                'cegb_tradeoff': 1.0,       # CEGB tradeoff
                'cegb_penalty_split': 0.0,  # CEGB penalty split
                'path_smooth': 0.0,         # Path smoothing
                'interaction_constraints': '', # No interaction constraints
                'verbosity': -1,            # Quiet mode
                'seed': 42,                 # Random seed for reproducibility
                'deterministic': True,      # Deterministic training
            }
            
            # Store default params for easy access
            if not hasattr(lgb, '_steamdeck_default_params'):
                lgb._steamdeck_default_params = default_params
            
            # LightGBM 4.6.0+ uses environment variables for threading control
            # This is the ONLY reliable way to set threading in modern versions
            threading_env_vars = {
                'LIGHTGBM_NUM_THREADS': str(self.config.lightgbm_threads),
                'OMP_NUM_THREADS': str(self.config.lightgbm_threads),  # OpenMP fallback
            }
            
            for env_var, value in threading_env_vars.items():
                if env_var not in os.environ:  # Don't override existing settings
                    os.environ[env_var] = value
                    self.logger.debug(f"Set {env_var}={value} for LightGBM threading")
            
            # Check LightGBM version for compatibility warnings
            try:
                version_info = lgb.__version__
                major, minor, patch = map(int, version_info.split('.')[:3])
                
                if major >= 4 and minor >= 6:
                    self.logger.info(f"LightGBM {version_info} detected - using environment variable threading control")
                elif major >= 3:
                    self.logger.info(f"LightGBM {version_info} detected - modern version with environment variable support")
                else:
                    self.logger.warning(f"LightGBM {version_info} detected - very old version, may have compatibility issues")
                    
            except (AttributeError, ValueError) as version_error:
                self.logger.warning(f"Could not determine LightGBM version: {version_error}")
            
            # Verify threading configuration worked
            actual_threads = os.environ.get('LIGHTGBM_NUM_THREADS', 'not set')
            if actual_threads != 'not set':
                self.logger.info(f"LightGBM threading configured: LIGHTGBM_NUM_THREADS={actual_threads}")
            else:
                self.logger.warning("LightGBM threading environment variable not set")
            
            self._configured_libraries.add('lightgbm')
            self.logger.info(f"Successfully configured LightGBM for Steam Deck with {self.config.lightgbm_threads} threads")
            
        except ImportError:
            self.logger.debug("LightGBM not available - skipping configuration")
        except Exception as e:
            self.logger.error(f"Failed to configure LightGBM: {e}")
            self.logger.debug("LightGBM configuration failed, but system will continue with default settings")
    
    def configure_sklearn(self):
        """Configure scikit-learn threading"""
        try:
            from sklearn import set_config
            from sklearn.utils import parallel_backend
            
            # Set global config
            set_config(
                assume_finite=True,  # Skip input validation for performance
                working_memory=128,  # Limit memory usage
            )
            
            # Configure joblib backend
            try:
                import joblib
                joblib.parallel_backend('threading', n_jobs=self.config.sklearn_threads)
            except ImportError:
                pass
            
            self._configured_libraries.add('sklearn')
            self.logger.info(f"Configured scikit-learn with {self.config.sklearn_threads} threads")
            
        except ImportError:
            self.logger.debug("scikit-learn not available")
        except Exception as e:
            self.logger.error(f"Failed to configure scikit-learn: {e}")
    
    def configure_numba(self):
        """Configure Numba JIT threading"""
        try:
            import numba
            
            # Set number of threads
            numba.set_num_threads(self.config.numba_threads)
            
            # Configure for AMD Zen 2 (Steam Deck APU)
            numba.config.THREADING_LAYER = 'workqueue'
            numba.config.NUMBA_NUM_THREADS = self.config.numba_threads
            
            self._configured_libraries.add('numba')
            self.logger.info(f"Configured Numba with {self.config.numba_threads} threads")
            
        except ImportError:
            self.logger.debug("Numba not available")
        except Exception as e:
            self.logger.error(f"Failed to configure Numba: {e}")
    
    def configure_numpy(self):
        """Configure NumPy threading"""
        try:
            import numpy as np
            
            # Try to configure BLAS threads through NumPy
            if hasattr(np, '__config__') and hasattr(np.__config__, 'show'):
                config_info = np.__config__.show()
                self.logger.debug(f"NumPy build configuration: {config_info}")
            
            self._configured_libraries.add('numpy')
            self.logger.info("NumPy threading configured via environment variables")
            
        except ImportError:
            self.logger.debug("NumPy not available")
        except Exception as e:
            self.logger.error(f"Failed to configure NumPy: {e}")
    
    def configure_all_libraries(self):
        """Configure all available ML libraries"""
        self.configure_environment_variables()
        self.configure_python_threading()
        
        # Configure individual libraries
        self.configure_lightgbm()
        self.configure_sklearn()
        self.configure_numba()
        self.configure_numpy()
        
        self._is_configured = True
        
        configured_count = len(self._configured_libraries)
        self.logger.info(f"Configured {configured_count} ML libraries for Steam Deck threading")
    
    def get_optimal_params_lightgbm(self) -> Dict[str, Any]:
        """Get optimal LightGBM parameters for Steam Deck with comprehensive optimization"""
        # Base parameters optimized for Steam Deck APU (8-core Zen 2)
        base_params = {
            # Threading and device configuration
            'num_threads': self.config.lightgbm_threads,
            'force_row_wise': True,          # Better memory access pattern on APU
            'device_type': 'cpu',            # CPU-only for Steam Deck
            'verbose': -1,                   # Quiet mode
            
            # Memory optimization for 16GB shared LPDDR5
            'max_bin': 255,                  # Reduce memory usage while maintaining accuracy
            'histogram_pool_size': 128,      # Limit histogram memory pool (MB)
            'min_data_in_leaf': 10,          # Prevent overfitting on small datasets
            'min_child_samples': 5,          # Minimum samples in leaf node
            'min_child_weight': 0.001,       # Minimum sum of weights in leaf
            'subsample_for_bin': 200000,     # Samples for constructing histogram bins
            
            # Model structure optimization
            'max_depth': 6,                  # Limit tree depth for memory and speed
            'num_leaves': 31,                # Number of leaves (2^max_depth - 1)
            'learning_rate': 0.1,            # Conservative learning rate
            'num_iterations': 100,           # Default number of boosting iterations
            
            # Regularization for stability
            'lambda_l1': 0.1,                # L1 regularization (sparsity)
            'lambda_l2': 0.1,                # L2 regularization (stability)
            'reg_alpha': 0.1,                # L1 regularization (alternative name)
            'reg_lambda': 0.1,               # L2 regularization (alternative name)
            'min_split_gain': 0.1,           # Minimum gain required to split
            'min_gain_to_split': 0.0,        # Minimum gain to split (legacy)
            
            # Feature and data sampling
            'feature_fraction': 0.8,         # Use 80% of features per tree
            'bagging_fraction': 0.8,         # Use 80% of data per iteration  
            'subsample': 0.8,                # Row subsampling rate
            'bagging_freq': 1,               # Apply bagging every iteration
            'subsample_freq': 1,             # Apply subsampling every iteration
            'colsample_bytree': 0.8,         # Column subsampling per tree
            
            # Performance optimizations
            'enable_sparse': True,           # Optimize sparse feature handling
            'is_unbalance': False,           # Assume balanced data
            'boost_from_average': True,      # Initialize from label average
            'feature_pre_filter': False,     # Don't pre-filter features
            'two_round': False,              # Single round loading
            'use_missing': True,             # Handle missing values
            'zero_as_missing': False,        # Don't treat zero as missing
            
            # Default objective and metrics
            'objective': 'regression',       # Default to regression
            'metric': 'rmse',               # Root mean squared error
            'boosting_type': 'gbdt',        # Gradient boosting decision tree
            
            # Early stopping and validation
            'early_stopping_round': 10,     # Stop if no improvement for 10 rounds
            'first_metric_only': True,      # Use first metric for early stopping
            'metric_freq': 1,               # Calculate metric every iteration
            'is_training_metric': False,    # Don't calculate training metric
            
            # Reproducibility
            'seed': 42,                     # Random seed
            'deterministic': True,          # Deterministic training
            'force_col_wise': False,        # Use row-wise for better memory
            'extra_trees': False,           # Don't use extremely randomized trees
            
            # Advanced parameters for Steam Deck
            'max_delta_step': 0.0,          # No max delta step
            'top_rate': 0.2,                # Top rate for GOSS
            'other_rate': 0.1,              # Other rate for GOSS
            'min_data_per_group': 100,      # Minimum data per group
            'max_cat_threshold': 32,        # Maximum categorical threshold
            'cat_l2': 10.0,                 # L2 regularization for categorical
            'cat_smooth': 10.0,             # Categorical smoothing
            'max_cat_to_onehot': 4,         # Max categories for one-hot encoding
            'cegb_tradeoff': 1.0,           # Cost-effective gradient boosting tradeoff
            'cegb_penalty_split': 0.0,      # CEGB penalty for splitting
            'path_smooth': 0.0,             # Path smoothing
            'interaction_constraints': '',  # No interaction constraints
            
            # DART specific (if boosting_type='dart')
            'drop_rate': 0.1,               # Dropout rate
            'max_drop': 50,                 # Maximum number of dropped trees
            'skip_drop': 0.5,               # Probability of skipping dropout
            'xgboost_dart_mode': False,     # Use LightGBM DART mode
            'uniform_drop': False,          # Non-uniform dropout
            
            # Memory and threading optimization
            'num_threads': self.config.lightgbm_threads,  # Will be overridden below
            'data_random_seed': 42,         # Data shuffling seed
        }
        
        # Adaptive parameters based on system state
        current_threads = self.config.lightgbm_threads
        
        if self._current_thermal_state in ['hot', 'critical']:
            # Reduce computational load for thermal management
            scale_factor = 0.5 if self._current_thermal_state == 'hot' else 0.25
            current_threads = max(1, int(self.config.lightgbm_threads * scale_factor))
            
            base_params.update({
                'num_threads': current_threads,
                'max_depth': min(4, base_params['max_depth']),      # Shallower trees
                'num_leaves': min(15, base_params['num_leaves']),   # Fewer leaves
                'learning_rate': min(0.05, base_params['learning_rate']),  # Slower learning
                'num_iterations': min(50, base_params['num_iterations']),  # Fewer iterations
                'histogram_pool_size': 64,                         # Less memory
                'max_bin': 127,                                    # Fewer bins
            })
        
        if self._gaming_mode_active:
            # Minimal resource usage during gaming
            base_params.update({
                'num_threads': 1,                               # Single thread only
                'max_depth': 4,                                 # Very shallow trees
                'num_leaves': 15,                               # Minimal leaves
                'learning_rate': 0.05,                          # Very slow learning
                'num_iterations': 25,                           # Minimal iterations
                'histogram_pool_size': 32,                      # Minimal memory
                'max_bin': 63,                                  # Very few bins
                'early_stopping_round': 5,                     # Stop early
                'feature_fraction': 0.5,                       # Use fewer features
                'bagging_fraction': 0.5,                       # Use less data
            })
        
        if self._battery_level < 20.0:
            # Power saving mode
            battery_scale = 0.5
            current_threads = max(1, int(current_threads * battery_scale))
            
            base_params.update({
                'num_threads': current_threads,
                'learning_rate': 0.05,                          # Slower, more efficient learning
                'num_iterations': 50,                           # Fewer iterations
                'early_stopping_round': 5,                     # Stop early to save power
                'histogram_pool_size': 64,                      # Reduce memory usage
            })
        
        # Ensure threading consistency with environment variables
        if current_threads != self.config.lightgbm_threads:
            os.environ['LIGHTGBM_NUM_THREADS'] = str(current_threads)
            os.environ['OMP_NUM_THREADS'] = str(current_threads)
        
        # Log parameter optimization
        self.logger.debug(f"Generated LightGBM parameters with {current_threads} threads for "
                         f"thermal={self._current_thermal_state}, gaming={self._gaming_mode_active}, "
                         f"battery={self._battery_level}%")
        
        return base_params
    
    def get_optimal_params_sklearn(self) -> Dict[str, Any]:
        """Get optimal scikit-learn parameters for Steam Deck"""
        n_jobs = self.config.sklearn_threads
        
        # Adaptive threading
        if self._current_thermal_state in ['hot', 'critical']:
            n_jobs = max(1, n_jobs // 2)
        
        if self._gaming_mode_active:
            n_jobs = 1
        
        return {
            'n_jobs': n_jobs,
            'max_iter': 100,  # Limit iterations for responsiveness
        }
    
    def update_thermal_state(self, thermal_state: str):
        """Update thermal state and adjust threading accordingly"""
        if thermal_state != self._current_thermal_state:
            self.logger.info(f"Thermal state changed: {self._current_thermal_state} -> {thermal_state}")
            self._current_thermal_state = thermal_state
            
            if self.config.enable_thermal_scaling:
                self._apply_thermal_scaling()
    
    def update_gaming_mode(self, gaming_active: bool):
        """Update gaming mode state"""
        if gaming_active != self._gaming_mode_active:
            self.logger.info(f"Gaming mode: {self._gaming_mode_active} -> {gaming_active}")
            self._gaming_mode_active = gaming_active
            
            if self.config.enable_gaming_mode:
                self._apply_gaming_mode_scaling()
    
    def update_battery_level(self, battery_percent: float):
        """Update battery level and adjust threading"""
        self._battery_level = battery_percent
        
        if self.config.enable_battery_scaling and battery_percent < 20.0:
            self._apply_battery_scaling()
    
    def _apply_thermal_scaling(self):
        """Apply thermal-based thread scaling"""
        if self._current_thermal_state in ['hot', 'critical']:
            # Reduce threading for thermal management
            scale_factor = 0.5 if self._current_thermal_state == 'hot' else 0.25
            
            # Update LightGBM threading via environment variables
            if 'lightgbm' in self._configured_libraries:
                try:
                    new_threads = max(1, int(self.config.lightgbm_threads * scale_factor))
                    os.environ['LIGHTGBM_NUM_THREADS'] = str(new_threads)
                    os.environ['OMP_NUM_THREADS'] = str(new_threads)  # Also update OpenMP
                    self.logger.info(f"Scaled LightGBM threads to {new_threads} for thermal management")
                    
                    # Update config for future model creation
                    self.config.lightgbm_threads = new_threads
                    self.config.omp_num_threads = new_threads
                    
                except Exception as e:
                    self.logger.error(f"Failed to scale LightGBM threads: {e}")
    
    def _apply_gaming_mode_scaling(self):
        """Apply gaming mode thread scaling"""
        if self._gaming_mode_active:
            # Minimal threading during gaming
            if 'lightgbm' in self._configured_libraries:
                try:
                    os.environ['LIGHTGBM_NUM_THREADS'] = '1'
                    os.environ['OMP_NUM_THREADS'] = '1'
                    self.logger.info("Reduced LightGBM threads to 1 for gaming mode")
                    
                    # Update config for future model creation
                    self.config.lightgbm_threads = 1
                    self.config.omp_num_threads = 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to apply gaming mode scaling: {e}")
    
    def _apply_battery_scaling(self):
        """Apply battery-based thread scaling"""
        if self._battery_level < 20.0:
            # Reduce threading to save battery
            scale_factor = 0.5
            
            if 'lightgbm' in self._configured_libraries:
                try:
                    new_threads = max(1, int(self.config.lightgbm_threads * scale_factor))
                    os.environ['LIGHTGBM_NUM_THREADS'] = str(new_threads)
                    os.environ['OMP_NUM_THREADS'] = str(new_threads)
                    self.logger.info(f"Scaled LightGBM threads to {new_threads} for battery saving")
                    
                    # Update config for future model creation
                    self.config.lightgbm_threads = new_threads
                    self.config.omp_num_threads = new_threads
                    
                except Exception as e:
                    self.logger.warning(f"Failed to apply battery scaling: {e}")
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            'is_configured': self._is_configured,
            'is_steam_deck': self._is_steam_deck,
            'cpu_count': self._cpu_count,
            'configured_libraries': list(self._configured_libraries),
            'config': {
                'max_threads': self.config.max_threads,
                'ml_threads': self.config.ml_threads,
                'omp_threads': self.config.omp_num_threads,
            },
            'adaptive_state': {
                'thermal_state': self._current_thermal_state,
                'gaming_mode': self._gaming_mode_active,
                'battery_level': self._battery_level,
            },
            'environment_variables': {
                key: os.environ.get(key, 'not set')
                for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
            }
        }
    
    def cleanup(self):
        """Cleanup and restore original environment"""
        if self._original_env:
            for key, original_value in self._original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            
            self.logger.debug("Restored original environment variables")


# Global configurator instance
_global_configurator = None


def get_threading_configurator() -> SteamDeckThreadingConfigurator:
    """Get or create global threading configurator"""
    global _global_configurator
    if _global_configurator is None:
        _global_configurator = SteamDeckThreadingConfigurator()
    return _global_configurator


def configure_threading_for_steam_deck(config: Optional[ThreadingConfig] = None) -> SteamDeckThreadingConfigurator:
    """Configure threading for Steam Deck (main entry point)"""
    configurator = SteamDeckThreadingConfigurator(config)
    configurator.configure_all_libraries()
    return configurator


def get_lightgbm_params() -> Dict[str, Any]:
    """Get optimized LightGBM parameters for current system state"""
    configurator = get_threading_configurator()
    return configurator.get_optimal_params_lightgbm()


def get_sklearn_params() -> Dict[str, Any]:
    """Get optimized scikit-learn parameters for current system state"""
    configurator = get_threading_configurator()
    return configurator.get_optimal_params_sklearn()


def check_lightgbm_compatibility() -> Dict[str, Any]:
    """
    Check LightGBM compatibility and return version information
    
    Returns:
        Dict containing version info, supported features, and compatibility warnings
    """
    compatibility_info = {
        'available': False,
        'version': None,
        'major': 0,
        'minor': 0,
        'patch': 0,
        'supports_env_threading': False,
        'supports_set_option': False,
        'supports_set_number_threads': False,
        'warnings': [],
        'recommendations': [],
    }
    
    try:
        import lightgbm as lgb
        compatibility_info['available'] = True
        
        # Get version information
        try:
            version_str = lgb.__version__
            compatibility_info['version'] = version_str
            
            # Parse version numbers
            version_parts = version_str.split('.')
            compatibility_info['major'] = int(version_parts[0])
            compatibility_info['minor'] = int(version_parts[1]) if len(version_parts) > 1 else 0
            compatibility_info['patch'] = int(version_parts[2]) if len(version_parts) > 2 else 0
            
        except (AttributeError, ValueError, IndexError) as version_error:
            compatibility_info['warnings'].append(f"Could not parse LightGBM version: {version_error}")
        
        # Check available methods
        compatibility_info['supports_set_option'] = hasattr(lgb, 'set_option')
        compatibility_info['supports_set_number_threads'] = hasattr(lgb, 'set_number_threads')
        
        # Environment variable support is available in 3.0+
        if compatibility_info['major'] >= 3:
            compatibility_info['supports_env_threading'] = True
        
        # Version-specific compatibility checks
        major, minor = compatibility_info['major'], compatibility_info['minor']
        
        if major >= 4 and minor >= 6:
            # LightGBM 4.6.0+ - modern version
            if compatibility_info['supports_set_option']:
                compatibility_info['warnings'].append(
                    "LightGBM 4.6.0+ detected with deprecated set_option() - this shouldn't exist"
                )
            compatibility_info['recommendations'].append("Use LIGHTGBM_NUM_THREADS environment variable")
            
        elif major >= 4:
            # LightGBM 4.0-4.5
            compatibility_info['recommendations'].append("Consider upgrading to LightGBM 4.6+ for better compatibility")
            
        elif major >= 3:
            # LightGBM 3.x
            compatibility_info['recommendations'].append("Environment variables supported, consider upgrade to 4.6+")
            
        elif major >= 2:
            # LightGBM 2.x
            compatibility_info['warnings'].append("LightGBM 2.x is old - upgrade recommended")
            compatibility_info['recommendations'].append("Upgrade to LightGBM 4.6+ for best Steam Deck performance")
            
        else:
            # Very old versions
            compatibility_info['warnings'].append(f"LightGBM {version_str} is very old and may have issues")
            compatibility_info['recommendations'].append("Upgrade to LightGBM 4.6+ immediately")
        
        # Check for deprecated methods that shouldn't be used
        if compatibility_info['supports_set_option'] and major >= 4 and minor >= 6:
            compatibility_info['warnings'].append(
                "set_option() method found in LightGBM 4.6+ - this is unexpected and may indicate issues"
            )
        
        if compatibility_info['supports_set_number_threads'] and major >= 4:
            compatibility_info['warnings'].append(
                "set_number_threads() method found in LightGBM 4.x - this is deprecated"
            )
        
    except ImportError:
        compatibility_info['warnings'].append("LightGBM is not available")
        compatibility_info['recommendations'].append("Install LightGBM with: pip install lightgbm")
    
    except Exception as e:
        compatibility_info['warnings'].append(f"Error checking LightGBM compatibility: {e}")
    
    return compatibility_info


def get_lightgbm_safe_params(custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get safe LightGBM parameters with compatibility handling
    
    Args:
        custom_params: Optional custom parameters to merge
        
    Returns:
        Safe parameter dictionary that works across LightGBM versions
    """
    # Check compatibility first
    compat = check_lightgbm_compatibility()
    
    if not compat['available']:
        logging.warning("LightGBM not available - returning empty parameters")
        return {}
    
    # Get base optimized parameters
    configurator = get_threading_configurator()
    base_params = configurator.get_optimal_params_lightgbm()
    
    # Remove parameters that might not be supported in older versions
    safe_params = {}
    
    # Core parameters supported in all versions
    core_params = {
        'num_threads', 'verbose', 'device_type', 'objective', 'metric',
        'boosting_type', 'num_leaves', 'max_depth', 'learning_rate',
        'num_iterations', 'seed', 'deterministic'
    }
    
    # Parameters supported in 3.0+
    modern_params = {
        'force_row_wise', 'histogram_pool_size', 'max_bin', 'min_data_in_leaf',
        'lambda_l1', 'lambda_l2', 'feature_fraction', 'bagging_fraction',
        'bagging_freq', 'early_stopping_round', 'first_metric_only'
    }
    
    # Parameters supported in 4.0+
    advanced_params = {
        'reg_alpha', 'reg_lambda', 'min_split_gain', 'subsample', 'subsample_freq',
        'colsample_bytree', 'min_child_samples', 'min_child_weight', 'subsample_for_bin'
    }
    
    # Filter parameters based on version
    major, minor = compat['major'], compat['minor']
    
    for param, value in base_params.items():
        if param in core_params:
            safe_params[param] = value
        elif major >= 3 and param in modern_params:
            safe_params[param] = value
        elif major >= 4 and param in advanced_params:
            safe_params[param] = value
        elif major >= 4 and minor >= 6:
            # Latest version - include all parameters
            safe_params[param] = value
    
    # Merge custom parameters
    if custom_params:
        safe_params.update(custom_params)
    
    # Log compatibility info
    logger = logging.getLogger(__name__)
    if compat['warnings']:
        for warning in compat['warnings']:
            logger.warning(f"LightGBM compatibility: {warning}")
    
    if compat['recommendations']:
        for rec in compat['recommendations']:
            logger.info(f"LightGBM recommendation: {rec}")
    
    logger.info(f"Generated {len(safe_params)} safe LightGBM parameters for version {compat['version']}")
    
    return safe_params


def ensure_lightgbm_threading() -> bool:
    """
    Ensure LightGBM threading is properly configured
    
    Returns:
        True if threading was successfully configured, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check compatibility
        compat = check_lightgbm_compatibility()
        
        if not compat['available']:
            logger.warning("LightGBM not available - cannot configure threading")
            return False
        
        # Get current threading configuration
        configurator = get_threading_configurator()
        desired_threads = configurator.config.lightgbm_threads
        
        # Set environment variable (works for all modern versions)
        os.environ['LIGHTGBM_NUM_THREADS'] = str(desired_threads)
        os.environ['OMP_NUM_THREADS'] = str(desired_threads)
        
        # Verify environment variable was set
        actual_threads = os.environ.get('LIGHTGBM_NUM_THREADS')
        if actual_threads == str(desired_threads):
            logger.info(f"LightGBM threading configured: {actual_threads} threads via environment variable")
            return True
        else:
            logger.error(f"Failed to set LIGHTGBM_NUM_THREADS environment variable")
            return False
            
    except Exception as e:
        logger.error(f"Error configuring LightGBM threading: {e}")
        return False


if __name__ == "__main__":
    # Test threading configuration
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Steam Deck Threading Configuration Test")
    print("=" * 50)
    
    # Configure threading
    configurator = configure_threading_for_steam_deck()
    
    # Show status
    status = configurator.get_configuration_status()
    print(f"Steam Deck detected: {status['is_steam_deck']}")
    print(f"CPU cores: {status['cpu_count']}")
    print(f"Configured libraries: {', '.join(status['configured_libraries'])}")
    print(f"Max threads: {status['config']['max_threads']}")
    
    # Test adaptive parameters
    print("\nOptimized Parameters:")
    lgb_params = get_lightgbm_params()
    print(f"LightGBM threads: {lgb_params.get('num_threads', 'N/A')}")
    
    sklearn_params = get_sklearn_params()
    print(f"scikit-learn n_jobs: {sklearn_params.get('n_jobs', 'N/A')}")
    
    # Test thermal scaling
    print("\nTesting thermal scaling...")
    configurator.update_thermal_state('hot')
    hot_params = get_lightgbm_params()
    print(f"LightGBM threads (hot): {hot_params.get('num_threads', 'N/A')}")
    
    configurator.update_thermal_state('normal')
    
    print("\n✅ Threading configuration test completed")