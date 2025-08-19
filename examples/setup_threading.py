#!/usr/bin/env python3
"""
Early Threading Environment Setup for Steam Deck
==============================================

CRITICAL: This module MUST be imported before ANY ML libraries to properly configure
threading environment variables. ML libraries cache threading configuration at import time.

Steam Deck OLED Optimizations:
- 8 logical cores (4 physical with SMT)
- 16GB LPDDR5 shared memory
- AMD Van Gogh APU with thermal constraints
- SteamOS gaming mode integration

Usage:
    import setup_threading  # FIRST import before anything else
    setup_threading.configure_for_steam_deck()
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Configure logging for early initialization
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteamDeckThreadingConfigurator:
    """Steam Deck specific threading configuration manager."""
    
    # Steam Deck Hardware Detection
    STEAM_DECK_IDENTIFIERS = [
        'Jupiter',     # Steam Deck LCD
        'Galileo',     # Steam Deck OLED  
        'Neptune',     # Steam Deck prototype
        'VALVE'        # Valve Corporation products
    ]
    
    # Conservative threading limits for Steam Deck
    STEAM_DECK_THREAD_LIMITS = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'LIGHTGBM_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'BLIS_NUM_THREADS': '1',
        'NUMBA_NUM_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': '1',
        'TF_NUM_INTEROP_THREADS': '1',
        'SKLEARN_N_JOBS': '1',
        'JOBLIB_N_JOBS': '1',
        'NPY_NUM_BUILD_JOBS': '1'
    }
    
    # OpenMP optimization for Steam Deck
    STEAM_DECK_OPENMP_CONFIG = {
        'OMP_DYNAMIC': 'FALSE',
        'OMP_NESTED': 'FALSE', 
        'OMP_WAIT_POLICY': 'PASSIVE',
        'OMP_PROC_BIND': 'FALSE',
        'OMP_PLACES': 'cores',
        'OMP_MAX_ACTIVE_LEVELS': '1'
    }
    
    # Memory optimization for shared LPDDR5
    STEAM_DECK_MEMORY_CONFIG = {
        'MALLOC_ARENA_MAX': '1',
        'MALLOC_MMAP_THRESHOLD_': '65536',
        'MALLOC_TRIM_THRESHOLD_': '262144',
        'PYTHON_GIL_REQUEST_INTERVAL': '0.005'  # 5ms for gaming responsiveness
    }
    
    def __init__(self):
        self.is_steam_deck = self._detect_steam_deck()
        self.steam_deck_model = self._get_steam_deck_model()
        self.configuration_applied = False
        logger.info(f"Threading configurator initialized - Steam Deck: {self.is_steam_deck}, Model: {self.steam_deck_model}")
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck hardware."""
        try:
            # Check DMI product name
            dmi_path = Path('/sys/class/dmi/id/product_name')
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip()
                if any(identifier in product_name for identifier in self.STEAM_DECK_IDENTIFIERS):
                    return True
            
            # Check board name as fallback
            board_path = Path('/sys/class/dmi/id/board_name') 
            if board_path.exists():
                board_name = board_path.read_text().strip()
                if any(identifier in board_name for identifier in self.STEAM_DECK_IDENTIFIERS):
                    return True
            
            # Check for SteamOS
            if Path('/etc/os-release').exists():
                with open('/etc/os-release', 'r') as f:
                    os_content = f.read()
                    if 'steamos' in os_content.lower() or 'holo' in os_content.lower():
                        return True
            
            return False
        except Exception as e:
            logger.warning(f"Steam Deck detection failed: {e}")
            return False
    
    def _get_steam_deck_model(self) -> Optional[str]:
        """Get specific Steam Deck model (LCD/OLED)."""
        if not self.is_steam_deck:
            return None
            
        try:
            product_path = Path('/sys/class/dmi/id/product_name')
            if product_path.exists():
                product_name = product_path.read_text().strip()
                if 'Jupiter' in product_name:
                    return 'steam_deck_lcd'
                elif 'Galileo' in product_name:
                    return 'steam_deck_oled'
            return 'steam_deck_unknown'
        except Exception as e:
            logger.warning(f"Steam Deck model detection failed: {e}")
            return 'steam_deck_unknown'
    
    def _get_thermal_state(self) -> Tuple[float, str]:
        """Get current thermal state for threading decisions."""
        try:
            thermal_zones = list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
            if not thermal_zones:
                return 40.0, 'normal'  # Safe default
            
            temps = []
            for zone in thermal_zones:
                try:
                    temp_millicelsius = int(zone.read_text().strip())
                    temp_celsius = temp_millicelsius / 1000.0
                    if 20.0 <= temp_celsius <= 100.0:  # Reasonable temperature range
                        temps.append(temp_celsius)
                except (ValueError, IOError):
                    continue
            
            if not temps:
                return 40.0, 'normal'
            
            max_temp = max(temps)
            if max_temp >= 85.0:
                return max_temp, 'critical'
            elif max_temp >= 75.0:
                return max_temp, 'hot'
            elif max_temp >= 65.0:
                return max_temp, 'warm'
            else:
                return max_temp, 'normal'
        except Exception as e:
            logger.warning(f"Thermal state detection failed: {e}")
            return 40.0, 'normal'
    
    def _check_steam_running(self) -> bool:
        """Check if Steam client is running."""
        try:
            result = subprocess.run(['pgrep', '-f', 'steam'], capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_available_thread_capacity(self) -> int:
        """Get available threading capacity on system."""
        try:
            # Check current thread count
            result = subprocess.run(['ps', '-eLf'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                current_threads = len(result.stdout.splitlines()) - 1  # Subtract header
            else:
                current_threads = 100  # Conservative estimate
            
            # Check system limits
            try:
                import resource
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
                available_capacity = max(0, min(soft_limit - current_threads, 50))  # Cap at 50
            except ImportError:
                available_capacity = max(0, 100 - current_threads)
            
            return available_capacity
        except Exception as e:
            logger.warning(f"Thread capacity check failed: {e}")
            return 10  # Very conservative fallback
    
    def configure_environment(self) -> bool:
        """Configure threading environment variables for Steam Deck."""
        if self.configuration_applied:
            logger.info("Threading configuration already applied")
            return True
        
        try:
            # Get current system state
            temp, thermal_state = self._get_thermal_state()
            steam_running = self._check_steam_running()
            thread_capacity = self._get_available_thread_capacity()
            
            logger.info(f"System state - Thermal: {temp:.1f}°C ({thermal_state}), Steam: {steam_running}, Threads available: {thread_capacity}")
            
            # Base configuration
            config_vars = {}
            config_vars.update(self.STEAM_DECK_THREAD_LIMITS)
            config_vars.update(self.STEAM_DECK_OPENMP_CONFIG)
            config_vars.update(self.STEAM_DECK_MEMORY_CONFIG)
            
            # Thermal adjustments
            if thermal_state == 'critical':
                # Ultra-conservative for critical temps
                config_vars.update({
                    'OMP_NUM_THREADS': '1',
                    'NUMEXPR_NUM_THREADS': '1',
                    'LIGHTGBM_NUM_THREADS': '1'
                })
            elif thermal_state == 'hot':
                # Conservative for hot temps  
                config_vars.update({
                    'OMP_NUM_THREADS': '1',
                    'NUMEXPR_NUM_THREADS': '1'
                })
            
            # Gaming mode adjustments (when Steam is running)
            if steam_running:
                config_vars.update({
                    'OMP_WAIT_POLICY': 'PASSIVE',
                    'PYTHON_GIL_REQUEST_INTERVAL': '0.001'  # 1ms for gaming
                })
            
            # Low thread capacity adjustments
            if thread_capacity < 20:
                logger.warning(f"Low thread capacity ({thread_capacity}), using ultra-conservative settings")
                config_vars.update({
                    'OMP_NUM_THREADS': '1',
                    'NUMEXPR_NUM_THREADS': '1',
                    'LIGHTGBM_NUM_THREADS': '1',
                    'NUMBA_NUM_THREADS': '1'
                })
            
            # Apply configuration
            for key, value in config_vars.items():
                old_value = os.environ.get(key, 'unset')
                os.environ[key] = value
                if old_value != value:
                    logger.debug(f"Set {key}='{value}' (was '{old_value}')")
            
            # Verify critical variables are set
            critical_vars = ['OMP_NUM_THREADS', 'LIGHTGBM_NUM_THREADS', 'NUMEXPR_NUM_THREADS']
            for var in critical_vars:
                if os.environ.get(var) != '1':
                    logger.error(f"Critical variable {var} not properly set!")
                    return False
            
            self.configuration_applied = True
            logger.info(f"Threading environment configured successfully for Steam Deck {self.steam_deck_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure threading environment: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate that threading configuration is properly applied."""
        try:
            # Check critical environment variables
            required_vars = {
                'OMP_NUM_THREADS': '1',
                'LIGHTGBM_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1'
            }
            
            for var, expected in required_vars.items():
                actual = os.environ.get(var)
                if actual != expected:
                    logger.error(f"Configuration validation failed: {var}='{actual}', expected '{expected}'")
                    return False
            
            logger.info("Threading configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configurator instance
_configurator = None

def get_configurator() -> SteamDeckThreadingConfigurator:
    """Get the global threading configurator instance."""
    global _configurator
    if _configurator is None:
        _configurator = SteamDeckThreadingConfigurator()
    return _configurator

def configure_for_steam_deck() -> bool:
    """Configure threading environment for Steam Deck (main entry point)."""
    configurator = get_configurator()
    
    if not configurator.is_steam_deck:
        logger.warning("Not running on Steam Deck, applying generic configuration")
        # Apply generic conservative configuration for non-Steam Deck systems
        generic_config = {
            'OMP_NUM_THREADS': '2',
            'LIGHTGBM_NUM_THREADS': '2', 
            'NUMEXPR_NUM_THREADS': '2'
        }
        for key, value in generic_config.items():
            os.environ[key] = value
        return True
    
    success = configurator.configure_environment()
    if success:
        success = configurator.validate_configuration()
    
    return success

def is_steam_deck() -> bool:
    """Check if running on Steam Deck."""
    return get_configurator().is_steam_deck

def get_steam_deck_model() -> Optional[str]:
    """Get Steam Deck model (lcd/oled/unknown).""" 
    return get_configurator().steam_deck_model

# Auto-configure when module is imported (critical for early setup)
if __name__ == '__main__' or os.environ.get('SHADER_PREDICT_AUTO_CONFIGURE', '1') == '1':
    success = configure_for_steam_deck()
    if success:
        logger.info("Early threading configuration completed successfully")
    else:
        logger.error("Early threading configuration failed - system may experience threading issues")