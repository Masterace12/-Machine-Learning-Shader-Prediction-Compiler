#!/usr/bin/env python3

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class RADVOptimization:
    game_id: str
    shader_version_override: int
    graphics_version: Optional[int] = None
    compute_version: Optional[int] = None
    ray_tracing_version: Optional[int] = None
    perftest_flags: List[str] = None
    device_select: str = "1002:163f"  # Steam Deck GPU

class RADVOptimizer:
    """
    RADV driver optimization manager implementing the latest 2025 improvements
    including forced shader recompilation and Steam Deck specific optimizations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('radv_optimizer')
        self.drirc_path = Path.home() / '.drirc'
        self.shader_cache_path = Path.home() / '.cache/mesa_shader_cache'
        
        # Steam Deck specific RADV optimizations
        self.steam_deck_optimizations = {
            'base_perftest': ['aco', 'nggc'],  # ACO compiler + NGG culling
            'oled_perftest': ['aco', 'nggc', 'rt'],  # Add ray tracing for OLED
            'device_select': '1002:163f',  # Van Gogh APU
            'shader_compiler_options': {
                'nir_lower_int64': True,
                'enable_mesh_shaders': True,
                'variable_rate_shading': True
            }
        }
        
        # Game-specific shader version overrides for compiler fixes
        self.known_problematic_games = {
            'Cyberpunk2077': {'graphics': 1, 'compute': 1, 'rt': 1},
            'EldenRing': {'graphics': 2, 'compute': 1},
            'HorizonZeroDawn': {'graphics': 1, 'compute': 1},
            'DeathStranding': {'graphics': 1},
            'ControlUltimateEdition': {'graphics': 2, 'rt': 1}
        }
        
        self.applied_optimizations = {}
        
    def detect_steam_deck_model(self) -> str:
        """Detect Steam Deck model for appropriate optimizations"""
        try:
            with open('/sys/devices/virtual/dmi/id/product_name', 'r') as f:
                product = f.read().strip()
                
            if 'Jupiter' not in product:
                return 'Unknown'
                
            # Try to detect OLED vs LCD
            try:
                # Check for newer APU (Phoenix for OLED)
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
                if '1002:15bf' in result.stdout:  # Phoenix APU
                    return 'OLED'
                elif '1002:163f' in result.stdout:  # Van Gogh APU
                    return 'LCD'
            except:
                pass
                
            return 'LCD'  # Default assumption
        except:
            return 'Unknown'
    
    def apply_global_radv_optimizations(self, steam_deck_model: str = None) -> bool:
        """Apply global RADV optimizations for Steam Deck"""
        if steam_deck_model is None:
            steam_deck_model = self.detect_steam_deck_model()
            
        env_vars = {
            'MESA_VK_DEVICE_SELECT': self.steam_deck_optimizations['device_select'],
            'RADV_DEBUG': 'noshaderdb,nocompute',  # Disable debug features for performance
            'MESA_SHADER_CACHE_MAX_SIZE': '2G',  # 2GB shader cache limit
        }
        
        # Model-specific optimizations
        if steam_deck_model == 'OLED':
            env_vars['RADV_PERFTEST'] = ','.join(self.steam_deck_optimizations['oled_perftest'])
            env_vars['RADV_RT'] = '1'  # Enable ray tracing features
        else:
            env_vars['RADV_PERFTEST'] = ','.join(self.steam_deck_optimizations['base_perftest'])
            
        # Additional Steam Deck optimizations
        env_vars.update({
            'RADV_LOWER_DISCARD_TO_DEMOTE': '1',  # Better performance on RDNA2
            'RADV_ENABLE_MRT_OUTPUT_NAN_FIXUP': '1',  # Fix rendering issues
            'mesa_glthread': 'true',  # Enable GL threading
            'MESA_DISK_CACHE_SINGLE_FILE': 'true',  # Faster cache access
        })
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            
        self.logger.info(f"Applied global RADV optimizations for {steam_deck_model} model")
        return True
    
    def create_drirc_optimization(self, game_id: str, optimization: RADVOptimization) -> bool:
        """Create or update drirc entry for game-specific optimizations"""
        try:
            # Load existing drirc or create new
            drirc_config = self._load_drirc_config()
            
            # Add/update game configuration
            game_config = {
                'executable': self._get_game_executable(game_id),
                'options': []
            }
            
            # Add shader version overrides for forced recompilation
            if optimization.graphics_version is not None:
                game_config['options'].append({
                    'option': 'radv_override_graphics_shader_version',
                    'value': str(optimization.graphics_version)
                })
                
            if optimization.compute_version is not None:
                game_config['options'].append({
                    'option': 'radv_override_compute_shader_version', 
                    'value': str(optimization.compute_version)
                })
                
            if optimization.ray_tracing_version is not None:
                game_config['options'].append({
                    'option': 'radv_override_ray_tracing_shader_version',
                    'value': str(optimization.ray_tracing_version)
                })
                
            # Add performance optimizations
            if optimization.perftest_flags:
                game_config['options'].append({
                    'option': 'radv_perftest',
                    'value': ','.join(optimization.perftest_flags)
                })
            
            # Steam Deck specific options
            game_config['options'].extend([
                {'option': 'radv_lower_discard_to_demote', 'value': 'true'},
                {'option': 'radv_invariant_geom', 'value': 'true'},
                {'option': 'radv_disable_shrink_image_store', 'value': 'true'}
            ])
            
            drirc_config['games'][game_id] = game_config
            self._save_drirc_config(drirc_config)
            
            self.applied_optimizations[game_id] = optimization
            self.logger.info(f"Applied RADV optimization for {game_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create drirc optimization for {game_id}: {e}")
            return False
    
    def apply_known_game_fixes(self) -> int:
        """Apply known fixes for problematic games"""
        applied_count = 0
        
        for game_id, versions in self.known_problematic_games.items():
            optimization = RADVOptimization(
                game_id=game_id,
                shader_version_override=1,
                graphics_version=versions.get('graphics'),
                compute_version=versions.get('compute'),
                ray_tracing_version=versions.get('rt'),
                perftest_flags=self.steam_deck_optimizations['base_perftest']
            )
            
            if self.create_drirc_optimization(game_id, optimization):
                applied_count += 1
                
        self.logger.info(f"Applied fixes for {applied_count} known problematic games")
        return applied_count
    
    def force_shader_recompilation(self, game_id: str, reasons: List[str]) -> bool:
        """Force shader recompilation for a specific game"""
        try:
            # Increment shader version to force recompilation
            current_optimization = self.applied_optimizations.get(game_id)
            if current_optimization:
                # Increment all version numbers
                new_optimization = RADVOptimization(
                    game_id=game_id,
                    shader_version_override=current_optimization.shader_version_override + 1,
                    graphics_version=(current_optimization.graphics_version or 0) + 1,
                    compute_version=(current_optimization.compute_version or 0) + 1,
                    ray_tracing_version=(current_optimization.ray_tracing_version or 0) + 1,
                    perftest_flags=current_optimization.perftest_flags
                )
            else:
                # Create new optimization with version 1
                new_optimization = RADVOptimization(
                    game_id=game_id,
                    shader_version_override=1,
                    graphics_version=1,
                    compute_version=1,
                    perftest_flags=self.steam_deck_optimizations['base_perftest']
                )
            
            # Apply the optimization
            if self.create_drirc_optimization(game_id, new_optimization):
                # Clear existing shader cache for this game
                self._clear_game_shader_cache(game_id)
                
                self.logger.info(f"Forced shader recompilation for {game_id}: {', '.join(reasons)}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to force recompilation for {game_id}: {e}")
            
        return False
    
    def optimize_shader_cache(self) -> Dict[str, int]:
        """Optimize shader cache structure and cleanup"""
        results = {
            'files_optimized': 0,
            'space_saved_mb': 0,
            'cache_entries': 0
        }
        
        try:
            if not self.shader_cache_path.exists():
                return results
                
            # Get cache statistics
            cache_files = list(self.shader_cache_path.rglob('*'))
            results['cache_entries'] = len([f for f in cache_files if f.is_file()])
            
            # Implement Fossilize single-file optimization mentioned in notes
            self._optimize_fossilize_cache()
            
            # Cleanup old cache entries (30 days as per notes)
            import time
            cutoff_time = time.time() - (30 * 24 * 3600)
            
            for cache_file in cache_files:
                if cache_file.is_file():
                    try:
                        if cache_file.stat().st_mtime < cutoff_time:
                            file_size = cache_file.stat().st_size
                            cache_file.unlink()
                            results['files_optimized'] += 1
                            results['space_saved_mb'] += file_size / (1024 * 1024)
                    except:
                        continue
            
            self.logger.info(f"Cache optimization complete: {results}")
            
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            
        return results
    
    def _optimize_fossilize_cache(self):
        """Optimize Fossilize cache using single-file database format"""
        try:
            fossilize_path = self.shader_cache_path / 'fossilize'
            if fossilize_path.exists():
                # Run fossilize optimization if available
                subprocess.run([
                    'fossilize-replay', '--optimize-cache',
                    '--compression', 'zstd',  # Use compression as mentioned in notes
                    str(fossilize_path)
                ], check=False, capture_output=True)
        except:
            pass
    
    def _load_drirc_config(self) -> Dict:
        """Load existing drirc configuration"""
        if self.drirc_path.exists():
            try:
                # Parse XML drirc format
                import xml.etree.ElementTree as ET
                tree = ET.parse(self.drirc_path)
                # Convert to internal format
                return {'games': {}}  # Simplified for this implementation
            except:
                pass
                
        return {'games': {}}
    
    def _save_drirc_config(self, config: Dict):
        """Save drirc configuration in XML format"""
        try:
            # Create XML structure
            xml_content = '<?xml version="1.0" standalone="yes"?>\n<driconf>\n'
            
            for game_id, game_config in config['games'].items():
                xml_content += f'  <device screen="0" driver="radv">\n'
                xml_content += f'    <application name="{game_config["executable"]}" executable="{game_config["executable"]}">\n'
                
                for option in game_config['options']:
                    xml_content += f'      <option name="{option["option"]}" value="{option["value"]}" />\n'
                    
                xml_content += '    </application>\n'
                xml_content += '  </device>\n'
            
            xml_content += '</driconf>\n'
            
            with open(self.drirc_path, 'w') as f:
                f.write(xml_content)
                
        except Exception as e:
            self.logger.error(f"Failed to save drirc config: {e}")
    
    def _get_game_executable(self, game_id: str) -> str:
        """Get executable name for game ID"""
        # This would normally query Steam's app info
        # For now, return a generic pattern
        return f"{game_id.lower()}.exe"
    
    def _clear_game_shader_cache(self, game_id: str):
        """Clear shader cache for specific game"""
        try:
            game_cache_pattern = game_id.lower()
            for cache_file in self.shader_cache_path.rglob('*'):
                if cache_file.is_file() and game_cache_pattern in cache_file.name.lower():
                    cache_file.unlink()
        except:
            pass
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        return {
            'steam_deck_model': self.detect_steam_deck_model(),
            'applied_optimizations': len(self.applied_optimizations),
            'shader_cache_size_mb': self._get_cache_size_mb(),
            'drirc_entries': len(self._load_drirc_config().get('games', {})),
            'environment_variables': {
                k: v for k, v in os.environ.items() 
                if k.startswith(('RADV_', 'MESA_'))
            }
        }
    
    def _get_cache_size_mb(self) -> float:
        """Get current shader cache size in MB"""
        try:
            total_size = 0
            for cache_file in self.shader_cache_path.rglob('*'):
                if cache_file.is_file():
                    total_size += cache_file.stat().st_size
            return total_size / (1024 * 1024)
        except:
            return 0.0

if __name__ == '__main__':
    # Example usage
    optimizer = RADVOptimizer()
    
    # Apply global optimizations
    optimizer.apply_global_radv_optimizations()
    
    # Apply known game fixes
    optimizer.apply_known_game_fixes()
    
    # Get status
    status = optimizer.get_optimization_status()
    print(f"Optimization Status: {status}")
    
    # Optimize cache
    cache_results = optimizer.optimize_shader_cache()
    print(f"Cache Optimization: {cache_results}")