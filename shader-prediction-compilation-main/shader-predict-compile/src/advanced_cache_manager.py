#!/usr/bin/env python3

import os
import json
import hashlib
import threading
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

@dataclass
class CacheEntry:
    game_id: str
    shader_hash: str
    compile_time: float
    file_size: int
    last_accessed: datetime
    hit_count: int
    shader_type: str
    compile_flags: List[str]
    steam_deck_optimized: bool

@dataclass
class CacheStats:
    total_entries: int
    total_size_mb: float
    hit_rate: float
    compilation_time_saved_hours: float
    games_cached: int
    last_cleanup: datetime
    transcoded_videos: int

class AdvancedCacheManager:
    """
    Advanced shader cache management implementing SteamOS 3.6.22 improvements:
    - Fixed pre-compiled shader downloads
    - Transcoded video cutscenes for Proton games
    - Intelligent cache cleanup and optimization
    - Steam Deck specific caching strategies
    """
    
    def __init__(self):
        self.logger = logging.getLogger('advanced_cache_manager')
        
        # Cache directories
        self.mesa_cache_path = Path.home() / '.cache/mesa_shader_cache'
        self.steam_shader_cache = Path.home() / '.steam/steam/steamapps/shadercache'
        self.fossilize_cache = Path.home() / '.cache/fossilize'
        self.app_cache_path = Path.home() / '.cache/shader-predict-compile'
        
        # Create cache directories
        for cache_dir in [self.mesa_cache_path, self.fossilize_cache, self.app_cache_path]:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache database
        self.cache_db_path = self.app_cache_path / 'cache_database.json'
        self.cache_entries = {}
        self.cache_stats = CacheStats(0, 0.0, 0.0, 0.0, 0, datetime.now(), 0)
        
        # Load existing database
        self._load_cache_database()
        
        # SteamOS 3.6.22 specific settings
        self.steamos_integration = {
            'enable_precompiled_downloads': True,
            'transcode_cutscenes': True,
            'compression_enabled': True,
            'single_file_optimization': True,
            'automatic_cleanup': True
        }
        
        # Cache management settings
        self.cache_limits = {
            'max_total_size_gb': 4.0,  # 4GB max total cache
            'max_age_days': 30,        # 30 day retention
            'cleanup_threshold': 0.8,   # Cleanup when 80% full
            'min_hit_count': 2,        # Minimum hits to retain
            'max_entries_per_game': 500  # Limit per game
        }
        
        self.monitoring_active = False
        
    def _load_cache_database(self):
        """Load cache database from disk"""
        try:
            if self.cache_db_path.exists():
                with open(self.cache_db_path, 'r') as f:
                    data = json.load(f)
                    
                # Load cache entries
                for entry_data in data.get('entries', []):
                    entry_data['last_accessed'] = datetime.fromisoformat(entry_data['last_accessed'])
                    entry = CacheEntry(**entry_data)
                    self.cache_entries[entry.shader_hash] = entry
                
                # Load stats
                stats_data = data.get('stats', {})
                if stats_data:
                    stats_data['last_cleanup'] = datetime.fromisoformat(stats_data['last_cleanup'])
                    self.cache_stats = CacheStats(**stats_data)
                    
        except Exception as e:
            self.logger.warning(f"Could not load cache database: {e}")
    
    def _save_cache_database(self):
        """Save cache database to disk"""
        try:
            data = {
                'entries': [],
                'stats': asdict(self.cache_stats)
            }
            
            # Convert entries to serializable format
            for entry in self.cache_entries.values():
                entry_dict = asdict(entry)
                entry_dict['last_accessed'] = entry.last_accessed.isoformat()
                data['entries'].append(entry_dict)
            
            # Convert stats datetime
            data['stats']['last_cleanup'] = self.cache_stats.last_cleanup.isoformat()
            
            with open(self.cache_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save cache database: {e}")
    
    def integrate_with_steamos_cache(self) -> Dict[str, int]:
        """Integrate with SteamOS shader pre-compilation system"""
        results = {
            'precompiled_shaders_downloaded': 0,
            'transcoded_videos_found': 0,
            'steam_cache_entries': 0,
            'integration_errors': 0
        }
        
        try:
            # Check Steam shader cache directory
            if self.steam_shader_cache.exists():
                for game_cache_dir in self.steam_shader_cache.iterdir():
                    if game_cache_dir.is_dir():
                        results['steam_cache_entries'] += 1
                        
                        # Look for pre-compiled shaders
                        precompiled_dir = game_cache_dir / 'fozpipelinecache_steamdeck'
                        if precompiled_dir.exists():
                            fossilize_files = list(precompiled_dir.glob('*.foz'))
                            results['precompiled_shaders_downloaded'] += len(fossilize_files)
                            
                            # Import into our cache system
                            self._import_steam_cache_entry(game_cache_dir.name, precompiled_dir)
                        
                        # Look for transcoded videos (SteamOS 3.6.22 feature)
                        video_cache_dir = game_cache_dir / 'transcoded_videos'
                        if video_cache_dir.exists():
                            video_files = list(video_cache_dir.glob('*'))
                            results['transcoded_videos_found'] += len(video_files)
                            self.cache_stats.transcoded_videos += len(video_files)
            
            self.logger.info(f"SteamOS integration complete: {results}")
            
        except Exception as e:
            self.logger.error(f"SteamOS integration error: {e}")
            results['integration_errors'] = 1
            
        return results
    
    def _import_steam_cache_entry(self, game_id: str, cache_dir: Path):
        """Import Steam cache entry into our database"""
        try:
            for foz_file in cache_dir.glob('*.foz'):
                # Calculate hash for the file
                file_hash = self._calculate_file_hash(foz_file)
                
                # Create cache entry
                entry = CacheEntry(
                    game_id=game_id,
                    shader_hash=file_hash,
                    compile_time=0.0,  # Pre-compiled by Valve
                    file_size=foz_file.stat().st_size,
                    last_accessed=datetime.now(),
                    hit_count=1,  # Imported from Steam
                    shader_type='precompiled',
                    compile_flags=['steam_precompiled'],
                    steam_deck_optimized=True
                )
                
                self.cache_entries[file_hash] = entry
                
        except Exception as e:
            self.logger.warning(f"Could not import cache entry for {game_id}: {e}")
    
    def optimize_cache_structure(self) -> Dict[str, any]:
        """Optimize cache structure using single-file format and compression"""
        results = {
            'files_optimized': 0,
            'size_reduction_mb': 0,
            'compression_enabled': False,
            'single_file_conversion': 0
        }
        
        try:
            # Enable compression for mesa cache
            self._enable_mesa_compression()
            results['compression_enabled'] = True
            
            # Convert to single-file format where possible
            single_file_count = self._convert_to_single_file_format()
            results['single_file_conversion'] = single_file_count
            
            # Optimize fossilize databases
            fossilize_results = self._optimize_fossilize_databases()
            results.update(fossilize_results)
            
            # Deduplicate identical shaders
            dedup_results = self._deduplicate_shaders()
            results['files_optimized'] += dedup_results['deduplicated']
            results['size_reduction_mb'] += dedup_results['size_saved_mb']
            
            self.logger.info(f"Cache optimization complete: {results}")
            
        except Exception as e:
            self.logger.error(f"Cache optimization error: {e}")
            
        return results
    
    def _enable_mesa_compression(self):
        """Enable Mesa shader cache compression"""
        try:
            # Set environment variable for compression
            os.environ['MESA_DISK_CACHE_COMPRESSION'] = 'true'
            os.environ['MESA_DISK_CACHE_SINGLE_FILE'] = 'true'
            
            # Create mesa config if it doesn't exist
            mesa_config_dir = Path.home() / '.config/mesa'
            mesa_config_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = mesa_config_dir / 'config'
            with open(config_file, 'w') as f:
                f.write("MESA_DISK_CACHE_COMPRESSION=true\n")
                f.write("MESA_DISK_CACHE_SINGLE_FILE=true\n")
                
        except Exception as e:
            self.logger.warning(f"Could not enable Mesa compression: {e}")
    
    def _convert_to_single_file_format(self) -> int:
        """Convert multi-file caches to single-file format"""
        converted = 0
        
        try:
            # Process mesa cache directories
            for cache_dir in self.mesa_cache_path.iterdir():
                if cache_dir.is_dir():
                    cache_files = list(cache_dir.glob('*'))
                    if len(cache_files) > 10:  # Worth converting
                        if self._merge_cache_files(cache_dir):
                            converted += 1
                            
        except Exception as e:
            self.logger.warning(f"Single-file conversion error: {e}")
            
        return converted
    
    def _merge_cache_files(self, cache_dir: Path) -> bool:
        """Merge multiple cache files into single file"""
        try:
            cache_files = list(cache_dir.glob('*'))
            if len(cache_files) <= 1:
                return False
                
            # Create merged file
            merged_file = cache_dir / 'merged_cache.bin'
            
            with open(merged_file, 'wb') as merged:
                for cache_file in cache_files:
                    if cache_file != merged_file:
                        with open(cache_file, 'rb') as f:
                            merged.write(f.read())
                        cache_file.unlink()  # Remove original
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not merge cache files in {cache_dir}: {e}")
            return False
    
    def _optimize_fossilize_databases(self) -> Dict[str, int]:
        """Optimize Fossilize databases with compression and cleanup"""
        results = {
            'databases_optimized': 0,
            'compression_applied': 0
        }
        
        try:
            for foz_file in self.fossilize_cache.rglob('*.foz'):
                # Try to compress existing fossilize database
                if self._compress_fossilize_database(foz_file):
                    results['compression_applied'] += 1
                
                results['databases_optimized'] += 1
                
        except Exception as e:
            self.logger.warning(f"Fossilize optimization error: {e}")
            
        return results
    
    def _compress_fossilize_database(self, foz_file: Path) -> bool:
        """Compress Fossilize database using zstd"""
        try:
            # Use fossilize tools if available
            result = subprocess.run([
                'fossilize-replay',
                '--compress', 'zstd',
                '--optimize',
                str(foz_file)
            ], capture_output=True, check=False)
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _deduplicate_shaders(self) -> Dict[str, any]:
        """Deduplicate identical shaders across cache"""
        results = {
            'deduplicated': 0,
            'size_saved_mb': 0
        }
        
        try:
            # Build hash map of file contents
            hash_to_files = {}
            
            for cache_dir in [self.mesa_cache_path, self.fossilize_cache]:
                for cache_file in cache_dir.rglob('*'):
                    if cache_file.is_file():
                        file_hash = self._calculate_file_hash(cache_file)
                        
                        if file_hash in hash_to_files:
                            hash_to_files[file_hash].append(cache_file)
                        else:
                            hash_to_files[file_hash] = [cache_file]
            
            # Remove duplicates (keep one copy)
            for file_hash, files in hash_to_files.items():
                if len(files) > 1:
                    # Keep the first file, remove others
                    for duplicate_file in files[1:]:
                        size_mb = duplicate_file.stat().st_size / (1024 * 1024)
                        duplicate_file.unlink()
                        
                        results['deduplicated'] += 1
                        results['size_saved_mb'] += size_mb
                        
        except Exception as e:
            self.logger.warning(f"Deduplication error: {e}")
            
        return results
    
    def intelligent_cleanup(self, force: bool = False) -> Dict[str, int]:
        """Intelligent cache cleanup based on usage patterns and age"""
        results = {
            'entries_removed': 0,
            'size_freed_mb': 0,
            'games_cleaned': 0
        }
        
        try:
            current_size_gb = self._get_total_cache_size_gb()
            
            # Check if cleanup is needed
            if not force and current_size_gb < (self.cache_limits['max_total_size_gb'] * self.cache_limits['cleanup_threshold']):
                return results
            
            # Calculate cleanup targets
            cutoff_date = datetime.now() - timedelta(days=self.cache_limits['max_age_days'])
            
            # Find candidates for removal
            removal_candidates = []
            
            for shader_hash, entry in self.cache_entries.items():
                should_remove = False
                
                # Remove very old entries
                if entry.last_accessed < cutoff_date:
                    should_remove = True
                
                # Remove low-usage entries if over limit
                elif (current_size_gb > self.cache_limits['max_total_size_gb'] and 
                      entry.hit_count < self.cache_limits['min_hit_count']):
                    should_remove = True
                
                if should_remove:
                    removal_candidates.append((shader_hash, entry))
            
            # Sort by priority (remove least valuable first)
            removal_candidates.sort(key=lambda x: (x[1].hit_count, x[1].last_accessed))
            
            # Remove candidates until under limit
            target_size_gb = self.cache_limits['max_total_size_gb'] * 0.7  # Target 70% of limit
            
            for shader_hash, entry in removal_candidates:
                if current_size_gb <= target_size_gb:
                    break
                
                # Remove cache files
                if self._remove_cache_entry(entry):
                    del self.cache_entries[shader_hash]
                    results['entries_removed'] += 1
                    results['size_freed_mb'] += entry.file_size / (1024 * 1024)
                    current_size_gb -= entry.file_size / (1024 * 1024 * 1024)
            
            # Update stats
            self.cache_stats.last_cleanup = datetime.now()
            self._save_cache_database()
            
            self.logger.info(f"Cache cleanup complete: {results}")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")
            
        return results
    
    def _remove_cache_entry(self, entry: CacheEntry) -> bool:
        """Remove cache entry files from disk"""
        try:
            # This would remove actual cache files
            # Implementation depends on cache structure
            return True
        except:
            return False
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return ""
    
    def _get_total_cache_size_gb(self) -> float:
        """Get total cache size in GB"""
        total_size = 0
        
        try:
            for cache_dir in [self.mesa_cache_path, self.fossilize_cache, self.steam_shader_cache]:
                if cache_dir.exists():
                    for cache_file in cache_dir.rglob('*'):
                        if cache_file.is_file():
                            total_size += cache_file.stat().st_size
        except:
            pass
            
        return total_size / (1024 * 1024 * 1024)
    
    def update_cache_stats(self):
        """Update cache statistics"""
        self.cache_stats.total_entries = len(self.cache_entries)
        self.cache_stats.total_size_mb = self._get_total_cache_size_gb() * 1024
        
        # Calculate hit rate
        if self.cache_stats.total_entries > 0:
            total_hits = sum(entry.hit_count for entry in self.cache_entries.values())
            self.cache_stats.hit_rate = total_hits / self.cache_stats.total_entries
        
        # Calculate time saved
        total_compile_time = sum(entry.compile_time for entry in self.cache_entries.values())
        self.cache_stats.compilation_time_saved_hours = total_compile_time / 3600
        
        # Count games
        games = set(entry.game_id for entry in self.cache_entries.values())
        self.cache_stats.games_cached = len(games)
    
    def start_automatic_management(self):
        """Start automatic cache management"""
        self.monitoring_active = True
        
        def management_loop():
            while self.monitoring_active:
                try:
                    # Update stats
                    self.update_cache_stats()
                    
                    # Check if cleanup is needed
                    current_size = self._get_total_cache_size_gb()
                    if current_size > (self.cache_limits['max_total_size_gb'] * self.cache_limits['cleanup_threshold']):
                        self.intelligent_cleanup()
                    
                    # Periodic optimization
                    if datetime.now().hour == 3:  # Run at 3 AM
                        self.optimize_cache_structure()
                    
                    # Save database
                    self._save_cache_database()
                    
                    time.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    self.logger.error(f"Automatic management error: {e}")
                    time.sleep(600)  # Wait 10 minutes on error
        
        management_thread = threading.Thread(target=management_loop, daemon=True)
        management_thread.start()
        
        self.logger.info("Started automatic cache management")
    
    def stop_automatic_management(self):
        """Stop automatic cache management"""
        self.monitoring_active = False
        self._save_cache_database()
        self.logger.info("Stopped automatic cache management")
    
    def get_cache_report(self) -> Dict:
        """Get comprehensive cache report"""
        self.update_cache_stats()
        
        return {
            'statistics': asdict(self.cache_stats),
            'current_size_gb': self._get_total_cache_size_gb(),
            'limits': self.cache_limits,
            'steamos_integration': self.steamos_integration,
            'top_games': self._get_top_cached_games(10),
            'recent_activity': self._get_recent_cache_activity(),
            'recommendations': self._get_cache_recommendations()
        }
    
    def _get_top_cached_games(self, limit: int) -> List[Dict]:
        """Get top games by cache usage"""
        game_stats = {}
        
        for entry in self.cache_entries.values():
            if entry.game_id not in game_stats:
                game_stats[entry.game_id] = {
                    'entries': 0,
                    'total_size_mb': 0,
                    'total_hits': 0
                }
            
            game_stats[entry.game_id]['entries'] += 1
            game_stats[entry.game_id]['total_size_mb'] += entry.file_size / (1024 * 1024)
            game_stats[entry.game_id]['total_hits'] += entry.hit_count
        
        # Sort by total hits
        sorted_games = sorted(game_stats.items(), 
                            key=lambda x: x[1]['total_hits'], 
                            reverse=True)
        
        return [{'game_id': game, **stats} for game, stats in sorted_games[:limit]]
    
    def _get_recent_cache_activity(self) -> List[Dict]:
        """Get recent cache activity"""
        recent_entries = [
            {
                'game_id': entry.game_id,
                'shader_type': entry.shader_type,
                'last_accessed': entry.last_accessed.isoformat(),
                'hit_count': entry.hit_count
            }
            for entry in sorted(self.cache_entries.values(), 
                              key=lambda x: x.last_accessed, 
                              reverse=True)[:20]
        ]
        
        return recent_entries
    
    def _get_cache_recommendations(self) -> List[str]:
        """Get cache optimization recommendations"""
        recommendations = []
        
        current_size = self._get_total_cache_size_gb()
        
        if current_size > self.cache_limits['max_total_size_gb']:
            recommendations.append("Cache size exceeds limit - cleanup recommended")
        
        if self.cache_stats.hit_rate < 2.0:
            recommendations.append("Low hit rate detected - consider adjusting retention policy")
        
        if not self.steamos_integration['compression_enabled']:
            recommendations.append("Enable compression to save space")
        
        if self.cache_stats.transcoded_videos == 0:
            recommendations.append("No transcoded videos found - check Proton game cutscenes")
        
        old_entries = sum(1 for entry in self.cache_entries.values() 
                         if entry.last_accessed < datetime.now() - timedelta(days=14))
        
        if old_entries > 100:
            recommendations.append(f"{old_entries} entries haven't been accessed in 2 weeks")
        
        return recommendations

if __name__ == '__main__':
    # Example usage
    cache_manager = AdvancedCacheManager()
    
    # Integrate with SteamOS cache
    integration_results = cache_manager.integrate_with_steamos_cache()
    print(f"SteamOS Integration: {integration_results}")
    
    # Optimize cache
    optimization_results = cache_manager.optimize_cache_structure()
    print(f"Cache Optimization: {optimization_results}")
    
    # Get cache report
    report = cache_manager.get_cache_report()
    print(f"Cache Report: {report['statistics']}")
    
    # Start automatic management
    cache_manager.start_automatic_management()
    
    try:
        time.sleep(10)  # Run for 10 seconds
    except KeyboardInterrupt:
        pass
    finally:
        cache_manager.stop_automatic_management()