#!/usr/bin/env python3

import os
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from queue import Queue, Empty
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Create a dummy psutil for basic operations
    class DummyPsutil:
        @staticmethod
        def cpu_count():
            return 4
        @staticmethod
        def cpu_percent(interval=None):
            return 25.0
        @staticmethod
        def virtual_memory():
            class Memory:
                percent = 50.0
            return Memory()
        @staticmethod
        def process_iter(*args, **kwargs):
            return []
        @staticmethod
        def disk_io_counters():
            class DiskIO:
                read_bytes = 0
                write_bytes = 0
            return DiskIO()
    psutil = DummyPsutil()

class FossilizeIntegration:
    def __init__(self):
        self.fossilize_path = self._find_fossilize()
        self.vulkan_icd_path = '/usr/share/vulkan/icd.d'
        self.compile_queue = Queue()
        self.compile_threads = []
        self.is_running = False
        self.stats = {
            'shaders_compiled': 0,
            'shaders_failed': 0,
            'total_compile_time': 0,
            'cache_hits': 0
        }
        
    def _find_fossilize(self) -> Optional[Path]:
        """Find Fossilize installation on Steam Deck"""
        possible_paths = [
            Path('/usr/bin/fossilize-replay'),
            Path('/usr/local/bin/fossilize-replay'),
            Path.home() / '.steam/steam/ubuntu12_32/fossilize_replay',
            Path.home() / '.steam/steam/ubuntu12_64/fossilize_replay'
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Try to find via Steam runtime
        try:
            result = subprocess.run(['which', 'fossilize-replay'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass
            
        return None
    
    def check_compatibility(self) -> Dict[str, bool]:
        """Check Steam Deck compatibility"""
        checks = {
            'fossilize_available': False,
            'vulkan_support': False,
            'adequate_storage': False,
            'steam_running': False,
            'correct_kernel': False
        }
        
        # Check Fossilize
        checks['fossilize_available'] = self.fossilize_path is not None
        
        # Check Vulkan support
        checks['vulkan_support'] = Path(self.vulkan_icd_path).exists()
        
        # Check storage (need at least 1GB free)
        statvfs = os.statvfs('/')
        free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        checks['adequate_storage'] = free_gb > 1.0
        
        # Check if Steam is running
        checks['steam_running'] = any('steam' in p.name().lower() 
                                     for p in psutil.process_iter(['name']))
        
        # Check kernel version (Steam Deck specific)
        try:
            kernel = subprocess.run(['uname', '-r'], 
                                  capture_output=True, text=True).stdout.strip()
            checks['correct_kernel'] = 'neptune' in kernel or 'valve' in kernel
        except:
            pass
            
        return checks
    
    def prepare_shader_cache(self, game_id: str, priorities: List[Dict]) -> Path:
        """Prepare shader cache directory and metadata"""
        cache_base = Path.home() / '.cache' / 'shader-predict-compile'
        game_cache = cache_base / game_id
        game_cache.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'game_id': game_id,
            'timestamp': time.time(),
            'priority_count': len(priorities),
            'steam_deck_optimized': True,
            'fossilize_version': self._get_fossilize_version()
        }
        
        with open(game_cache / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return game_cache
    
    def _get_fossilize_version(self) -> str:
        """Get Fossilize version"""
        if not self.fossilize_path:
            return 'unknown'
            
        try:
            result = subprocess.run([str(self.fossilize_path), '--version'],
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except:
            return 'unknown'
    
    def compile_shaders(self, game_id: str, priorities: List[Dict], 
                       progress_callback=None) -> Dict:
        """Compile shaders with Fossilize integration"""
        if not self.fossilize_path:
            return {'error': 'Fossilize not found'}
            
        cache_dir = self.prepare_shader_cache(game_id, priorities)
        results = {
            'compiled': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0
        }
        
        # Start compilation threads
        self.is_running = True
        thread_count = min(4, psutil.cpu_count())  # Steam Deck has 4 cores
        
        for i in range(thread_count):
            thread = threading.Thread(target=self._compile_worker, 
                                    args=(cache_dir, progress_callback))
            thread.start()
            self.compile_threads.append(thread)
            
        # Queue shader compilation tasks
        for priority in priorities:
            self.compile_queue.put(priority)
            
        # Wait for completion
        self.compile_queue.join()
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.compile_threads:
            thread.join()
            
        results.update(self.stats)
        return results
    
    def _compile_worker(self, cache_dir: Path, progress_callback):
        """Worker thread for shader compilation"""
        while self.is_running:
            try:
                shader_info = self.compile_queue.get(timeout=1)
                
                start_time = time.time()
                success = self._compile_single_shader(shader_info, cache_dir)
                compile_time = time.time() - start_time
                
                if success:
                    self.stats['shaders_compiled'] += 1
                else:
                    self.stats['shaders_failed'] += 1
                    
                self.stats['total_compile_time'] += compile_time
                
                if progress_callback:
                    progress_callback({
                        'shader': shader_info['variant'],
                        'success': success,
                        'time': compile_time,
                        'total_compiled': self.stats['shaders_compiled']
                    })
                    
                self.compile_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Compilation error: {e}")
                self.compile_queue.task_done()
    
    def _compile_single_shader(self, shader_info: Dict, cache_dir: Path) -> bool:
        """Compile a single shader variant"""
        # Generate shader specification for Fossilize
        shader_spec = {
            'type': shader_info['shader_type'],
            'variant': shader_info['variant'],
            'flags': shader_info.get('compile_flags', []),
            'priority': shader_info['priority']
        }
        
        spec_file = cache_dir / f"{shader_info['variant']}.json"
        with open(spec_file, 'w') as f:
            json.dump(shader_spec, f)
            
        # Prepare Fossilize command
        cmd = [
            str(self.fossilize_path),
            '--enable-pipeline-cache',
            '--shader-spec', str(spec_file),
            '--output', str(cache_dir / f"{shader_info['variant']}.foz")
        ]
        
        # Add Steam Deck optimizations
        if shader_info.get('steam_deck_optimized'):
            cmd.extend(['--opt-level', '2'])
            cmd.extend(['--target-cpu', 'znver2'])
            
        # Run compilation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def integrate_with_steam(self, game_id: str, compiled_cache: Path) -> bool:
        """Integrate compiled shaders with Steam's shader cache"""
        steam_shader_cache = Path.home() / '.steam/steam/steamapps/shadercache' / game_id
        
        try:
            # Create shader cache directory if it doesn't exist
            steam_shader_cache.mkdir(parents=True, exist_ok=True)
            
            # Copy our compiled shaders to Steam's cache
            for shader_file in compiled_cache.glob('*.foz'):
                target = steam_shader_cache / shader_file.name
                
                # Check if shader already exists and is newer
                if target.exists() and target.stat().st_mtime > shader_file.stat().st_mtime:
                    self.stats['cache_hits'] += 1
                    continue
                    
                # Copy shader to Steam cache
                import shutil
                shutil.copy2(shader_file, target)
                
            # Update Steam's shader cache metadata
            metadata_file = steam_shader_cache / 'shader_cache_metadata.json'
            metadata = {}
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
            metadata['last_update'] = time.time()
            metadata['predictive_compile'] = True
            metadata['shader_count'] = len(list(steam_shader_cache.glob('*.foz')))
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Steam integration error: {e}")
            return False
    
    def monitor_performance(self) -> Dict:
        """Monitor shader compilation performance"""
        performance = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_io': {},
            'gpu_usage': self._get_gpu_usage(),
            'compilation_rate': 0
        }
        
        # Calculate compilation rate
        if self.stats['total_compile_time'] > 0:
            performance['compilation_rate'] = (
                self.stats['shaders_compiled'] / self.stats['total_compile_time']
            )
            
        # Get disk I/O stats
        disk_io = psutil.disk_io_counters()
        performance['disk_io'] = {
            'read_mb': disk_io.read_bytes / (1024**2),
            'write_mb': disk_io.write_bytes / (1024**2)
        }
        
        return performance
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage for Steam Deck (AMD APU)"""
        try:
            # Try to read from sysfs
            gpu_busy_path = Path('/sys/class/drm/card0/device/gpu_busy_percent')
            if gpu_busy_path.exists():
                with open(gpu_busy_path, 'r') as f:
                    return float(f.read().strip())
        except:
            pass
            
        return 0.0
    
    def cleanup_old_caches(self, days_old: int = 30) -> int:
        """Clean up old shader caches"""
        cache_base = Path.home() / '.cache' / 'shader-predict-compile'
        cleaned = 0
        
        if not cache_base.exists():
            return 0
            
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        for game_cache in cache_base.iterdir():
            if game_cache.is_dir():
                metadata_file = game_cache / 'metadata.json'
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                        if metadata.get('timestamp', 0) < cutoff_time:
                            import shutil
                            shutil.rmtree(game_cache)
                            cleaned += 1
                    except:
                        pass
                        
        return cleaned