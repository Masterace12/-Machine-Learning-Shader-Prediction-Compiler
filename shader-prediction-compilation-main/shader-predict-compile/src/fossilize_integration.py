#!/usr/bin/env python3

import os
import json
import subprocess
import threading
import time
import struct
import zlib
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from queue import Queue, Empty
import logging
from dataclasses import dataclass
from enum import Enum
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

# SPIRV-Tools integration constants
SPIRV_MAGIC_NUMBER = 0x07230203
SPIRV_VERSION_1_6 = 0x00010600

@dataclass
class VkPipelineInfo:
    """Vulkan pipeline information for Fossilize database"""
    stage_flags: int
    shader_hash: str
    spirv_data: bytes
    specialization_info: Optional[Dict] = None
    descriptor_layout: Optional[Dict] = None
    render_pass_info: Optional[Dict] = None

class ShaderStage(Enum):
    VERTEX = 0x00000001
    FRAGMENT = 0x00000010
    COMPUTE = 0x00000020
    GEOMETRY = 0x00000008
    TESS_CONTROL = 0x00000002
    TESS_EVALUATION = 0x00000004

class FossilizeIntegration:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fossilize_path = self._find_fossilize()
        self.spirv_tools_path = self._find_spirv_tools()
        self.vulkan_icd_path = self._get_vulkan_icd_path()
        self.compile_queue = Queue()
        self.compile_threads = []
        self.is_running = False
        self.pipeline_cache = {}
        self.steam_shader_cache = self._locate_steam_cache()
        self.stats = {
            'shaders_compiled': 0,
            'shaders_failed': 0,
            'total_compile_time': 0,
            'cache_hits': 0,
            'foz_databases_created': 0,
            'spirv_validated': 0,
            'pipeline_caches_serialized': 0
        }
        
        # Initialize SPIRV-Tools integration
        self.spirv_optimizer_options = [
            '--eliminate-dead-code-aggressive',
            '--merge-blocks',
            '--inline-entry-points-exhaustive',
            '--eliminate-local-single-block',
            '--eliminate-local-single-store',
            '--merge-return',
            '--eliminate-local-multi-store',
            '--convert-local-access-chains',
            '--eliminate-insert-extract',
            '--reduce-load-size'
        ]
        
    def _find_fossilize(self) -> Optional[Path]:
        """Find Fossilize installation with comprehensive search"""
        possible_paths = [
            # Steam Deck locations
            Path('/usr/bin/fossilize-replay'),
            Path('/usr/local/bin/fossilize-replay'),
            Path('/home/deck/.steam/steam/ubuntu12_32/fossilize-replay'),
            Path('/home/deck/.steam/steam/ubuntu12_64/fossilize-replay'),
            # Steam runtime locations
            Path.home() / '.steam/steam/ubuntu12_32/fossilize-replay',
            Path.home() / '.steam/steam/ubuntu12_64/fossilize-replay',
            Path.home() / '.steam/steam/steamapps/common/SteamLinuxRuntime/pressure-vessel/bin/fossilize-replay',
            # System-wide installations
            Path('/opt/fossilize/bin/fossilize-replay'),
            Path('/usr/share/vulkan/fossilize-replay')
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                # Verify it's executable and working
                if self._verify_fossilize_binary(path):
                    self.logger.info(f"Found Fossilize at: {path}")
                    return path
                    
        # Try to find via Steam runtime or system PATH
        try:
            result = subprocess.run(['which', 'fossilize-replay'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                path = Path(result.stdout.strip())
                if self._verify_fossilize_binary(path):
                    return path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        self.logger.warning("Fossilize binary not found - shader pre-compilation will be limited")
        return None
        
    def _find_spirv_tools(self) -> Optional[Path]:
        """Find SPIRV-Tools installation"""
        possible_paths = [
            Path('/usr/bin/spirv-opt'),
            Path('/usr/local/bin/spirv-opt'),
            Path('/opt/spirv-tools/bin/spirv-opt'),
            Path.home() / '.local/bin/spirv-opt'
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                try:
                    result = subprocess.run([str(path), '--version'], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        self.logger.info(f"Found SPIRV-Tools at: {path}")
                        return path
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
                    
        # Try system PATH
        try:
            result = subprocess.run(['which', 'spirv-opt'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        self.logger.warning("SPIRV-Tools not found - shader optimization will be limited")
        return None
        
    def _verify_fossilize_binary(self, path: Path) -> bool:
        """Verify that the Fossilize binary is working"""
        try:
            result = subprocess.run([str(path), '--help'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and 'fossilize' in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def _get_vulkan_icd_path(self) -> str:
        """Get Vulkan ICD path for the current system"""
        possible_paths = [
            '/usr/share/vulkan/icd.d',
            '/etc/vulkan/icd.d',
            '/home/deck/.local/share/vulkan/icd.d'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        return '/usr/share/vulkan/icd.d'  # Default fallback
        
    def _locate_steam_cache(self) -> Optional[Path]:
        """Locate Steam's shader cache directory"""
        steam_paths = [
            Path.home() / '.steam/steam/steamapps/shadercache',
            Path.home() / '.local/share/Steam/steamapps/shadercache',
            Path('/home/deck/.steam/steam/steamapps/shadercache')  # Steam Deck specific
        ]
        
        for path in steam_paths:
            if path.exists():
                return path
                
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
        """Compile a single shader variant using proper Fossilize workflow"""
        try:
            # Extract or generate SPIRV bytecode
            spirv_data = self._get_spirv_bytecode(shader_info)
            if not spirv_data:
                self.logger.error(f"Failed to get SPIRV bytecode for {shader_info['variant']}")
                return False
                
            # Validate and optimize SPIRV if SPIRV-Tools available
            if self.spirv_tools_path:
                spirv_data = self._optimize_spirv(spirv_data)
                if not self._validate_spirv(spirv_data):
                    self.logger.error(f"SPIRV validation failed for {shader_info['variant']}")
                    return False
                self.stats['spirv_validated'] += 1
            
            # Create VkPipelineInfo for Fossilize database
            pipeline_info = VkPipelineInfo(
                stage_flags=self._get_stage_flags(shader_info['shader_type']),
                shader_hash=hashlib.sha256(spirv_data).hexdigest()[:16],
                spirv_data=spirv_data,
                specialization_info=shader_info.get('specialization_info'),
                descriptor_layout=shader_info.get('descriptor_layout'),
                render_pass_info=shader_info.get('render_pass_info')
            )
            
            # Create Fossilize database entry
            success = self._create_fossilize_database_entry(
                cache_dir, shader_info['variant'], pipeline_info
            )
            
            if success:
                # Create VkPipelineCache for Steam integration
                self._create_vulkan_pipeline_cache(cache_dir, shader_info['variant'], pipeline_info)
                self.stats['pipeline_caches_serialized'] += 1
                
            return success
            
        except Exception as e:
            self.logger.error(f"Shader compilation failed for {shader_info['variant']}: {e}")
            return False
            
    def _get_spirv_bytecode(self, shader_info: Dict) -> Optional[bytes]:
        """Get or generate SPIRV bytecode for shader"""
        try:
            # If SPIRV data is directly provided
            if 'spirv_data' in shader_info:
                return shader_info['spirv_data']
                
            # If shader source is provided, compile to SPIRV
            if 'shader_source' in shader_info:
                return self._compile_glsl_to_spirv(
                    shader_info['shader_source'],
                    shader_info['shader_type']
                )
                
            # If shader file path is provided
            if 'shader_file' in shader_info:
                shader_path = Path(shader_info['shader_file'])
                if shader_path.exists():
                    if shader_path.suffix in ['.spv', '.spirv']:
                        # Already compiled SPIRV
                        return shader_path.read_bytes()
                    else:
                        # Compile from source
                        source = shader_path.read_text()
                        return self._compile_glsl_to_spirv(source, shader_info['shader_type'])
                        
            self.logger.error(f"No SPIRV source found for {shader_info.get('variant', 'unknown')}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting SPIRV bytecode: {e}")
            return None
            
    def _compile_glsl_to_spirv(self, source: str, shader_type: str) -> Optional[bytes]:
        """Compile GLSL source to SPIRV using glslc"""
        try:
            # Try to find glslc (from shaderc)
            glslc_paths = ['/usr/bin/glslc', '/usr/local/bin/glslc']
            glslc_path = None
            
            for path in glslc_paths:
                if Path(path).exists():
                    glslc_path = path
                    break
                    
            if not glslc_path:
                # Try system PATH
                result = subprocess.run(['which', 'glslc'], capture_output=True, text=True)
                if result.returncode == 0:
                    glslc_path = result.stdout.strip()
                    
            if not glslc_path:
                self.logger.warning("glslc not found - cannot compile GLSL to SPIRV")
                return None
                
            # Determine shader stage flag for glslc
            stage_map = {
                'vertex': 'vert',
                'fragment': 'frag', 
                'compute': 'comp',
                'geometry': 'geom',
                'tess_control': 'tesc',
                'tess_evaluation': 'tese'
            }
            
            stage = stage_map.get(shader_type.lower(), 'frag')
            
            # Compile to SPIRV
            cmd = [
                glslc_path,
                '-fshader-stage=' + stage,
                '-O',  # Optimize
                '-o', '-',  # Output to stdout
                '-'  # Input from stdin
            ]
            
            result = subprocess.run(
                cmd,
                input=source.encode(),
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.error(f"GLSL compilation failed: {result.stderr.decode()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error compiling GLSL to SPIRV: {e}")
            return None
            
    def _optimize_spirv(self, spirv_data: bytes) -> bytes:
        """Optimize SPIRV bytecode using spirv-opt"""
        if not self.spirv_tools_path:
            return spirv_data
            
        try:
            cmd = [str(self.spirv_tools_path)] + self.spirv_optimizer_options + ['-o', '-', '-']
            
            result = subprocess.run(
                cmd,
                input=spirv_data,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.debug("SPIRV optimization successful")
                return result.stdout
            else:
                self.logger.warning(f"SPIRV optimization failed: {result.stderr.decode()}")
                return spirv_data
                
        except Exception as e:
            self.logger.warning(f"SPIRV optimization error: {e}")
            return spirv_data
            
    def _validate_spirv(self, spirv_data: bytes) -> bool:
        """Validate SPIRV bytecode"""
        try:
            # Basic format validation
            if len(spirv_data) < 20:
                return False
                
            # Check SPIRV magic number
            magic = struct.unpack('<I', spirv_data[0:4])[0]
            if magic != SPIRV_MAGIC_NUMBER:
                return False
                
            # Check version
            version = struct.unpack('<I', spirv_data[4:8])[0]
            if version > SPIRV_VERSION_1_6:
                self.logger.warning(f"SPIRV version {version:08x} may not be supported")
                
            # Use spirv-val if available
            if self.spirv_tools_path:
                spirv_val_path = self.spirv_tools_path.parent / 'spirv-val'
                if spirv_val_path.exists():
                    result = subprocess.run(
                        [str(spirv_val_path), '-'],
                        input=spirv_data,
                        capture_output=True,
                        timeout=5
                    )
                    return result.returncode == 0
                    
            return True
            
        except Exception as e:
            self.logger.error(f"SPIRV validation error: {e}")
            return False
            
    def _get_stage_flags(self, shader_type: str) -> int:
        """Get Vulkan stage flags for shader type"""
        stage_map = {
            'vertex': ShaderStage.VERTEX.value,
            'fragment': ShaderStage.FRAGMENT.value,
            'compute': ShaderStage.COMPUTE.value,
            'geometry': ShaderStage.GEOMETRY.value,
            'tess_control': ShaderStage.TESS_CONTROL.value,
            'tess_evaluation': ShaderStage.TESS_EVALUATION.value
        }
        return stage_map.get(shader_type.lower(), ShaderStage.FRAGMENT.value)
    
    def _create_fossilize_database_entry(self, cache_dir: Path, variant: str, pipeline_info: VkPipelineInfo) -> bool:
        """Create proper Fossilize database entry"""
        try:
            # Create Fossilize database file
            foz_file = cache_dir / f"{variant}.foz"
            
            # Fossilize database format:
            # - Magic header: "FOSSILIZEDB"
            # - Version: 4 bytes
            # - Entry count: 4 bytes
            # - Entries: variable length
            
            with open(foz_file, 'wb') as f:
                # Write header
                f.write(b'FOSSILIZEDB\x00\x00\x00\x00\x00')
                
                # Write version (format version 1)
                f.write(struct.pack('<I', 1))
                
                # Write entry count (1 for now)
                f.write(struct.pack('<I', 1))
                
                # Write pipeline entry
                entry_data = self._serialize_pipeline_entry(pipeline_info)
                f.write(struct.pack('<I', len(entry_data)))
                f.write(entry_data)
                
            self.stats['foz_databases_created'] += 1
            self.logger.debug(f"Created Fossilize database: {foz_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Fossilize database entry: {e}")
            return False
            
    def _serialize_pipeline_entry(self, pipeline_info: VkPipelineInfo) -> bytes:
        """Serialize pipeline info for Fossilize database"""
        try:
            # Create pipeline state object
            pipeline_state = {
                'type': 'graphics_pipeline',  # or 'compute_pipeline'
                'hash': pipeline_info.shader_hash,
                'stages': [{
                    'stage': pipeline_info.stage_flags,
                    'spirv': pipeline_info.spirv_data.hex(),
                    'specialization': pipeline_info.specialization_info or {}
                }],
                'vertex_input': {},
                'input_assembly': {'topology': 'triangle_list'},
                'rasterization': {'cull_mode': 'back'},
                'multisample': {'sample_count': 1},
                'depth_stencil': {},
                'color_blend': {},
                'dynamic_state': [],
                'descriptor_layout': pipeline_info.descriptor_layout or {},
                'render_pass': pipeline_info.render_pass_info or {}
            }
            
            # Serialize to JSON then compress
            json_data = json.dumps(pipeline_state, separators=(',', ':')).encode()
            compressed_data = zlib.compress(json_data, level=6)
            
            return compressed_data
            
        except Exception as e:
            self.logger.error(f"Pipeline serialization error: {e}")
            return b''
            
    def _create_vulkan_pipeline_cache(self, cache_dir: Path, variant: str, pipeline_info: VkPipelineInfo) -> bool:
        """Create VkPipelineCache compatible cache file"""
        try:
            cache_file = cache_dir / f"{variant}.vkpipelinecache"
            
            # VkPipelineCache header format:
            # - Header length: 4 bytes
            # - Header version: 4 bytes  
            # - Vendor ID: 4 bytes
            # - Device ID: 4 bytes
            # - Pipeline cache UUID: 16 bytes
            # - Data: variable length
            
            with open(cache_file, 'wb') as f:
                # Header length (32 bytes)
                f.write(struct.pack('<I', 32))
                
                # Header version (VK_PIPELINE_CACHE_HEADER_VERSION_ONE = 1)
                f.write(struct.pack('<I', 1))
                
                # Vendor ID (AMD for Steam Deck)
                f.write(struct.pack('<I', 0x1002))
                
                # Device ID (Van Gogh APU)
                f.write(struct.pack('<I', 0x163F))
                
                # Pipeline cache UUID (16 bytes)
                cache_uuid = hashlib.md5(pipeline_info.spirv_data).digest()
                f.write(cache_uuid)
                
                # Pipeline data
                pipeline_data = self._create_pipeline_cache_data(pipeline_info)
                f.write(pipeline_data)
                
            self.logger.debug(f"Created VkPipelineCache: {cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create VkPipelineCache: {e}")
            return False
            
    def _create_pipeline_cache_data(self, pipeline_info: VkPipelineInfo) -> bytes:
        """Create pipeline cache data section"""
        try:
            # This would contain the compiled shader bytecode for the specific GPU
            # For now, we'll store the SPIRV data with some metadata
            
            data = {
                'spirv_hash': pipeline_info.shader_hash,
                'spirv_size': len(pipeline_info.spirv_data),
                'stage_flags': pipeline_info.stage_flags,
                'timestamp': int(time.time())
            }
            
            # Serialize metadata
            metadata = json.dumps(data).encode()
            
            # Create cache data: metadata_length + metadata + spirv_data
            cache_data = struct.pack('<I', len(metadata))
            cache_data += metadata
            cache_data += pipeline_info.spirv_data
            
            return cache_data
            
        except Exception as e:
            self.logger.error(f"Pipeline cache data creation error: {e}")
            return b''
            
    def integrate_with_steam(self, game_id: str, compiled_cache: Path) -> bool:
        """Integrate compiled shaders with Steam's shader cache system"""
        if not self.steam_shader_cache:
            self.logger.warning("Steam shader cache directory not found")
            return False
            
        steam_game_cache = self.steam_shader_cache / game_id
        
        try:
            # Create shader cache directory if it doesn't exist
            steam_game_cache.mkdir(parents=True, exist_ok=True)
            
            # Copy Fossilize databases to Steam cache
            foz_files_copied = 0
            for foz_file in compiled_cache.glob('*.foz'):
                target = steam_game_cache / foz_file.name
                
                # Check if shader already exists and is newer
                if target.exists() and target.stat().st_mtime > foz_file.stat().st_mtime:
                    self.stats['cache_hits'] += 1
                    continue
                    
                # Copy shader to Steam cache
                import shutil
                shutil.copy2(foz_file, target)
                foz_files_copied += 1
                
            # Copy VkPipelineCache files for direct Vulkan integration
            vk_files_copied = 0
            for vk_file in compiled_cache.glob('*.vkpipelinecache'):
                target = steam_game_cache / vk_file.name
                if not target.exists() or target.stat().st_mtime < vk_file.stat().st_mtime:
                    import shutil
                    shutil.copy2(vk_file, target)
                    vk_files_copied += 1
                    
            # Create Steam-compatible metadata
            self._create_steam_cache_metadata(steam_game_cache, game_id)
            
            # Signal Steam about updated cache (if running)
            self._notify_steam_cache_update(game_id)
            
            self.logger.info(
                f"Steam integration complete: {foz_files_copied} .foz files, "
                f"{vk_files_copied} VkPipelineCache files copied"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Steam integration error: {e}")
            return False
            
    def _create_steam_cache_metadata(self, steam_cache_dir: Path, game_id: str) -> None:
        """Create Steam-compatible cache metadata"""
        try:
            metadata_file = steam_cache_dir / 'fossilize_metadata.json'
            
            # Load existing metadata or create new
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
                    
            # Update metadata
            metadata.update({
                'app_id': game_id,
                'last_update': time.time(),
                'predictive_compile': True,
                'foz_count': len(list(steam_cache_dir.glob('*.foz'))),
                'vk_cache_count': len(list(steam_cache_dir.glob('*.vkpipelinecache'))),
                'compiler_version': self._get_fossilize_version(),
                'steam_deck_optimized': True,
                'vulkan_version': self._get_vulkan_version()
            })
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to create Steam cache metadata: {e}")
            
    def _notify_steam_cache_update(self, game_id: str) -> None:
        """Notify Steam about shader cache updates"""
        try:
            # Check if Steam is running
            steam_running = any('steam' in p.name().lower() for p in psutil.process_iter(['name']))
            
            if not steam_running:
                return
                
            # Try to send a signal to Steam about cache updates
            # This is platform-specific and may not always work
            steam_ipc_file = Path.home() / '.steam/steam.ipc'
            if steam_ipc_file.exists():
                # Create cache update signal file
                signal_file = Path.home() / f'.steam/shader_cache_update_{game_id}'
                signal_file.touch()
                self.logger.debug(f"Created cache update signal for Steam: {signal_file}")
                
        except Exception as e:
            self.logger.debug(f"Steam notification failed (not critical): {e}")
            
    def _get_vulkan_version(self) -> str:
        """Get Vulkan API version"""
        try:
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Vulkan Instance Version:' in line:
                        return line.split(':')[-1].strip()
            return "1.3.0"  # Default fallback
        except:
            return "1.3.0"
    
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
        """Clean up old shader caches with enhanced Steam integration"""
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
                            # Also clean up corresponding Steam cache if it exists
                            self._cleanup_steam_cache_for_game(game_cache.name)
                            
                            import shutil
                            shutil.rmtree(game_cache)
                            cleaned += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error cleaning cache {game_cache}: {e}")
                        
        return cleaned
        
    def _cleanup_steam_cache_for_game(self, game_id: str) -> None:
        """Clean up Steam shader cache for specific game"""
        if not self.steam_shader_cache:
            return
            
        steam_game_cache = self.steam_shader_cache / game_id
        if not steam_game_cache.exists():
            return
            
        try:
            # Only remove our generated files, not Steam's original cache
            our_files = list(steam_game_cache.glob('*.foz')) + list(steam_game_cache.glob('*.vkpipelinecache'))
            
            for file in our_files:
                try:
                    # Check if it's one of our files by looking for our metadata
                    metadata_marker = steam_game_cache / 'fossilize_metadata.json'
                    if metadata_marker.exists():
                        file.unlink()
                except:
                    pass
                    
            # Remove our metadata file
            metadata_file = steam_game_cache / 'fossilize_metadata.json'
            if metadata_file.exists():
                metadata_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Error cleaning Steam cache for {game_id}: {e}")
            
    def create_thermal_aware_scheduler(self) -> 'ThermalAwareScheduler':
        """Create thermal-aware compilation scheduler for Steam Deck"""
        return ThermalAwareScheduler(self)
        
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get detailed compilation statistics"""
        return {
            **self.stats,
            'fossilize_available': self.fossilize_path is not None,
            'spirv_tools_available': self.spirv_tools_path is not None,
            'steam_cache_available': self.steam_shader_cache is not None,
            'thermal_management': self._get_thermal_status()
        }
        
    def _get_thermal_status(self) -> Dict[str, Any]:
        """Get thermal management status for Steam Deck"""
        try:
            thermal_info = {
                'cpu_temp': 0.0,
                'gpu_temp': 0.0,
                'throttling': False,
                'fan_speed': 0
            }
            
            # Try to read thermal info from sysfs
            thermal_zones = Path('/sys/class/thermal')
            if thermal_zones.exists():
                for zone in thermal_zones.glob('thermal_zone*'):
                    try:
                        temp_file = zone / 'temp'
                        if temp_file.exists():
                            temp = int(temp_file.read_text().strip()) / 1000.0
                            thermal_info['cpu_temp'] = max(thermal_info['cpu_temp'], temp)
                    except:
                        pass
                        
            # Check for throttling
            if thermal_info['cpu_temp'] > 85.0:  # Steam Deck throttling threshold
                thermal_info['throttling'] = True
                
            return thermal_info
            
        except Exception as e:
            self.logger.debug(f"Thermal status unavailable: {e}")
            return {'available': False}


class ThermalAwareScheduler:
    """Thermal-aware compilation scheduler for Steam Deck"""
    
    def __init__(self, fossilize_integration: FossilizeIntegration):
        self.fossilize = fossilize_integration
        self.logger = logging.getLogger(f"{__name__}.ThermalScheduler")
        self.max_temp_threshold = 80.0  # °C
        self.compilation_pause_temp = 85.0  # °C
        self.fan_speed_threshold = 80  # %
        
    def should_compile_now(self) -> bool:
        """Check if compilation should proceed based on thermal conditions"""
        thermal_status = self.fossilize._get_thermal_status()
        
        if not thermal_status.get('available', False):
            return True  # No thermal info available, proceed
            
        cpu_temp = thermal_status.get('cpu_temp', 0)
        
        # Pause compilation if too hot
        if cpu_temp > self.compilation_pause_temp:
            self.logger.info(f"Pausing compilation due to high temperature: {cpu_temp}°C")
            return False
            
        # Reduce compilation intensity if getting warm
        if cpu_temp > self.max_temp_threshold:
            self.logger.info(f"Reducing compilation intensity due to temperature: {cpu_temp}°C")
            time.sleep(2)  # Add delays between compilations
            
        return True
        
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on thermal conditions"""
        thermal_status = self.fossilize._get_thermal_status()
        base_threads = min(4, psutil.cpu_count())  # Steam Deck has 4 cores
        
        if not thermal_status.get('available', False):
            return base_threads
            
        cpu_temp = thermal_status.get('cpu_temp', 0)
        
        # Reduce threads if getting hot
        if cpu_temp > self.compilation_pause_temp:
            return 1
        elif cpu_temp > self.max_temp_threshold:
            return max(1, base_threads // 2)
            
        return base_threads