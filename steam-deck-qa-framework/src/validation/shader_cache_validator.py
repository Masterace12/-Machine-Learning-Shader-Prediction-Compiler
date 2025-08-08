#!/usr/bin/env python3
"""
Shader Cache Validation Module
Comprehensive validation of shader cache integrity and performance
"""

import os
import asyncio
import logging
import json
import hashlib
import struct
import zlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import mmap
import subprocess

class ShaderCacheValidator:
    """Comprehensive shader cache validation and integrity checking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}")
        self.cache_directory = config.get("steam_deck", {}).get("cache_directory", "/home/deck/.steam/steam/steamapps/shadercache")
        self.validation_schemas = self._load_validation_schemas()
    
    def _load_validation_schemas(self) -> Dict:
        """Load validation schemas for different shader cache formats"""
        return {
            "dxvk_cache": {
                "magic_bytes": [b"DXVK", b"DX11", b"DX12"],
                "min_size": 64,
                "max_size": 100 * 1024 * 1024,  # 100MB per cache file
                "required_sections": ["header", "shaders", "metadata"]
            },
            "spirv_cache": {
                "magic_bytes": [b"\x03\x02\x23\x07"],  # SPIR-V magic number
                "min_size": 20,
                "max_size": 50 * 1024 * 1024,
                "required_sections": ["header", "instructions"]
            },
            "mesa_cache": {
                "magic_bytes": [b"MESA", b"RADV"],
                "min_size": 32,
                "max_size": 200 * 1024 * 1024,
                "required_sections": ["header", "pipeline_cache"]
            },
            "valve_fossilize": {
                "magic_bytes": [b"FOSSILIZEDB"],
                "min_size": 32,
                "max_size": 500 * 1024 * 1024,
                "required_sections": ["header", "version", "entries"]
            },
            "vk_pipeline_cache": {
                "magic_bytes": [],  # No specific magic, validated by header structure
                "min_size": 32,
                "max_size": 100 * 1024 * 1024,
                "required_sections": ["header", "cache_data"]
            }
        }
    
    async def validate_shader_cache(self, app_id: str, expected_shaders: int = 0) -> Dict[str, Any]:
        """Comprehensive shader cache validation for a specific game"""
        self.logger.info(f"Validating shader cache for app {app_id}")
        
        validation_result = {
            "app_id": app_id,
            "cache_directory": f"{self.cache_directory}/{app_id}",
            "valid": False,
            "cache_files": {},
            "integrity_check": {},
            "performance_metrics": {},
            "compatibility_check": {},
            "issues": []
        }
        
        try:
            cache_dir = f"{self.cache_directory}/{app_id}"
            
            if not os.path.exists(cache_dir):
                validation_result["issues"].append("Shader cache directory not found")
                return validation_result
            
            # 1. File System Validation
            file_validation = await self._validate_cache_files(cache_dir)
            validation_result["cache_files"] = file_validation
            
            # 2. Integrity Checking
            integrity_check = await self._perform_integrity_check(cache_dir)
            validation_result["integrity_check"] = integrity_check
            
            # 3. Performance Metrics
            performance_metrics = await self._analyze_cache_performance(cache_dir)
            validation_result["performance_metrics"] = performance_metrics
            
            # 4. Compatibility Checking
            compatibility_check = await self._check_cache_compatibility(cache_dir)
            validation_result["compatibility_check"] = compatibility_check
            
            # 5. Expected Shader Count Validation
            if expected_shaders > 0:
                shader_count_validation = await self._validate_shader_count(cache_dir, expected_shaders)
                validation_result["shader_count_validation"] = shader_count_validation
            
            # Determine overall validity
            validation_result["valid"] = self._determine_cache_validity(validation_result)
            
        except Exception as e:
            self.logger.error(f"Cache validation failed: {e}")
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _validate_cache_files(self, cache_dir: str) -> Dict[str, Any]:
        """Validate cache file structure and basic properties"""
        file_validation = {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "corrupted_files": [],
            "valid_files": [],
            "issues": []
        }
        
        try:
            cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    cache_files.append(os.path.join(root, file))
            
            file_validation["total_files"] = len(cache_files)
            
            for cache_file in cache_files:
                try:
                    file_stat = os.stat(cache_file)
                    file_size = file_stat.st_size
                    file_validation["total_size"] += file_size
                    
                    # Determine file type
                    file_type = await self._identify_cache_file_type(cache_file)
                    if file_type not in file_validation["file_types"]:
                        file_validation["file_types"][file_type] = {"count": 0, "size": 0}
                    
                    file_validation["file_types"][file_type]["count"] += 1
                    file_validation["file_types"][file_type]["size"] += file_size
                    
                    # Validate file integrity
                    if await self._validate_single_cache_file(cache_file, file_type):
                        file_validation["valid_files"].append(cache_file)
                    else:
                        file_validation["corrupted_files"].append(cache_file)
                
                except Exception as e:
                    file_validation["corrupted_files"].append(cache_file)
                    file_validation["issues"].append(f"Error validating {cache_file}: {str(e)}")
            
            # Calculate corruption rate
            corruption_rate = len(file_validation["corrupted_files"]) / len(cache_files) if cache_files else 0
            file_validation["corruption_rate"] = corruption_rate
            
            if corruption_rate > 0.1:  # More than 10% corruption
                file_validation["issues"].append(f"High corruption rate: {corruption_rate:.2%}")
        
        except Exception as e:
            file_validation["issues"].append(f"File validation error: {str(e)}")
        
        return file_validation
    
    async def _identify_cache_file_type(self, file_path: str) -> str:
        """Identify the type of cache file based on content and name"""
        try:
            # Check file extension first
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.dxvk', '.d3d11', '.d3d12']:
                return "dxvk_cache"
            elif file_ext in ['.spirv', '.spv']:
                return "spirv_cache"
            elif file_ext in ['.mesa', '.radv']:
                return "mesa_cache"
            elif file_ext in ['.foz', '.fossilize']:
                return "valve_fossilize"
            elif file_ext in ['.vkpipelinecache', '.vkcache']:
                return "vk_pipeline_cache"
            
            # Check magic bytes
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
                for cache_type, schema in self.validation_schemas.items():
                    for magic_bytes in schema["magic_bytes"]:
                        if header.startswith(magic_bytes):
                            return cache_type
            
            return "unknown"
        
        except Exception:
            return "unknown"
    
    async def _validate_single_cache_file(self, file_path: str, file_type: str) -> bool:
        """Validate a single cache file"""
        try:
            if file_type not in self.validation_schemas:
                return True  # Unknown types are considered valid
            
            schema = self.validation_schemas[file_type]
            file_stat = os.stat(file_path)
            
            # Check file size
            if file_stat.st_size < schema["min_size"] or file_stat.st_size > schema["max_size"]:
                return False
            
            # Check magic bytes
            with open(file_path, 'rb') as f:
                header = f.read(16)
                magic_valid = any(header.startswith(magic) for magic in schema["magic_bytes"])
                
                if not magic_valid:
                    return False
                
                # Perform format-specific validation
                if file_type == "dxvk_cache":
                    return await self._validate_dxvk_cache(f)
                elif file_type == "spirv_cache":
                    return await self._validate_spirv_cache(f)
                elif file_type == "mesa_cache":
                    return await self._validate_mesa_cache(f)
                elif file_type == "valve_fossilize":
                    return await self._validate_fossilize_cache(f)
                elif file_type == "vk_pipeline_cache":
                    return await self._validate_vk_pipeline_cache(f)
            
            return True
        
        except Exception:
            return False
    
    async def _validate_dxvk_cache(self, file_handle) -> bool:
        """Validate DXVK cache file format"""
        try:
            # Read DXVK cache header
            file_handle.seek(0)
            header = file_handle.read(32)
            
            # Basic DXVK validation
            if len(header) < 32:
                return False
            
            # Check version field (typically at offset 4)
            version = struct.unpack('<I', header[4:8])[0]
            if version > 1000:  # Reasonable version check
                return False
            
            # Check entry count (typically at offset 8)
            entry_count = struct.unpack('<I', header[8:12])[0]
            if entry_count > 100000:  # Reasonable entry count
                return False
            
            return True
        
        except Exception:
            return False
    
    async def _validate_spirv_cache(self, file_handle) -> bool:
        """Validate SPIR-V cache file format"""
        try:
            file_handle.seek(0)
            header = file_handle.read(20)
            
            if len(header) < 20:
                return False
            
            # SPIR-V magic number validation
            magic = struct.unpack('<I', header[0:4])[0]
            if magic != 0x07230203:
                return False
            
            # Version validation
            version = struct.unpack('<I', header[4:8])[0]
            if version > 0x00010600:  # SPIR-V 1.6
                return False
            
            return True
        
        except Exception:
            return False
    
    async def _validate_mesa_cache(self, file_handle) -> bool:
        """Validate Mesa/RADV cache file format"""
        try:
            file_handle.seek(0)
            header = file_handle.read(64)
            
            if len(header) < 32:
                return False
            
            # Basic Mesa cache validation
            # This would include checking cache timestamps, versions, etc.
            return True
        
        except Exception:
            return False
    
    async def _validate_fossilize_cache(self, file_handle) -> bool:
        """Validate Valve Fossilize cache format with proper .foz structure"""
        try:
            file_handle.seek(0)
            header = file_handle.read(32)
            
            if len(header) < 32:
                return False
            
            # Check Fossilize magic header: "FOSSILIZEDB" + padding
            if not header.startswith(b'FOSSILIZEDB'):
                return False
                
            # Read version (4 bytes after magic + padding)
            version_offset = 16  # Magic + padding
            file_handle.seek(version_offset)
            version_bytes = file_handle.read(4)
            if len(version_bytes) < 4:
                return False
                
            version = struct.unpack('<I', version_bytes)[0]
            if version > 10:  # Reasonable version check
                return False
                
            # Read entry count
            entry_count_bytes = file_handle.read(4)
            if len(entry_count_bytes) < 4:
                return False
                
            entry_count = struct.unpack('<I', entry_count_bytes)[0]
            if entry_count > 100000:  # Reasonable entry count limit
                return False
                
            # Validate entries structure
            for i in range(min(entry_count, 10)):  # Validate first 10 entries
                entry_size_bytes = file_handle.read(4)
                if len(entry_size_bytes) < 4:
                    break
                    
                entry_size = struct.unpack('<I', entry_size_bytes)[0]
                if entry_size > 10 * 1024 * 1024:  # Max 10MB per entry
                    return False
                    
                # Skip entry data
                current_pos = file_handle.tell()
                file_handle.seek(current_pos + entry_size)
                
            return True
        
        except Exception as e:
            return False
            
    async def _validate_vk_pipeline_cache(self, file_handle) -> bool:
        """Validate VkPipelineCache format"""
        try:
            file_handle.seek(0)
            header = file_handle.read(32)
            
            if len(header) < 32:
                return False
            
            # Parse VkPipelineCache header
            header_length = struct.unpack('<I', header[0:4])[0]
            header_version = struct.unpack('<I', header[4:8])[0]
            vendor_id = struct.unpack('<I', header[8:12])[0]
            device_id = struct.unpack('<I', header[12:16])[0]
            
            # Validate header
            if header_length != 32:
                return False
                
            if header_version != 1:  # VK_PIPELINE_CACHE_HEADER_VERSION_ONE
                return False
                
            # Vendor ID should be a valid PCI vendor ID
            valid_vendors = [0x1002, 0x10DE, 0x8086]  # AMD, NVIDIA, Intel
            if vendor_id not in valid_vendors:
                return False
                
            # UUID should be 16 bytes
            uuid = header[16:32]
            if len(uuid) != 16:
                return False
                
            return True
        
        except Exception as e:
            return False
    
    async def _perform_integrity_check(self, cache_dir: str) -> Dict[str, Any]:
        """Perform comprehensive integrity checking"""
        integrity_check = {
            "checksum_validation": {},
            "cross_reference_check": {},
            "temporal_consistency": {},
            "dependency_validation": {},
            "issues": []
        }
        
        try:
            # 1. Checksum validation
            checksum_results = await self._validate_checksums(cache_dir)
            integrity_check["checksum_validation"] = checksum_results
            
            # 2. Cross-reference validation
            cross_ref_results = await self._validate_cross_references(cache_dir)
            integrity_check["cross_reference_check"] = cross_ref_results
            
            # 3. Temporal consistency
            temporal_results = await self._check_temporal_consistency(cache_dir)
            integrity_check["temporal_consistency"] = temporal_results
            
            # 4. Dependency validation
            dependency_results = await self._validate_dependencies(cache_dir)
            integrity_check["dependency_validation"] = dependency_results
        
        except Exception as e:
            integrity_check["issues"].append(f"Integrity check error: {str(e)}")
        
        return integrity_check
    
    async def _validate_checksums(self, cache_dir: str) -> Dict[str, Any]:
        """Validate file checksums against stored metadata"""
        checksum_results = {
            "files_checked": 0,
            "checksum_mismatches": [],
            "missing_checksums": [],
            "valid_checksums": 0
        }
        
        try:
            # Look for checksum files
            checksum_files = []
            for root, dirs, files in os.walk(cache_dir):
                checksum_files.extend([
                    os.path.join(root, f) for f in files 
                    if f.endswith(('.md5', '.sha1', '.sha256', '.crc'))
                ])
            
            for checksum_file in checksum_files:
                stored_checksums = await self._read_checksum_file(checksum_file)
                
                for file_path, expected_checksum in stored_checksums.items():
                    if os.path.exists(file_path):
                        actual_checksum = await self._calculate_checksum(file_path)
                        checksum_results["files_checked"] += 1
                        
                        if actual_checksum == expected_checksum:
                            checksum_results["valid_checksums"] += 1
                        else:
                            checksum_results["checksum_mismatches"].append({
                                "file": file_path,
                                "expected": expected_checksum,
                                "actual": actual_checksum
                            })
                    else:
                        checksum_results["missing_checksums"].append(file_path)
        
        except Exception as e:
            self.logger.error(f"Checksum validation error: {e}")
        
        return checksum_results
    
    async def _read_checksum_file(self, checksum_file: str) -> Dict[str, str]:
        """Read checksums from checksum file"""
        checksums = {}
        try:
            with open(checksum_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        checksum = parts[0]
                        filename = ' '.join(parts[1:])
                        checksums[filename] = checksum
        except Exception as e:
            self.logger.error(f"Error reading checksum file {checksum_file}: {e}")
        
        return checksums
    
    async def _calculate_checksum(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate file checksum"""
        try:
            hasher = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    async def _validate_cross_references(self, cache_dir: str) -> Dict[str, Any]:
        """Validate cross-references between cache files"""
        cross_ref_results = {
            "reference_files": 0,
            "broken_references": [],
            "circular_references": [],
            "orphaned_files": []
        }
        
        try:
            # Build reference map
            reference_map = {}
            cache_files = []
            
            for root, dirs, files in os.walk(cache_dir):
                cache_files.extend([os.path.join(root, f) for f in files])
            
            # Analyze references
            for cache_file in cache_files:
                references = await self._extract_file_references(cache_file)
                if references:
                    reference_map[cache_file] = references
                    cross_ref_results["reference_files"] += 1
            
            # Check for broken references
            for file_path, references in reference_map.items():
                for ref in references:
                    ref_path = os.path.join(cache_dir, ref)
                    if not os.path.exists(ref_path):
                        cross_ref_results["broken_references"].append({
                            "source": file_path,
                            "missing_reference": ref
                        })
            
            # Check for circular references
            circular_refs = await self._detect_circular_references(reference_map)
            cross_ref_results["circular_references"] = circular_refs
            
            # Check for orphaned files
            referenced_files = set()
            for references in reference_map.values():
                referenced_files.update(references)
            
            for cache_file in cache_files:
                if (cache_file not in reference_map and 
                    cache_file not in referenced_files):
                    cross_ref_results["orphaned_files"].append(cache_file)
        
        except Exception as e:
            self.logger.error(f"Cross-reference validation error: {e}")
        
        return cross_ref_results
    
    async def _extract_file_references(self, file_path: str) -> List[str]:
        """Extract file references from cache file"""
        references = []
        try:
            # This would be format-specific
            # For now, return empty list
            pass
        except Exception:
            pass
        
        return references
    
    async def _detect_circular_references(self, reference_map: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular references in cache files"""
        circular_refs = []
        
        try:
            visited = set()
            rec_stack = set()
            
            def dfs(file_path: str, path: List[str]) -> bool:
                if file_path in rec_stack:
                    # Found circular reference
                    cycle_start = path.index(file_path)
                    circular_refs.append(path[cycle_start:] + [file_path])
                    return True
                
                if file_path in visited:
                    return False
                
                visited.add(file_path)
                rec_stack.add(file_path)
                
                for ref in reference_map.get(file_path, []):
                    if dfs(ref, path + [ref]):
                        return True
                
                rec_stack.remove(file_path)
                return False
            
            for file_path in reference_map:
                if file_path not in visited:
                    dfs(file_path, [file_path])
        
        except Exception as e:
            self.logger.error(f"Circular reference detection error: {e}")
        
        return circular_refs
    
    async def _check_temporal_consistency(self, cache_dir: str) -> Dict[str, Any]:
        """Check temporal consistency of cache files"""
        temporal_results = {
            "timestamp_anomalies": [],
            "modification_conflicts": [],
            "creation_order_issues": []
        }
        
        try:
            cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    stat = os.stat(file_path)
                    cache_files.append({
                        "path": file_path,
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                        "accessed": stat.st_atime
                    })
            
            # Sort by creation time
            cache_files.sort(key=lambda x: x["created"])
            
            # Check for timestamp anomalies
            current_time = time.time()
            for file_info in cache_files:
                # Check for future timestamps
                if (file_info["created"] > current_time or 
                    file_info["modified"] > current_time):
                    temporal_results["timestamp_anomalies"].append({
                        "file": file_info["path"],
                        "issue": "future_timestamp"
                    })
                
                # Check for modification before creation
                if file_info["modified"] < file_info["created"]:
                    temporal_results["modification_conflicts"].append({
                        "file": file_info["path"],
                        "created": file_info["created"],
                        "modified": file_info["modified"]
                    })
        
        except Exception as e:
            self.logger.error(f"Temporal consistency check error: {e}")
        
        return temporal_results
    
    async def _validate_dependencies(self, cache_dir: str) -> Dict[str, Any]:
        """Validate cache file dependencies"""
        dependency_results = {
            "missing_dependencies": [],
            "version_mismatches": [],
            "dependency_tree_valid": True
        }
        
        try:
            # This would check for required driver versions, libraries, etc.
            # Implementation would be system-specific
            pass
        
        except Exception as e:
            dependency_results["issues"] = [f"Dependency validation error: {str(e)}"]
        
        return dependency_results
    
    async def _analyze_cache_performance(self, cache_dir: str) -> Dict[str, Any]:
        """Analyze cache performance characteristics with Fossilize-specific metrics"""
        performance_metrics = {
            "access_patterns": {},
            "hit_ratio_estimate": 0.0,
            "fragmentation_analysis": {},
            "size_efficiency": {},
            "fossilize_metrics": {},
            "steam_integration_status": {},
            "issues": []
        }
        
        try:
            # 1. Analyze access patterns
            access_analysis = await self._analyze_access_patterns(cache_dir)
            performance_metrics["access_patterns"] = access_analysis
            
            # 2. Estimate hit ratio
            hit_ratio = await self._estimate_cache_hit_ratio(cache_dir)
            performance_metrics["hit_ratio_estimate"] = hit_ratio
            
            # 3. Fragmentation analysis
            fragmentation = await self._analyze_fragmentation(cache_dir)
            performance_metrics["fragmentation_analysis"] = fragmentation
            
            # 4. Size efficiency
            size_efficiency = await self._analyze_size_efficiency(cache_dir)
            performance_metrics["size_efficiency"] = size_efficiency
            
            # 5. Fossilize-specific metrics
            fossilize_metrics = await self._analyze_fossilize_metrics(cache_dir)
            performance_metrics["fossilize_metrics"] = fossilize_metrics
            
            # 6. Steam integration status
            steam_status = await self._analyze_steam_integration(cache_dir)
            performance_metrics["steam_integration_status"] = steam_status
        
        except Exception as e:
            performance_metrics["issues"].append(f"Performance analysis error: {str(e)}")
        
        return performance_metrics
        
    async def _analyze_fossilize_metrics(self, cache_dir: str) -> Dict[str, Any]:
        """Analyze Fossilize-specific cache metrics"""
        fossilize_metrics = {
            "foz_file_count": 0,
            "total_pipeline_entries": 0,
            "average_entries_per_file": 0,
            "spirv_shader_count": 0,
            "compression_efficiency": 0.0,
            "version_compatibility": {},
            "issues": []
        }
        
        try:
            foz_files = []
            for root, dirs, files in os.walk(cache_dir):
                foz_files.extend([
                    os.path.join(root, f) for f in files 
                    if f.endswith(('.foz', '.fossilize'))
                ])
            
            fossilize_metrics["foz_file_count"] = len(foz_files)
            
            total_entries = 0
            total_spirv_shaders = 0
            version_counts = {}
            
            for foz_file in foz_files[:20]:  # Analyze up to 20 files to avoid performance issues
                try:
                    with open(foz_file, 'rb') as f:
                        # Validate header
                        header = f.read(32)
                        if len(header) < 32 or not header.startswith(b'FOSSILIZEDB'):
                            continue
                            
                        # Read version and entry count
                        f.seek(16)  # Skip to version
                        version = struct.unpack('<I', f.read(4))[0]
                        entry_count = struct.unpack('<I', f.read(4))[0]
                        
                        total_entries += entry_count
                        version_counts[version] = version_counts.get(version, 0) + 1
                        
                        # Analyze entries for SPIRV content
                        spirv_count = await self._count_spirv_in_fossilize(f, entry_count)
                        total_spirv_shaders += spirv_count
                        
                except Exception as e:
                    fossilize_metrics["issues"].append(f"Error analyzing {foz_file}: {str(e)}")
            
            fossilize_metrics["total_pipeline_entries"] = total_entries
            fossilize_metrics["spirv_shader_count"] = total_spirv_shaders
            fossilize_metrics["version_compatibility"] = version_counts
            
            if len(foz_files) > 0:
                fossilize_metrics["average_entries_per_file"] = total_entries / len(foz_files)
            
        except Exception as e:
            fossilize_metrics["issues"].append(f"Fossilize metrics analysis error: {str(e)}")
        
        return fossilize_metrics
        
    async def _count_spirv_in_fossilize(self, file_handle, entry_count: int) -> int:
        """Count SPIRV shaders in Fossilize database entries"""
        spirv_count = 0
        
        try:
            for i in range(min(entry_count, 100)):  # Limit to avoid performance issues
                # Read entry size
                entry_size_bytes = file_handle.read(4)
                if len(entry_size_bytes) < 4:
                    break
                    
                entry_size = struct.unpack('<I', entry_size_bytes)[0]
                
                # Read entry data
                entry_data = file_handle.read(entry_size)
                if len(entry_data) < entry_size:
                    break
                    
                # Try to decompress and parse JSON
                try:
                    decompressed = zlib.decompress(entry_data)
                    pipeline_data = json.loads(decompressed.decode())
                    
                    # Count SPIRV shaders in stages
                    stages = pipeline_data.get('stages', [])
                    for stage in stages:
                        if 'spirv' in stage and stage['spirv']:
                            spirv_count += 1
                            
                except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
                    # Entry might not be compressed JSON, skip
                    continue
                    
        except Exception:
            pass
            
        return spirv_count
        
    async def _analyze_steam_integration(self, cache_dir: str) -> Dict[str, Any]:
        """Analyze Steam integration status"""
        steam_status = {
            "metadata_present": False,
            "steam_cache_linked": False,
            "fossilize_metadata_valid": False,
            "vk_cache_present": False,
            "last_steam_update": 0,
            "issues": []
        }
        
        try:
            # Check for Steam metadata files
            metadata_files = [
                'fossilize_metadata.json',
                'shader_cache_metadata.json',
                'steam_metadata.json'
            ]
            
            for metadata_file in metadata_files:
                metadata_path = os.path.join(cache_dir, metadata_file)
                if os.path.exists(metadata_path):
                    steam_status["metadata_present"] = True
                    
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            
                        if metadata_file == 'fossilize_metadata.json':
                            steam_status["fossilize_metadata_valid"] = True
                            steam_status["last_steam_update"] = metadata.get('last_update', 0)
                            
                    except (json.JSONDecodeError, IOError):
                        steam_status["issues"].append(f"Invalid metadata file: {metadata_file}")
            
            # Check for VkPipelineCache files
            vk_cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                vk_cache_files.extend([
                    f for f in files if f.endswith(('.vkpipelinecache', '.vkcache'))
                ])
                
            steam_status["vk_cache_present"] = len(vk_cache_files) > 0
            
            # Check Steam cache directory linkage
            steam_cache_base = os.path.expanduser("~/.steam/steam/steamapps/shadercache")
            if os.path.exists(steam_cache_base):
                # Extract app ID from cache directory path
                cache_parts = cache_dir.split(os.sep)
                if 'shadercache' in cache_parts:
                    idx = cache_parts.index('shadercache')
                    if idx + 1 < len(cache_parts):
                        app_id = cache_parts[idx + 1]
                        steam_app_cache = os.path.join(steam_cache_base, app_id)
                        steam_status["steam_cache_linked"] = os.path.exists(steam_app_cache)
                        
        except Exception as e:
            steam_status["issues"].append(f"Steam integration analysis error: {str(e)}")
        
        return steam_status
    
    async def _analyze_access_patterns(self, cache_dir: str) -> Dict[str, Any]:
        """Analyze file access patterns"""
        access_patterns = {
            "frequently_accessed": [],
            "rarely_accessed": [],
            "access_distribution": {}
        }
        
        try:
            cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    stat = os.stat(file_path)
                    cache_files.append({
                        "path": file_path,
                        "size": stat.st_size,
                        "accessed": stat.st_atime
                    })
            
            # Sort by access time
            cache_files.sort(key=lambda x: x["accessed"], reverse=True)
            
            # Identify frequently and rarely accessed files
            total_files = len(cache_files)
            if total_files > 0:
                frequently_accessed = cache_files[:int(total_files * 0.2)]  # Top 20%
                rarely_accessed = cache_files[-int(total_files * 0.2):]     # Bottom 20%
                
                access_patterns["frequently_accessed"] = [f["path"] for f in frequently_accessed]
                access_patterns["rarely_accessed"] = [f["path"] for f in rarely_accessed]
        
        except Exception as e:
            self.logger.error(f"Access pattern analysis error: {e}")
        
        return access_patterns
    
    async def _estimate_cache_hit_ratio(self, cache_dir: str) -> float:
        """Estimate cache hit ratio based on file patterns"""
        try:
            # This is a simplified estimation
            # Real implementation would track actual hit/miss statistics
            
            cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                cache_files.extend(files)
            
            # Estimate based on file count and recency
            if len(cache_files) > 100:  # Good cache population
                return 0.85  # 85% hit ratio estimate
            elif len(cache_files) > 50:
                return 0.70  # 70% hit ratio estimate
            else:
                return 0.50  # 50% hit ratio estimate
        
        except Exception:
            return 0.0
    
    async def _analyze_fragmentation(self, cache_dir: str) -> Dict[str, Any]:
        """Analyze cache fragmentation"""
        fragmentation_analysis = {
            "fragmentation_ratio": 0.0,
            "large_files": [],
            "small_files": [],
            "recommendations": []
        }
        
        try:
            file_sizes = []
            large_file_threshold = 10 * 1024 * 1024  # 10MB
            small_file_threshold = 1024  # 1KB
            
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    file_sizes.append(size)
                    
                    if size > large_file_threshold:
                        fragmentation_analysis["large_files"].append(file_path)
                    elif size < small_file_threshold:
                        fragmentation_analysis["small_files"].append(file_path)
            
            if file_sizes:
                # Calculate fragmentation ratio
                total_size = sum(file_sizes)
                avg_size = total_size / len(file_sizes)
                size_variance = sum((size - avg_size) ** 2 for size in file_sizes) / len(file_sizes)
                fragmentation_ratio = size_variance / (avg_size ** 2) if avg_size > 0 else 0
                
                fragmentation_analysis["fragmentation_ratio"] = fragmentation_ratio
                
                # Generate recommendations
                if fragmentation_ratio > 0.5:
                    fragmentation_analysis["recommendations"].append("Consider cache defragmentation")
                
                if len(fragmentation_analysis["small_files"]) > len(file_sizes) * 0.5:
                    fragmentation_analysis["recommendations"].append("Many small files detected - consider consolidation")
        
        except Exception as e:
            self.logger.error(f"Fragmentation analysis error: {e}")
        
        return fragmentation_analysis
    
    async def _analyze_size_efficiency(self, cache_dir: str) -> Dict[str, Any]:
        """Analyze cache size efficiency"""
        size_efficiency = {
            "total_size": 0,
            "compression_ratio": 0.0,
            "space_utilization": 0.0,
            "duplicate_detection": {}
        }
        
        try:
            total_size = 0
            file_hashes = {}
            
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    total_size += size
                    
                    # Calculate hash for duplicate detection
                    file_hash = await self._calculate_checksum(file_path)
                    if file_hash in file_hashes:
                        file_hashes[file_hash].append(file_path)
                    else:
                        file_hashes[file_hash] = [file_path]
            
            size_efficiency["total_size"] = total_size
            
            # Detect duplicates
            duplicates = {hash_val: paths for hash_val, paths in file_hashes.items() if len(paths) > 1}
            size_efficiency["duplicate_detection"] = {
                "duplicate_groups": len(duplicates),
                "duplicate_files": sum(len(paths) - 1 for paths in duplicates.values()),
                "wasted_space": sum(
                    os.path.getsize(paths[0]) * (len(paths) - 1) 
                    for paths in duplicates.values()
                )
            }
        
        except Exception as e:
            self.logger.error(f"Size efficiency analysis error: {e}")
        
        return size_efficiency
    
    async def _check_cache_compatibility(self, cache_dir: str) -> Dict[str, Any]:
        """Check cache compatibility with current system"""
        compatibility_check = {
            "driver_compatibility": {},
            "version_compatibility": {},
            "architecture_compatibility": {},
            "issues": []
        }
        
        try:
            # 1. Driver compatibility
            driver_compat = await self._check_driver_compatibility(cache_dir)
            compatibility_check["driver_compatibility"] = driver_compat
            
            # 2. Version compatibility
            version_compat = await self._check_version_compatibility(cache_dir)
            compatibility_check["version_compatibility"] = version_compat
            
            # 3. Architecture compatibility
            arch_compat = await self._check_architecture_compatibility(cache_dir)
            compatibility_check["architecture_compatibility"] = arch_compat
        
        except Exception as e:
            compatibility_check["issues"].append(f"Compatibility check error: {str(e)}")
        
        return compatibility_check
    
    async def _check_driver_compatibility(self, cache_dir: str) -> Dict[str, Any]:
        """Check driver compatibility"""
        driver_compat = {
            "current_driver": "unknown",
            "cache_driver": "unknown",
            "compatible": True,
            "issues": []
        }
        
        try:
            # Get current driver version
            current_driver = await self._get_current_driver_version()
            driver_compat["current_driver"] = current_driver
            
            # Check cache metadata for driver info
            metadata_file = os.path.join(cache_dir, "driver_info.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    cache_driver = metadata.get("driver_version", "unknown")
                    driver_compat["cache_driver"] = cache_driver
                    
                    # Simple version comparison
                    if cache_driver != current_driver and cache_driver != "unknown":
                        driver_compat["compatible"] = False
                        driver_compat["issues"].append(
                            f"Driver version mismatch: cache={cache_driver}, current={current_driver}"
                        )
        
        except Exception as e:
            driver_compat["issues"].append(f"Driver compatibility check error: {str(e)}")
        
        return driver_compat
    
    async def _get_current_driver_version(self) -> str:
        """Get current graphics driver version"""
        try:
            # For AMD RADV on Steam Deck
            result = await self._run_command("glxinfo | grep 'OpenGL version'")
            if result.returncode == 0:
                return result.stdout.decode().strip()
            
            return "unknown"
        except Exception:
            return "unknown"
    
    async def _check_version_compatibility(self, cache_dir: str) -> Dict[str, Any]:
        """Check version compatibility"""
        version_compat = {
            "cache_version": "unknown",
            "system_version": "unknown",
            "compatible": True,
            "issues": []
        }
        
        # Implementation would check DXVK, Mesa, Proton versions
        return version_compat
    
    async def _check_architecture_compatibility(self, cache_dir: str) -> Dict[str, Any]:
        """Check architecture compatibility"""
        arch_compat = {
            "cache_architecture": "unknown",
            "system_architecture": "unknown",
            "compatible": True,
            "issues": []
        }
        
        # Implementation would check x86_64, ARM compatibility
        return arch_compat
    
    async def _validate_shader_count(self, cache_dir: str, expected_count: int) -> Dict[str, Any]:
        """Validate expected shader count"""
        count_validation = {
            "expected_shaders": expected_count,
            "actual_shaders": 0,
            "count_difference": 0,
            "within_tolerance": False,
            "tolerance_percentage": 0.1  # 10% tolerance
        }
        
        try:
            # Count shader entries across all cache files
            shader_count = await self._count_shaders_in_cache(cache_dir)
            count_validation["actual_shaders"] = shader_count
            count_validation["count_difference"] = abs(shader_count - expected_count)
            
            # Check if within tolerance
            tolerance = expected_count * count_validation["tolerance_percentage"]
            count_validation["within_tolerance"] = count_validation["count_difference"] <= tolerance
        
        except Exception as e:
            self.logger.error(f"Shader count validation error: {e}")
        
        return count_validation
    
    async def _count_shaders_in_cache(self, cache_dir: str) -> int:
        """Count total shaders in cache directory"""
        total_shaders = 0
        
        try:
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_type = await self._identify_cache_file_type(file_path)
                    
                    # Count shaders based on file type
                    if file_type in ["dxvk_cache", "spirv_cache", "mesa_cache"]:
                        shader_count = await self._count_shaders_in_file(file_path, file_type)
                        total_shaders += shader_count
        
        except Exception as e:
            self.logger.error(f"Error counting shaders: {e}")
        
        return total_shaders
    
    async def _count_shaders_in_file(self, file_path: str, file_type: str) -> int:
        """Count shaders in a specific cache file"""
        try:
            if file_type == "dxvk_cache":
                with open(file_path, 'rb') as f:
                    header = f.read(32)
                    if len(header) >= 12:
                        return struct.unpack('<I', header[8:12])[0]
            
            # For other types, estimate based on file size
            file_size = os.path.getsize(file_path)
            avg_shader_size = 2048  # Estimated average shader size
            return max(1, file_size // avg_shader_size)
        
        except Exception:
            return 0
    
    def _determine_cache_validity(self, validation_result: Dict[str, Any]) -> bool:
        """Determine overall cache validity with Fossilize-specific checks"""
        try:
            # Check critical failure conditions
            cache_files = validation_result.get("cache_files", {})
            if cache_files.get("corruption_rate", 1.0) > 0.2:  # >20% corruption
                return False
            
            integrity_check = validation_result.get("integrity_check", {})
            checksum_validation = integrity_check.get("checksum_validation", {})
            if (checksum_validation.get("files_checked", 0) > 0 and
                len(checksum_validation.get("checksum_mismatches", [])) > 0):
                return False
            
            # Check performance metrics
            performance = validation_result.get("performance_metrics", {})
            hit_ratio = performance.get("hit_ratio_estimate", 0.0)
            if hit_ratio < 0.3:  # Less than 30% hit ratio
                return False
            
            # Fossilize-specific validation
            fossilize_metrics = performance.get("fossilize_metrics", {})
            if fossilize_metrics:
                # Check if Fossilize files have reasonable content
                foz_files = fossilize_metrics.get("foz_file_count", 0)
                total_entries = fossilize_metrics.get("total_pipeline_entries", 0)
                
                if foz_files > 0 and total_entries == 0:
                    return False  # Empty Fossilize files
                    
                if len(fossilize_metrics.get("issues", [])) > foz_files / 2:
                    return False  # More than half the files have issues
            
            # Steam integration validation
            steam_status = performance.get("steam_integration_status", {})
            if steam_status and len(steam_status.get("issues", [])) > 5:
                return False  # Too many Steam integration issues
            
            return True
        
        except Exception:
            return False
    
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Run shell command and return result"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise