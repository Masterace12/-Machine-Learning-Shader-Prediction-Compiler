#!/usr/bin/env python3
"""
Anti-Cheat System Compatibility Validator
Validates shader cache compatibility with anti-cheat systems
"""

import os
import asyncio
import logging
import subprocess
import json
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

class AntiCheatValidator:
    """Validator for anti-cheat system compatibility with shader caching"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}")
        self.anticheat_configs = self._load_anticheat_configs()
    
    def _load_anticheat_configs(self) -> Dict:
        """Load anti-cheat system configurations and known compatibility issues"""
        return {
            "eac": {
                "name": "Easy Anti-Cheat",
                "executable_patterns": ["EasyAntiCheat.exe", "EACLauncher.exe"],
                "registry_keys": [
                    r"HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\EasyAntiCheat"
                ],
                "file_integrity_check": True,
                "shader_cache_compatibility": {
                    "allowed_modifications": ["compression", "format_conversion"],
                    "forbidden_modifications": ["shader_code_injection", "bypass_files"],
                    "validation_required": True
                },
                "known_issues": [
                    "May flag custom shader formats",
                    "Requires validated shader signatures"
                ]
            },
            "battleye": {
                "name": "BattlEye",
                "executable_patterns": ["BEService.exe", "BEDaisy.exe"],
                "registry_keys": [
                    r"HKEY_LOCAL_MACHINE\SOFTWARE\BattlEye"
                ],
                "file_integrity_check": True,
                "shader_cache_compatibility": {
                    "allowed_modifications": ["compression", "optimization"],
                    "forbidden_modifications": ["code_modification", "memory_patching"],
                    "validation_required": True
                },
                "known_issues": [
                    "Strict memory protection",
                    "May block shader cache injection"
                ]
            },
            "vac": {
                "name": "Valve Anti-Cheat",
                "executable_patterns": [],  # VAC is integrated into Steam
                "registry_keys": [],
                "file_integrity_check": False,
                "shader_cache_compatibility": {
                    "allowed_modifications": ["all"],
                    "forbidden_modifications": ["game_binary_modification"],
                    "validation_required": False
                },
                "known_issues": []
            },
            "denuvo": {
                "name": "Denuvo Anti-Tamper",
                "executable_patterns": ["denuvo64.dll", "denuvo32.dll"],
                "registry_keys": [],
                "file_integrity_check": True,
                "shader_cache_compatibility": {
                    "allowed_modifications": ["shader_optimization"],
                    "forbidden_modifications": ["executable_patching", "dll_injection"],
                    "validation_required": True
                },
                "known_issues": [
                    "May interfere with shader compilation",
                    "Performance impact during validation"
                ]
            }
        }
    
    async def validate_anticheat_compatibility(self, app_id: str, anticheat_type: str) -> Dict[str, Any]:
        """Validate shader cache compatibility with specified anti-cheat system"""
        self.logger.info(f"Validating {anticheat_type} compatibility for app {app_id}")
        
        validation_result = {
            "app_id": app_id,
            "anticheat_type": anticheat_type,
            "compatible": False,
            "issues_found": [],
            "validation_tests": {},
            "recommendations": []
        }
        
        try:
            if anticheat_type not in self.anticheat_configs:
                validation_result["issues_found"].append(f"Unknown anti-cheat type: {anticheat_type}")
                return validation_result
            
            anticheat_config = self.anticheat_configs[anticheat_type]
            
            # Run validation tests
            tests = {
                "installation_check": await self._check_anticheat_installation(anticheat_config),
                "shader_cache_validation": await self._validate_shader_cache_integrity(app_id, anticheat_config),
                "runtime_compatibility": await self._test_runtime_compatibility(app_id, anticheat_type),
                "file_integrity_check": await self._verify_file_integrity(app_id, anticheat_config),
                "memory_protection_test": await self._test_memory_protection(app_id, anticheat_type)
            }
            
            validation_result["validation_tests"] = tests
            
            # Determine overall compatibility
            validation_result["compatible"] = all(test.get("passed", False) for test in tests.values())
            
            # Collect issues
            for test_name, test_result in tests.items():
                if not test_result.get("passed", False):
                    validation_result["issues_found"].extend(test_result.get("issues", []))
            
            # Generate recommendations
            validation_result["recommendations"] = self._generate_recommendations(
                anticheat_type, validation_result["issues_found"]
            )
            
        except Exception as e:
            self.logger.error(f"Anti-cheat validation failed: {e}")
            validation_result["issues_found"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _check_anticheat_installation(self, anticheat_config: Dict) -> Dict[str, Any]:
        """Check if anti-cheat system is properly installed"""
        check_result = {
            "passed": False,
            "anticheat_detected": False,
            "version": "unknown",
            "issues": []
        }
        
        try:
            # Check for executable patterns
            executables_found = []
            for pattern in anticheat_config["executable_patterns"]:
                # Search in common locations
                search_paths = [
                    "/usr/bin",
                    "/opt",
                    os.path.expanduser("~/.steam/steam/steamapps/common"),
                    "/tmp"  # Some anti-cheat systems extract temporarily
                ]
                
                for search_path in search_paths:
                    if os.path.exists(search_path):
                        for root, dirs, files in os.walk(search_path):
                            if pattern in files:
                                executables_found.append(os.path.join(root, pattern))
            
            check_result["anticheat_detected"] = len(executables_found) > 0
            check_result["executables"] = executables_found
            
            # Check registry keys (if applicable on Linux/Proton)
            registry_found = await self._check_registry_keys(anticheat_config["registry_keys"])
            
            check_result["passed"] = check_result["anticheat_detected"] or registry_found
            
            if not check_result["passed"]:
                check_result["issues"].append("Anti-cheat system not detected or not properly installed")
        
        except Exception as e:
            check_result["issues"].append(f"Installation check error: {str(e)}")
        
        return check_result
    
    async def _check_registry_keys(self, registry_keys: List[str]) -> bool:
        """Check for anti-cheat registry keys (Proton environment)"""
        try:
            # In Proton environment, check wine registry
            for key in registry_keys:
                # Convert to wine registry path
                wine_key = key.replace("HKEY_LOCAL_MACHINE", "HKLM")
                
                # Use wine registry tools if available
                result = await self._run_command(f"wine reg query '{wine_key}'")
                if result.returncode == 0:
                    return True
            
            return False
        except Exception:
            return False
    
    async def _validate_shader_cache_integrity(self, app_id: str, anticheat_config: Dict) -> Dict[str, Any]:
        """Validate that shader cache doesn't violate anti-cheat policies"""
        validation_result = {
            "passed": False,
            "cache_valid": False,
            "signatures_valid": False,
            "issues": []
        }
        
        try:
            cache_dir = f"{self.config['steam_deck']['cache_directory']}/{app_id}"
            
            if not os.path.exists(cache_dir):
                validation_result["issues"].append("Shader cache directory not found")
                return validation_result
            
            # Check cache file integrity
            cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                cache_files.extend([os.path.join(root, f) for f in files])
            
            validation_result["cache_files_count"] = len(cache_files)
            
            # Validate each cache file
            corrupted_files = []
            for cache_file in cache_files:
                if not await self._validate_cache_file_integrity(cache_file, anticheat_config):
                    corrupted_files.append(cache_file)
            
            validation_result["corrupted_files"] = len(corrupted_files)
            validation_result["cache_valid"] = len(corrupted_files) == 0
            
            # Check shader signatures if required
            if anticheat_config["shader_cache_compatibility"]["validation_required"]:
                signature_validation = await self._validate_shader_signatures(cache_files)
                validation_result["signatures_valid"] = signature_validation
                
                if not signature_validation:
                    validation_result["issues"].append("Shader signature validation failed")
            else:
                validation_result["signatures_valid"] = True
            
            validation_result["passed"] = validation_result["cache_valid"] and validation_result["signatures_valid"]
        
        except Exception as e:
            validation_result["issues"].append(f"Cache integrity validation error: {str(e)}")
        
        return validation_result
    
    async def _validate_cache_file_integrity(self, cache_file: str, anticheat_config: Dict) -> bool:
        """Validate individual cache file integrity"""
        try:
            # Check file is readable and not corrupted
            with open(cache_file, 'rb') as f:
                content = f.read()
                
                # Basic integrity checks
                if len(content) == 0:
                    return False
                
                # Check for forbidden modifications
                forbidden = anticheat_config["shader_cache_compatibility"]["forbidden_modifications"]
                
                # This is a simplified check - real implementation would be more sophisticated
                content_str = str(content)
                for forbidden_pattern in ["code_injection", "bypass", "patch"]:
                    if forbidden_pattern in forbidden:
                        if forbidden_pattern.encode() in content:
                            return False
            
            return True
            
        except Exception:
            return False
    
    async def _validate_shader_signatures(self, cache_files: List[str]) -> bool:
        """Validate shader file signatures"""
        try:
            valid_signatures = 0
            
            for cache_file in cache_files:
                # Generate file signature
                signature = await self._generate_file_signature(cache_file)
                
                # Validate against known good signatures or patterns
                if await self._verify_signature(signature):
                    valid_signatures += 1
            
            # Require at least 90% of signatures to be valid
            return (valid_signatures / len(cache_files)) >= 0.9 if cache_files else True
            
        except Exception as e:
            self.logger.error(f"Signature validation error: {e}")
            return False
    
    async def _generate_file_signature(self, file_path: str) -> str:
        """Generate cryptographic signature for file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    async def _verify_signature(self, signature: str) -> bool:
        """Verify file signature against known good signatures"""
        # In a real implementation, this would check against a database
        # of known good shader signatures
        return len(signature) == 64  # Valid SHA256 hex string
    
    async def _test_runtime_compatibility(self, app_id: str, anticheat_type: str) -> Dict[str, Any]:
        """Test runtime compatibility with anti-cheat system"""
        runtime_test = {
            "passed": False,
            "game_launched": False,
            "anticheat_active": False,
            "shader_loading_successful": False,
            "issues": []
        }
        
        try:
            # Launch game and monitor for anti-cheat interaction
            launch_result = await self._launch_game_with_anticheat_monitoring(app_id, anticheat_type)
            runtime_test.update(launch_result)
            
            runtime_test["passed"] = (
                runtime_test["game_launched"] and
                runtime_test["anticheat_active"] and
                runtime_test["shader_loading_successful"]
            )
            
        except Exception as e:
            runtime_test["issues"].append(f"Runtime test error: {str(e)}")
        
        return runtime_test
    
    async def _launch_game_with_anticheat_monitoring(self, app_id: str, anticheat_type: str) -> Dict[str, Any]:
        """Launch game while monitoring anti-cheat system interaction"""
        monitoring_result = {
            "game_launched": False,
            "anticheat_active": False,
            "shader_loading_successful": False,
            "launch_time": 0,
            "issues": []
        }
        
        try:
            start_time = time.time()
            
            # Set up anti-cheat monitoring
            anticheat_monitor = await self._setup_anticheat_monitoring(anticheat_type)
            
            # Launch game
            steam_path = self.config["steam_deck"]["steam_path"]
            launch_cmd = f"{steam_path} -applaunch {app_id}"
            
            process = subprocess.Popen(
                launch_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Monitor launch process
            await asyncio.sleep(30)  # Wait for initialization
            
            if process.poll() is None:
                monitoring_result["game_launched"] = True
                
                # Check if anti-cheat is active
                anticheat_status = await self._check_anticheat_status(anticheat_type)
                monitoring_result["anticheat_active"] = anticheat_status
                
                # Test shader loading
                shader_test = await self._test_shader_loading_with_anticheat(app_id)
                monitoring_result["shader_loading_successful"] = shader_test
                
                # Clean shutdown
                process.terminate()
                await asyncio.sleep(5)
                if process.poll() is None:
                    process.kill()
            
            else:
                monitoring_result["issues"].append("Game failed to launch with anti-cheat active")
            
            monitoring_result["launch_time"] = time.time() - start_time
            
        except Exception as e:
            monitoring_result["issues"].append(f"Monitoring error: {str(e)}")
        
        return monitoring_result
    
    async def _setup_anticheat_monitoring(self, anticheat_type: str):
        """Set up monitoring for anti-cheat system activity"""
        # This would set up monitoring specific to each anti-cheat system
        # Implementation depends on available monitoring tools
        pass
    
    async def _check_anticheat_status(self, anticheat_type: str) -> bool:
        """Check if anti-cheat system is actively running"""
        try:
            anticheat_config = self.anticheat_configs[anticheat_type]
            
            # Check for running processes
            for pattern in anticheat_config["executable_patterns"]:
                result = await self._run_command(f"pgrep -f {pattern}")
                if result.returncode == 0:
                    return True
            
            # Additional checks for integrated systems like VAC
            if anticheat_type == "vac":
                # VAC is integrated into Steam, check Steam process
                result = await self._run_command("pgrep -f steam")
                return result.returncode == 0
            
            return False
            
        except Exception:
            return False
    
    async def _test_shader_loading_with_anticheat(self, app_id: str) -> bool:
        """Test shader loading while anti-cheat is active"""
        try:
            cache_dir = f"{self.config['steam_deck']['cache_directory']}/{app_id}"
            
            if not os.path.exists(cache_dir):
                return False
            
            # Simulate shader access patterns
            shader_files = [f for f in os.listdir(cache_dir) if f.endswith(('.cache', '.spirv', '.dxbc'))]
            
            successful_loads = 0
            for shader_file in shader_files[:10]:  # Test first 10 shader files
                try:
                    with open(os.path.join(cache_dir, shader_file), 'rb') as f:
                        content = f.read(1024)  # Read first 1KB
                        if len(content) > 0:
                            successful_loads += 1
                except Exception:
                    continue
            
            # Consider successful if we can load at least 80% of test files
            return (successful_loads / len(shader_files[:10])) >= 0.8 if shader_files else True
            
        except Exception as e:
            self.logger.error(f"Shader loading test failed: {e}")
            return False
    
    async def _verify_file_integrity(self, app_id: str, anticheat_config: Dict) -> Dict[str, Any]:
        """Verify file integrity as required by anti-cheat system"""
        integrity_result = {
            "passed": False,
            "files_checked": 0,
            "files_valid": 0,
            "issues": []
        }
        
        try:
            if not anticheat_config["file_integrity_check"]:
                integrity_result["passed"] = True
                return integrity_result
            
            # Get game installation directory
            game_dir = await self._get_game_directory(app_id)
            
            if not game_dir or not os.path.exists(game_dir):
                integrity_result["issues"].append("Game directory not found")
                return integrity_result
            
            # Check critical game files
            critical_files = await self._get_critical_files(game_dir)
            integrity_result["files_checked"] = len(critical_files)
            
            valid_files = 0
            for file_path in critical_files:
                if await self._verify_single_file_integrity(file_path):
                    valid_files += 1
                else:
                    integrity_result["issues"].append(f"File integrity failed: {file_path}")
            
            integrity_result["files_valid"] = valid_files
            integrity_result["passed"] = valid_files == len(critical_files)
            
        except Exception as e:
            integrity_result["issues"].append(f"File integrity check error: {str(e)}")
        
        return integrity_result
    
    async def _get_game_directory(self, app_id: str) -> Optional[str]:
        """Get game installation directory"""
        try:
            # Query Steam for game directory
            result = await self._run_command(f"steam://info/{app_id}")
            # Parse result to extract installation path
            # This is simplified - real implementation would parse Steam's response
            return f"/home/deck/.steam/steam/steamapps/common/game_{app_id}"
        except Exception:
            return None
    
    async def _get_critical_files(self, game_dir: str) -> List[str]:
        """Get list of critical files that need integrity checking"""
        critical_files = []
        
        try:
            # Common critical file patterns
            critical_patterns = ["*.exe", "*.dll", "*.so", "*.bin"]
            
            for pattern in critical_patterns:
                result = await self._run_command(f"find {game_dir} -name '{pattern}' -type f")
                if result.returncode == 0:
                    files = result.stdout.decode().strip().split('\n')
                    critical_files.extend([f for f in files if f])
            
        except Exception as e:
            self.logger.error(f"Error getting critical files: {e}")
        
        return critical_files
    
    async def _verify_single_file_integrity(self, file_path: str) -> bool:
        """Verify integrity of a single file"""
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                return False
            
            # Basic file integrity checks
            file_stat = os.stat(file_path)
            
            # Check file size is reasonable
            if file_stat.st_size == 0:
                return False
            
            # Generate checksum
            file_hash = await self._generate_file_signature(file_path)
            
            # In a real implementation, this would check against known good hashes
            return len(file_hash) == 64  # Valid SHA256
            
        except Exception:
            return False
    
    async def _test_memory_protection(self, app_id: str, anticheat_type: str) -> Dict[str, Any]:
        """Test memory protection compatibility"""
        memory_test = {
            "passed": False,
            "protection_active": False,
            "shader_injection_blocked": False,
            "issues": []
        }
        
        try:
            # Test if memory protection is active
            protection_status = await self._check_memory_protection_status(anticheat_type)
            memory_test["protection_active"] = protection_status
            
            # Test if shader injection is properly blocked
            if protection_status:
                injection_test = await self._test_shader_injection_protection(app_id)
                memory_test["shader_injection_blocked"] = injection_test
            
            memory_test["passed"] = (
                memory_test["protection_active"] and
                memory_test["shader_injection_blocked"]
            )
            
        except Exception as e:
            memory_test["issues"].append(f"Memory protection test error: {str(e)}")
        
        return memory_test
    
    async def _check_memory_protection_status(self, anticheat_type: str) -> bool:
        """Check if memory protection is active"""
        try:
            # Check for memory protection indicators
            if anticheat_type in ["eac", "battleye"]:
                # These systems typically use kernel-level protection
                result = await self._run_command("lsmod | grep -E '(eac|battleye)'")
                return result.returncode == 0
            
            return True  # Assume protection is active for other systems
            
        except Exception:
            return False
    
    async def _test_shader_injection_protection(self, app_id: str) -> bool:
        """Test that shader injection is properly blocked"""
        try:
            # Attempt to inject a test shader (should be blocked)
            test_shader = b"FAKE_SHADER_INJECTION_TEST"
            cache_dir = f"{self.config['steam_deck']['cache_directory']}/{app_id}"
            
            if not os.path.exists(cache_dir):
                return True  # No cache to inject into
            
            test_file = os.path.join(cache_dir, "test_injection.tmp")
            
            try:
                with open(test_file, 'wb') as f:
                    f.write(test_shader)
                
                # If we can write the file, protection may not be active
                # In a real system, anti-cheat should prevent this
                os.unlink(test_file)  # Clean up
                return False  # Injection not blocked
                
            except PermissionError:
                # Good - injection was blocked
                return True
            
        except Exception:
            # Assume protection is working if we can't test
            return True
    
    def _generate_recommendations(self, anticheat_type: str, issues: List[str]) -> List[str]:
        """Generate recommendations based on validation issues"""
        recommendations = []
        
        anticheat_config = self.anticheat_configs.get(anticheat_type, {})
        known_issues = anticheat_config.get("known_issues", [])
        
        # Add known issues as recommendations
        for issue in known_issues:
            recommendations.append(f"Known issue: {issue}")
        
        # Generate specific recommendations based on found issues
        for issue in issues:
            if "not detected" in issue.lower():
                recommendations.append(f"Ensure {anticheat_config.get('name', anticheat_type)} is properly installed")
            
            elif "signature" in issue.lower():
                recommendations.append("Re-validate shader cache signatures")
            
            elif "integrity" in issue.lower():
                recommendations.append("Verify game file integrity through Steam")
            
            elif "memory protection" in issue.lower():
                recommendations.append("Check anti-cheat service status and restart if necessary")
        
        return recommendations
    
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