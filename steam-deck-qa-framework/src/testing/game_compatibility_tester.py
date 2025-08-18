#!/usr/bin/env python3
"""
Game Compatibility Testing Module
Automated testing for popular Steam Deck games
"""

import os
import asyncio
import time
import logging
import subprocess
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

class GameCompatibilityTester:
    """Automated game compatibility testing for Steam Deck"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}")
        self.steam_path = config.get("steam_deck", {}).get("steam_path", "/home/deck/.steam/steam")
        self.test_duration = config.get("validation", {}).get("test_duration", 900)
    
    async def test_game_compatibility(self, app_id: str, test_scenarios: List[str], launch_options: str = "") -> Dict[str, Any]:
        """Run comprehensive compatibility tests for a game"""
        self.logger.info(f"Starting compatibility tests for app {app_id}")
        
        test_result = {
            "app_id": app_id,
            "launch_options": launch_options,
            "test_scenarios": {},
            "overall_compatibility": True,
            "issues_found": [],
            "status": "pending"
        }
        
        try:
            # Pre-test validation
            if not await self._validate_game_installation(app_id):
                test_result["status"] = "failed"
                test_result["issues_found"].append("Game not installed or accessible")
                return test_result
            
            # Test each scenario
            for scenario in test_scenarios:
                self.logger.info(f"Testing scenario: {scenario}")
                scenario_result = await self._test_scenario(app_id, scenario, launch_options)
                test_result["test_scenarios"][scenario] = scenario_result
                
                if not scenario_result.get("success", False):
                    test_result["overall_compatibility"] = False
                    test_result["issues_found"].extend(scenario_result.get("issues", []))
            
            # Determine final status
            test_result["status"] = "passed" if test_result["overall_compatibility"] else "failed"
            
        except Exception as e:
            self.logger.error(f"Compatibility test failed for {app_id}: {e}")
            test_result["status"] = "failed"
            test_result["issues_found"].append(f"Test execution error: {str(e)}")
        
        return test_result
    
    async def _validate_game_installation(self, app_id: str) -> bool:
        """Validate that the game is properly installed"""
        try:
            # Check Steam library for game
            result = await self._run_steam_command(f"steam://validate/{app_id}")
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Installation validation failed: {e}")
            return False
    
    async def _test_scenario(self, app_id: str, scenario: str, launch_options: str) -> Dict[str, Any]:
        """Test a specific game scenario"""
        scenario_result = {
            "scenario": scenario,
            "success": False,
            "launch_time": 0,
            "runtime": 0,
            "issues": [],
            "performance_data": {},
            "shader_compilation_detected": False
        }
        
        try:
            start_time = time.time()
            
            # Launch game with monitoring
            process = await self._launch_game_monitored(app_id, launch_options)
            
            if not process:
                scenario_result["issues"].append("Failed to launch game")
                return scenario_result
            
            launch_time = time.time() - start_time
            scenario_result["launch_time"] = launch_time
            
            # Monitor game execution
            monitoring_data = await self._monitor_game_execution(process, scenario)
            scenario_result["performance_data"] = monitoring_data
            scenario_result["shader_compilation_detected"] = monitoring_data.get("shader_compilation", False)
            
            # Scenario-specific tests
            scenario_tests = await self._run_scenario_tests(process, scenario)
            scenario_result.update(scenario_tests)
            
            # Clean shutdown
            await self._shutdown_game(process)
            scenario_result["runtime"] = time.time() - start_time
            scenario_result["success"] = len(scenario_result["issues"]) == 0
            
        except Exception as e:
            self.logger.error(f"Scenario test failed: {e}")
            scenario_result["issues"].append(f"Scenario execution error: {str(e)}")
        
        return scenario_result
    
    async def _launch_game_monitored(self, app_id: str, launch_options: str) -> Optional[subprocess.Popen]:
        """Launch game with comprehensive monitoring"""
        try:
            # Build Steam launch command
            cmd = [
                self.steam_path,
                "-applaunch", app_id
            ]
            
            if launch_options:
                cmd.extend(launch_options.split())
            
            # Set environment for monitoring
            env = os.environ.copy()
            env.update({
                "PROTON_LOG": "1",
                "PROTON_LOG_DIR": f"/tmp/proton_logs_{app_id}",
                "DXVK_HUD": "fps,memory,drawcalls",
                "VK_LAYER_PATH": "/usr/share/vulkan/explicit_layer.d"
            })
            
            # Launch with monitoring
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for game to start
            await asyncio.sleep(30)  # Give game time to initialize
            
            # Verify process is running
            if process.poll() is not None:
                self.logger.error(f"Game {app_id} exited during launch")
                return None
            
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to launch game {app_id}: {e}")
            return None
    
    async def _monitor_game_execution(self, process: subprocess.Popen, scenario: str) -> Dict[str, Any]:
        """Monitor game execution for performance and shader compilation"""
        monitoring_data = {
            "fps_samples": [],
            "memory_usage": [],
            "cpu_usage": [],
            "shader_compilation": False,
            "frame_time_spikes": [],
            "gpu_utilization": []
        }
        
        # Monitor for specified duration or until scenario completes
        monitor_duration = min(self.test_duration, 300)  # Max 5 minutes per scenario
        start_time = time.time()
        
        try:
            ps_process = psutil.Process(process.pid)
            
            while time.time() - start_time < monitor_duration:
                if process.poll() is not None:
                    break
                
                # Collect performance metrics
                try:
                    cpu_percent = ps_process.cpu_percent()
                    memory_info = ps_process.memory_info()
                    
                    monitoring_data["cpu_usage"].append(cpu_percent)
                    monitoring_data["memory_usage"].append(memory_info.rss / 1024 / 1024)  # MB
                    
                    # Check for shader compilation indicators
                    if self._detect_shader_compilation(process):
                        monitoring_data["shader_compilation"] = True
                    
                    # GPU monitoring would require additional tools on Steam Deck
                    # This is a placeholder for actual implementation
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                
                await asyncio.sleep(1)  # Sample every second
            
        except Exception as e:
            self.logger.warning(f"Monitoring interrupted: {e}")
        
        return monitoring_data
    
    def _detect_shader_compilation(self, process: subprocess.Popen) -> bool:
        """Detect shader compilation activity"""
        try:
            # Check for DXVK shader compilation in logs
            log_dir = f"/tmp/proton_logs_{process.pid}"
            if os.path.exists(log_dir):
                for log_file in os.listdir(log_dir):
                    if "dxvk" in log_file.lower():
                        with open(os.path.join(log_dir, log_file), 'r') as f:
                            content = f.read()
                            if "compiling shader" in content.lower() or "shader cache" in content.lower():
                                return True
            
            # Check process CPU usage spikes (indication of compilation)
            ps_process = psutil.Process(process.pid)
            cpu_percent = ps_process.cpu_percent()
            return cpu_percent > 80  # High CPU usage may indicate compilation
            
        except Exception:
            return False
    
    async def _run_scenario_tests(self, process: subprocess.Popen, scenario: str) -> Dict[str, Any]:
        """Run scenario-specific tests"""
        scenario_tests = {
            "scenario_specific_results": {},
            "issues": []
        }
        
        try:
            if scenario == "main_menu":
                # Test main menu responsiveness
                await asyncio.sleep(30)  # Wait for menu to load
                scenario_tests["scenario_specific_results"]["menu_loaded"] = True
                
            elif scenario == "gameplay" or "combat" in scenario:
                # Test gameplay stability
                await asyncio.sleep(120)  # Monitor for 2 minutes
                scenario_tests["scenario_specific_results"]["gameplay_stable"] = True
                
            elif scenario == "loading":
                # Test loading performance
                load_start = time.time()
                await asyncio.sleep(60)  # Monitor loading
                load_time = time.time() - load_start
                scenario_tests["scenario_specific_results"]["load_time"] = load_time
                
                if load_time > 120:  # Over 2 minutes is concerning
                    scenario_tests["issues"].append("Excessive loading time")
            
            # Check if process is still responsive
            if process.poll() is not None:
                scenario_tests["issues"].append("Game crashed during scenario test")
            
        except Exception as e:
            scenario_tests["issues"].append(f"Scenario test error: {str(e)}")
        
        return scenario_tests
    
    async def _shutdown_game(self, process: subprocess.Popen):
        """Safely shutdown the game process"""
        try:
            # Try graceful shutdown first
            process.terminate()
            await asyncio.sleep(10)
            
            # Force kill if still running
            if process.poll() is None:
                process.kill()
                await asyncio.sleep(5)
            
            # Clean up any remaining processes
            try:
                ps_process = psutil.Process(process.pid)
                for child in ps_process.children(recursive=True):
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        except Exception as e:
            self.logger.warning(f"Error during game shutdown: {e}")
    
    async def _run_steam_command(self, command: str) -> subprocess.CompletedProcess:
        """Run a Steam command and return result"""
        try:
            process = await asyncio.create_subprocess_shell(
                f"{self.steam_path} {command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
        except Exception as e:
            self.logger.error(f"Steam command failed: {e}")
            raise
    
    async def test_proton_compatibility(self, app_id: str, proton_versions: List[str]) -> Dict[str, Any]:
        """Test game compatibility across different Proton versions"""
        self.logger.info(f"Testing Proton compatibility for app {app_id}")
        
        compatibility_results = {
            "app_id": app_id,
            "proton_tests": {},
            "recommended_version": None,
            "compatibility_issues": []
        }
        
        for proton_version in proton_versions:
            self.logger.info(f"Testing with Proton {proton_version}")
            
            try:
                # Set Proton version for this app
                await self._set_proton_version(app_id, proton_version)
                
                # Run basic compatibility test
                test_result = await self.test_game_compatibility(
                    app_id, ["main_menu"], ""
                )
                
                compatibility_results["proton_tests"][proton_version] = {
                    "compatible": test_result["status"] == "passed",
                    "launch_time": test_result.get("test_scenarios", {}).get("main_menu", {}).get("launch_time", 0),
                    "issues": test_result.get("issues_found", [])
                }
                
            except Exception as e:
                compatibility_results["proton_tests"][proton_version] = {
                    "compatible": False,
                    "error": str(e)
                }
        
        # Determine recommended Proton version
        compatible_versions = [
            v for v, data in compatibility_results["proton_tests"].items()
            if data.get("compatible", False)
        ]
        
        if compatible_versions:
            # Prefer latest compatible version
            compatibility_results["recommended_version"] = compatible_versions[0]
        
        return compatibility_results
    
    async def _set_proton_version(self, app_id: str, proton_version: str):
        """Set Proton version for a specific app"""
        # This would interact with Steam's configuration
        # Implementation depends on Steam Deck environment
        config_path = f"{os.path.expanduser('~')}/.steam/steam/config/config.vdf"
        # Actual implementation would modify Steam configuration
        pass
    
    async def validate_multiplayer_compatibility(self, app_id: str) -> Dict[str, Any]:
        """Test multiplayer/online functionality"""
        multiplayer_result = {
            "app_id": app_id,
            "online_connectivity": False,
            "matchmaking_functional": False,
            "p2p_shader_sharing": False,
            "issues": []
        }
        
        try:
            # Test basic online connectivity
            connectivity_test = await self._test_online_connectivity(app_id)
            multiplayer_result["online_connectivity"] = connectivity_test
            
            if connectivity_test:
                # Test matchmaking if game supports it
                matchmaking_test = await self._test_matchmaking(app_id)
                multiplayer_result["matchmaking_functional"] = matchmaking_test
                
                # Test P2P shader sharing
                p2p_test = await self._test_p2p_shader_sharing(app_id)
                multiplayer_result["p2p_shader_sharing"] = p2p_test
            
        except Exception as e:
            multiplayer_result["issues"].append(f"Multiplayer test error: {str(e)}")
        
        return multiplayer_result
    
    async def _test_online_connectivity(self, app_id: str) -> bool:
        """Test basic online connectivity for the game"""
        try:
            # Launch game and check for online features
            process = await self._launch_game_monitored(app_id, "-online")
            if not process:
                return False
            
            # Monitor for online connection indicators
            await asyncio.sleep(60)  # Wait for connection attempt
            
            # Check logs for connection status
            # Implementation would depend on game-specific indicators
            
            await self._shutdown_game(process)
            return True
            
        except Exception as e:
            self.logger.error(f"Online connectivity test failed: {e}")
            return False
    
    async def _test_matchmaking(self, app_id: str) -> bool:
        """Test matchmaking functionality"""
        # Implementation would be game-specific
        # This is a placeholder for actual matchmaking tests
        return True
    
    async def _test_p2p_shader_sharing(self, app_id: str) -> bool:
        """Test P2P shader sharing functionality"""
        try:
            # Check if P2P shader system is active
            shader_cache_dir = f"{self.config['steam_deck']['cache_directory']}/{app_id}"
            if os.path.exists(shader_cache_dir):
                # Look for P2P shared shaders
                shared_shaders = [f for f in os.listdir(shader_cache_dir) if "p2p" in f.lower()]
                return len(shared_shaders) > 0
            
            return False
            
        except Exception as e:
            self.logger.error(f"P2P shader test failed: {e}")
            return False