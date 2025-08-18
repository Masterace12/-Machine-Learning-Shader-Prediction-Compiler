#!/usr/bin/env python3
"""
Steam Deck Shader Prediction Compiler QA Framework
Main orchestrator for comprehensive game compatibility testing
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from ..testing.game_compatibility_tester import GameCompatibilityTester
from ..testing.anticheat_validator import AntiCheatValidator
from ..validation.shader_cache_validator import ShaderCacheValidator
from ..benchmarking.performance_analyzer import PerformanceAnalyzer
from ..telemetry.ml_metrics_collector import MLMetricsCollector
from ..reporting.qa_reporter import QAReporter

class SteamDeckQAFramework:
    """Main QA Framework for Steam Deck shader prediction testing"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/qa_config.json"
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.game_tester = GameCompatibilityTester(self.config)
        self.anticheat_validator = AntiCheatValidator(self.config)
        self.cache_validator = ShaderCacheValidator(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.metrics_collector = MLMetricsCollector(self.config)
        self.reporter = QAReporter(self.config)
        
        self.test_results = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _load_config(self) -> Dict:
        """Load QA framework configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for QA framework"""
        return {
            "steam_deck": {
                "steam_path": "/home/deck/.steam/steam",
                "proton_versions": ["8.0", "Experimental", "7.0", "6.3", "5.13"],
                "cache_directory": "/home/deck/.steam/steam/steamapps/shadercache",
                "performance_metrics": {
                    "target_fps": 60,
                    "acceptable_stutter_threshold": 16.67,  # ms
                    "cache_hit_target": 0.85
                }
            },
            "test_games": {
                "cyberpunk_2077": {
                    "app_id": "1091500",
                    "launch_options": "-windowed -novid",
                    "test_scenarios": ["main_menu", "driving", "combat", "raytracing"],
                    "expected_shaders": 1500,
                    "anticheat": None
                },
                "elden_ring": {
                    "app_id": "1245620",
                    "launch_options": "-windowed",
                    "test_scenarios": ["main_menu", "open_world", "boss_fight"],
                    "expected_shaders": 800,
                    "anticheat": "eac"
                },
                "spider_man_remastered": {
                    "app_id": "1817070",
                    "launch_options": "-windowed -dx12",
                    "test_scenarios": ["swinging", "combat", "cutscenes"],
                    "expected_shaders": 1200,
                    "anticheat": None
                },
                "portal_2": {
                    "app_id": "620",
                    "launch_options": "-windowed",
                    "test_scenarios": ["main_menu", "puzzle_solving", "coop"],
                    "expected_shaders": 200,
                    "anticheat": "vac"
                },
                "apex_legends": {
                    "app_id": "1172470",
                    "launch_options": "-windowed",
                    "test_scenarios": ["main_menu", "training", "multiplayer"],
                    "expected_shaders": 900,
                    "anticheat": "eac"
                },
                "destiny_2": {
                    "app_id": "1085660",
                    "launch_options": "-windowed",
                    "test_scenarios": ["main_menu", "patrol", "strikes"],
                    "expected_shaders": 1100,
                    "anticheat": "battleye"
                }
            },
            "validation": {
                "shader_timeout": 300,  # seconds
                "test_duration": 900,   # 15 minutes per scenario
                "regression_threshold": 0.05,  # 5% performance regression
                "cache_validation": True,
                "memory_leak_detection": True
            },
            "telemetry": {
                "collect_ml_data": True,
                "upload_anonymized_data": False,
                "local_storage_days": 30
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("SteamDeckQA")
        logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        log_file = f"data/logs/qa_session_{self.session_id}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite on all configured games"""
        self.logger.info("Starting Steam Deck QA Framework full test suite")
        
        test_results = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "system_info": await self._get_system_info(),
            "game_results": {},
            "summary": {}
        }
        
        # Test each configured game
        for game_name, game_config in self.config["test_games"].items():
            self.logger.info(f"Starting tests for {game_name}")
            
            try:
                game_result = await self._test_single_game(game_name, game_config)
                test_results["game_results"][game_name] = game_result
                self.logger.info(f"Completed tests for {game_name}")
                
            except Exception as e:
                self.logger.error(f"Failed testing {game_name}: {e}")
                test_results["game_results"][game_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate summary
        test_results["summary"] = self._generate_test_summary(test_results["game_results"])
        
        # Save results and generate reports
        await self._save_results(test_results)
        await self._generate_reports(test_results)
        
        self.logger.info("Full test suite completed")
        return test_results
    
    async def _test_single_game(self, game_name: str, game_config: Dict) -> Dict[str, Any]:
        """Run comprehensive tests for a single game"""
        game_result = {
            "game_name": game_name,
            "app_id": game_config["app_id"],
            "start_time": datetime.now().isoformat(),
            "compatibility_test": {},
            "anticheat_validation": {},
            "cache_validation": {},
            "performance_analysis": {},
            "ml_metrics": {},
            "status": "pending"
        }
        
        try:
            # 1. Game Compatibility Testing
            self.logger.info(f"Running compatibility tests for {game_name}")
            compatibility_result = await self.game_tester.test_game_compatibility(
                game_config["app_id"], 
                game_config["test_scenarios"],
                game_config.get("launch_options", "")
            )
            game_result["compatibility_test"] = compatibility_result
            
            # 2. Anti-cheat Validation (if applicable)
            if game_config.get("anticheat"):
                self.logger.info(f"Running anti-cheat validation for {game_name}")
                anticheat_result = await self.anticheat_validator.validate_anticheat_compatibility(
                    game_config["app_id"],
                    game_config["anticheat"]
                )
                game_result["anticheat_validation"] = anticheat_result
            
            # 3. Shader Cache Validation
            self.logger.info(f"Validating shader cache for {game_name}")
            cache_result = await self.cache_validator.validate_shader_cache(
                game_config["app_id"],
                game_config.get("expected_shaders", 0)
            )
            game_result["cache_validation"] = cache_result
            
            # 4. Performance Analysis
            self.logger.info(f"Running performance analysis for {game_name}")
            perf_result = await self.performance_analyzer.analyze_game_performance(
                game_config["app_id"],
                game_config["test_scenarios"]
            )
            game_result["performance_analysis"] = perf_result
            
            # 5. ML Metrics Collection
            self.logger.info(f"Collecting ML metrics for {game_name}")
            ml_result = await self.metrics_collector.collect_prediction_metrics(
                game_config["app_id"]
            )
            game_result["ml_metrics"] = ml_result
            
            # Determine overall status
            game_result["status"] = self._determine_game_status(game_result)
            game_result["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error testing {game_name}: {e}")
            game_result["status"] = "failed"
            game_result["error"] = str(e)
        
        return game_result
    
    def _determine_game_status(self, game_result: Dict) -> str:
        """Determine overall status based on test results"""
        compatibility = game_result.get("compatibility_test", {}).get("status", "failed")
        cache_valid = game_result.get("cache_validation", {}).get("valid", False)
        performance = game_result.get("performance_analysis", {}).get("status", "failed")
        
        if compatibility == "passed" and cache_valid and performance == "passed":
            return "passed"
        elif compatibility == "failed" or performance == "critical_failure":
            return "failed"
        else:
            return "warning"
    
    def _generate_test_summary(self, game_results: Dict) -> Dict:
        """Generate summary statistics from test results"""
        total_games = len(game_results)
        passed_games = sum(1 for result in game_results.values() 
                          if result.get("status") == "passed")
        failed_games = sum(1 for result in game_results.values() 
                          if result.get("status") == "failed")
        warning_games = sum(1 for result in game_results.values() 
                           if result.get("status") == "warning")
        
        return {
            "total_games_tested": total_games,
            "passed": passed_games,
            "failed": failed_games,
            "warnings": warning_games,
            "pass_rate": passed_games / total_games if total_games > 0 else 0,
            "critical_issues": self._identify_critical_issues(game_results)
        }
    
    def _identify_critical_issues(self, game_results: Dict) -> List[str]:
        """Identify critical issues from test results"""
        issues = []
        
        for game, result in game_results.items():
            if result.get("status") == "failed":
                issues.append(f"Game {game} failed compatibility tests")
            
            # Check for anti-cheat issues
            anticheat = result.get("anticheat_validation", {})
            if not anticheat.get("compatible", True):
                issues.append(f"Anti-cheat compatibility issue in {game}")
            
            # Check for cache corruption
            cache = result.get("cache_validation", {})
            if cache.get("corrupted", False):
                issues.append(f"Shader cache corruption detected in {game}")
            
            # Check for severe performance issues
            perf = result.get("performance_analysis", {})
            if perf.get("severe_stuttering", False):
                issues.append(f"Severe performance degradation in {game}")
        
        return issues
    
    async def _get_system_info(self) -> Dict:
        """Collect system information for the test report"""
        try:
            # This would need to be adapted for actual Steam Deck environment
            return {
                "platform": "Steam Deck",
                "kernel_version": await self._run_command("uname -r"),
                "proton_version": await self._get_proton_version(),
                "gpu_driver": await self._get_gpu_driver_info(),
                "steam_version": await self._get_steam_version(),
                "test_framework_version": "1.0.0"
            }
        except Exception as e:
            self.logger.warning(f"Could not collect system info: {e}")
            return {"error": str(e)}
    
    async def _run_command(self, command: str) -> str:
        """Run shell command and return output"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return stdout.decode().strip()
        except Exception:
            return "unknown"
    
    async def _get_proton_version(self) -> str:
        """Get current Proton version"""
        # Implementation would check Steam's Proton installation
        return "8.0-3"
    
    async def _get_gpu_driver_info(self) -> str:
        """Get GPU driver information"""
        return await self._run_command("glxinfo | grep 'OpenGL version'")
    
    async def _get_steam_version(self) -> str:
        """Get Steam client version"""
        return await self._run_command("steam --version")
    
    async def _save_results(self, results: Dict):
        """Save test results to disk"""
        results_file = f"data/results/qa_results_{self.session_id}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    async def _generate_reports(self, results: Dict):
        """Generate comprehensive reports"""
        await self.reporter.generate_comprehensive_report(results, self.session_id)
        await self.reporter.generate_executive_summary(results, self.session_id)
        await self.reporter.generate_compatibility_matrix(results, self.session_id)
    
    async def run_regression_tests(self, baseline_session_id: str) -> Dict[str, Any]:
        """Run regression tests against a baseline session"""
        self.logger.info(f"Running regression tests against baseline {baseline_session_id}")
        
        # Load baseline results
        baseline_file = f"data/results/qa_results_{baseline_session_id}.json"
        try:
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Baseline results not found: {baseline_file}")
        
        # Run current tests
        current_results = await self.run_full_test_suite()
        
        # Compare results
        regression_analysis = self._analyze_regression(baseline_results, current_results)
        
        # Generate regression report
        await self.reporter.generate_regression_report(
            baseline_results, 
            current_results, 
            regression_analysis, 
            self.session_id
        )
        
        return regression_analysis
    
    def _analyze_regression(self, baseline: Dict, current: Dict) -> Dict:
        """Analyze regression between baseline and current results"""
        regression_analysis = {
            "overall_regression": False,
            "performance_regressions": [],
            "new_failures": [],
            "fixed_issues": [],
            "cache_efficiency_changes": {}
        }
        
        baseline_games = baseline.get("game_results", {})
        current_games = current.get("game_results", {})
        
        for game in baseline_games:
            if game not in current_games:
                continue
            
            baseline_game = baseline_games[game]
            current_game = current_games[game]
            
            # Check for new failures
            if (baseline_game.get("status") == "passed" and 
                current_game.get("status") == "failed"):
                regression_analysis["new_failures"].append(game)
                regression_analysis["overall_regression"] = True
            
            # Check for fixed issues
            elif (baseline_game.get("status") == "failed" and 
                  current_game.get("status") == "passed"):
                regression_analysis["fixed_issues"].append(game)
            
            # Analyze performance regressions
            baseline_perf = baseline_game.get("performance_analysis", {})
            current_perf = current_game.get("performance_analysis", {})
            
            perf_regression = self._check_performance_regression(
                baseline_perf, current_perf
            )
            if perf_regression:
                regression_analysis["performance_regressions"].append({
                    "game": game,
                    "details": perf_regression
                })
                regression_analysis["overall_regression"] = True
        
        return regression_analysis
    
    def _check_performance_regression(self, baseline: Dict, current: Dict) -> Optional[Dict]:
        """Check for performance regression between baseline and current"""
        threshold = self.config["validation"]["regression_threshold"]
        
        baseline_fps = baseline.get("average_fps", 0)
        current_fps = current.get("average_fps", 0)
        
        if baseline_fps > 0 and current_fps > 0:
            fps_change = (baseline_fps - current_fps) / baseline_fps
            if fps_change > threshold:
                return {
                    "type": "fps_regression",
                    "baseline_fps": baseline_fps,
                    "current_fps": current_fps,
                    "regression_percent": fps_change * 100
                }
        
        baseline_stutter = baseline.get("stutter_events", 0)
        current_stutter = current.get("stutter_events", 0)
        
        if current_stutter > baseline_stutter * (1 + threshold):
            return {
                "type": "stutter_regression",
                "baseline_stutter": baseline_stutter,
                "current_stutter": current_stutter
            }
        
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Steam Deck QA Framework")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--regression", help="Run regression tests against baseline session ID")
    args = parser.parse_args()
    
    framework = SteamDeckQAFramework(args.config)
    
    if args.regression:
        results = asyncio.run(framework.run_regression_tests(args.regression))
    else:
        results = asyncio.run(framework.run_full_test_suite())
    
    print(f"Testing completed. Session ID: {framework.session_id}")