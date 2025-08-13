#!/usr/bin/env python3
"""
Performance Analysis Module
Comprehensive performance benchmarking and stutter reduction measurement
"""

import os
import asyncio
import logging
import json
import time
import statistics
import subprocess
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class FrameTimeData:
    """Frame time measurement data"""
    timestamp: float
    frame_time: float
    fps: float
    gpu_util: float
    cpu_util: float
    memory_usage: float

@dataclass
class StutterEvent:
    """Stutter event data"""
    timestamp: float
    duration: float
    severity: str  # 'minor', 'moderate', 'severe'
    frame_drop_count: int
    preceding_frame_time: float
    recovery_time: float

class PerformanceAnalyzer:
    """Comprehensive performance analysis and benchmarking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}")
        self.target_fps = config.get("steam_deck", {}).get("performance_metrics", {}).get("target_fps", 60)
        self.stutter_threshold = config.get("steam_deck", {}).get("performance_metrics", {}).get("acceptable_stutter_threshold", 16.67)
        self.monitoring_tools = self._initialize_monitoring_tools()
    
    def _initialize_monitoring_tools(self) -> Dict[str, Any]:
        """Initialize performance monitoring tools"""
        return {
            "fps_counter": "mangohud",  # MangoHUD for Steam Deck
            "gpu_monitor": "radeontop",
            "cpu_monitor": "htop",
            "memory_monitor": "free",
            "frame_capture": "obs",
            "profiler": "perf"
        }
    
    async def analyze_game_performance(self, app_id: str, test_scenarios: List[str]) -> Dict[str, Any]:
        """Comprehensive performance analysis for a game"""
        self.logger.info(f"Starting performance analysis for app {app_id}")
        
        analysis_result = {
            "app_id": app_id,
            "test_scenarios": {},
            "overall_performance": {},
            "shader_compilation_impact": {},
            "stutter_analysis": {},
            "baseline_comparison": {},
            "status": "pending"
        }
        
        try:
            # Pre-analysis setup
            await self._setup_performance_monitoring()
            
            # Analyze each test scenario
            for scenario in test_scenarios:
                self.logger.info(f"Analyzing performance for scenario: {scenario}")
                scenario_result = await self._analyze_scenario_performance(app_id, scenario)
                analysis_result["test_scenarios"][scenario] = scenario_result
            
            # Generate overall performance metrics
            overall_perf = await self._calculate_overall_performance(analysis_result["test_scenarios"])
            analysis_result["overall_performance"] = overall_perf
            
            # Analyze shader compilation impact
            shader_impact = await self._analyze_shader_compilation_impact(app_id)
            analysis_result["shader_compilation_impact"] = shader_impact
            
            # Comprehensive stutter analysis
            stutter_analysis = await self._perform_stutter_analysis(analysis_result["test_scenarios"])
            analysis_result["stutter_analysis"] = stutter_analysis
            
            # Compare with baseline if available
            baseline_comparison = await self._compare_with_baseline(app_id, analysis_result)
            analysis_result["baseline_comparison"] = baseline_comparison
            
            # Determine overall status
            analysis_result["status"] = self._determine_performance_status(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            analysis_result["status"] = "failed"
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    async def _setup_performance_monitoring(self):
        """Setup performance monitoring environment"""
        try:
            # Configure MangoHUD for frame time logging
            mangohud_config = {
                "fps_limit": 0,
                "fps_sampling_period": 500,
                "output_folder": "/tmp/mangohud_logs",
                "log_duration": 300,
                "cpu_stats": True,
                "gpu_stats": True,
                "ram": True,
                "vram": True,
                "frame_timing": True
            }
            
            # Write MangoHUD config
            config_dir = os.path.expanduser("~/.config/MangoHud")
            os.makedirs(config_dir, exist_ok=True)
            
            with open(os.path.join(config_dir, "MangoHud.conf"), 'w') as f:
                for key, value in mangohud_config.items():
                    f.write(f"{key}={value}\n")
            
            # Setup additional monitoring
            await self._setup_system_monitoring()
            
        except Exception as e:
            self.logger.error(f"Error setting up performance monitoring: {e}")
    
    async def _setup_system_monitoring(self):
        """Setup system-level performance monitoring"""
        try:
            # Start system monitoring processes
            monitoring_processes = {}
            
            # GPU monitoring
            gpu_cmd = "radeontop -d /tmp/gpu_stats.log -t 1"
            monitoring_processes["gpu"] = subprocess.Popen(
                gpu_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # CPU monitoring
            cpu_cmd = "sar -u 1 -o /tmp/cpu_stats.log"
            monitoring_processes["cpu"] = subprocess.Popen(
                cpu_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Store monitoring processes for cleanup
            self.monitoring_processes = monitoring_processes
            
        except Exception as e:
            self.logger.warning(f"Could not setup system monitoring: {e}")
    
    async def _analyze_scenario_performance(self, app_id: str, scenario: str) -> Dict[str, Any]:
        """Analyze performance for a specific scenario"""
        scenario_result = {
            "scenario": scenario,
            "frame_time_data": [],
            "performance_metrics": {},
            "stutter_events": [],
            "compilation_events": [],
            "resource_usage": {},
            "issues": []
        }
        
        try:
            # Launch game with performance monitoring
            game_process = await self._launch_game_with_monitoring(app_id, scenario)
            
            if not game_process:
                scenario_result["issues"].append("Failed to launch game for performance analysis")
                return scenario_result
            
            # Monitor performance during scenario execution
            monitoring_data = await self._monitor_scenario_performance(game_process, scenario)
            
            # Process monitoring data
            scenario_result["frame_time_data"] = monitoring_data["frame_times"]
            scenario_result["resource_usage"] = monitoring_data["resource_usage"]
            scenario_result["compilation_events"] = monitoring_data["compilation_events"]
            
            # Calculate performance metrics
            perf_metrics = await self._calculate_performance_metrics(monitoring_data["frame_times"])
            scenario_result["performance_metrics"] = perf_metrics
            
            # Detect stutter events
            stutter_events = await self._detect_stutter_events(monitoring_data["frame_times"])
            scenario_result["stutter_events"] = stutter_events
            
            # Clean shutdown
            await self._shutdown_game_monitoring(game_process)
            
        except Exception as e:
            scenario_result["issues"].append(f"Scenario analysis error: {str(e)}")
        
        return scenario_result
    
    async def _launch_game_with_monitoring(self, app_id: str, scenario: str) -> Optional[subprocess.Popen]:
        """Launch game with comprehensive performance monitoring"""
        try:
            steam_path = self.config["steam_deck"]["steam_path"]
            
            # Set environment variables for monitoring
            env = os.environ.copy()
            env.update({
                "MANGOHUD": "1",
                "MANGOHUD_DLSYM": "1",
                "MANGOHUD_CONFIG": "fps_sampling_period=100,output_folder=/tmp/mangohud_logs",
                "PROTON_LOG": "1",
                "PROTON_LOG_DIR": f"/tmp/proton_logs_{app_id}",
                "DXVK_HUD": "fps,memory,drawcalls,pipelines",
                "VK_LAYER_PATH": "/usr/share/vulkan/explicit_layer.d"
            })
            
            # Launch command
            launch_cmd = f"{steam_path} -applaunch {app_id}"
            
            process = subprocess.Popen(
                launch_cmd.split(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for game to start
            await asyncio.sleep(30)
            
            if process.poll() is not None:
                self.logger.error(f"Game {app_id} failed to start for performance monitoring")
                return None
            
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to launch game with monitoring: {e}")
            return None
    
    async def _monitor_scenario_performance(self, game_process: subprocess.Popen, scenario: str) -> Dict[str, Any]:
        """Monitor performance during scenario execution"""
        monitoring_data = {
            "frame_times": [],
            "resource_usage": {
                "cpu": [],
                "gpu": [],
                "memory": [],
                "vram": []
            },
            "compilation_events": []
        }
        
        try:
            # Monitor for scenario duration
            scenario_duration = self._get_scenario_duration(scenario)
            start_time = time.time()
            
            while time.time() - start_time < scenario_duration:
                if game_process.poll() is not None:
                    break
                
                # Collect frame time data
                frame_data = await self._collect_frame_time_data()
                if frame_data:
                    monitoring_data["frame_times"].append(frame_data)
                
                # Collect resource usage
                resource_data = await self._collect_resource_usage(game_process)
                for resource, value in resource_data.items():
                    monitoring_data["resource_usage"][resource].append(value)
                
                # Check for shader compilation
                compilation_event = await self._detect_compilation_event(game_process)
                if compilation_event:
                    monitoring_data["compilation_events"].append(compilation_event)
                
                await asyncio.sleep(0.1)  # 100ms sampling rate
            
        except Exception as e:
            self.logger.error(f"Performance monitoring error: {e}")
        
        return monitoring_data
    
    def _get_scenario_duration(self, scenario: str) -> int:
        """Get monitoring duration for scenario"""
        scenario_durations = {
            "main_menu": 60,      # 1 minute
            "gameplay": 300,      # 5 minutes
            "loading": 120,       # 2 minutes
            "combat": 180,        # 3 minutes
            "driving": 240,       # 4 minutes
            "cutscenes": 120,     # 2 minutes
            "multiplayer": 300    # 5 minutes
        }
        
        return scenario_durations.get(scenario, 180)  # Default 3 minutes
    
    async def _collect_frame_time_data(self) -> Optional[FrameTimeData]:
        """Collect current frame time data"""
        try:
            # Read from MangoHUD log
            mangohud_log = "/tmp/mangohud_logs/MangoHud.log"
            
            if os.path.exists(mangohud_log):
                # Parse latest frame time data
                with open(mangohud_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        latest_line = lines[-1].strip()
                        # Parse MangoHUD log format
                        parts = latest_line.split(',')
                        if len(parts) >= 6:
                            return FrameTimeData(
                                timestamp=float(parts[0]),
                                frame_time=float(parts[1]),
                                fps=float(parts[2]),
                                gpu_util=float(parts[3]),
                                cpu_util=float(parts[4]),
                                memory_usage=float(parts[5])
                            )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error collecting frame time data: {e}")
            return None
    
    async def _collect_resource_usage(self, game_process: subprocess.Popen) -> Dict[str, float]:
        """Collect current resource usage"""
        resource_usage = {
            "cpu": 0.0,
            "gpu": 0.0,
            "memory": 0.0,
            "vram": 0.0
        }
        
        try:
            # Get process CPU and memory usage
            ps_process = psutil.Process(game_process.pid)
            resource_usage["cpu"] = ps_process.cpu_percent()
            resource_usage["memory"] = ps_process.memory_info().rss / 1024 / 1024  # MB
            
            # Get GPU usage (from radeontop log)
            gpu_usage = await self._read_gpu_usage()
            resource_usage["gpu"] = gpu_usage
            
            # Get VRAM usage
            vram_usage = await self._read_vram_usage()
            resource_usage["vram"] = vram_usage
            
        except Exception as e:
            self.logger.warning(f"Error collecting resource usage: {e}")
        
        return resource_usage
    
    async def _read_gpu_usage(self) -> float:
        """Read GPU usage from monitoring logs"""
        try:
            gpu_log = "/tmp/gpu_stats.log"
            if os.path.exists(gpu_log):
                with open(gpu_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Parse radeontop output
                        latest = lines[-1].strip()
                        # Extract GPU usage percentage
                        if "gpu" in latest:
                            parts = latest.split()
                            for i, part in enumerate(parts):
                                if "gpu" in part.lower() and i + 1 < len(parts):
                                    usage_str = parts[i + 1].replace('%', '')
                                    return float(usage_str)
            
            return 0.0
        except Exception:
            return 0.0
    
    async def _read_vram_usage(self) -> float:
        """Read VRAM usage"""
        try:
            # Read from /sys filesystem for AMD GPU
            vram_used_file = "/sys/class/drm/card0/device/mem_info_vram_used"
            vram_total_file = "/sys/class/drm/card0/device/mem_info_vram_total"
            
            if os.path.exists(vram_used_file) and os.path.exists(vram_total_file):
                with open(vram_used_file, 'r') as f:
                    vram_used = int(f.read().strip())
                with open(vram_total_file, 'r') as f:
                    vram_total = int(f.read().strip())
                
                return (vram_used / vram_total) * 100  # Percentage
            
            return 0.0
        except Exception:
            return 0.0
    
    async def _detect_compilation_event(self, game_process: subprocess.Popen) -> Optional[Dict[str, Any]]:
        """Detect shader compilation events"""
        try:
            # Check DXVK logs for compilation activity
            log_dir = f"/tmp/proton_logs_{game_process.pid}"
            if os.path.exists(log_dir):
                for log_file in os.listdir(log_dir):
                    if "dxvk" in log_file.lower():
                        log_path = os.path.join(log_dir, log_file)
                        with open(log_path, 'r') as f:
                            content = f.read()
                            # Look for compilation indicators
                            if ("compiling shader" in content.lower() or 
                                "creating pipeline" in content.lower()):
                                return {
                                    "timestamp": time.time(),
                                    "type": "shader_compilation",
                                    "source": log_file
                                }
            
            # Check CPU usage spikes as compilation indicator
            ps_process = psutil.Process(game_process.pid)
            cpu_percent = ps_process.cpu_percent()
            if cpu_percent > 90:  # High CPU usage may indicate compilation
                return {
                    "timestamp": time.time(),
                    "type": "high_cpu_compilation",
                    "cpu_usage": cpu_percent
                }
            
            return None
            
        except Exception:
            return None
    
    async def _calculate_performance_metrics(self, frame_time_data: List[FrameTimeData]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not frame_time_data:
            return {"error": "No frame time data available"}
        
        fps_values = [data.fps for data in frame_time_data]
        frame_times = [data.frame_time for data in frame_time_data]
        
        metrics = {
            "average_fps": statistics.mean(fps_values),
            "median_fps": statistics.median(fps_values),
            "min_fps": min(fps_values),
            "max_fps": max(fps_values),
            "fps_std_dev": statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
            
            "average_frame_time": statistics.mean(frame_times),
            "percentile_1_low": np.percentile(fps_values, 1),
            "percentile_0_1_low": np.percentile(fps_values, 0.1),
            
            "frame_time_variance": statistics.variance(frame_times) if len(frame_times) > 1 else 0,
            "consistency_score": self._calculate_consistency_score(fps_values),
            "target_fps_achievement": (sum(1 for fps in fps_values if fps >= self.target_fps) / len(fps_values)) * 100
        }
        
        # Performance classification
        if metrics["average_fps"] >= self.target_fps * 0.95:
            metrics["performance_class"] = "excellent"
        elif metrics["average_fps"] >= self.target_fps * 0.80:
            metrics["performance_class"] = "good"
        elif metrics["average_fps"] >= self.target_fps * 0.60:
            metrics["performance_class"] = "acceptable"
        else:
            metrics["performance_class"] = "poor"
        
        return metrics
    
    def _calculate_consistency_score(self, fps_values: List[float]) -> float:
        """Calculate frame rate consistency score (0-100)"""
        if len(fps_values) < 2:
            return 100.0
        
        # Calculate coefficient of variation
        mean_fps = statistics.mean(fps_values)
        std_dev = statistics.stdev(fps_values)
        
        if mean_fps == 0:
            return 0.0
        
        cv = (std_dev / mean_fps) * 100
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency_score = max(0, 100 - cv)
        return consistency_score
    
    async def _detect_stutter_events(self, frame_time_data: List[FrameTimeData]) -> List[StutterEvent]:
        """Detect and classify stutter events"""
        stutter_events = []
        
        if len(frame_time_data) < 2:
            return stutter_events
        
        frame_times = [data.frame_time for data in frame_time_data]
        
        # Calculate stutter detection thresholds
        median_frame_time = statistics.median(frame_times)
        stutter_threshold = max(self.stutter_threshold, median_frame_time * 2.5)
        
        i = 0
        while i < len(frame_times) - 1:
            current_time = frame_times[i]
            
            if current_time > stutter_threshold:
                # Found potential stutter event
                stutter_start = i
                stutter_duration = current_time
                frame_drops = 1
                
                # Check for consecutive high frame times
                j = i + 1
                while j < len(frame_times) and frame_times[j] > stutter_threshold:
                    stutter_duration += frame_times[j]
                    frame_drops += 1
                    j += 1
                
                # Classify stutter severity
                severity = self._classify_stutter_severity(stutter_duration, frame_drops)
                
                # Calculate recovery time
                recovery_time = self._calculate_recovery_time(frame_times, j)
                
                stutter_event = StutterEvent(
                    timestamp=frame_time_data[stutter_start].timestamp,
                    duration=stutter_duration,
                    severity=severity,
                    frame_drop_count=frame_drops,
                    preceding_frame_time=frame_times[max(0, stutter_start - 1)],
                    recovery_time=recovery_time
                )
                
                stutter_events.append(stutter_event)
                i = j  # Skip past this stutter event
            else:
                i += 1
        
        return stutter_events
    
    def _classify_stutter_severity(self, duration: float, frame_drops: int) -> str:
        """Classify stutter severity based on duration and frame drops"""
        if duration > 100 or frame_drops > 5:  # >100ms or >5 frames
            return "severe"
        elif duration > 50 or frame_drops > 3:  # >50ms or >3 frames
            return "moderate"
        else:
            return "minor"
    
    def _calculate_recovery_time(self, frame_times: List[float], stutter_end_index: int) -> float:
        """Calculate time to recover to normal frame times after stutter"""
        if stutter_end_index >= len(frame_times):
            return 0.0
        
        # Find median frame time for normal performance
        median_frame_time = statistics.median(frame_times)
        recovery_threshold = median_frame_time * 1.2  # 20% above median
        
        recovery_time = 0.0
        for i in range(stutter_end_index, min(stutter_end_index + 10, len(frame_times))):
            recovery_time += frame_times[i]
            if frame_times[i] <= recovery_threshold:
                break
        
        return recovery_time
    
    async def _calculate_overall_performance(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics across all scenarios"""
        overall_performance = {
            "average_fps_all_scenarios": 0.0,
            "worst_case_fps": 0.0,
            "best_case_fps": 0.0,
            "total_stutter_events": 0,
            "severe_stutter_count": 0,
            "consistency_rating": "unknown",
            "performance_summary": {}
        }
        
        try:
            all_fps_values = []
            all_stutter_events = []
            scenario_performances = {}
            
            for scenario, result in scenario_results.items():
                perf_metrics = result.get("performance_metrics", {})
                stutter_events = result.get("stutter_events", [])
                
                if perf_metrics:
                    avg_fps = perf_metrics.get("average_fps", 0)
                    all_fps_values.append(avg_fps)
                    scenario_performances[scenario] = {
                        "fps": avg_fps,
                        "class": perf_metrics.get("performance_class", "unknown")
                    }
                
                all_stutter_events.extend(stutter_events)
            
            if all_fps_values:
                overall_performance["average_fps_all_scenarios"] = statistics.mean(all_fps_values)
                overall_performance["worst_case_fps"] = min(all_fps_values)
                overall_performance["best_case_fps"] = max(all_fps_values)
                
                # Consistency rating based on FPS variation
                fps_variance = statistics.variance(all_fps_values) if len(all_fps_values) > 1 else 0
                if fps_variance < 10:
                    overall_performance["consistency_rating"] = "excellent"
                elif fps_variance < 25:
                    overall_performance["consistency_rating"] = "good"
                elif fps_variance < 50:
                    overall_performance["consistency_rating"] = "fair"
                else:
                    overall_performance["consistency_rating"] = "poor"
            
            overall_performance["total_stutter_events"] = len(all_stutter_events)
            overall_performance["severe_stutter_count"] = sum(
                1 for event in all_stutter_events if event.severity == "severe"
            )
            overall_performance["performance_summary"] = scenario_performances
            
        except Exception as e:
            self.logger.error(f"Error calculating overall performance: {e}")
        
        return overall_performance
    
    async def _analyze_shader_compilation_impact(self, app_id: str) -> Dict[str, Any]:
        """Analyze the impact of shader compilation on performance"""
        compilation_impact = {
            "compilation_events_detected": 0,
            "performance_degradation": {},
            "cache_effectiveness": {},
            "compilation_patterns": [],
            "recommendations": []
        }
        
        try:
            # Read compilation logs
            log_dir = f"/tmp/proton_logs_{app_id}"
            compilation_events = []
            
            if os.path.exists(log_dir):
                for log_file in os.listdir(log_dir):
                    if "dxvk" in log_file.lower():
                        log_path = os.path.join(log_dir, log_file)
                        events = await self._parse_compilation_log(log_path)
                        compilation_events.extend(events)
            
            compilation_impact["compilation_events_detected"] = len(compilation_events)
            
            # Analyze performance impact during compilation
            if compilation_events:
                perf_impact = await self._analyze_compilation_performance_impact(compilation_events)
                compilation_impact["performance_degradation"] = perf_impact
                
                # Analyze cache effectiveness
                cache_effectiveness = await self._analyze_cache_effectiveness(app_id, compilation_events)
                compilation_impact["cache_effectiveness"] = cache_effectiveness
                
                # Identify compilation patterns
                patterns = await self._identify_compilation_patterns(compilation_events)
                compilation_impact["compilation_patterns"] = patterns
                
                # Generate recommendations
                recommendations = self._generate_compilation_recommendations(compilation_impact)
                compilation_impact["recommendations"] = recommendations
        
        except Exception as e:
            self.logger.error(f"Shader compilation analysis error: {e}")
        
        return compilation_impact
    
    async def _parse_compilation_log(self, log_path: str) -> List[Dict[str, Any]]:
        """Parse shader compilation events from log file"""
        events = []
        
        try:
            with open(log_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if ("compiling shader" in line.lower() or 
                        "creating pipeline" in line.lower() or
                        "shader cache" in line.lower()):
                        
                        event = {
                            "line_number": line_num,
                            "content": line.strip(),
                            "type": "compilation",
                            "timestamp": self._extract_timestamp(line)
                        }
                        events.append(event)
        
        except Exception as e:
            self.logger.error(f"Error parsing compilation log {log_path}: {e}")
        
        return events
    
    def _extract_timestamp(self, log_line: str) -> Optional[float]:
        """Extract timestamp from log line"""
        try:
            # Try to extract timestamp from various log formats
            # This is simplified - real implementation would be more robust
            import re
            timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
            match = re.search(timestamp_pattern, log_line)
            if match:
                timestamp_str = match.group()
                return time.mktime(time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"))
            
            return None
        except Exception:
            return None
    
    async def _analyze_compilation_performance_impact(self, compilation_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance impact during shader compilation"""
        impact_analysis = {
            "average_fps_during_compilation": 0.0,
            "fps_drop_percentage": 0.0,
            "compilation_stutter_events": 0,
            "longest_compilation_stutter": 0.0
        }
        
        try:
            # This would correlate compilation events with frame time data
            # For now, provide estimated impact based on event count
            event_count = len(compilation_events)
            
            if event_count > 100:
                impact_analysis["fps_drop_percentage"] = 30.0  # 30% FPS drop
                impact_analysis["compilation_stutter_events"] = event_count // 5
            elif event_count > 50:
                impact_analysis["fps_drop_percentage"] = 20.0
                impact_analysis["compilation_stutter_events"] = event_count // 8
            elif event_count > 0:
                impact_analysis["fps_drop_percentage"] = 10.0
                impact_analysis["compilation_stutter_events"] = event_count // 10
        
        except Exception as e:
            self.logger.error(f"Compilation impact analysis error: {e}")
        
        return impact_analysis
    
    async def _analyze_cache_effectiveness(self, app_id: str, compilation_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze shader cache effectiveness"""
        cache_effectiveness = {
            "cache_hit_rate": 0.0,
            "cache_miss_events": len(compilation_events),
            "cache_size": 0,
            "prediction_accuracy": 0.0
        }
        
        try:
            # Get cache directory info
            cache_dir = f"{self.config['steam_deck']['cache_directory']}/{app_id}"
            if os.path.exists(cache_dir):
                cache_size = await self._calculate_directory_size(cache_dir)
                cache_effectiveness["cache_size"] = cache_size
                
                # Estimate hit rate based on compilation events
                expected_shaders = 1000  # Estimated total shaders
                miss_count = len(compilation_events)
                hit_rate = max(0, (expected_shaders - miss_count) / expected_shaders)
                cache_effectiveness["cache_hit_rate"] = hit_rate
        
        except Exception as e:
            self.logger.error(f"Cache effectiveness analysis error: {e}")
        
        return cache_effectiveness
    
    async def _calculate_directory_size(self, directory: str) -> int:
        """Calculate total size of directory"""
        total_size = 0
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        except Exception as e:
            self.logger.error(f"Error calculating directory size: {e}")
        
        return total_size
    
    async def _identify_compilation_patterns(self, compilation_events: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in shader compilation"""
        patterns = []
        
        try:
            event_count = len(compilation_events)
            
            if event_count > 500:
                patterns.append("Excessive compilation - cache may be ineffective")
            elif event_count > 200:
                patterns.append("High compilation activity - optimization needed")
            elif event_count > 50:
                patterns.append("Moderate compilation - within normal range")
            else:
                patterns.append("Low compilation - good cache performance")
            
            # Analyze timing patterns
            timestamps = [event.get("timestamp") for event in compilation_events if event.get("timestamp")]
            if timestamps:
                timestamps.sort()
                if len(timestamps) > 1:
                    time_span = timestamps[-1] - timestamps[0]
                    if time_span < 60:  # All within 1 minute
                        patterns.append("Burst compilation pattern detected")
                    elif time_span > 600:  # Spread over 10+ minutes
                        patterns.append("Distributed compilation pattern detected")
        
        except Exception as e:
            self.logger.error(f"Pattern identification error: {e}")
        
        return patterns
    
    def _generate_compilation_recommendations(self, compilation_impact: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on compilation analysis"""
        recommendations = []
        
        try:
            event_count = compilation_impact.get("compilation_events_detected", 0)
            fps_drop = compilation_impact.get("performance_degradation", {}).get("fps_drop_percentage", 0)
            cache_hit_rate = compilation_impact.get("cache_effectiveness", {}).get("cache_hit_rate", 1.0)
            
            if event_count > 200:
                recommendations.append("Consider pre-compiling shaders for this game")
                recommendations.append("Verify shader cache is properly configured")
            
            if fps_drop > 20:
                recommendations.append("Enable shader pre-compilation in game settings")
                recommendations.append("Consider limiting frame rate during compilation")
            
            if cache_hit_rate < 0.7:
                recommendations.append("Shader cache may need rebuilding")
                recommendations.append("Check for cache corruption or invalidation")
            
            if not recommendations:
                recommendations.append("Shader compilation performance is within acceptable limits")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    async def _perform_stutter_analysis(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive stutter analysis across all scenarios"""
        stutter_analysis = {
            "total_stutter_events": 0,
            "stutter_severity_breakdown": {"minor": 0, "moderate": 0, "severe": 0},
            "average_stutter_duration": 0.0,
            "stutter_frequency": 0.0,
            "worst_scenario": "",
            "stutter_patterns": [],
            "improvement_recommendations": []
        }
        
        try:
            all_stutter_events = []
            scenario_stutter_counts = {}
            
            # Collect all stutter events
            for scenario, result in scenario_results.items():
                stutter_events = result.get("stutter_events", [])
                all_stutter_events.extend(stutter_events)
                scenario_stutter_counts[scenario] = len(stutter_events)
            
            stutter_analysis["total_stutter_events"] = len(all_stutter_events)
            
            if all_stutter_events:
                # Severity breakdown
                for event in all_stutter_events:
                    severity = event.severity
                    stutter_analysis["stutter_severity_breakdown"][severity] += 1
                
                # Average duration
                total_duration = sum(event.duration for event in all_stutter_events)
                stutter_analysis["average_stutter_duration"] = total_duration / len(all_stutter_events)
                
                # Worst scenario
                if scenario_stutter_counts:
                    worst_scenario = max(scenario_stutter_counts, key=scenario_stutter_counts.get)
                    stutter_analysis["worst_scenario"] = worst_scenario
                
                # Stutter frequency (events per minute)
                total_test_time = len(scenario_results) * 180  # Assume 3 minutes per scenario
                stutter_analysis["stutter_frequency"] = (len(all_stutter_events) / total_test_time) * 60
                
                # Generate improvement recommendations
                recommendations = self._generate_stutter_recommendations(stutter_analysis)
                stutter_analysis["improvement_recommendations"] = recommendations
        
        except Exception as e:
            self.logger.error(f"Stutter analysis error: {e}")
        
        return stutter_analysis
    
    def _generate_stutter_recommendations(self, stutter_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations to reduce stuttering"""
        recommendations = []
        
        severe_count = stutter_analysis["stutter_severity_breakdown"]["severe"]
        total_events = stutter_analysis["total_stutter_events"]
        avg_duration = stutter_analysis["average_stutter_duration"]
        frequency = stutter_analysis["stutter_frequency"]
        
        if severe_count > 5:
            recommendations.append("Critical: Address severe stuttering issues immediately")
            recommendations.append("Consider enabling V-Sync or frame rate limiting")
        
        if total_events > 50:
            recommendations.append("High stutter count detected - investigate shader compilation")
            recommendations.append("Consider pre-warming shader cache before gameplay")
        
        if avg_duration > 50:  # >50ms average stutter
            recommendations.append("Long stutter duration detected - check for background processes")
            recommendations.append("Monitor system resources during gameplay")
        
        if frequency > 2:  # More than 2 stutters per minute
            recommendations.append("High stutter frequency - consider graphics settings optimization")
        
        if not recommendations:
            recommendations.append("Stutter performance is within acceptable parameters")
        
        return recommendations
    
    async def _compare_with_baseline(self, app_id: str, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current performance with baseline results"""
        baseline_comparison = {
            "baseline_available": False,
            "performance_delta": {},
            "regression_detected": False,
            "improvement_detected": False,
            "comparison_details": {}
        }
        
        try:
            # Look for baseline results
            baseline_file = f"data/baselines/performance_{app_id}_baseline.json"
            
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                
                baseline_comparison["baseline_available"] = True
                
                # Compare overall performance metrics
                current_overall = current_results.get("overall_performance", {})
                baseline_overall = baseline_data.get("overall_performance", {})
                
                if current_overall and baseline_overall:
                    current_fps = current_overall.get("average_fps_all_scenarios", 0)
                    baseline_fps = baseline_overall.get("average_fps_all_scenarios", 0)
                    
                    if baseline_fps > 0:
                        fps_delta = ((current_fps - baseline_fps) / baseline_fps) * 100
                        baseline_comparison["performance_delta"]["fps_change_percent"] = fps_delta
                        
                        # Detect regression/improvement
                        if fps_delta < -5:  # >5% FPS drop
                            baseline_comparison["regression_detected"] = True
                        elif fps_delta > 5:  # >5% FPS improvement
                            baseline_comparison["improvement_detected"] = True
                    
                    # Compare stutter metrics
                    current_stutters = current_results.get("stutter_analysis", {}).get("total_stutter_events", 0)
                    baseline_stutters = baseline_data.get("stutter_analysis", {}).get("total_stutter_events", 0)
                    
                    stutter_delta = current_stutters - baseline_stutters
                    baseline_comparison["performance_delta"]["stutter_change"] = stutter_delta
        
        except Exception as e:
            self.logger.error(f"Baseline comparison error: {e}")
        
        return baseline_comparison
    
    def _determine_performance_status(self, analysis_result: Dict[str, Any]) -> str:
        """Determine overall performance status"""
        try:
            overall_perf = analysis_result.get("overall_performance", {})
            stutter_analysis = analysis_result.get("stutter_analysis", {})
            baseline_comparison = analysis_result.get("baseline_comparison", {})
            
            # Check for critical failures
            severe_stutters = stutter_analysis.get("stutter_severity_breakdown", {}).get("severe", 0)
            if severe_stutters > 10:
                return "critical_failure"
            
            # Check for regression
            if baseline_comparison.get("regression_detected", False):
                return "regression_detected"
            
            # Check average FPS
            avg_fps = overall_perf.get("average_fps_all_scenarios", 0)
            target_fps = self.target_fps
            
            if avg_fps >= target_fps * 0.90:
                return "passed"
            elif avg_fps >= target_fps * 0.70:
                return "acceptable"
            else:
                return "failed"
        
        except Exception:
            return "failed"
    
    async def _shutdown_game_monitoring(self, game_process: subprocess.Popen):
        """Clean shutdown of game and monitoring processes"""
        try:
            # Shutdown game process
            game_process.terminate()
            await asyncio.sleep(5)
            if game_process.poll() is None:
                game_process.kill()
            
            # Shutdown monitoring processes
            if hasattr(self, 'monitoring_processes'):
                for process in self.monitoring_processes.values():
                    try:
                        process.terminate()
                        await asyncio.sleep(2)
                        if process.poll() is None:
                            process.kill()
                    except Exception:
                        pass
        
        except Exception as e:
            self.logger.error(f"Error during monitoring shutdown: {e}")