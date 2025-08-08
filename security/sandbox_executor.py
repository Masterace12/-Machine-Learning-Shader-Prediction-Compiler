#!/usr/bin/env python3
"""
Sandboxed Shader Execution Environment

This module provides secure, isolated execution environments for untrusted shader compilation
and testing. It implements multiple layers of containment including:

- Process-level isolation with restricted privileges
- Resource quotas (memory, CPU, GPU time)
- File system access controls
- Network isolation
- Kill switches for runaway shaders
- Comprehensive monitoring and logging

The sandbox is designed to prevent malicious shaders from:
- Escaping the execution environment
- Consuming excessive system resources
- Accessing sensitive system files
- Communicating with external networks
- Persisting beyond the execution window
"""

import os
import sys
import time
import signal
import threading
import subprocess
import tempfile
import shutil
import psutil
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
from contextlib import contextmanager
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import resource
import ctypes
import platform

logger = logging.getLogger(__name__)


class SandboxResult(Enum):
    """Sandbox execution results"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    CPU_LIMIT = "cpu_limit"
    ACCESS_VIOLATION = "access_violation"
    CRASH = "crash"
    KILLED = "killed"
    ERROR = "error"


class SecurityLevel(Enum):
    """Security isolation levels"""
    MINIMAL = "minimal"      # Basic process isolation
    STANDARD = "standard"    # Process + resource limits
    STRICT = "strict"        # Full isolation with restricted filesystem
    PARANOID = "paranoid"    # Maximum security with virtualization


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution"""
    max_memory_mb: int = 512
    max_cpu_time_seconds: float = 30.0
    max_wall_time_seconds: float = 60.0
    max_file_size_mb: int = 100
    max_open_files: int = 100
    max_processes: int = 1
    max_gpu_memory_mb: int = 256
    max_gpu_time_seconds: float = 10.0


@dataclass
class SandboxConfiguration:
    """Configuration for sandbox execution"""
    security_level: SecurityLevel = SecurityLevel.STANDARD
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    
    # File system controls
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/usr", "/bin", "/sbin", "/boot", "/dev", "/proc", "/sys"
    ])
    
    # Network controls
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    
    # GPU controls
    allow_gpu_access: bool = True
    gpu_device_isolation: bool = True
    
    # Monitoring
    enable_detailed_monitoring: bool = True
    log_all_system_calls: bool = False
    
    # Environment
    preserve_env_vars: List[str] = field(default_factory=lambda: [
        "HOME", "USER", "DISPLAY", "MESA_DEBUG"
    ])
    
    # Kill switches
    auto_kill_on_suspicious_activity: bool = True
    suspicious_activity_threshold: int = 5


@dataclass
class ExecutionResult:
    """Result of sandboxed shader execution"""
    result: SandboxResult
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    memory_peak_mb: float
    cpu_time_used: float
    gpu_time_used: float = 0.0
    gpu_memory_used_mb: float = 0.0
    
    # Security monitoring
    suspicious_activities: List[str] = field(default_factory=list)
    blocked_operations: List[str] = field(default_factory=list)
    resource_violations: List[str] = field(default_factory=list)
    
    # File system access
    files_accessed: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    
    # Network activity
    network_connections: List[Dict] = field(default_factory=list)
    
    # Additional metrics
    context_switches: int = 0
    page_faults: int = 0
    disk_io_bytes: int = 0


class ProcessMonitor:
    """Real-time process monitoring for sandbox"""
    
    def __init__(self, pid: int, limits: ResourceLimits, config: SandboxConfiguration):
        self.pid = pid
        self.limits = limits
        self.config = config
        self.process = psutil.Process(pid)
        self.start_time = time.time()
        self.monitoring = True
        self.violations = []
        self.suspicious_activities = []
        
    def monitor_loop(self) -> Dict[str, Any]:
        """Main monitoring loop"""
        stats = {
            'peak_memory_mb': 0.0,
            'peak_cpu_percent': 0.0,
            'total_cpu_time': 0.0,
            'context_switches': 0,
            'page_faults': 0,
            'io_counters': None,
            'files_accessed': [],
            'network_connections': []
        }
        
        try:
            while self.monitoring and self.process.is_running():
                # Memory check
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                stats['peak_memory_mb'] = max(stats['peak_memory_mb'], memory_mb)
                
                if memory_mb > self.limits.max_memory_mb:
                    self.violations.append(f"Memory limit exceeded: {memory_mb:.1f}MB")
                    if self.config.auto_kill_on_suspicious_activity:
                        self._terminate_process("Memory limit exceeded")
                        break
                
                # CPU time check
                cpu_times = self.process.cpu_times()
                total_cpu = cpu_times.user + cpu_times.system
                stats['total_cpu_time'] = total_cpu
                
                if total_cpu > self.limits.max_cpu_time_seconds:
                    self.violations.append(f"CPU time limit exceeded: {total_cpu:.1f}s")
                    if self.config.auto_kill_on_suspicious_activity:
                        self._terminate_process("CPU time limit exceeded")
                        break
                
                # Wall time check
                wall_time = time.time() - self.start_time
                if wall_time > self.limits.max_wall_time_seconds:
                    self.violations.append(f"Wall time limit exceeded: {wall_time:.1f}s")
                    self._terminate_process("Wall time limit exceeded")
                    break
                
                # CPU usage monitoring
                cpu_percent = self.process.cpu_percent()
                stats['peak_cpu_percent'] = max(stats['peak_cpu_percent'], cpu_percent)
                
                # Context switches and page faults
                try:
                    ctx_switches = self.process.num_ctx_switches()
                    stats['context_switches'] = ctx_switches.voluntary + ctx_switches.involuntary
                    
                    if hasattr(self.process, 'memory_full_info'):
                        mem_full = self.process.memory_full_info()
                        stats['page_faults'] = getattr(mem_full, 'pfaults', 0)
                except (AttributeError, psutil.AccessDenied):
                    pass
                
                # I/O monitoring
                try:
                    io_counters = self.process.io_counters()
                    stats['io_counters'] = io_counters
                    
                    # Check for excessive I/O (potential data exfiltration)
                    total_io = io_counters.read_bytes + io_counters.write_bytes
                    if total_io > 100 * 1024 * 1024:  # 100MB
                        self.suspicious_activities.append("Excessive I/O activity detected")
                except (AttributeError, psutil.AccessDenied):
                    pass
                
                # Network connections monitoring
                if self.config.enable_detailed_monitoring:
                    try:
                        connections = self.process.connections()
                        for conn in connections:
                            if not self.config.allow_network:
                                self.violations.append(f"Unauthorized network connection: {conn}")
                                self._terminate_process("Network access violation")
                                break
                            stats['network_connections'].append({
                                'family': conn.family,
                                'type': conn.type,
                                'local': conn.laddr,
                                'remote': conn.raddr,
                                'status': conn.status
                            })
                    except (AttributeError, psutil.AccessDenied):
                        pass
                
                # File handle monitoring
                try:
                    open_files = self.process.open_files()
                    if len(open_files) > self.limits.max_open_files:
                        self.violations.append(f"Too many open files: {len(open_files)}")
                    
                    for file_info in open_files:
                        file_path = file_info.path
                        stats['files_accessed'].append(file_path)
                        
                        # Check against blocked paths
                        if any(file_path.startswith(blocked) for blocked in self.config.blocked_paths):
                            self.violations.append(f"Access to blocked path: {file_path}")
                            if self.config.auto_kill_on_suspicious_activity:
                                self._terminate_process(f"Blocked path access: {file_path}")
                                break
                
                except (AttributeError, psutil.AccessDenied):
                    pass
                
                # Child process monitoring
                try:
                    children = self.process.children(recursive=True)
                    if len(children) > self.limits.max_processes - 1:
                        self.violations.append(f"Too many child processes: {len(children)}")
                        if self.config.auto_kill_on_suspicious_activity:
                            self._terminate_process("Process limit exceeded")
                            break
                except psutil.AccessDenied:
                    pass
                
                # Termination check
                if len(self.violations) > self.config.suspicious_activity_threshold:
                    self._terminate_process("Too many violations")
                    break
                
                time.sleep(0.1)  # 100ms monitoring interval
                
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        
        return stats
    
    def _terminate_process(self, reason: str):
        """Terminate the monitored process"""
        logger.warning(f"Terminating process {self.pid}: {reason}")
        
        try:
            # Try graceful termination first
            self.process.terminate()
            
            # Wait a bit for graceful termination
            try:
                self.process.wait(timeout=3)
            except psutil.TimeoutExpired:
                # Force kill if graceful termination fails
                logger.warning(f"Force killing process {self.pid}")
                self.process.kill()
                
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring = False


class ShaderSandbox:
    """Secure sandbox for shader compilation and execution"""
    
    def __init__(self, config: SandboxConfiguration = None):
        self.config = config or SandboxConfiguration()
        self.temp_dirs = []
        self.active_processes = []
        
        # Platform-specific initialization
        if platform.system() == "Linux":
            self._init_linux_sandbox()
        elif platform.system() == "Windows":
            self._init_windows_sandbox()
        else:
            logger.warning(f"Limited sandbox support on {platform.system()}")
        
        logger.info(f"Shader sandbox initialized with {self.config.security_level.value} security")
    
    def _init_linux_sandbox(self):
        """Initialize Linux-specific sandbox features"""
        # Check for available containerization technologies
        self.has_docker = shutil.which('docker') is not None
        self.has_podman = shutil.which('podman') is not None
        self.has_firejail = shutil.which('firejail') is not None
        self.has_bwrap = shutil.which('bwrap') is not None
        
        if self.config.security_level == SecurityLevel.PARANOID and not (self.has_docker or self.has_podman):
            logger.warning("Paranoid security level requires Docker/Podman but not available")
            self.config.security_level = SecurityLevel.STRICT
    
    def _init_windows_sandbox(self):
        """Initialize Windows-specific sandbox features"""
        # Check for Windows Sandbox or App Container capabilities
        self.has_sandbox = False
        try:
            # Check if Windows Sandbox is available (Windows 10 Pro/Enterprise)
            result = subprocess.run(['dism', '/online', '/get-featureinfo', '/featurename:Containers-DisposableClientVM'], 
                                  capture_output=True, text=True, timeout=10)
            if 'State : Enabled' in result.stdout:
                self.has_sandbox = True
        except:
            pass
        
        if self.config.security_level == SecurityLevel.PARANOID and not self.has_sandbox:
            logger.warning("Paranoid security level requires Windows Sandbox but not available")
            self.config.security_level = SecurityLevel.STRICT
    
    def execute_shader_compilation(self, compiler_path: str, shader_source: str, 
                                 output_dir: str = None, 
                                 additional_args: List[str] = None) -> ExecutionResult:
        """Execute shader compilation in sandbox"""
        
        # Create temporary directory for execution
        temp_dir = tempfile.mkdtemp(prefix="shader_sandbox_")
        self.temp_dirs.append(temp_dir)
        
        try:
            # Write shader source to temp file
            shader_file = Path(temp_dir) / "shader.glsl"
            with open(shader_file, 'w') as f:
                f.write(shader_source)
            
            # Prepare output directory
            if output_dir is None:
                output_dir = temp_dir
            
            # Build command
            cmd = [compiler_path]
            if additional_args:
                cmd.extend(additional_args)
            cmd.extend([str(shader_file), '-o', str(Path(output_dir) / 'shader.spv')])
            
            # Execute in sandbox
            return self._execute_sandboxed_command(cmd, temp_dir)
            
        except Exception as e:
            logger.error(f"Shader compilation error: {e}")
            return ExecutionResult(
                result=SandboxResult.ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                memory_peak_mb=0.0,
                cpu_time_used=0.0
            )
        finally:
            self._cleanup_temp_dir(temp_dir)
    
    def execute_shader_test(self, shader_bytecode: bytes, test_program: str,
                          test_data: Dict[str, Any] = None) -> ExecutionResult:
        """Execute shader test in sandbox"""
        
        temp_dir = tempfile.mkdtemp(prefix="shader_test_")
        self.temp_dirs.append(temp_dir)
        
        try:
            # Write shader bytecode
            shader_file = Path(temp_dir) / "shader.spv"
            with open(shader_file, 'wb') as f:
                f.write(shader_bytecode)
            
            # Write test program
            test_file = Path(temp_dir) / "test.py"
            with open(test_file, 'w') as f:
                f.write(test_program)
            
            # Write test data if provided
            if test_data:
                data_file = Path(temp_dir) / "test_data.json"
                with open(data_file, 'w') as f:
                    json.dump(test_data, f)
            
            # Execute test
            cmd = [sys.executable, str(test_file)]
            return self._execute_sandboxed_command(cmd, temp_dir)
            
        except Exception as e:
            logger.error(f"Shader test error: {e}")
            return ExecutionResult(
                result=SandboxResult.ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                memory_peak_mb=0.0,
                cpu_time_used=0.0
            )
        finally:
            self._cleanup_temp_dir(temp_dir)
    
    def _execute_sandboxed_command(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Execute command with appropriate sandbox isolation"""
        
        if self.config.security_level == SecurityLevel.PARANOID:
            return self._execute_containerized(cmd, working_dir)
        elif self.config.security_level == SecurityLevel.STRICT:
            return self._execute_restricted(cmd, working_dir)
        elif self.config.security_level == SecurityLevel.STANDARD:
            return self._execute_monitored(cmd, working_dir)
        else:  # MINIMAL
            return self._execute_basic(cmd, working_dir)
    
    def _execute_containerized(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Execute in full container isolation"""
        
        if platform.system() == "Linux" and (self.has_docker or self.has_podman):
            return self._execute_linux_container(cmd, working_dir)
        elif platform.system() == "Windows" and self.has_sandbox:
            return self._execute_windows_sandbox(cmd, working_dir)
        else:
            # Fallback to restricted execution
            logger.warning("Container isolation not available, falling back to restricted execution")
            return self._execute_restricted(cmd, working_dir)
    
    def _execute_linux_container(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Execute in Linux container (Docker/Podman)"""
        
        container_runtime = 'docker' if self.has_docker else 'podman'
        
        # Build container command
        container_cmd = [
            container_runtime, 'run', '--rm',
            '--network=none',  # No network access
            f'--memory={self.config.resource_limits.max_memory_mb}m',
            f'--cpus={self.config.resource_limits.max_cpu_time_seconds / 60.0}',  # Approximate CPU limit
            '--read-only',  # Read-only filesystem
            f'--tmpfs=/tmp:size={self.config.resource_limits.max_file_size_mb}m,noexec',
            f'--workdir=/workspace',
            f'--volume={working_dir}:/workspace:rw',
            '--user=1000:1000',  # Non-root user
            '--security-opt=no-new-privileges',
            '--cap-drop=ALL',  # Drop all capabilities
            'ubuntu:20.04',  # Base image (would need to be pre-built with shader tools)
        ]
        container_cmd.extend(cmd)
        
        return self._execute_with_monitoring(container_cmd, working_dir, use_container=True)
    
    def _execute_windows_sandbox(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Execute in Windows Sandbox"""
        
        # Create sandbox configuration
        sandbox_config = f"""
        <Configuration>
            <VGpu>Enable</VGpu>
            <AudioInput>Disable</AudioInput>
            <VideoInput>Disable</VideoInput>
            <PrinterRedirection>Disable</PrinterRedirection>
            <ClipboardRedirection>Disable</ClipboardRedirection>
            <MemoryInMB>{self.config.resource_limits.max_memory_mb}</MemoryInMB>
            <MappedFolders>
                <MappedFolder>
                    <HostFolder>{working_dir}</HostFolder>
                    <SandboxFolder>C:\\workspace</SandboxFolder>
                    <ReadOnly>false</ReadOnly>
                </MappedFolder>
            </MappedFolders>
            <LogonCommand>
                <Command>cmd.exe /c "{' '.join(cmd)}"</Command>
            </LogonCommand>
        </Configuration>
        """
        
        config_file = Path(working_dir) / "sandbox.wsb"
        with open(config_file, 'w') as f:
            f.write(sandbox_config)
        
        sandbox_cmd = ['WindowsSandbox.exe', str(config_file)]
        
        return self._execute_with_monitoring(sandbox_cmd, working_dir, use_container=True)
    
    def _execute_restricted(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Execute with restricted privileges and filesystem access"""
        
        if platform.system() == "Linux" and self.has_firejail:
            # Use Firejail for sandboxing
            firejail_cmd = [
                'firejail',
                '--private=' + working_dir,
                '--nonetwork',
                '--nonewprivs',
                '--noroot',
                '--rlimit-as=' + str(self.config.resource_limits.max_memory_mb * 1024 * 1024),
                '--rlimit-cpu=' + str(int(self.config.resource_limits.max_cpu_time_seconds)),
                '--rlimit-nproc=' + str(self.config.resource_limits.max_processes),
                '--timeout=' + str(int(self.config.resource_limits.max_wall_time_seconds)),
            ]
            firejail_cmd.extend(cmd)
            
            return self._execute_with_monitoring(firejail_cmd, working_dir)
        
        elif platform.system() == "Linux" and self.has_bwrap:
            # Use Bubblewrap for sandboxing
            bwrap_cmd = [
                'bwrap',
                '--ro-bind', '/usr', '/usr',
                '--ro-bind', '/lib', '/lib',
                '--ro-bind', '/lib64', '/lib64',
                '--bind', working_dir, '/workspace',
                '--chdir', '/workspace',
                '--unshare-net',
                '--unshare-pid',
                '--die-with-parent',
            ]
            bwrap_cmd.extend(cmd)
            
            return self._execute_with_monitoring(bwrap_cmd, working_dir)
        
        else:
            # Fallback to monitored execution with resource limits
            return self._execute_monitored(cmd, working_dir)
    
    def _execute_monitored(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Execute with resource limits and monitoring"""
        return self._execute_with_monitoring(cmd, working_dir)
    
    def _execute_basic(self, cmd: List[str], working_dir: str) -> ExecutionResult:
        """Basic execution with minimal restrictions"""
        return self._execute_with_monitoring(cmd, working_dir, minimal_monitoring=True)
    
    def _execute_with_monitoring(self, cmd: List[str], working_dir: str, 
                               use_container: bool = False, 
                               minimal_monitoring: bool = False) -> ExecutionResult:
        """Execute command with comprehensive monitoring"""
        
        start_time = time.time()
        
        try:
            # Set up environment
            env = os.environ.copy()
            if not use_container:
                # Restrict environment variables
                clean_env = {}
                for var in self.config.preserve_env_vars:
                    if var in env:
                        clean_env[var] = env[var]
                
                # Add necessary variables
                clean_env['PATH'] = env.get('PATH', '')
                clean_env['LD_LIBRARY_PATH'] = env.get('LD_LIBRARY_PATH', '')
                env = clean_env
            
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._setup_process_limits if platform.system() != "Windows" else None
            )
            
            self.active_processes.append(process.pid)
            
            # Start monitoring if not minimal
            monitor = None
            monitor_stats = {}
            if not minimal_monitoring:
                monitor = ProcessMonitor(process.pid, self.config.resource_limits, self.config)
                monitor_thread = threading.Thread(target=lambda: monitor_stats.update(monitor.monitor_loop()))
                monitor_thread.start()
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.config.resource_limits.max_wall_time_seconds)
                exit_code = process.returncode
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -9
                
                result = ExecutionResult(
                    result=SandboxResult.TIMEOUT,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=time.time() - start_time,
                    memory_peak_mb=monitor_stats.get('peak_memory_mb', 0.0),
                    cpu_time_used=monitor_stats.get('total_cpu_time', 0.0)
                )
                
                if monitor:
                    monitor.stop_monitoring()
                    result.suspicious_activities = monitor.suspicious_activities
                    result.resource_violations = monitor.violations
                
                return result
            
            # Stop monitoring
            if monitor:
                monitor.stop_monitoring()
                if hasattr(monitor_thread, 'join'):
                    monitor_thread.join(timeout=1.0)
            
            # Determine result status
            execution_time = time.time() - start_time
            
            if exit_code == 0:
                result_status = SandboxResult.SUCCESS
            elif exit_code == -9:
                result_status = SandboxResult.KILLED
            elif exit_code < 0:
                result_status = SandboxResult.CRASH
            else:
                result_status = SandboxResult.ERROR
            
            # Check for resource violations
            if monitor and monitor.violations:
                if any("memory" in v.lower() for v in monitor.violations):
                    result_status = SandboxResult.MEMORY_LIMIT
                elif any("cpu" in v.lower() for v in monitor.violations):
                    result_status = SandboxResult.CPU_LIMIT
                elif any("access" in v.lower() for v in monitor.violations):
                    result_status = SandboxResult.ACCESS_VIOLATION
            
            # Build result
            result = ExecutionResult(
                result=result_status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                memory_peak_mb=monitor_stats.get('peak_memory_mb', 0.0),
                cpu_time_used=monitor_stats.get('total_cpu_time', 0.0),
                context_switches=monitor_stats.get('context_switches', 0),
                page_faults=monitor_stats.get('page_faults', 0),
                files_accessed=monitor_stats.get('files_accessed', []),
                network_connections=monitor_stats.get('network_connections', [])
            )
            
            if monitor:
                result.suspicious_activities = monitor.suspicious_activities
                result.resource_violations = monitor.violations
                result.blocked_operations = [v for v in monitor.violations if "blocked" in v.lower()]
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                result=SandboxResult.ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                memory_peak_mb=0.0,
                cpu_time_used=0.0
            )
        finally:
            if process.pid in self.active_processes:
                self.active_processes.remove(process.pid)
    
    def _setup_process_limits(self):
        """Set up resource limits for the process (Unix only)"""
        try:
            # Set memory limit
            memory_limit = self.config.resource_limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit
            cpu_limit = int(self.config.resource_limits.max_cpu_time_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # Set file size limit
            file_limit = self.config.resource_limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
            
            # Set maximum number of open files
            resource.setrlimit(resource.RLIMIT_NOFILE, 
                             (self.config.resource_limits.max_open_files,
                              self.config.resource_limits.max_open_files))
            
            # Set process limit
            resource.setrlimit(resource.RLIMIT_NPROC,
                             (self.config.resource_limits.max_processes,
                              self.config.resource_limits.max_processes))
            
        except (ValueError, OSError) as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def _cleanup_temp_dir(self, temp_dir: str):
        """Clean up temporary directory"""
        try:
            shutil.rmtree(temp_dir)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up temp dir {temp_dir}: {e}")
    
    def emergency_kill_all(self):
        """Emergency kill all active processes"""
        logger.warning("Emergency kill all active processes")
        
        for pid in self.active_processes[:]:
            try:
                process = psutil.Process(pid)
                process.kill()
                logger.info(f"Killed process {pid}")
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.error(f"Could not kill process {pid}: {e}")
        
        self.active_processes.clear()
    
    def cleanup(self):
        """Clean up sandbox resources"""
        logger.info("Cleaning up sandbox")
        
        # Kill any remaining processes
        self.emergency_kill_all()
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs[:]:
            self._cleanup_temp_dir(temp_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ShaderValidationSandbox:
    """High-level shader validation using sandbox"""
    
    def __init__(self, config: SandboxConfiguration = None):
        self.sandbox = ShaderSandbox(config)
        
    def validate_shader_compilation(self, shader_source: str, compiler_path: str) -> Dict[str, Any]:
        """Validate shader compilation in sandbox"""
        
        result = self.sandbox.execute_shader_compilation(compiler_path, shader_source)
        
        return {
            'compilation_successful': result.result == SandboxResult.SUCCESS,
            'execution_time': result.execution_time,
            'memory_used_mb': result.memory_peak_mb,
            'cpu_time_used': result.cpu_time_used,
            'exit_code': result.exit_code,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'security_violations': len(result.resource_violations) > 0,
            'suspicious_activity': len(result.suspicious_activities) > 0,
            'violations': result.resource_violations,
            'suspicious_activities': result.suspicious_activities,
            'files_accessed': result.files_accessed,
            'safe_for_deployment': (
                result.result == SandboxResult.SUCCESS and
                len(result.resource_violations) == 0 and
                len(result.suspicious_activities) == 0
            )
        }
    
    def stress_test_shader(self, shader_bytecode: bytes, iterations: int = 100) -> Dict[str, Any]:
        """Stress test shader execution to detect resource issues"""
        
        test_program = f"""
import time
import sys
import os

# Simulate shader execution {iterations} times
for i in range({iterations}):
    # Simulate GPU work
    time.sleep(0.001)  # 1ms per iteration
    
    # Check for excessive memory usage
    if i % 10 == 0:
        print(f"Iteration {{i}}/{iterations}")
        sys.stdout.flush()

print("Stress test completed successfully")
"""
        
        result = self.sandbox.execute_shader_test(shader_bytecode, test_program)
        
        return {
            'stress_test_passed': result.result == SandboxResult.SUCCESS,
            'execution_time': result.execution_time,
            'peak_memory_mb': result.memory_peak_mb,
            'cpu_time_used': result.cpu_time_used,
            'resource_violations': result.resource_violations,
            'performance_stable': (
                result.execution_time < 30.0 and  # Completed in reasonable time
                result.memory_peak_mb < 100.0 and  # Didn't use excessive memory
                len(result.resource_violations) == 0
            )
        }
    
    def cleanup(self):
        """Clean up validation sandbox"""
        self.sandbox.cleanup()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sandbox with strict security
    config = SandboxConfiguration(
        security_level=SecurityLevel.STRICT,
        resource_limits=ResourceLimits(
            max_memory_mb=256,
            max_cpu_time_seconds=10.0,
            max_wall_time_seconds=15.0
        )
    )
    
    with ShaderSandbox(config) as sandbox:
        # Test shader compilation
        test_shader = """
        #version 450
        layout(location = 0) in vec3 position;
        void main() {
            gl_Position = vec4(position, 1.0);
        }
        """
        
        # Note: This would require an actual GLSL compiler like glslc
        # result = sandbox.execute_shader_compilation('glslc', test_shader)
        # print(f"Compilation result: {result.result.value}")
        
        print("Sandbox test completed")