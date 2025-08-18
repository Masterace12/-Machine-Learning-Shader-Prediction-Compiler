#!/usr/bin/env python3
"""
Pytest configuration and fixtures for shader-predict-compile tests
"""

import pytest
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any
import time
import json
import numpy as np

# Test data and fixtures
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(scope="session")
def test_config():
    """Test configuration data"""
    return {
        "ml": {
            "model_type": "lightgbm",
            "max_memory_mb": 50,
            "enable_async": True,
            "cache_size": 100
        },
        "thermal": {
            "monitoring_interval": 0.1,  # Fast for testing
            "prediction_interval": 1.0,
            "temp_limits": {
                "apu_max": 95.0,
                "cpu_max": 85.0,
                "gpu_max": 90.0
            }
        },
        "cache": {
            "hot_cache_size": 10,
            "warm_cache_size": 50,
            "max_memory_mb": 20,
            "enable_compression": True
        }
    }


@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(prefix="shader_test_"))
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def clean_temp_dir(temp_dir):
    """Clean temporary directory for each test"""
    test_dir = temp_dir / f"test_{int(time.time() * 1000000)}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def mock_steam_deck_sensors():
    """Mock Steam Deck thermal sensors"""
    sensors = {
        "k10temp_temp1_input": 75.5,  # APU temperature
        "amdgpu_temp1_input": 68.2,   # GPU temperature
        "jupiter_fan1_input": 2500,    # Fan RPM
    }
    return sensors


@pytest.fixture(scope="function")
def mock_thermal_manager(mock_steam_deck_sensors):
    """Mock thermal manager for testing"""
    from src.thermal.optimized_thermal_manager import OptimizedThermalManager, ThermalSample, ThermalState
    
    manager = OptimizedThermalManager()
    
    # Mock sensor reading
    def mock_read_sensors():
        return mock_steam_deck_sensors
    
    manager._read_sensors = mock_read_sensors
    
    # Mock hardware detection
    from src.thermal.optimized_thermal_manager import SteamDeckModel
    manager.steam_deck_model = SteamDeckModel.LCD
    
    return manager


@pytest.fixture(scope="function")
def sample_shader_features():
    """Sample shader features for testing"""
    try:
        from src.ml.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
        
        return UnifiedShaderFeatures(
            shader_hash="test_shader_123456",
            shader_type=ShaderType.FRAGMENT,
            instruction_count=500,
            register_usage=32,
            texture_samples=4,
            memory_operations=10,
            control_flow_complexity=5,
            wave_size=64,
            uses_derivatives=True,
            uses_tessellation=False,
            uses_geometry_shader=False,
            optimization_level=3,
            cache_priority=0.8
        )
    except ImportError:
        # Fallback mock object
        mock_features = Mock()
        mock_features.shader_hash = "test_shader_123456"
        mock_features.shader_type = Mock()
        mock_features.shader_type.value = "fragment"
        mock_features.instruction_count = 500
        mock_features.register_usage = 32
        mock_features.texture_samples = 4
        mock_features.memory_operations = 10
        mock_features.control_flow_complexity = 5
        mock_features.wave_size = 64
        mock_features.uses_derivatives = True
        mock_features.uses_tessellation = False
        mock_features.uses_geometry_shader = False
        mock_features.optimization_level = 3
        mock_features.cache_priority = 0.8
        return mock_features


@pytest.fixture(scope="function")
def sample_cache_entry(clean_temp_dir):
    """Sample shader cache entry for testing"""
    try:
        from src.cache.optimized_shader_cache import ShaderCacheEntry
        
        return ShaderCacheEntry(
            shader_hash="test_cache_entry_789",
            game_id="game_001",
            shader_type="fragment",
            bytecode=b"mock_shader_bytecode" * 100,  # Some test data
            compilation_time=15.5,
            access_count=5,
            last_access=time.time(),
            size_bytes=2000,
            compression_ratio=2.0,
            priority=10.5
        )
    except ImportError:
        # Fallback mock object
        mock_entry = Mock()
        mock_entry.shader_hash = "test_cache_entry_789"
        mock_entry.game_id = "game_001"
        mock_entry.shader_type = "fragment"
        mock_entry.bytecode = b"mock_shader_bytecode" * 100
        mock_entry.compilation_time = 15.5
        mock_entry.access_count = 5
        mock_entry.last_access = time.time()
        mock_entry.size_bytes = 2000
        mock_entry.compression_ratio = 2.0
        mock_entry.priority = 10.5
        return mock_entry


@pytest.fixture(scope="function")
def mock_ml_model():
    """Mock ML model for testing"""
    model = Mock()
    model.predict = Mock(return_value=np.array([12.5]))
    model.fit = Mock()
    model.n_estimators = 50
    return model


@pytest.fixture(scope="function")
def performance_data():
    """Sample performance data for benchmarks"""
    return {
        "prediction_times": np.random.exponential(5.0, 1000),  # milliseconds
        "cache_hit_rates": np.random.beta(8, 2, 100),  # 0-1 range
        "memory_usage": np.random.normal(50, 10, 100),  # MB
        "thermal_readings": np.random.normal(75, 5, 500)  # °C
    }


@pytest.fixture(scope="function")
def steam_deck_hardware_mock():
    """Mock Steam Deck hardware detection"""
    hardware_info = {
        "model": "lcd",
        "apu": "van_gogh",
        "memory": "16GB LPDDR5",
        "storage": "512GB NVMe",
        "display": "1280x800 LCD",
        "battery": "40Wh",
        "thermal_design_power": "15W"
    }
    return hardware_info


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Suppress verbose third-party logs during testing
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


@pytest.fixture(scope="function")
def benchmark_environment():
    """Setup environment for benchmarking"""
    import psutil
    import gc
    
    # Clean up memory before benchmarks
    gc.collect()
    
    initial_memory = psutil.Process().memory_info().rss
    
    yield {
        "initial_memory": initial_memory,
        "pid": psutil.Process().pid
    }
    
    # Clean up after benchmarks
    gc.collect()


class MockSteamProcess:
    """Mock Steam process for integration tests"""
    
    def __init__(self, game_id: str = "480"):
        self.game_id = game_id
        self.running = False
        self.pid = 12345
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def is_running(self):
        return self.running


@pytest.fixture(scope="function")
def mock_steam_process():
    """Mock Steam game process"""
    return MockSteamProcess()


# Custom pytest markers and hooks

def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "requires_steamdeck: mark test as requiring Steam Deck hardware")
    config.addinivalue_line("markers", "memory_intensive: mark test as memory intensive")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    # Skip GPU tests if no GPU available
    gpu_marker = pytest.mark.skip(reason="GPU not available")
    steamdeck_marker = pytest.mark.skip(reason="Steam Deck hardware not available")
    
    for item in items:
        if "requires_gpu" in item.keywords:
            # Check for GPU availability (simplified)
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(gpu_marker)
            except ImportError:
                item.add_marker(gpu_marker)
        
        if "requires_steamdeck" in item.keywords:
            # Check for Steam Deck hardware
            if not Path("/sys/class/dmi/id/product_name").exists():
                item.add_marker(steamdeck_marker)


@pytest.fixture(scope="function")
def async_test_environment():
    """Setup for async tests"""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    yield loop
    
    loop.close()


# Utility functions for tests

def assert_performance_within_bounds(value: float, expected: float, tolerance: float = 0.1):
    """Assert that a performance metric is within acceptable bounds"""
    assert abs(value - expected) <= expected * tolerance, (
        f"Performance metric {value} not within {tolerance*100}% of expected {expected}"
    )


def create_test_shader_data(count: int = 100):
    """Create test shader data for ML training"""
    data = []
    for i in range(count):
        data.append({
            "shader_hash": f"test_shader_{i:04d}",
            "instruction_count": np.random.randint(100, 2000),
            "register_usage": np.random.randint(8, 64),
            "texture_samples": np.random.randint(0, 16),
            "compilation_time": np.random.exponential(10.0),
            "success": np.random.random() > 0.1
        })
    return data


# Benchmark fixtures

@pytest.fixture
def benchmark_data():
    """Data for benchmark tests"""
    return create_test_shader_data(1000)


# Memory profiling fixtures

@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests"""
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    yield process
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
    
    if memory_increase > 50:  # More than 50MB increase
        pytest.fail(f"Memory leak detected: {memory_increase:.1f}MB increase")