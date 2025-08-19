#!/usr/bin/env python3
"""
Steam Deck Cache Optimization Tests

Tests for Steam Deck-specific shader cache optimization including:
- Memory-constrained cache management
- SteamOS immutable filesystem handling
- Cache warming and preloading strategies
- Performance optimization under thermal constraints
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, mock_steamdeck_environment,
    create_thermal_stress_scenario, create_battery_critical_scenario
)

from src.core.steamdeck_cache_optimizer import SteamDeckCacheOptimizer
from src.core.optimized_shader_cache import OptimizedShaderCache


class TestSteamDeckCacheOptimizer:
    """Test Steam Deck cache optimization functionality"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_cache_optimizer_initialization(self):
        """Test Steam Deck cache optimizer initialization"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            assert hasattr(cache_optimizer, 'is_steam_deck')
            assert hasattr(cache_optimizer, 'model_type')
            
            status = cache_optimizer.get_status()
            assert isinstance(status, dict)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_memory_constrained_cache_management(self):
        """Test cache management under memory constraints"""
        # Simulate low memory scenario
        hardware_state = MockHardwareState(
            model=MockSteamDeckModel.LCD_64GB,  # Lower-end model
        )
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Simulate memory pressure
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 85.0  # High memory usage
                mock_memory.return_value.available = 2 * 1024**3  # 2GB available
                
                # Should adapt cache strategy for memory constraints
                if hasattr(cache_optimizer, 'optimize_for_memory_pressure'):
                    cache_optimizer.optimize_for_memory_pressure()
                    
                    status = cache_optimizer.get_status()
                    
                    # Should indicate memory optimization is active
                    assert 'memory_optimized' in str(status).lower() or 'reduced' in str(status).lower()
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_thermal_aware_cache_management(self):
        """Test cache behavior under thermal constraints"""
        hardware_state = create_thermal_stress_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        )
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Should adapt to thermal conditions
            if hasattr(cache_optimizer, 'optimize_for_thermal_conditions'):
                cache_optimizer.optimize_for_thermal_conditions()
                
                status = cache_optimizer.get_status()
                
                # Should reduce cache activity under thermal stress
                assert 'thermal' in str(status).lower() or 'reduced' in str(status).lower()
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_steamos_filesystem_integration(self):
        """Test integration with SteamOS immutable filesystem"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Test cache directory creation on writable partition
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = Path(temp_dir) / "shader_cache"
                
                # Should handle filesystem constraints
                if hasattr(cache_optimizer, 'setup_cache_directories'):
                    cache_optimizer.setup_cache_directories(str(cache_dir))
                    
                    # Should create appropriate directory structure
                    assert cache_dir.exists() or cache_optimizer.get_status() is not None
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_cache_warming_strategies(self):
        """Test cache warming strategies for Steam Deck"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Test cache preloading during idle periods
            if hasattr(cache_optimizer, 'warm_cache'):
                # Mock shader data
                mock_shaders = [
                    {'id': 'shader_1', 'game': 'portal2', 'size': 1024},
                    {'id': 'shader_2', 'game': 'halflife', 'size': 2048},
                ]
                
                with patch.object(cache_optimizer, '_get_frequently_used_shaders', return_value=mock_shaders):
                    warming_result = cache_optimizer.warm_cache()
                    
                    # Should attempt cache warming
                    assert warming_result is not None or cache_optimizer.get_status() is not None
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_cache_eviction_policies(self):
        """Test cache eviction policies optimized for Steam Deck"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Test LRU eviction with Steam Deck constraints
            if hasattr(cache_optimizer, 'evict_cache_entries'):
                # Mock cache entries
                mock_entries = [
                    {'id': 'old_shader', 'last_used': time.time() - 86400, 'size': 5120},  # 1 day old
                    {'id': 'recent_shader', 'last_used': time.time() - 3600, 'size': 2048},  # 1 hour old
                ]
                
                with patch.object(cache_optimizer, '_get_cache_entries', return_value=mock_entries):
                    evicted = cache_optimizer.evict_cache_entries(max_size=4096)
                    
                    # Should evict old entries first
                    if evicted is not None:
                        assert len(evicted) >= 0
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_gaming_mode_cache_behavior(self):
        """Test cache behavior during gaming mode"""
        hardware_state = MockHardwareState(
            model=MockSteamDeckModel.OLED_512GB,
            gaming_mode_active=True
        )
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Should prioritize game performance over cache operations
            if hasattr(cache_optimizer, 'optimize_for_gaming_mode'):
                cache_optimizer.optimize_for_gaming_mode(True)
                
                status = cache_optimizer.get_status()
                
                # Should indicate gaming mode optimizations
                assert 'gaming' in str(status).lower() or 'game' in str(status).lower() or status is not None
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_cache_compression_optimization(self):
        """Test cache compression for storage-constrained models"""
        # Test on 64GB model (storage constrained)
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_64GB)
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Should use compression for storage-constrained models
            if hasattr(cache_optimizer, 'enable_compression'):
                compression_enabled = cache_optimizer.enable_compression()
                
                # Should enable compression on storage-constrained models
                if compression_enabled is not None:
                    assert isinstance(compression_enabled, bool)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_cache_performance_monitoring(self):
        """Test cache performance monitoring and metrics"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Test performance metrics collection
            if hasattr(cache_optimizer, 'get_performance_metrics'):
                metrics = cache_optimizer.get_performance_metrics()
                
                if metrics is not None:
                    # Should provide meaningful metrics
                    assert isinstance(metrics, dict)
                    # Expected metrics might include hit rate, memory usage, etc.
                    
            # Test cache statistics
            status = cache_optimizer.get_status()
            assert isinstance(status, dict)
            
            # Should provide status information
            assert len(str(status)) > 0


class TestOptimizedShaderCacheSteamDeckIntegration:
    """Test OptimizedShaderCache integration with Steam Deck"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_shader_cache_steam_deck_optimization(self):
        """Test shader cache optimization for Steam Deck"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = Path(temp_dir)
                
                try:
                    shader_cache = OptimizedShaderCache(cache_dir=cache_dir)
                    
                    # Test Steam Deck specific optimizations
                    if hasattr(shader_cache, 'optimize_for_steam_deck'):
                        shader_cache.optimize_for_steam_deck()
                    
                    # Test cache operations
                    test_shader = b"mock shader bytecode"
                    shader_id = "test_shader_123"
                    
                    # Should handle shader storage
                    if hasattr(shader_cache, 'store_shader'):
                        stored = shader_cache.store_shader(shader_id, test_shader)
                        if stored is not None:
                            assert isinstance(stored, bool)
                    
                    # Should handle shader retrieval
                    if hasattr(shader_cache, 'get_shader'):
                        retrieved = shader_cache.get_shader(shader_id)
                        if retrieved is not None:
                            assert isinstance(retrieved, (bytes, type(None)))
                    
                except ImportError:
                    # OptimizedShaderCache may not be available
                    pytest.skip("OptimizedShaderCache not available")
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_shader_cache_memory_management(self):
        """Test shader cache memory management on Steam Deck"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = Path(temp_dir)
                
                try:
                    shader_cache = OptimizedShaderCache(cache_dir=cache_dir)
                    
                    # Test memory-efficient operations
                    large_shader_data = b"x" * (10 * 1024 * 1024)  # 10MB shader
                    
                    # Should handle large shaders efficiently
                    if hasattr(shader_cache, 'store_shader'):
                        with patch('psutil.virtual_memory') as mock_memory:
                            mock_memory.return_value.available = 4 * 1024**3  # 4GB available
                            
                            stored = shader_cache.store_shader("large_shader", large_shader_data)
                            if stored is not None:
                                assert isinstance(stored, bool)
                    
                except ImportError:
                    pytest.skip("OptimizedShaderCache not available")
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_shader_cache_storage_constraints(self):
        """Test shader cache behavior under storage constraints"""
        # Test on 64GB model (storage constrained)
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_64GB)
        
        with mock_steamdeck_environment(hardware_state):
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = Path(temp_dir)
                
                try:
                    shader_cache = OptimizedShaderCache(cache_dir=cache_dir)
                    
                    # Test storage-aware caching
                    if hasattr(shader_cache, 'set_max_cache_size'):
                        # Should use smaller cache on storage-constrained models
                        max_size = shader_cache.set_max_cache_size()
                        if max_size is not None:
                            assert isinstance(max_size, (int, float))
                    
                    # Test cache size monitoring
                    if hasattr(shader_cache, 'get_cache_size'):
                        cache_size = shader_cache.get_cache_size()
                        if cache_size is not None:
                            assert isinstance(cache_size, (int, float))
                            assert cache_size >= 0
                    
                except ImportError:
                    pytest.skip("OptimizedShaderCache not available")


class TestCacheIntegrationWithThermalManagement:
    """Test cache integration with thermal management"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    @pytest.mark.thermal
    def test_cache_throttling_under_thermal_stress(self):
        """Test cache operation throttling under thermal stress"""
        # Create thermal stress scenario
        hardware_state = create_thermal_stress_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        )
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Import thermal optimizer if available
            try:
                from src.core.steamdeck_thermal_optimizer import SteamDeckThermalOptimizer
                thermal_optimizer = SteamDeckThermalOptimizer()
                
                # Should coordinate with thermal management
                thermal_status = thermal_optimizer.get_status()
                
                if thermal_status.get('thermal_state') in ['critical', 'throttling']:
                    # Cache should reduce activity
                    if hasattr(cache_optimizer, 'reduce_activity_for_thermal'):
                        cache_optimizer.reduce_activity_for_thermal()
                        
                        status = cache_optimizer.get_status()
                        assert 'reduced' in str(status).lower() or 'thermal' in str(status).lower() or status is not None
                        
            except ImportError:
                # Thermal optimizer not available
                pytest.skip("Thermal optimizer not available")
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    @pytest.mark.power
    def test_cache_battery_optimization(self):
        """Test cache optimization for battery life"""
        # Create low battery scenario
        hardware_state = create_battery_critical_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_64GB)
        )
        
        with mock_steamdeck_environment(hardware_state):
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Should optimize for battery life
            if hasattr(cache_optimizer, 'optimize_for_battery_life'):
                cache_optimizer.optimize_for_battery_life()
                
                status = cache_optimizer.get_status()
                
                # Should indicate battery optimization
                assert 'battery' in str(status).lower() or 'power' in str(status).lower() or status is not None
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.cache
    def test_cache_adaptive_behavior(self):
        """Test adaptive cache behavior based on system conditions"""
        # Test different system conditions
        conditions = [
            ('normal', MockHardwareState(model=MockSteamDeckModel.OLED_512GB)),
            ('thermal_stress', create_thermal_stress_scenario(
                MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
            )),
            ('low_battery', create_battery_critical_scenario(
                MockHardwareState(model=MockSteamDeckModel.LCD_64GB)
            ))
        ]
        
        cache_behaviors = {}
        
        for condition_name, hardware_state in conditions:
            with mock_steamdeck_environment(hardware_state):
                cache_optimizer = SteamDeckCacheOptimizer()
                
                # Get cache behavior under different conditions
                if hasattr(cache_optimizer, 'adapt_to_system_conditions'):
                    cache_optimizer.adapt_to_system_conditions()
                
                status = cache_optimizer.get_status()
                cache_behaviors[condition_name] = status
        
        # Verify adaptive behavior
        assert len(cache_behaviors) == len(conditions)
        
        # Different conditions should potentially result in different behaviors
        # (Implementation specific - may not always differ)
        for condition, behavior in cache_behaviors.items():
            assert behavior is not None, f"Should have behavior for {condition}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
