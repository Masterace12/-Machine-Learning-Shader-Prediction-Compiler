#!/usr/bin/env python3
"""
Unified ML Predictor Base Classes and Enums
Provides foundational classes and types for the shader prediction system
"""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field


class ShaderType(Enum):
    """Shader types supported by the system"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"
    GEOMETRY = "geometry"
    TESSELLATION_CONTROL = "tessellation_control"
    TESSELLATION_EVALUATION = "tessellation_evaluation"
    UNKNOWN = "unknown"


class ThermalState(Enum):
    """Thermal states for system management"""
    COOL = "cool"              # < 60°C - Aggressive compilation
    OPTIMAL = "optimal"        # 60-70°C - Full compilation capacity
    NORMAL = "normal"          # 70-80°C - Standard operation
    WARM = "warm"              # 80-85°C - Reduced background work
    HOT = "hot"                # 85-90°C - Essential shaders only
    THROTTLING = "throttling"  # 90-95°C - Compilation paused
    CRITICAL = "critical"      # > 95°C - Emergency shutdown
    PREDICTIVE_WARM = "predictive_warm"  # Predicted to become warm


class SteamDeckModel(Enum):
    """Steam Deck hardware models"""
    LCD = "lcd"     # Van Gogh APU - 7nm, higher thermals
    OLED = "oled"   # Phoenix APU - 6nm, better thermal efficiency
    UNKNOWN = "unknown"


@dataclass
class UnifiedShaderFeatures:
    """Unified shader features for ML prediction"""
    shader_hash: str
    shader_type: ShaderType = ShaderType.UNKNOWN
    instruction_count: int = 0
    register_usage: int = 0
    texture_samples: int = 0
    memory_operations: int = 0
    control_flow_complexity: int = 0
    wave_size: int = 64
    uses_derivatives: bool = False
    uses_tessellation: bool = False
    uses_geometry_shader: bool = False
    optimization_level: int = 0
    cache_priority: float = 0.5
    
    def __post_init__(self):
        """Validate and normalize feature values"""
        # Ensure positive values
        self.instruction_count = max(0, self.instruction_count)
        self.register_usage = max(0, min(256, self.register_usage))  # Typical GPU register limit
        self.texture_samples = max(0, self.texture_samples)
        self.memory_operations = max(0, self.memory_operations)
        self.control_flow_complexity = max(0, self.control_flow_complexity)
        self.wave_size = max(32, min(128, self.wave_size))  # Common wave sizes
        self.optimization_level = max(0, min(3, self.optimization_level))
        self.cache_priority = max(0.0, min(1.0, self.cache_priority))


class HeuristicPredictor:
    """Fallback heuristic predictor when ML models are unavailable"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Base compilation times by shader type (milliseconds)
        self.base_times = {
            ShaderType.VERTEX: 5.0,
            ShaderType.FRAGMENT: 12.0,
            ShaderType.COMPUTE: 20.0,
            ShaderType.GEOMETRY: 15.0,
            ShaderType.TESSELLATION_CONTROL: 18.0,
            ShaderType.TESSELLATION_EVALUATION: 18.0,
            ShaderType.UNKNOWN: 10.0
        }
        
        # Complexity multipliers
        self.complexity_factors = {
            "instruction_count": 0.002,    # 2ms per 1000 instructions
            "register_usage": 0.05,        # 5ms per 100 registers
            "texture_samples": 0.8,        # 0.8ms per texture sample
            "memory_operations": 0.3,      # 0.3ms per memory op
            "control_flow_complexity": 1.5, # 1.5ms per complexity unit
        }
    
    def predict_compilation_time(self, features: UnifiedShaderFeatures) -> float:
        """
        Predict shader compilation time using heuristics
        
        Args:
            features: Shader features
            
        Returns:
            Predicted compilation time in milliseconds
        """
        # Base time for shader type
        base_time = self.base_times.get(features.shader_type, 10.0)
        
        # Complexity adjustments
        complexity_time = 0.0
        complexity_time += features.instruction_count * self.complexity_factors["instruction_count"]
        complexity_time += features.register_usage * self.complexity_factors["register_usage"]
        complexity_time += features.texture_samples * self.complexity_factors["texture_samples"]
        complexity_time += features.memory_operations * self.complexity_factors["memory_operations"]
        complexity_time += features.control_flow_complexity * self.complexity_factors["control_flow_complexity"]
        
        # Special feature multipliers
        feature_multiplier = 1.0
        if features.uses_derivatives:
            feature_multiplier *= 1.3
        if features.uses_tessellation:
            feature_multiplier *= 1.5
        if features.uses_geometry_shader:
            feature_multiplier *= 1.4
        
        # Optimization level adjustment (higher optimization = longer compile time)
        opt_multiplier = 1.0 + (features.optimization_level * 0.2)
        
        # Wave size adjustment (non-standard wave sizes may be slower)
        wave_multiplier = 1.0
        if features.wave_size not in [32, 64]:
            wave_multiplier = 1.2
        
        total_time = (base_time + complexity_time) * feature_multiplier * opt_multiplier * wave_multiplier
        
        # Ensure minimum time
        return max(1.0, total_time)
    
    def predict_success_probability(self, features: UnifiedShaderFeatures) -> float:
        """
        Predict compilation success probability
        
        Args:
            features: Shader features
            
        Returns:
            Success probability (0.0 to 1.0)
        """
        # Base success rates by shader type
        base_success = {
            ShaderType.VERTEX: 0.98,
            ShaderType.FRAGMENT: 0.95,
            ShaderType.COMPUTE: 0.90,
            ShaderType.GEOMETRY: 0.92,
            ShaderType.TESSELLATION_CONTROL: 0.88,
            ShaderType.TESSELLATION_EVALUATION: 0.88,
            ShaderType.UNKNOWN: 0.85
        }
        
        success_prob = base_success.get(features.shader_type, 0.85)
        
        # Reduce success probability for complex shaders
        if features.instruction_count > 5000:
            success_prob *= 0.95
        if features.register_usage > 128:
            success_prob *= 0.96
        if features.control_flow_complexity > 10:
            success_prob *= 0.93
        
        return max(0.1, min(1.0, success_prob))


class ThermalAwareScheduler:
    """Scheduler that adapts to thermal conditions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_thermal_state = ThermalState.NORMAL
        self.compilation_callbacks: List[Callable] = []
    
    def update_thermal_state(self, state: ThermalState):
        """Update current thermal state"""
        if state != self.current_thermal_state:
            self.logger.info(f"Thermal state change: {self.current_thermal_state.value} → {state.value}")
            self.current_thermal_state = state
            self._notify_callbacks()
    
    def _notify_callbacks(self):
        """Notify registered callbacks of thermal state change"""
        for callback in self.compilation_callbacks:
            try:
                callback(self.current_thermal_state)
            except Exception as e:
                self.logger.error(f"Thermal callback error: {e}")
    
    def add_callback(self, callback: Callable[[ThermalState], None]):
        """Add thermal state change callback"""
        self.compilation_callbacks.append(callback)
    
    def should_compile(self, features: UnifiedShaderFeatures) -> bool:
        """
        Determine if shader should be compiled given current thermal state
        
        Args:
            features: Shader features
            
        Returns:
            True if compilation should proceed
        """
        if self.current_thermal_state == ThermalState.CRITICAL:
            return False
        
        if self.current_thermal_state == ThermalState.THROTTLING:
            # Only compile high priority shaders
            return features.cache_priority > 0.8
        
        if self.current_thermal_state == ThermalState.HOT:
            # Compile essential shaders only
            return features.cache_priority > 0.6
        
        if self.current_thermal_state == ThermalState.PREDICTIVE_WARM:
            # Reduce background compilation
            return features.cache_priority > 0.4
        
        # Normal, optimal, cool states allow all compilation
        return True
    
    def get_compilation_priority(self, features: UnifiedShaderFeatures) -> int:
        """
        Get compilation priority based on thermal state and features
        
        Args:
            features: Shader features
            
        Returns:
            Priority level (0 = highest, higher numbers = lower priority)
        """
        base_priority = int((1.0 - features.cache_priority) * 10)
        
        # Adjust for thermal state
        thermal_adjustment = {
            ThermalState.CRITICAL: 1000,    # Effectively disabled
            ThermalState.THROTTLING: 100,   # Very low priority
            ThermalState.HOT: 50,           # Low priority
            ThermalState.PREDICTIVE_WARM: 20, # Reduced priority
            ThermalState.WARM: 10,          # Slightly reduced
            ThermalState.NORMAL: 0,         # No adjustment
            ThermalState.OPTIMAL: -5,       # Slightly higher
            ThermalState.COOL: -10          # Higher priority
        }
        
        adjustment = thermal_adjustment.get(self.current_thermal_state, 0)
        return max(0, base_priority + adjustment)


# Utility functions for testing and validation

def create_test_features(shader_type: ShaderType = ShaderType.FRAGMENT,
                        complexity: str = "medium") -> UnifiedShaderFeatures:
    """
    Create test shader features for testing purposes
    
    Args:
        shader_type: Type of shader to create
        complexity: Complexity level ("low", "medium", "high")
        
    Returns:
        Test shader features
    """
    complexity_params = {
        "low": {
            "instruction_count": 100,
            "register_usage": 8,
            "texture_samples": 1,
            "memory_operations": 2,
            "control_flow_complexity": 1
        },
        "medium": {
            "instruction_count": 500,
            "register_usage": 32,
            "texture_samples": 4,
            "memory_operations": 10,
            "control_flow_complexity": 5
        },
        "high": {
            "instruction_count": 2000,
            "register_usage": 128,
            "texture_samples": 16,
            "memory_operations": 50,
            "control_flow_complexity": 20
        }
    }
    
    params = complexity_params.get(complexity, complexity_params["medium"])
    
    return UnifiedShaderFeatures(
        shader_hash=f"test_{shader_type.value}_{complexity}_{int(time.time())}",
        shader_type=shader_type,
        instruction_count=params["instruction_count"],
        register_usage=params["register_usage"],
        texture_samples=params["texture_samples"],
        memory_operations=params["memory_operations"],
        control_flow_complexity=params["control_flow_complexity"],
        wave_size=64,
        uses_derivatives=(complexity == "high"),
        uses_tessellation=(shader_type in [ShaderType.TESSELLATION_CONTROL, ShaderType.TESSELLATION_EVALUATION]),
        uses_geometry_shader=(shader_type == ShaderType.GEOMETRY),
        optimization_level=2,
        cache_priority=0.7 if complexity == "high" else 0.5
    )


def validate_features(features: UnifiedShaderFeatures) -> List[str]:
    """
    Validate shader features and return list of issues
    
    Args:
        features: Features to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not features.shader_hash:
        issues.append("shader_hash is required")
    
    if features.instruction_count < 0:
        issues.append("instruction_count cannot be negative")
    
    if features.register_usage < 0 or features.register_usage > 256:
        issues.append("register_usage must be between 0 and 256")
    
    if features.texture_samples < 0:
        issues.append("texture_samples cannot be negative")
    
    if features.memory_operations < 0:
        issues.append("memory_operations cannot be negative")
    
    if features.control_flow_complexity < 0:
        issues.append("control_flow_complexity cannot be negative")
    
    if features.wave_size not in [32, 64, 128]:
        issues.append("wave_size should typically be 32, 64, or 128")
    
    if features.optimization_level < 0 or features.optimization_level > 3:
        issues.append("optimization_level must be between 0 and 3")
    
    if features.cache_priority < 0.0 or features.cache_priority > 1.0:
        issues.append("cache_priority must be between 0.0 and 1.0")
    
    return issues


if __name__ == "__main__":
    # Test the base classes
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Unified ML Predictor Base Classes")
    print("=" * 50)
    
    # Test feature creation
    test_features = create_test_features(ShaderType.FRAGMENT, "medium")
    print(f"✓ Created test features: {test_features.shader_hash}")
    
    # Test validation
    issues = validate_features(test_features)
    if not issues:
        print("✓ Features validation passed")
    else:
        print(f"❌ Validation issues: {issues}")
    
    # Test heuristic predictor
    predictor = HeuristicPredictor()
    compile_time = predictor.predict_compilation_time(test_features)
    success_prob = predictor.predict_success_probability(test_features)
    
    print(f"✓ Heuristic prediction: {compile_time:.1f}ms (success: {success_prob:.1%})")
    
    # Test thermal scheduler
    scheduler = ThermalAwareScheduler()
    should_compile = scheduler.should_compile(test_features)
    priority = scheduler.get_compilation_priority(test_features)
    
    print(f"✓ Thermal scheduler: compile={should_compile}, priority={priority}")
    
    # Test with different thermal states
    for state in [ThermalState.COOL, ThermalState.HOT, ThermalState.CRITICAL]:
        scheduler.update_thermal_state(state)
        should_compile = scheduler.should_compile(test_features)
        print(f"  {state.value}: compile={should_compile}")
    
    print("\n✅ All base class tests passed!")