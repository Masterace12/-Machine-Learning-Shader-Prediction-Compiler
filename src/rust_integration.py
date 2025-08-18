#!/usr/bin/env python3
"""
Python integration layer for the Rust-based shader prediction system.

This module provides a seamless interface between the existing Python codebase
and the high-performance Rust components.
"""

import sys
import logging
import importlib.util
from typing import Optional, Dict, List, Any
import numpy as np

# Try to import the Rust module
try:
    import shader_predict_rust
    RUST_AVAILABLE = True
    logging.info("Rust acceleration available")
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust acceleration not available, falling back to Python implementation")

class HybridMLPredictor:
    """
    Hybrid ML predictor that uses Rust acceleration when available,
    with fallback to Python implementation.
    """
    
    def __init__(self, model_path: Optional[str] = None, force_python: bool = False):
        self.use_rust = RUST_AVAILABLE and not force_python
        
        if self.use_rust:
            try:
                self.rust_predictor = shader_predict_rust.RustMLPredictor(model_path)
                logging.info("Initialized Rust ML predictor")
            except Exception as e:
                logging.warning(f"Failed to initialize Rust predictor: {e}")
                self.use_rust = False
                self._init_python_fallback(model_path)
        else:
            self._init_python_fallback(model_path)
    
    def _init_python_fallback(self, model_path: Optional[str]):
        """Initialize Python fallback implementation"""
        try:
            # Import existing Python ML predictor
            from .ml.optimized_ml_predictor import OptimizedMLPredictor
            self.python_predictor = OptimizedMLPredictor(model_path)
            logging.info("Initialized Python ML predictor fallback")
        except ImportError:
            logging.error("Could not import Python ML predictor")
            raise
    
    def predict_compilation_time(self, features: Dict[str, Any]) -> float:
        """
        Predict shader compilation time.
        
        Args:
            features: Dictionary of shader features
            
        Returns:
            Predicted compilation time in milliseconds
        """
        if self.use_rust:
            try:
                return self.rust_predictor.predict_compilation_time(features)
            except Exception as e:
                logging.warning(f"Rust prediction failed: {e}, falling back to Python")
                self.use_rust = False
                return self.python_predictor.predict(features)
        else:
            return self.python_predictor.predict(features)
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """
        Predict compilation times for multiple shaders.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of predicted compilation times
        """
        if self.use_rust:
            try:
                predictions = self.rust_predictor.predict_batch(features_list)
                return predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            except Exception as e:
                logging.warning(f"Rust batch prediction failed: {e}, falling back to Python")
                self.use_rust = False
        
        # Python fallback
        return [self.python_predictor.predict(features) for features in features_list]
    
    def extract_features_from_spirv(self, spirv_data: bytes) -> Dict[str, Any]:
        """
        Extract features from SPIR-V bytecode.
        
        Args:
            spirv_data: SPIR-V bytecode as bytes
            
        Returns:
            Dictionary of extracted features
        """
        if self.use_rust:
            try:
                rust_features = self.rust_predictor.extract_features_from_spirv(spirv_data)
                return rust_features.to_dict()
            except Exception as e:
                logging.warning(f"Rust feature extraction failed: {e}, falling back to Python")
                self.use_rust = False
        
        # Python fallback - basic feature extraction
        return self._extract_features_python_fallback(spirv_data)
    
    def extract_features_from_source(self, source: str, shader_type: str) -> Dict[str, Any]:
        """
        Extract features from shader source code.
        
        Args:
            source: Shader source code
            shader_type: Type of shader (vertex, fragment, etc.)
            
        Returns:
            Dictionary of extracted features
        """
        if self.use_rust:
            try:
                rust_features = self.rust_predictor.extract_features_from_source(source, shader_type)
                return rust_features.to_dict()
            except Exception as e:
                logging.warning(f"Rust source extraction failed: {e}, falling back to Python")
                self.use_rust = False
        
        # Python fallback
        return self._extract_from_source_python(source, shader_type)
    
    def add_feedback(self, features: Dict[str, Any], predicted: float, actual: float):
        """
        Add feedback for model improvement.
        
        Args:
            features: Shader features used for prediction
            predicted: Predicted compilation time
            actual: Actual compilation time
        """
        if self.use_rust:
            try:
                self.rust_predictor.add_feedback(features, predicted, actual)
                return
            except Exception as e:
                logging.warning(f"Rust feedback failed: {e}")
        
        # Python fallback
        if hasattr(self.python_predictor, 'add_feedback'):
            self.python_predictor.add_feedback(features, predicted, actual)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get prediction metrics and statistics.
        
        Returns:
            Dictionary of metrics
        """
        if self.use_rust:
            try:
                return self.rust_predictor.get_metrics()
            except Exception as e:
                logging.warning(f"Failed to get Rust metrics: {e}")
        
        # Python fallback
        if hasattr(self.python_predictor, 'get_metrics'):
            return self.python_predictor.get_metrics()
        return {"backend": "python_fallback"}
    
    def _extract_features_python_fallback(self, spirv_data: bytes) -> Dict[str, Any]:
        """Basic Python fallback for SPIR-V feature extraction"""
        # Very basic analysis - count instructions
        instruction_count = len(spirv_data) // 4  # Rough estimate
        
        return {
            "instruction_count": float(instruction_count),
            "register_usage": 16.0,  # Default estimate
            "texture_samples": 2.0,  # Default estimate
            "memory_operations": 5.0,  # Default estimate
            "control_flow_complexity": 3.0,  # Default estimate
            "wave_size": 64.0,  # RDNA2 default
            "uses_derivatives": False,
            "uses_tessellation": False,
            "uses_geometry_shader": False,
            "shader_type_hash": 2.0,  # Assume fragment shader
            "optimization_level": 1.0,
            "cache_priority": 0.5,
            "van_gogh_optimized": True,
            "rdna2_features": True,
            "thermal_state": 0.5,
            "power_mode": 1.0,
        }
    
    def _extract_from_source_python(self, source: str, shader_type: str) -> Dict[str, Any]:
        """Basic Python fallback for source code feature extraction"""
        features = {
            "instruction_count": float(len(source.split('\n'))),
            "register_usage": 16.0,
            "texture_samples": float(source.count('texture') + source.count('sample')),
            "memory_operations": float(source.count('load') + source.count('store')),
            "control_flow_complexity": float(source.count('if') + source.count('for') + source.count('while')),
            "wave_size": 64.0,
            "uses_derivatives": 'ddx' in source or 'ddy' in source or 'fwidth' in source,
            "uses_tessellation": 'tessellation' in source.lower(),
            "uses_geometry_shader": shader_type.lower() == 'geometry',
            "shader_type_hash": self._get_shader_type_hash(shader_type),
            "optimization_level": 1.0,
            "cache_priority": 0.5,
            "van_gogh_optimized": True,
            "rdna2_features": True,
            "thermal_state": 0.5,
            "power_mode": 1.0,
        }
        
        return features
    
    def _get_shader_type_hash(self, shader_type: str) -> float:
        """Convert shader type to hash value"""
        type_map = {
            'vertex': 1.0,
            'fragment': 2.0,
            'pixel': 2.0,
            'geometry': 3.0,
            'compute': 4.0,
            'tess_ctrl': 5.0,
            'tessellation_control': 5.0,
            'tess_eval': 6.0,
            'tessellation_evaluation': 6.0,
        }
        return type_map.get(shader_type.lower(), 2.0)


class HybridVulkanCache:
    """
    Hybrid Vulkan cache that uses Rust acceleration when available.
    """
    
    def __init__(self, force_python: bool = False):
        self.use_rust = RUST_AVAILABLE and not force_python
        
        if self.use_rust:
            try:
                self.rust_cache = shader_predict_rust.RustVulkanCache()
                logging.info("Initialized Rust Vulkan cache")
            except Exception as e:
                logging.warning(f"Failed to initialize Rust cache: {e}")
                self.use_rust = False
                self._init_python_fallback()
        else:
            self._init_python_fallback()
    
    def _init_python_fallback(self):
        """Initialize Python fallback cache"""
        try:
            from .cache.optimized_shader_cache import OptimizedShaderCache
            self.python_cache = OptimizedShaderCache()
            logging.info("Initialized Python cache fallback")
        except ImportError:
            logging.error("Could not import Python cache")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.use_rust:
            try:
                return self.rust_cache.get_stats()
            except Exception as e:
                logging.warning(f"Failed to get Rust cache stats: {e}")
        
        # Python fallback
        if hasattr(self.python_cache, 'get_stats'):
            return self.python_cache.get_stats()
        return {"backend": "python_fallback"}


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for optimization decisions.
    
    Returns:
        Dictionary of system information
    """
    if RUST_AVAILABLE:
        try:
            return shader_predict_rust.get_system_info()
        except Exception as e:
            logging.warning(f"Failed to get Rust system info: {e}")
    
    # Python fallback
    import platform
    import os
    
    info = {
        "is_steam_deck": os.path.exists("/home/deck"),
        "cpu_count": os.cpu_count() or 4,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "optimization_target": "steam_deck" if os.path.exists("/home/deck") else "desktop",
        "backend": "python_fallback"
    }
    
    return info


def is_steam_deck() -> bool:
    """Check if running on Steam Deck"""
    if RUST_AVAILABLE:
        try:
            return shader_predict_rust.is_steam_deck()
        except Exception:
            pass
    
    # Python fallback
    import os
    return os.path.exists("/home/deck")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test system detection
    print("System Info:", get_system_info())
    print("Is Steam Deck:", is_steam_deck())
    
    # Test ML predictor
    try:
        predictor = HybridMLPredictor()
        
        # Test feature extraction
        test_source = """
        #version 450
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;
        layout(location = 0) out vec2 fragTexCoord;
        
        void main() {
            gl_Position = vec4(position, 1.0);
            fragTexCoord = texCoord;
        }
        """
        
        features = predictor.extract_features_from_source(test_source, "vertex")
        print("Extracted features:", features)
        
        # Test prediction
        prediction = predictor.predict_compilation_time(features)
        print(f"Predicted compilation time: {prediction:.2f}ms")
        
        # Test metrics
        metrics = predictor.get_metrics()
        print("Metrics:", metrics)
        
    except Exception as e:
        print(f"Error testing predictor: {e}")
    
    # Test cache
    try:
        cache = HybridVulkanCache()
        stats = cache.get_stats()
        print("Cache stats:", stats)
        
    except Exception as e:
        print(f"Error testing cache: {e}")