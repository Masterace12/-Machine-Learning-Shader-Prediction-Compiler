#!/usr/bin/env python3
"""
Basic usage example for the ML Shader Prediction Compiler
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimized_main import OptimizedShaderSystem, SystemConfig


async def main():
    """Basic usage example"""
    
    # Create configuration
    config = SystemConfig(
        enable_ml_prediction=True,
        enable_cache=True,
        enable_thermal_management=True,
        max_memory_mb=100  # Limit for example
    )
    
    # Create system
    system = OptimizedShaderSystem(config)
    
    print("Starting shader prediction system...")
    
    # Start system (this would run indefinitely in production)
    try:
        # For demo, just initialize components
        _ = system.ml_predictor
        _ = system.cache_manager
        _ = system.thermal_manager
        
        # Show status
        status = system.get_system_status()
        print(f"System Status: {status}")
        
        print("Example completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
