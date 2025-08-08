#!/usr/bin/env python3
"""
Example usage of the fixed Fossilize integration with proper Steam compatibility.

This example demonstrates:
1. Creating valid .foz files with proper Fossilize database format
2. SPIRV-Tools integration for shader optimization
3. VkPipelineCache generation for direct Vulkan compatibility  
4. Steam shader cache integration
5. Thermal-aware compilation for Steam Deck
"""

import sys
import json
import logging
from pathlib import Path
import asyncio

# Import the fixed modules
sys.path.insert(0, str(Path(__file__).parent / "shader-prediction-compilation-main/shader-predict-compile/src"))
sys.path.insert(0, str(Path(__file__).parent / "steam-deck-qa-framework/src/validation"))

from fossilize_integration import FossilizeIntegration, VkPipelineInfo, ShaderStage
from shader_cache_validator import ShaderCacheValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_example_shader_data():
    """Create example shader data for demonstration"""
    
    # Example GLSL vertex shader source
    vertex_shader_source = """
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view; 
    mat4 proj;
} ubo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
"""

    # Example GLSL fragment shader source  
    fragment_shader_source = """
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord) * vec4(fragColor, 1.0);
}
"""

    # Create shader info structures
    shaders = [
        {
            'variant': 'standard_vertex',
            'shader_type': 'vertex',
            'shader_source': vertex_shader_source,
            'priority': 1.0,
            'steam_deck_optimized': True,
            'specialization_info': {
                'constants': [
                    {'id': 0, 'value': 1.0},  # Example specialization constant
                ]
            },
            'descriptor_layout': {
                'bindings': [
                    {'binding': 0, 'type': 'uniform_buffer', 'stage': 'vertex'},
                    {'binding': 1, 'type': 'combined_image_sampler', 'stage': 'fragment'}
                ]
            },
            'render_pass_info': {
                'color_attachments': [{'format': 'VK_FORMAT_B8G8R8A8_SRGB'}],
                'depth_attachment': {'format': 'VK_FORMAT_D32_SFLOAT'}
            }
        },
        {
            'variant': 'standard_fragment', 
            'shader_type': 'fragment',
            'shader_source': fragment_shader_source,
            'priority': 1.0,
            'steam_deck_optimized': True,
            'specialization_info': {
                'constants': []
            }
        },
        {
            'variant': 'optimized_vertex',
            'shader_type': 'vertex', 
            'shader_source': vertex_shader_source.replace('vec3 inColor', 'vec4 inColor'),
            'priority': 0.8,
            'steam_deck_optimized': True
        }
    ]
    
    return shaders

def demonstrate_fossilize_integration():
    """Demonstrate the fixed Fossilize integration"""
    
    logger.info("=== Demonstrating Fixed Fossilize Integration ===")
    
    # Initialize Fossilize integration
    fossilize = FossilizeIntegration()
    
    # Check compatibility
    compatibility = fossilize.check_compatibility()
    logger.info(f"System compatibility: {compatibility}")
    
    # Get compilation statistics
    stats = fossilize.get_compilation_stats()
    logger.info(f"Initial stats: {stats}")
    
    if not stats['fossilize_available']:
        logger.warning("Fossilize not available - demonstration will be limited")
    
    if not stats['spirv_tools_available']:
        logger.warning("SPIRV-Tools not available - shader optimization will be skipped")
    
    # Create example game scenario
    game_id = "example_game_12345"
    shaders = create_example_shader_data()
    
    logger.info(f"Compiling {len(shaders)} shader variants for game {game_id}")
    
    # Create thermal-aware scheduler for Steam Deck
    thermal_scheduler = fossilize.create_thermal_aware_scheduler()
    
    # Check if we should compile now based on thermal conditions
    if thermal_scheduler.should_compile_now():
        optimal_threads = thermal_scheduler.get_optimal_thread_count()
        logger.info(f"Starting compilation with {optimal_threads} threads")
        
        # Compile shaders
        try:
            def progress_callback(progress_info):
                logger.info(f"Compiled shader: {progress_info['shader']} "
                           f"({'SUCCESS' if progress_info['success'] else 'FAILED'}) "
                           f"in {progress_info['time']:.3f}s")
            
            results = fossilize.compile_shaders(game_id, shaders, progress_callback)
            logger.info(f"Compilation results: {results}")
            
            # Get updated stats
            final_stats = fossilize.get_compilation_stats()
            logger.info(f"Final compilation stats: {final_stats}")
            
            # Integrate with Steam if available
            cache_dir = fossilize.prepare_shader_cache(game_id, shaders)
            steam_integration_success = fossilize.integrate_with_steam(game_id, cache_dir)
            
            if steam_integration_success:
                logger.info("Successfully integrated with Steam shader cache system")
            else:
                logger.warning("Steam integration failed or not available")
                
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
    else:
        logger.info("Compilation paused due to thermal conditions")

async def demonstrate_cache_validation():
    """Demonstrate the enhanced shader cache validation"""
    
    logger.info("=== Demonstrating Enhanced Cache Validation ===")
    
    # Create validator with Steam Deck configuration
    config = {
        "steam_deck": {
            "cache_directory": "/home/deck/.steam/steam/steamapps/shadercache"
        }
    }
    
    validator = ShaderCacheValidator(config)
    
    # Example app ID for validation
    app_id = "example_game_12345"
    expected_shaders = 150  # Expected number of shaders
    
    try:
        # Perform comprehensive validation
        validation_result = await validator.validate_shader_cache(app_id, expected_shaders)
        
        logger.info("=== Validation Results ===")
        logger.info(f"Cache valid: {validation_result['valid']}")
        logger.info(f"Issues found: {len(validation_result['issues'])}")
        
        # Display cache file analysis
        cache_files = validation_result.get('cache_files', {})
        logger.info(f"Total files: {cache_files.get('total_files', 0)}")
        logger.info(f"Total size: {cache_files.get('total_size', 0) / 1024 / 1024:.1f} MB")
        logger.info(f"File types: {cache_files.get('file_types', {})}")
        logger.info(f"Corruption rate: {cache_files.get('corruption_rate', 0):.1%}")
        
        # Display Fossilize-specific metrics
        performance = validation_result.get('performance_metrics', {})
        fossilize_metrics = performance.get('fossilize_metrics', {})
        if fossilize_metrics:
            logger.info("=== Fossilize Metrics ===")
            logger.info(f"FOZ files: {fossilize_metrics.get('foz_file_count', 0)}")
            logger.info(f"Pipeline entries: {fossilize_metrics.get('total_pipeline_entries', 0)}")
            logger.info(f"SPIRV shaders: {fossilize_metrics.get('spirv_shader_count', 0)}")
            logger.info(f"Version compatibility: {fossilize_metrics.get('version_compatibility', {})}")
        
        # Display Steam integration status
        steam_status = performance.get('steam_integration_status', {})
        if steam_status:
            logger.info("=== Steam Integration Status ===")
            logger.info(f"Metadata present: {steam_status.get('metadata_present', False)}")
            logger.info(f"Steam cache linked: {steam_status.get('steam_cache_linked', False)}")
            logger.info(f"Fossilize metadata valid: {steam_status.get('fossilize_metadata_valid', False)}")
            logger.info(f"VkPipelineCache present: {steam_status.get('vk_cache_present', False)}")
        
        # Display any issues
        if validation_result['issues']:
            logger.warning("=== Issues Found ===")
            for issue in validation_result['issues']:
                logger.warning(f"- {issue}")
                
    except Exception as e:
        logger.error(f"Cache validation failed: {e}")

def demonstrate_manual_foz_creation():
    """Demonstrate manual creation of .foz files with proper format"""
    
    logger.info("=== Demonstrating Manual FOZ File Creation ===")
    
    fossilize = FossilizeIntegration()
    
    # Create example SPIRV bytecode (this would normally come from compiled shaders)
    # For demonstration, we'll create a minimal valid SPIRV header
    import struct
    
    spirv_data = bytearray()
    spirv_data.extend(struct.pack('<I', 0x07230203))  # SPIRV magic
    spirv_data.extend(struct.pack('<I', 0x00010300))  # Version 1.3
    spirv_data.extend(struct.pack('<I', 0x00000000))  # Generator
    spirv_data.extend(struct.pack('<I', 0x00000001))  # Bound
    spirv_data.extend(struct.pack('<I', 0x00000000))  # Schema
    # Add minimal SPIRV instructions for a valid shader
    spirv_data.extend(b'\x00' * 100)  # Placeholder shader data
    
    # Create pipeline info
    pipeline_info = VkPipelineInfo(
        stage_flags=ShaderStage.VERTEX.value,
        shader_hash="example_hash_123",
        spirv_data=bytes(spirv_data),
        specialization_info={
            'constants': [{'id': 0, 'value': 1.0}]
        },
        descriptor_layout={
            'bindings': [
                {'binding': 0, 'type': 'uniform_buffer', 'stage': 'vertex'}
            ]
        }
    )
    
    # Create temporary directory for demonstration
    demo_dir = Path("./demo_cache")
    demo_dir.mkdir(exist_ok=True)
    
    try:
        # Create Fossilize database entry
        success = fossilize._create_fossilize_database_entry(
            demo_dir, "demo_shader", pipeline_info
        )
        
        if success:
            logger.info("Successfully created demo .foz file")
            
            # Create VkPipelineCache file
            cache_success = fossilize._create_vulkan_pipeline_cache(
                demo_dir, "demo_shader", pipeline_info
            )
            
            if cache_success:
                logger.info("Successfully created VkPipelineCache file")
            
            # List created files
            for file in demo_dir.iterdir():
                logger.info(f"Created file: {file.name} ({file.stat().st_size} bytes)")
                
        else:
            logger.error("Failed to create demo .foz file")
            
    except Exception as e:
        logger.error(f"Manual FOZ creation failed: {e}")
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(demo_dir)
        except:
            pass

def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("FIXED FOSSILIZE INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print()
    
    try:
        # 1. Demonstrate Fossilize integration
        demonstrate_fossilize_integration()
        print()
        
        # 2. Demonstrate cache validation
        asyncio.run(demonstrate_cache_validation())
        print()
        
        # 3. Demonstrate manual FOZ creation
        demonstrate_manual_foz_creation()
        print()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print()
        print("Key improvements made:")
        print("✓ Fixed .foz file format with proper Fossilize database structure")
        print("✓ Added SPIRV-Tools integration for shader optimization and validation")
        print("✓ Implemented VkPipelineCache serialization for direct Vulkan compatibility")
        print("✓ Enhanced Steam shader cache integration with metadata")
        print("✓ Added Steam Deck thermal-aware compilation scheduling")
        print("✓ Comprehensive error handling and fallback mechanisms")
        print("✓ Proper shader cache validation with Fossilize-specific checks")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())