#!/usr/bin/env python3
"""
Final Project Cleanup Script
Organizes the optimized shader-predict-compile project by removing redundant files
and creating a clean, professional structure.
"""

import os
import shutil
import time
from pathlib import Path
import json
import logging

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'  # No Color

def log(message, color=Colors.GREEN):
    print(f"{color}[INFO]{Colors.NC} {message}")

def warn(message):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")

def error(message):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

class ProjectCleanup:
    """Comprehensive project cleanup"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.backup_dir = self.project_dir / f"cleanup_final_backup_{int(time.time())}"
        
        # Files and directories to remove (redundant/old)
        self.files_to_remove = [
            # Old backup directories
            "cleanup_backup",
            "config_backup", 
            "venv",  # Virtual environment shouldn't be in repo
            
            # Redundant installation scripts
            "install_user.sh",
            "install_user_enhanced.sh",
            "uninstall.sh",  # Keep uninstall_enhanced.sh as primary
            
            # Old/duplicate documentation
            "OPTIMIZATION_RESULTS.md",
            "STEAM_DECK_QA_FINAL_REPORT.md",
            "STEAM_DECK_SETUP.md",
            
            # Old scripts and utilities
            "activate_steam_deck.sh",
            "cleanup_redundant_files.py",
            "config_migration.py",
            "steam_deck_env.py",
            "system_health_check.py",
            "test_games_realtime.py",
            "thermal_pattern_test.py",
            
            # Old/duplicate config and data files
            "Cyberpunk 2077_performance.json",
            "cleanup_report.txt", 
            "health_check_results.json",
            "steam_deck_detection_report.json",
            "unified_config.json",
            
            # Duplicate/old source files
            "src/desktop_integration.py",
            "src/gaming_mode_integration.py", 
            "src/ml_shader_predictor_gui.py",
            "src/radv_optimizer_enhanced.py",
            "src/shader_prediction_system.py",
            "src/steam_deck_hardware_enhanced.py",
            
            # Duplicate cache implementation (keep optimized version)
            "src/cache/advanced_shader_cache.py",
            
            # Old ML implementation (keep optimized version)
            "src/ml/unified_ml_predictor.py",
            
            # Old thermal manager (keep optimized version)
            "src/steam/thermal_manager.py",
            "src/steam/comprehensive_steam_integration.py",
            
            # Empty thermal directory (files moved)
            "src/thermal",
            
            # Duplicate packaging directories
            "steamdeck-deployment",  # Keep packaging/ as primary
            
            # Old requirements file
            "requirements.txt",  # Keep requirements-optimized.txt
        ]
        
        # Directories to create for clean structure
        self.new_structure = {
            "docs": ["architecture.md", "api.md", "contributing.md"],
            "scripts": ["utilities and helper scripts"],
            "examples": ["usage examples and demos"],
            "config": ["configuration templates"]
        }
    
    def create_backup(self):
        """Create backup of current state"""
        log(f"Creating backup at {self.backup_dir}...")
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup files that will be removed
        for item_name in self.files_to_remove:
            item_path = self.project_dir / item_name
            if item_path.exists():
                backup_path = self.backup_dir / item_name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                if item_path.is_file():
                    shutil.copy2(item_path, backup_path)
                elif item_path.is_dir():
                    shutil.copytree(item_path, backup_path)
                
                log(f"  Backed up: {item_name}")
        
        success("Backup completed")
    
    def remove_redundant_files(self):
        """Remove redundant and outdated files"""
        log("Removing redundant files and directories...")
        
        removed_count = 0
        total_size_saved = 0
        
        for item_name in self.files_to_remove:
            item_path = self.project_dir / item_name
            
            if item_path.exists():
                try:
                    # Calculate size before removal
                    if item_path.is_file():
                        size = item_path.stat().st_size
                    else:
                        size = sum(f.stat().st_size for f in item_path.rglob("*") if f.is_file())
                    
                    total_size_saved += size
                    
                    # Remove the item
                    if item_path.is_file():
                        item_path.unlink()
                    else:
                        shutil.rmtree(item_path)
                    
                    log(f"  Removed: {item_name}")
                    removed_count += 1
                    
                except Exception as e:
                    warn(f"Could not remove {item_name}: {e}")
        
        success(f"Removed {removed_count} items, saved {total_size_saved / (1024*1024):.1f}MB")
    
    def organize_remaining_files(self):
        """Organize remaining files into clean structure"""
        log("Organizing remaining files...")
        
        # Create new directories
        for dir_name in ["docs", "scripts", "examples", "config"]:
            dir_path = self.project_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            
            # Create __init__.py for Python packages if needed
            if dir_name in ["scripts"]:
                (dir_path / "__init__.py").write_text('"""Package initialization"""\n')
        
        # Move utility scripts to scripts/
        scripts_to_move = [
            "migrate_dependencies.py",
            "reorganize_structure.py",
            "final_cleanup.py"
        ]
        
        scripts_dir = self.project_dir / "scripts"
        for script_name in scripts_to_move:
            script_path = self.project_dir / script_name
            if script_path.exists() and script_name != "final_cleanup.py":  # Don't move this script while running
                new_path = scripts_dir / script_name
                shutil.move(str(script_path), str(new_path))
                log(f"  Moved {script_name} to scripts/")
        
        # Rename main installation script to be primary
        old_install = self.project_dir / "install_optimized.sh"
        new_install = self.project_dir / "install.sh" 
        if old_install.exists() and not new_install.exists():
            shutil.move(str(old_install), str(new_install))
            log("  Renamed install_optimized.sh to install.sh")
        
        # Keep primary uninstall script
        old_uninstall = self.project_dir / "uninstall_enhanced.sh"
        new_uninstall = self.project_dir / "uninstall.sh"
        if old_uninstall.exists() and not new_uninstall.exists():
            shutil.move(str(old_uninstall), str(new_uninstall))
            log("  Renamed uninstall_enhanced.sh to uninstall.sh")
    
    def clean_src_directory(self):
        """Clean up the src directory structure"""
        log("Cleaning src directory...")
        
        src_dir = self.project_dir / "src"
        
        # Remove empty __init__.py files or update them
        for init_file in src_dir.rglob("__init__.py"):
            if init_file.stat().st_size == 0:
                init_file.write_text('"""Package initialization"""\n')
        
        # Ensure clean module structure
        expected_modules = ["ml", "cache", "thermal", "monitoring"]
        for module in expected_modules:
            module_dir = src_dir / module
            if module_dir.exists():
                init_file = module_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text(f'"""{module.title()} module initialization"""\n')
        
        log("  src directory cleaned")
    
    def create_clean_docs(self):
        """Create clean documentation structure"""
        log("Creating documentation structure...")
        
        docs_dir = self.project_dir / "docs"
        
        # Create architecture documentation
        arch_doc = docs_dir / "ARCHITECTURE.md"
        if not arch_doc.exists():
            arch_content = """# System Architecture

## Overview
The ML Shader Prediction Compiler uses a modular architecture with the following components:

## Core Modules

### ML Prediction Engine (`src/ml/`)
- Optimized machine learning models for shader compilation time prediction
- LightGBM backend for high-performance inference
- Memory pooling and caching for efficiency

### Cache System (`src/cache/`)
- Multi-tier caching (hot/warm/cold storage)
- LZ4 compression for space efficiency
- Async I/O operations for performance

### Thermal Management (`src/thermal/`)
- Predictive thermal modeling
- Game-specific thermal profiles
- Hardware-aware throttling

### Performance Monitoring (`src/monitoring/`)
- Real-time metrics collection
- Health scoring and alerting
- System optimization recommendations

## Data Flow
1. Steam launches detected via D-Bus
2. Shader features extracted and cached
3. ML models predict compilation requirements
4. Thermal manager adjusts compilation strategy
5. Shaders pre-compiled based on predictions
6. Performance metrics collected and analyzed

## Integration Points
- Steam client integration via D-Bus
- Vulkan layer interception
- System thermal sensors
- GPU driver optimization hooks
"""
            arch_doc.write_text(arch_content)
            log("  Created ARCHITECTURE.md")
        
        # Create API documentation
        api_doc = docs_dir / "API.md"  
        if not api_doc.exists():
            api_content = """# API Reference

## Command Line Interface

### Main Commands
- `shader-predict-compile` - Start the prediction system
- `shader-predict-status` - Show system status  
- `shader-predict-test` - Run system diagnostics

### Options
- `--config` - Show configuration
- `--stats` - Performance statistics
- `--export-metrics` - Export performance data
- `--test` - Run comprehensive tests

## Python API

### ML Predictor
```python
from src.ml.optimized_ml_predictor import get_optimized_predictor

predictor = get_optimized_predictor()
prediction = predictor.predict_compilation_time(shader_features)
```

### Cache Manager
```python
from src.cache.optimized_shader_cache import get_optimized_cache

cache = get_optimized_cache()
cache.put(shader_entry)
entry = cache.get(shader_hash)
```

### Thermal Manager
```python
from src.thermal.optimized_thermal_manager import get_thermal_manager

thermal = get_thermal_manager()
thermal.start_monitoring()
status = thermal.get_status()
```
"""
            api_doc.write_text(api_content)
            log("  Created API.md")
        
        # Create contributing guide
        contrib_doc = docs_dir / "CONTRIBUTING.md"
        if not contrib_doc.exists():
            contrib_content = """# Contributing Guide

## Development Setup

1. Clone the repository
2. Run `./install.sh --dev` for development installation
3. Install pre-commit hooks: `pre-commit install`

## Testing

Run the test suite with:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Code Style

- Use Black for code formatting
- Use Ruff for linting  
- Add type hints to all functions
- Write comprehensive docstrings

## Performance Guidelines

- Profile code changes with benchmarks
- Test on Steam Deck hardware when possible
- Measure memory usage impact
- Validate thermal behavior

## Submitting Changes

1. Create a feature branch
2. Make your changes with tests
3. Run the full test suite
4. Submit a pull request with description

## Architecture Guidelines

- Maintain async/await patterns
- Use object pooling for frequently allocated objects
- Implement proper error handling and logging
- Follow the existing module structure
"""
            contrib_doc.write_text(contrib_content)
            log("  Created CONTRIBUTING.md")
    
    def create_examples(self):
        """Create usage examples"""
        log("Creating examples...")
        
        examples_dir = self.project_dir / "examples"
        
        # Create basic usage example
        basic_example = examples_dir / "basic_usage.py"
        if not basic_example.exists():
            example_content = """#!/usr/bin/env python3
\"\"\"
Basic usage example for the ML Shader Prediction Compiler
\"\"\"

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimized_main import OptimizedShaderSystem, SystemConfig


async def main():
    \"\"\"Basic usage example\"\"\"
    
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
"""
            basic_example.write_text(example_content)
            log("  Created basic_usage.py example")
    
    def create_config_templates(self):
        """Create configuration templates"""
        log("Creating configuration templates...")
        
        config_dir = self.project_dir / "config"
        
        # Steam Deck LCD config template
        lcd_config = config_dir / "steamdeck_lcd_config.json"
        if not lcd_config.exists():
            config_content = {
                "version": "2.0.0-optimized",
                "hardware_profile": "steamdeck_lcd",
                "system": {
                    "max_memory_mb": 150,
                    "max_compilation_threads": 4,
                    "enable_async": True,
                    "enable_thermal_management": True
                },
                "thermal": {
                    "apu_max": 95.0,
                    "cpu_max": 85.0,
                    "gpu_max": 90.0,
                    "prediction_threshold": 80.0
                },
                "ml": {
                    "backend": "lightgbm",
                    "cache_size": 500,
                    "enable_training": True
                },
                "cache": {
                    "hot_cache_size": 50,
                    "warm_cache_size": 200,
                    "enable_compression": True
                }
            }
            
            lcd_config.write_text(json.dumps(config_content, indent=2))
            log("  Created steamdeck_lcd_config.json")
        
        # Steam Deck OLED config template
        oled_config = config_dir / "steamdeck_oled_config.json"
        if not oled_config.exists():
            config_content = {
                "version": "2.0.0-optimized", 
                "hardware_profile": "steamdeck_oled",
                "system": {
                    "max_memory_mb": 200,
                    "max_compilation_threads": 6,
                    "enable_async": True,
                    "enable_thermal_management": True
                },
                "thermal": {
                    "apu_max": 97.0,
                    "cpu_max": 87.0,
                    "gpu_max": 92.0,
                    "prediction_threshold": 82.0
                },
                "ml": {
                    "backend": "lightgbm",
                    "cache_size": 1000,
                    "enable_training": True
                },
                "cache": {
                    "hot_cache_size": 100,
                    "warm_cache_size": 500,
                    "enable_compression": True
                }
            }
            
            oled_config.write_text(json.dumps(config_content, indent=2))
            log("  Created steamdeck_oled_config.json")
    
    def update_main_script(self):
        """Update main script name for consistency"""
        log("Updating main script...")
        
        old_main = self.project_dir / "optimized_main.py"
        new_main = self.project_dir / "main.py"
        
        if old_main.exists() and not new_main.exists():
            shutil.move(str(old_main), str(new_main))
            log("  Renamed optimized_main.py to main.py")
    
    def generate_cleanup_report(self):
        """Generate cleanup report"""
        log("Generating cleanup report...")
        
        report = {
            "cleanup_timestamp": time.time(),
            "cleanup_version": "1.0.0",
            "files_removed": len(self.files_to_remove),
            "backup_location": str(self.backup_dir),
            "new_structure": {
                "main_files": ["main.py", "install.sh", "uninstall.sh", "README.md"],
                "source_modules": ["ml", "cache", "thermal", "monitoring"], 
                "documentation": ["docs/ARCHITECTURE.md", "docs/API.md", "docs/CONTRIBUTING.md"],
                "examples": ["examples/basic_usage.py"],
                "configuration": ["config/steamdeck_lcd_config.json", "config/steamdeck_oled_config.json"],
                "testing": ["tests/unit/", "pytest.ini", "conftest.py"],
                "utilities": ["scripts/migrate_dependencies.py", "scripts/reorganize_structure.py"]
            }
        }
        
        report_file = self.project_dir / "cleanup_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        
        log(f"  Report saved to: cleanup_report.json")
    
    def show_final_structure(self):
        """Show the final clean structure"""
        log("Final project structure:")
        
        def show_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            
            items = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    show_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        show_tree(self.project_dir)
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        print(f"{Colors.BLUE}🧹 Final Project Cleanup{Colors.NC}")
        print("=" * 50)
        
        # Confirm with user
        print(f"\n{Colors.YELLOW}⚠️  This will:{Colors.NC}")
        print("  1. Backup redundant files")
        print("  2. Remove duplicate/outdated files")
        print("  3. Organize remaining files") 
        print("  4. Create clean documentation structure")
        print("  5. Add usage examples and config templates")
        
        # Auto-proceed for automated cleanup
        print(f"\n{Colors.GREEN}Proceeding with automated cleanup...{Colors.NC}")
        
        # Run cleanup steps
        self.create_backup()
        self.remove_redundant_files()
        self.organize_remaining_files()
        self.clean_src_directory()
        self.create_clean_docs()
        self.create_examples()
        self.create_config_templates()
        self.update_main_script()
        self.generate_cleanup_report()
        
        print()
        success("🎉 Project cleanup completed!")
        print(f"\n{Colors.PURPLE}📊 Cleanup Summary:{Colors.NC}")
        print(f"  • Backup location: {self.backup_dir}")
        print(f"  • Files removed: {len(self.files_to_remove)}")
        print(f"  • Clean structure created")
        print(f"  • Documentation added")
        print(f"  • Examples and templates created")
        
        print(f"\n{Colors.BLUE}📁 Final Structure:{Colors.NC}")
        self.show_final_structure()


if __name__ == "__main__":
    cleanup = ProjectCleanup()
    cleanup.run_cleanup()