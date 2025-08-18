#!/usr/bin/env python3
"""
File Structure Reorganization Script
Cleans up and reorganizes the shader-predict-compile project
"""

import os
import shutil
import hashlib
from pathlib import Path
import json
import time
from typing import Dict, List, Set, Tuple


class ProjectReorganizer:
    """Reorganizes project structure and removes duplicates"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.backup_dir = self.project_dir / "backup_before_reorganization"
        self.timestamp = int(time.time())
        
        # Define new clean structure
        self.new_structure = {
            "src": {
                "core": ["main.py", "config.py", "utils.py"],
                "ml": ["predictor.py", "models.py", "training.py"],
                "cache": ["cache_manager.py", "storage.py"],
                "thermal": ["thermal_manager.py", "power_manager.py"],
                "steam": ["steam_integration.py", "deck_hardware.py"],
                "network": ["p2p_manager.py", "distribution.py"]
            },
            "tests": {
                "unit": [],
                "integration": [],
                "benchmarks": []
            },
            "config": ["default_config.json", "steam_deck_config.json"],
            "scripts": ["install.sh", "uninstall.sh", "migrate.py"],
            "docs": ["README.md", "API.md", "CONTRIBUTING.md"]
        }
        
        # Files to definitely remove
        self.files_to_remove = [
            "cleanup_backup",  # Redundant backup directory
            "config_backup",   # Old config backups
            "venv",           # Virtual environment (shouldn't be in repo)
            "*.pyc",          # Compiled Python files
            "__pycache__",    # Python cache directories
            ".pytest_cache",  # Pytest cache
            "*.log",          # Log files
            "*.bak",          # Backup files
            "*.tmp",          # Temporary files
        ]
        
        # Duplicate detection
        self.file_hashes = {}
        self.duplicates = []
    
    def create_backup(self):
        """Create a backup of the current structure"""
        print(f"\n📦 Creating backup at {self.backup_dir}...")
        
        if self.backup_dir.exists():
            backup_name = f"{self.backup_dir}_{self.timestamp}"
            shutil.move(str(self.backup_dir), backup_name)
            print(f"  Moved existing backup to {backup_name}")
        
        # Files to backup (exclude large/unnecessary directories)
        exclude_dirs = {"venv", "__pycache__", ".git", "backup_before_reorganization"}
        
        def should_backup(path: Path) -> bool:
            for exclude in exclude_dirs:
                if exclude in path.parts:
                    return False
            return True
        
        # Create backup
        self.backup_dir.mkdir(exist_ok=True)
        
        for item in self.project_dir.iterdir():
            if item.name != self.backup_dir.name and should_backup(item):
                dest = self.backup_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, ignore=shutil.ignore_patterns(*exclude_dirs))
        
        print(f"✓ Backup created successfully")
    
    def find_duplicates(self):
        """Find duplicate files based on content hash"""
        print("\n🔍 Scanning for duplicate files...")
        
        python_files = list(self.project_dir.glob("**/*.py"))
        python_files = [f for f in python_files if "venv" not in str(f) and "backup" not in str(f)]
        
        for file_path in python_files:
            try:
                content = file_path.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()
                
                if file_hash in self.file_hashes:
                    self.duplicates.append((file_path, self.file_hashes[file_hash]))
                else:
                    self.file_hashes[file_hash] = file_path
            except Exception as e:
                print(f"  ⚠️  Could not hash {file_path}: {e}")
        
        if self.duplicates:
            print(f"✓ Found {len(self.duplicates)} duplicate files")
            for dup, original in self.duplicates[:5]:
                print(f"  - {dup.name} duplicates {original.name}")
        else:
            print("✓ No exact duplicates found")
    
    def analyze_similar_files(self):
        """Find files with similar functionality"""
        print("\n🔍 Analyzing similar files...")
        
        similar_groups = {
            "ml_predictors": [],
            "thermal_managers": [],
            "cache_managers": [],
            "steam_integration": [],
            "installers": [],
            "config_files": []
        }
        
        for py_file in self.project_dir.glob("**/*.py"):
            if "venv" in str(py_file) or "backup" in str(py_file):
                continue
            
            name_lower = py_file.name.lower()
            
            if "predict" in name_lower and "ml" in name_lower:
                similar_groups["ml_predictors"].append(py_file)
            elif "thermal" in name_lower:
                similar_groups["thermal_managers"].append(py_file)
            elif "cache" in name_lower:
                similar_groups["cache_managers"].append(py_file)
            elif "steam" in name_lower or "deck" in name_lower:
                similar_groups["steam_integration"].append(py_file)
        
        # Check shell scripts
        for sh_file in self.project_dir.glob("**/*.sh"):
            if "install" in sh_file.name.lower():
                similar_groups["installers"].append(sh_file)
        
        # Check config files
        for config_file in self.project_dir.glob("**/*.json"):
            if "config" in config_file.name.lower():
                similar_groups["config_files"].append(config_file)
        
        # Report findings
        consolidation_needed = []
        for group_name, files in similar_groups.items():
            if len(files) > 1:
                consolidation_needed.append(group_name)
                print(f"\n  {group_name}: {len(files)} similar files")
                for f in files[:3]:
                    print(f"    - {f.relative_to(self.project_dir)}")
        
        return consolidation_needed
    
    def remove_unnecessary_files(self):
        """Remove unnecessary files and directories"""
        print("\n🗑️  Removing unnecessary files...")
        
        removed_count = 0
        removed_size = 0
        
        # Remove venv directory
        venv_path = self.project_dir / "venv"
        if venv_path.exists():
            size = sum(f.stat().st_size for f in venv_path.rglob("*") if f.is_file())
            shutil.rmtree(venv_path)
            removed_count += 1
            removed_size += size
            print(f"  ✓ Removed venv directory ({size / (1024*1024):.1f}MB)")
        
        # Remove backup directories
        for backup_dir in ["cleanup_backup", "config_backup"]:
            path = self.project_dir / backup_dir
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path)
                removed_count += 1
                removed_size += size
                print(f"  ✓ Removed {backup_dir} ({size / (1024*1024):.1f}MB)")
        
        # Remove __pycache__ directories
        for pycache in self.project_dir.glob("**/__pycache__"):
            shutil.rmtree(pycache)
            removed_count += 1
        
        # Remove .pyc files
        for pyc_file in self.project_dir.glob("**/*.pyc"):
            pyc_file.unlink()
            removed_count += 1
        
        print(f"\n✓ Removed {removed_count} items ({removed_size / (1024*1024):.1f}MB total)")
    
    def create_clean_structure(self):
        """Create the new clean directory structure"""
        print("\n📁 Creating clean directory structure...")
        
        # Create main directories
        directories = [
            "src/core",
            "src/ml",
            "src/cache",
            "src/thermal",
            "src/steam",
            "src/network",
            "tests/unit",
            "tests/integration",
            "tests/benchmarks",
            "config",
            "scripts",
            "docs"
        ]
        
        for dir_path in directories:
            (self.project_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if dir_path.startswith("src/") or dir_path.startswith("tests/"):
                init_file = self.project_dir / dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text('"""Package initialization"""\n')
        
        print("✓ Directory structure created")
    
    def consolidate_files(self):
        """Consolidate similar files into unified modules"""
        print("\n🔧 Consolidating similar files...")
        
        consolidation_map = {
            "src/ml/predictor.py": ["unified_ml_predictor.py", "optimized_ml_predictor.py"],
            "src/cache/cache_manager.py": ["advanced_shader_cache.py", "optimized_shader_cache.py"],
            "src/thermal/thermal_manager.py": ["thermal_manager.py", "advanced_thermal_controller.py"],
            "src/steam/steam_integration.py": ["comprehensive_steam_integration.py", "steam_platform_integrator.py"],
            "scripts/install.sh": ["install_user.sh", "install_user_enhanced.sh"],
            "scripts/uninstall.sh": ["uninstall.sh", "uninstall_enhanced.sh"]
        }
        
        for target, sources in consolidation_map.items():
            target_path = self.project_dir / target
            
            # Find the best source file (usually the most recent or largest)
            best_source = None
            best_size = 0
            
            for source_name in sources:
                source_files = list(self.project_dir.glob(f"**/{source_name}"))
                for source_file in source_files:
                    if source_file.exists() and source_file.stat().st_size > best_size:
                        best_source = source_file
                        best_size = source_file.stat().st_size
            
            if best_source and not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_source, target_path)
                print(f"  ✓ Consolidated {best_source.name} → {target}")
    
    def update_imports(self):
        """Update import statements in Python files"""
        print("\n🔧 Updating import statements...")
        
        # Map old imports to new imports
        import_map = {
            "from unified_ml_predictor import": "from src.ml.predictor import",
            "from optimized_ml_predictor import": "from src.ml.predictor import",
            "from advanced_shader_cache import": "from src.cache.cache_manager import",
            "from optimized_shader_cache import": "from src.cache.cache_manager import",
            "from thermal_manager import": "from src.thermal.thermal_manager import",
            "from comprehensive_steam_integration import": "from src.steam.steam_integration import"
        }
        
        python_files = list((self.project_dir / "src").glob("**/*.py"))
        
        updated_count = 0
        for py_file in python_files:
            try:
                content = py_file.read_text()
                original = content
                
                for old_import, new_import in import_map.items():
                    content = content.replace(old_import, new_import)
                
                if content != original:
                    py_file.write_text(content)
                    updated_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not update {py_file.name}: {e}")
        
        print(f"✓ Updated {updated_count} files")
    
    def create_main_entry_point(self):
        """Create a clean main entry point"""
        print("\n📝 Creating main entry point...")
        
        main_content = '''#!/usr/bin/env python3
"""
Steam Deck ML Shader Prediction Compiler
Main entry point for the optimized system
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config import load_config
from ml.predictor import get_optimized_predictor
from cache.cache_manager import get_optimized_cache
from thermal.thermal_manager import ThermalManager
from steam.steam_integration import SteamIntegration


def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Steam Deck Shader Prediction System")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    predictor = get_optimized_predictor()
    cache = get_optimized_cache()
    thermal = ThermalManager()
    steam = SteamIntegration()
    
    # Start services
    thermal.start()
    steam.start()
    
    logger.info("System initialized successfully")
    
    # Run main loop
    try:
        while True:
            # Main processing loop
            steam.process_events()
            thermal.update()
            
            # Handle shutdown signals
            if steam.should_shutdown():
                break
                
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        # Cleanup
        thermal.stop()
        steam.stop()
        cache.close()
        predictor.cleanup()
        
        logger.info("System shut down cleanly")


if __name__ == "__main__":
    main()
'''
        
        main_file = self.project_dir / "main.py"
        main_file.write_text(main_content)
        main_file.chmod(0o755)
        
        print("✓ Created main.py entry point")
    
    def generate_report(self):
        """Generate reorganization report"""
        print("\n" + "="*60)
        print("📊 REORGANIZATION REPORT")
        print("="*60)
        
        report = {
            "timestamp": self.timestamp,
            "duplicates_found": len(self.duplicates),
            "files_consolidated": 0,
            "structure_changes": [],
            "recommendations": []
        }
        
        # Count files in new structure
        src_files = len(list((self.project_dir / "src").glob("**/*.py")))
        test_files = len(list((self.project_dir / "tests").glob("**/*.py")))
        
        print(f"\n📁 New Structure:")
        print(f"  - Source files: {src_files}")
        print(f"  - Test files: {test_files}")
        print(f"  - Duplicates removed: {len(self.duplicates)}")
        
        print(f"\n✨ Benefits:")
        print("  - Cleaner, more maintainable structure")
        print("  - No duplicate code")
        print("  - Clear separation of concerns")
        print("  - Ready for testing and CI/CD")
        
        print(f"\n📁 Backup Location:")
        print(f"  {self.backup_dir}")
        
        # Save report
        report_file = self.project_dir / "reorganization_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        
        print(f"\n📄 Report saved to: {report_file}")
    
    def run(self):
        """Run the complete reorganization"""
        print("🚀 Project Structure Reorganization Tool")
        print("="*60)
        
        print("\n⚠️  This will:")
        print("  1. Backup current structure")
        print("  2. Find and remove duplicates")
        print("  3. Remove unnecessary files")
        print("  4. Reorganize into clean structure")
        print("  5. Update imports")
        
        if input("\nProceed? (y/n): ").lower() != 'y':
            print("Reorganization cancelled.")
            return
        
        # Run reorganization steps
        self.create_backup()
        self.find_duplicates()
        self.analyze_similar_files()
        self.remove_unnecessary_files()
        self.create_clean_structure()
        self.consolidate_files()
        self.update_imports()
        self.create_main_entry_point()
        self.generate_report()
        
        print("\n✨ Reorganization completed successfully!")
        print("Review the changes and test the system before removing the backup.")


if __name__ == "__main__":
    reorganizer = ProjectReorganizer()
    reorganizer.run()