#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SettingsSchema:
    """Settings schema with defaults and validation"""
    # Performance settings
    auto_start: bool = True
    low_power: bool = True
    memory_limit: int = 2048
    cleanup_days: int = 30
    
    # Compilation options
    continue_background: bool = False
    prioritize_common: bool = True
    steam_deck_optimize: bool = True
    thread_count: int = 4
    
    # Advanced settings
    cache_location: Optional[str] = None
    debug_mode: bool = False
    
    # Metadata
    schema_version: str = "1.0"
    last_modified: Optional[str] = None

class SettingsManager:
    """Comprehensive settings manager following best practices from notes.txt"""
    
    def __init__(self, config_name: str = "shader-predict-compile"):
        self.config_dir = Path.home() / '.config' / config_name
        self.settings_file = self.config_dir / 'settings.json'
        self.backup_file = self.config_dir / 'settings.json.backup'
        self.logger = logging.getLogger(__name__)
        
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Current settings
        self._settings = SettingsSchema()
        
        # Load existing settings
        self.load_settings()
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default settings schema"""
        return asdict(SettingsSchema())
    
    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize settings"""
        defaults = self.get_defaults()
        validated = {}
        
        for key, default_value in defaults.items():
            if key in settings:
                value = settings[key]
                
                # Type validation
                if isinstance(default_value, bool):
                    validated[key] = bool(value)
                elif isinstance(default_value, int):
                    try:
                        validated[key] = int(value)
                        # Range validation for specific settings
                        if key == 'memory_limit':
                            validated[key] = max(512, min(16384, validated[key]))
                        elif key == 'cleanup_days':
                            validated[key] = max(1, min(365, validated[key]))
                        elif key == 'thread_count':
                            validated[key] = max(1, min(16, validated[key]))
                    except (ValueError, TypeError):
                        validated[key] = default_value
                elif isinstance(default_value, str):
                    validated[key] = str(value) if value is not None else default_value
                else:
                    validated[key] = value
            else:
                validated[key] = default_value
        
        # Update metadata
        validated['last_modified'] = datetime.now().isoformat()
        
        return validated
    
    def load_settings(self) -> bool:
        """Load settings from file with error handling and recovery"""
        try:
            if not self.settings_file.exists():
                self.logger.info("No settings file found, using defaults")
                return True
            
            # Try to load main settings file
            with open(self.settings_file, 'r') as f:
                raw_settings = json.load(f)
            
            # Validate and apply settings
            validated_settings = self.validate_settings(raw_settings)
            self._settings = SettingsSchema(**validated_settings)
            
            self.logger.info(f"Settings loaded successfully from {self.settings_file}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON in settings file: {e}")
            return self._try_backup_recovery()
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return self._try_backup_recovery()
    
    def _try_backup_recovery(self) -> bool:
        """Try to recover from backup file"""
        try:
            if self.backup_file.exists():
                self.logger.info("Attempting recovery from backup")
                with open(self.backup_file, 'r') as f:
                    backup_settings = json.load(f)
                
                validated_settings = self.validate_settings(backup_settings)
                self._settings = SettingsSchema(**validated_settings)
                
                # Restore the main file from backup
                self.save_settings()
                self.logger.info("Successfully recovered from backup")
                return True
        except Exception as e:
            self.logger.error(f"Backup recovery failed: {e}")
        
        # If all else fails, use defaults
        self.logger.info("Using default settings")
        self._settings = SettingsSchema()
        return True
    
    def save_settings(self) -> bool:
        """Save settings with atomic write and backup"""
        try:
            # Create backup of existing file
            if self.settings_file.exists():
                import shutil
                shutil.copy2(self.settings_file, self.backup_file)
            
            # Update last modified time
            settings_dict = asdict(self._settings)
            settings_dict['last_modified'] = datetime.now().isoformat()
            
            # Atomic write using temporary file
            temp_file = self.settings_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(settings_dict, f, indent=2, sort_keys=True)
            
            # Move temp file to final location (atomic on most filesystems)
            temp_file.replace(self.settings_file)
            
            self.logger.info(f"Settings saved successfully to {self.settings_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return getattr(self._settings, key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Set a setting value with validation"""
        if hasattr(self._settings, key):
            # Validate the new value
            temp_dict = asdict(self._settings)
            temp_dict[key] = value
            validated = self.validate_settings(temp_dict)
            
            # Apply the validated value
            setattr(self._settings, key, validated[key])
            return True
        else:
            self.logger.warning(f"Unknown setting: {key}")
            return False
    
    def update(self, settings_dict: Dict[str, Any]) -> bool:
        """Update multiple settings at once"""
        try:
            current_dict = asdict(self._settings)
            current_dict.update(settings_dict)
            validated = self.validate_settings(current_dict)
            
            self._settings = SettingsSchema(**validated)
            return True
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults"""
        self._settings = SettingsSchema()
        return self.save_settings()
    
    def restore_to_defaults(self, include_cache: bool = False, include_logs: bool = False) -> Dict[str, bool]:
        """
        Comprehensive restore to default settings feature
        
        Args:
            include_cache: If True, also clears shader cache
            include_logs: If True, also clears log files
            
        Returns:
            Dictionary with results of each restoration step
        """
        results = {
            'settings_reset': False,
            'enhanced_settings_reset': False,
            'cache_cleared': False,
            'logs_cleared': False,
            'backup_created': False
        }
        
        try:
            # Create backup before resetting
            backup_dir = self.config_dir / 'backups'
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f'settings_backup_{timestamp}.json'
            
            if self.export_settings(backup_file):
                results['backup_created'] = True
                self.logger.info(f"Settings backup created: {backup_file}")
            
            # Reset main settings
            self._settings = SettingsSchema()
            if self.save_settings():
                results['settings_reset'] = True
                self.logger.info("Main settings reset to defaults")
            
            # Reset enhanced settings (if exists)
            enhanced_settings_file = self.config_dir / 'enhanced_settings.json'
            if enhanced_settings_file.exists():
                try:
                    # Create a default enhanced settings from the config template
                    config_enhanced = Path(__file__).parent.parent / 'config' / 'enhanced_settings.json'
                    if config_enhanced.exists():
                        import shutil
                        shutil.copy2(config_enhanced, enhanced_settings_file)
                        results['enhanced_settings_reset'] = True
                        self.logger.info("Enhanced settings reset to defaults")
                except Exception as e:
                    self.logger.warning(f"Could not reset enhanced settings: {e}")
            
            # Clear cache if requested
            if include_cache:
                cache_paths = [
                    Path.home() / '.cache/shader-predict-compile',
                    Path.home() / '.cache/fossilize',
                    self.config_dir / 'cache'
                ]
                
                cache_cleared = False
                for cache_path in cache_paths:
                    if cache_path.exists():
                        try:
                            import shutil
                            shutil.rmtree(cache_path)
                            cache_path.mkdir(parents=True, exist_ok=True)
                            cache_cleared = True
                            self.logger.info(f"Cleared cache: {cache_path}")
                        except Exception as e:
                            self.logger.warning(f"Could not clear cache {cache_path}: {e}")
                
                results['cache_cleared'] = cache_cleared
            
            # Clear logs if requested
            if include_logs:
                log_paths = [
                    Path.home() / '.cache/shader-predict-compile/logs',
                    self.config_dir / 'logs',
                    Path('/var/log/shader-predict-compile')
                ]
                
                logs_cleared = False
                for log_path in log_paths:
                    if log_path.exists():
                        try:
                            for log_file in log_path.glob('*.log'):
                                log_file.unlink()
                            logs_cleared = True
                            self.logger.info(f"Cleared logs: {log_path}")
                        except Exception as e:
                            self.logger.warning(f"Could not clear logs {log_path}: {e}")
                
                results['logs_cleared'] = logs_cleared
            
            # Log summary
            successful_operations = sum(1 for result in results.values() if result)
            total_operations = len([k for k in results.keys() if k != 'backup_created' or include_cache or include_logs])
            
            self.logger.info(f"Restore completed: {successful_operations}/{total_operations} operations successful")
            
        except Exception as e:
            self.logger.error(f"Error during restore to defaults: {e}")
        
        return results
    
    def export_settings(self, export_path: Path) -> bool:
        """Export settings to a file"""
        try:
            settings_dict = asdict(self._settings)
            with open(export_path, 'w') as f:
                json.dump(settings_dict, f, indent=2, sort_keys=True)
            return True
        except Exception as e:
            self.logger.error(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, import_path: Path) -> bool:
        """Import settings from a file"""
        try:
            with open(import_path, 'r') as f:
                imported_settings = json.load(f)
            
            return self.update(imported_settings)
        except Exception as e:
            self.logger.error(f"Error importing settings: {e}")
            return False
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings as a dictionary"""
        return asdict(self._settings)
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information"""
        return {
            'version': self._settings.schema_version,
            'last_modified': self._settings.last_modified,
            'settings_file': str(self.settings_file),
            'backup_file': str(self.backup_file)
        }

# Global settings manager instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager