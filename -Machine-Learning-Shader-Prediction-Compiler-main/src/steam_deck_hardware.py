#!/usr/bin/env python3
"""
Steam Deck Hardware Detection and Optimization Module

This module provides comprehensive Steam Deck hardware detection,
model identification, and hardware-specific optimizations for the
shader prediction compiler system.

Features:
- Accurate LCD vs OLED model detection
- Performance characteristics analysis
- Hardware-specific optimization settings
- Thermal and power management integration
- Memory and storage optimization
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDeckModel(Enum):
    """Steam Deck model enumeration"""
    UNKNOWN = "unknown"
    LCD_64GB = "lcd_64gb"
    LCD_256GB = "lcd_256gb"
    LCD_512GB = "lcd_512gb"
    OLED_512GB = "oled_512gb"
    OLED_1TB = "oled_1tb"


class SteamDeckGeneration(Enum):
    """Steam Deck generation enumeration"""
    UNKNOWN = "unknown"
    LCD = "lcd"      # Jupiter - Original Steam Deck
    OLED = "oled"    # Galileo - Steam Deck OLED


@dataclass
class HardwareSpecs:
    """Hardware specifications data class"""
    model: SteamDeckModel
    generation: SteamDeckGeneration
    apu_codename: str  # Aerith (LCD) or Sephiroth (OLED)
    display_size: float
    display_resolution: Tuple[int, int]
    display_refresh_rate: int
    display_brightness_nits: int
    battery_capacity_whr: int
    memory_speed_mt_s: int
    wifi_standard: str
    weight_grams: int
    manufacturing_process_nm: int
    estimated_release_year: int


@dataclass
class PerformanceProfile:
    """Performance optimization profile"""
    recommended_cache_size_mb: int
    max_parallel_jobs: int
    thermal_limit_celsius: int
    power_limit_watts: int
    memory_bandwidth_gb_s: float
    gpu_compute_units: int
    cpu_cores: int
    cpu_base_clock_mhz: int


class SteamDeckDetector:
    """Steam Deck hardware detection and identification system"""
    
    def __init__(self):
        self.detected_model = SteamDeckModel.UNKNOWN
        self.detected_generation = SteamDeckGeneration.UNKNOWN
        self.hardware_specs = None
        self.performance_profile = None
        self._detection_confidence = 0.0
        
        # Hardware specifications database
        self._hardware_database = self._initialize_hardware_database()
        
        # Perform detection on initialization
        self._detect_hardware()
    
    def _initialize_hardware_database(self) -> Dict[SteamDeckModel, HardwareSpecs]:
        """Initialize comprehensive hardware specifications database"""
        return {
            SteamDeckModel.LCD_64GB: HardwareSpecs(
                model=SteamDeckModel.LCD_64GB,
                generation=SteamDeckGeneration.LCD,
                apu_codename="Aerith",
                display_size=7.0,
                display_resolution=(1280, 800),
                display_refresh_rate=60,
                display_brightness_nits=400,
                battery_capacity_whr=40,
                memory_speed_mt_s=5500,
                wifi_standard="Wi-Fi 5",
                weight_grams=669,
                manufacturing_process_nm=7,
                estimated_release_year=2022
            ),
            SteamDeckModel.LCD_256GB: HardwareSpecs(
                model=SteamDeckModel.LCD_256GB,
                generation=SteamDeckGeneration.LCD,
                apu_codename="Aerith",
                display_size=7.0,
                display_resolution=(1280, 800),
                display_refresh_rate=60,
                display_brightness_nits=400,
                battery_capacity_whr=40,
                memory_speed_mt_s=5500,
                wifi_standard="Wi-Fi 5",
                weight_grams=669,
                manufacturing_process_nm=7,
                estimated_release_year=2022
            ),
            SteamDeckModel.LCD_512GB: HardwareSpecs(
                model=SteamDeckModel.LCD_512GB,
                generation=SteamDeckGeneration.LCD,
                apu_codename="Aerith",
                display_size=7.0,
                display_resolution=(1280, 800),
                display_refresh_rate=60,
                display_brightness_nits=400,
                battery_capacity_whr=40,
                memory_speed_mt_s=5500,
                wifi_standard="Wi-Fi 5",
                weight_grams=669,
                manufacturing_process_nm=7,
                estimated_release_year=2022
            ),
            SteamDeckModel.OLED_512GB: HardwareSpecs(
                model=SteamDeckModel.OLED_512GB,
                generation=SteamDeckGeneration.OLED,
                apu_codename="Sephiroth",
                display_size=7.4,
                display_resolution=(1280, 800),
                display_refresh_rate=90,
                display_brightness_nits=1000,
                battery_capacity_whr=50,
                memory_speed_mt_s=6400,
                wifi_standard="Wi-Fi 6E",
                weight_grams=640,
                manufacturing_process_nm=6,
                estimated_release_year=2023
            ),
            SteamDeckModel.OLED_1TB: HardwareSpecs(
                model=SteamDeckModel.OLED_1TB,
                generation=SteamDeckGeneration.OLED,
                apu_codename="Sephiroth",
                display_size=7.4,
                display_resolution=(1280, 800),
                display_refresh_rate=90,
                display_brightness_nits=1000,
                battery_capacity_whr=50,
                memory_speed_mt_s=6400,
                wifi_standard="Wi-Fi 6E",
                weight_grams=640,
                manufacturing_process_nm=6,
                estimated_release_year=2023
            )
        }
    
    def _detect_hardware(self) -> None:
        """Perform comprehensive hardware detection"""
        logger.info("Starting Steam Deck hardware detection...")
        
        detection_methods = [
            self._detect_via_dmi,
            self._detect_via_product_name,
            self._detect_via_cpu_info,
            self._detect_via_display_info,
            self._detect_via_battery_info,
            self._detect_via_network_info
        ]
        
        detection_scores = {}
        
        for method in detection_methods:
            try:
                result = method()
                if result and result[0] != SteamDeckModel.UNKNOWN:
                    model, confidence = result
                    if model not in detection_scores:
                        detection_scores[model] = []
                    detection_scores[model].append(confidence)
                    logger.debug(f"Detection method {method.__name__}: {model.value} (confidence: {confidence:.2f})")
            except Exception as e:
                logger.debug(f"Detection method {method.__name__} failed: {e}")
        
        # Aggregate detection results
        if detection_scores:
            best_model = max(detection_scores.keys(), 
                           key=lambda m: sum(detection_scores[m]) / len(detection_scores[m]))
            self.detected_model = best_model
            self.detected_generation = self._hardware_database[best_model].generation
            self.hardware_specs = self._hardware_database[best_model]
            self._detection_confidence = sum(detection_scores[best_model]) / len(detection_scores[best_model])
            
            # Generate performance profile
            self.performance_profile = self._generate_performance_profile()
            
            logger.info(f"Detected Steam Deck: {best_model.value} (confidence: {self._detection_confidence:.2f})")
        else:
            logger.warning("Could not reliably detect Steam Deck model")
    
    def _detect_via_dmi(self) -> Optional[Tuple[SteamDeckModel, float]]:
        """Detect Steam Deck via DMI (Desktop Management Interface) information"""
        try:
            dmi_paths = [
                "/sys/class/dmi/id/board_name",
                "/sys/devices/virtual/dmi/id/product_name",
                "/sys/class/dmi/id/product_name",
                "/sys/class/dmi/id/product_family"
            ]
            
            dmi_info = {}
            for path in dmi_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            dmi_info[path] = f.read().strip()
                    except:
                        continue
            
            if not dmi_info:
                return None
            
            logger.debug(f"DMI info: {dmi_info}")
            
            # Check for Jupiter (LCD models)
            if any("Jupiter" in value for value in dmi_info.values()):
                # Try to determine specific LCD model by storage info
                return self._determine_lcd_model(), 0.9
            
            # Check for Galileo (OLED models)  
            if any("Galileo" in value for value in dmi_info.values()):
                # Try to determine specific OLED model by storage info
                return self._determine_oled_model(), 0.9
            
            # Check for Valve as manufacturer
            if any("Valve" in value for value in dmi_info.values()):
                return SteamDeckModel.UNKNOWN, 0.5  # Valve device, but model unclear
            
            return None
            
        except Exception as e:
            logger.debug(f"DMI detection failed: {e}")
            return None
    
    def _detect_via_product_name(self) -> Optional[Tuple[SteamDeckModel, float]]:
        """Detect via product name in system information"""
        try:
            # Check /proc/version
            if os.path.exists("/proc/version"):
                with open("/proc/version", 'r') as f:
                    version_info = f.read()
                if "steamdeck" in version_info.lower() or "steamos" in version_info.lower():
                    return SteamDeckModel.UNKNOWN, 0.6  # SteamOS detected
            
            # Check for SteamOS-specific files
            steamos_files = [
                "/etc/os-release",
                "/usr/share/steamos-release"
            ]
            
            for file_path in steamos_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().lower()
                        if "steamos" in content:
                            return SteamDeckModel.UNKNOWN, 0.7
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Product name detection failed: {e}")
            return None
    
    def _detect_via_cpu_info(self) -> Optional[Tuple[SteamDeckModel, float]]:
        """Detect via CPU information"""
        try:
            if not os.path.exists("/proc/cpuinfo"):
                return None
            
            with open("/proc/cpuinfo", 'r') as f:
                cpu_info = f.read()
            
            # Look for AMD Van Gogh (Steam Deck APU)
            if "AMD Custom APU 0405" in cpu_info:
                # This is the Steam Deck APU
                # Check for specific model indicators
                if "Sephiroth" in cpu_info or "6nm" in cpu_info:
                    return SteamDeckModel.OLED_512GB, 0.8  # Default to base OLED model
                else:
                    return SteamDeckModel.LCD_256GB, 0.8   # Default to base LCD model
            
            # Look for Van Gogh architecture
            if "Van Gogh" in cpu_info or "VanGogh" in cpu_info:
                return SteamDeckModel.UNKNOWN, 0.6
            
            return None
            
        except Exception as e:
            logger.debug(f"CPU info detection failed: {e}")
            return None
    
    def _detect_via_display_info(self) -> Optional[Tuple[SteamDeckModel, float]]:
        """Detect via display information"""
        try:
            # Try to get display information
            display_commands = [
                ["xrandr", "--query"],
                ["drm_info"],
                ["cat", "/sys/class/drm/card0*/modes"]
            ]
            
            for cmd in display_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout
                        
                        # Check resolution (Steam Deck uses 1280x800)
                        if "1280x800" in output:
                            # Check for 90Hz (OLED) vs 60Hz (LCD)
                            if "90.00" in output or "90Hz" in output:
                                return SteamDeckModel.OLED_512GB, 0.7
                            elif "60.00" in output or "60Hz" in output:
                                return SteamDeckModel.LCD_256GB, 0.7
                            else:
                                return SteamDeckModel.UNKNOWN, 0.5
                        break
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Display info detection failed: {e}")
            return None
    
    def _detect_via_battery_info(self) -> Optional[Tuple[SteamDeckModel, float]]:
        """Detect via battery information"""
        try:
            battery_paths = [
                "/sys/class/power_supply/BAT1/energy_full_design",
                "/sys/class/power_supply/BAT0/energy_full_design"
            ]
            
            for path in battery_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            energy_uwh = int(f.read().strip())
                            energy_whr = energy_uwh / 1000000  # Convert µWh to Wh
                        
                        # Steam Deck LCD: ~40Wh, OLED: ~50Wh
                        if 48 <= energy_whr <= 52:  # OLED range
                            return SteamDeckModel.OLED_512GB, 0.8
                        elif 38 <= energy_whr <= 42:  # LCD range
                            return SteamDeckModel.LCD_256GB, 0.8
                        elif 35 <= energy_whr <= 55:  # Broader Steam Deck range
                            return SteamDeckModel.UNKNOWN, 0.5
                        
                        break
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Battery info detection failed: {e}")
            return None
    
    def _detect_via_network_info(self) -> Optional[Tuple[SteamDeckModel, float]]:
        """Detect via network adapter information"""
        try:
            # Check for specific WiFi chipsets
            network_commands = [
                ["lspci", "-nn"],
                ["lsusb"],
                ["iwconfig"]
            ]
            
            for cmd in network_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout.lower()
                        
                        # Check for Steam Deck specific network adapters
                        # OLED models have Wi-Fi 6E, LCD models have Wi-Fi 5
                        if "wi-fi 6e" in output or "ax210" in output:
                            return SteamDeckModel.OLED_512GB, 0.6
                        elif "wi-fi 5" in output or "rtl8822ce" in output:
                            return SteamDeckModel.LCD_256GB, 0.6
                        
                        break
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Network info detection failed: {e}")
            return None
    
    def _determine_lcd_model(self) -> SteamDeckModel:
        """Determine specific LCD model based on storage information"""
        try:
            # Check storage size to determine model
            storage_info = self._get_storage_info()
            if storage_info:
                total_gb = storage_info.get("total_gb", 0)
                
                if total_gb >= 480:  # 512GB model (allowing for formatting overhead)
                    return SteamDeckModel.LCD_512GB
                elif total_gb >= 200:  # 256GB model
                    return SteamDeckModel.LCD_256GB
                elif total_gb >= 50:   # 64GB model
                    return SteamDeckModel.LCD_64GB
            
            # Default to most common LCD model
            return SteamDeckModel.LCD_256GB
            
        except Exception as e:
            logger.debug(f"LCD model determination failed: {e}")
            return SteamDeckModel.LCD_256GB
    
    def _determine_oled_model(self) -> SteamDeckModel:
        """Determine specific OLED model based on storage information"""
        try:
            # Check storage size to determine model
            storage_info = self._get_storage_info()
            if storage_info:
                total_gb = storage_info.get("total_gb", 0)
                
                if total_gb >= 950:  # 1TB model (allowing for formatting overhead)
                    return SteamDeckModel.OLED_1TB
                elif total_gb >= 480:  # 512GB model
                    return SteamDeckModel.OLED_512GB
            
            # Default to base OLED model
            return SteamDeckModel.OLED_512GB
            
        except Exception as e:
            logger.debug(f"OLED model determination failed: {e}")
            return SteamDeckModel.OLED_512GB
    
    def _get_storage_info(self) -> Optional[Dict[str, Any]]:
        """Get storage information for model identification"""
        try:
            # Get storage info from df command
            result = subprocess.run(["df", "/"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    fields = lines[1].split()
                    if len(fields) >= 2:
                        total_kb = int(fields[1])
                        total_gb = total_kb / (1024 * 1024)
                        return {"total_gb": total_gb}
            
            return None
            
        except Exception as e:
            logger.debug(f"Storage info retrieval failed: {e}")
            return None
    
    def _generate_performance_profile(self) -> PerformanceProfile:
        """Generate performance optimization profile based on detected hardware"""
        if not self.hardware_specs:
            # Default profile for unknown hardware
            return PerformanceProfile(
                recommended_cache_size_mb=1024,
                max_parallel_jobs=4,
                thermal_limit_celsius=85,
                power_limit_watts=15,
                memory_bandwidth_gb_s=88.0,
                gpu_compute_units=8,
                cpu_cores=4,
                cpu_base_clock_mhz=2400
            )
        
        # Generate profile based on hardware specs
        if self.detected_generation == SteamDeckGeneration.OLED:
            return PerformanceProfile(
                recommended_cache_size_mb=1536,  # More cache for better hardware
                max_parallel_jobs=6,
                thermal_limit_celsius=87,       # Better cooling
                power_limit_watts=20,           # Higher power budget
                memory_bandwidth_gb_s=102.4,    # 6400 MT/s memory
                gpu_compute_units=8,
                cpu_cores=4,
                cpu_base_clock_mhz=2800        # Higher clocks on 6nm
            )
        else:  # LCD
            return PerformanceProfile(
                recommended_cache_size_mb=1024,
                max_parallel_jobs=4,
                thermal_limit_celsius=85,
                power_limit_watts=15,
                memory_bandwidth_gb_s=88.0,     # 5500 MT/s memory
                gpu_compute_units=8,
                cpu_cores=4,
                cpu_base_clock_mhz=2400
            )
    
    def is_steam_deck(self) -> bool:
        """Check if the current system is a Steam Deck"""
        return self.detected_model != SteamDeckModel.UNKNOWN
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "detected": self.is_steam_deck(),
            "model": self.detected_model.value,
            "generation": self.detected_generation.value,
            "confidence": self._detection_confidence,
            "hardware_specs": self.hardware_specs.__dict__ if self.hardware_specs else None,
            "performance_profile": self.performance_profile.__dict__ if self.performance_profile else None
        }
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get optimization settings for shader compilation"""
        if not self.performance_profile:
            return {}
        
        settings = {
            "cache_size_mb": self.performance_profile.recommended_cache_size_mb,
            "parallel_jobs": self.performance_profile.max_parallel_jobs,
            "thermal_aware": True,
            "power_aware": True,
            "memory_bandwidth_optimization": True
        }
        
        # Add hardware-specific optimizations
        if self.detected_generation == SteamDeckGeneration.OLED:
            settings.update({
                "enable_90hz_optimization": True,
                "oled_power_optimization": True,
                "wifi6e_p2p_optimization": True,
                "memory_speed_boost": True
            })
        elif self.detected_generation == SteamDeckGeneration.LCD:
            settings.update({
                "enable_60hz_optimization": True,
                "battery_conservation": True,
                "thermal_throttling_aware": True
            })
        
        return settings
    
    def export_detection_report(self, filepath: Optional[str] = None) -> str:
        """Export detailed detection report"""
        report = {
            "detection_timestamp": __import__('datetime').datetime.now().isoformat(),
            "detection_results": self.get_model_info(),
            "optimization_settings": self.get_optimization_settings(),
            "system_info": {
                "platform": sys.platform,
                "python_version": sys.version,
                "architecture": os.uname() if hasattr(os, 'uname') else "Windows"
            }
        }
        
        report_json = json.dumps(report, indent=2)
        
        if filepath:
            Path(filepath).write_text(report_json)
            logger.info(f"Detection report exported to {filepath}")
        
        return report_json


def main():
    """Main function for command-line usage"""
    detector = SteamDeckDetector()
    
    print("🎮 Steam Deck Hardware Detection Report")
    print("=" * 50)
    
    model_info = detector.get_model_info()
    
    if detector.is_steam_deck():
        print(f"✅ Steam Deck Detected: {model_info['model']}")
        print(f"   Generation: {model_info['generation']}")
        print(f"   Confidence: {model_info['confidence']:.1%}")
        
        if detector.hardware_specs:
            specs = detector.hardware_specs
            print(f"\n📋 Hardware Specifications:")
            print(f"   Display: {specs.display_size}\" {specs.display_resolution[0]}x{specs.display_resolution[1]} @ {specs.display_refresh_rate}Hz")
            print(f"   Battery: {specs.battery_capacity_whr}Wh")
            print(f"   Memory: {specs.memory_speed_mt_s} MT/s")
            print(f"   WiFi: {specs.wifi_standard}")
            print(f"   APU: {specs.apu_codename} ({specs.manufacturing_process_nm}nm)")
        
        if detector.performance_profile:
            profile = detector.performance_profile
            print(f"\n⚡ Performance Profile:")
            print(f"   Recommended Cache: {profile.recommended_cache_size_mb}MB")
            print(f"   Max Parallel Jobs: {profile.max_parallel_jobs}")
            print(f"   Memory Bandwidth: {profile.memory_bandwidth_gb_s:.1f} GB/s")
            print(f"   Power Limit: {profile.power_limit_watts}W")
        
        optimization = detector.get_optimization_settings()
        print(f"\n🔧 Optimization Settings:")
        for key, value in optimization.items():
            print(f"   {key}: {value}")
            
    else:
        print("❌ Not a Steam Deck or unable to detect model")
        print("   This system may not be a Steam Deck or detection failed")
    
    # Export report
    report_path = "steam_deck_detection_report.json"
    detector.export_detection_report(report_path)
    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()