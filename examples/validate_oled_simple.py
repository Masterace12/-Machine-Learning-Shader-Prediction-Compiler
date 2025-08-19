#!/usr/bin/env python3
"""
Simple OLED Steam Deck Optimization Validation
Basic validation without complex dependencies
"""

import os
import sys
import time
import json
from pathlib import Path


def detect_steam_deck_model():
    """Detect Steam Deck model"""
    try:
        dmi_path = Path("/sys/class/dmi/id/product_name")
        if dmi_path.exists():
            product = dmi_path.read_text().strip()
            print(f"DMI Product Name: {product}")
            
            if "Galileo" in product:
                return "OLED"
            elif "Jupiter" in product:
                return "LCD"
        
        # Check board name as fallback
        board_path = Path("/sys/class/dmi/id/board_name") 
        if board_path.exists():
            board = board_path.read_text().strip()
            print(f"DMI Board Name: {board}")
            
            if "Galileo" in board:
                return "OLED"
            elif "Jupiter" in board:
                return "LCD"
                
        return "Unknown"
    except Exception as e:
        print(f"Model detection error: {e}")
        return "Error"


def check_thermal_sensors():
    """Check available thermal sensors"""
    sensors_found = []
    
    # Check thermal zones
    thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    for zone in thermal_zones:
        try:
            temp = int(zone.read_text()) / 1000.0
            sensors_found.append(f"Thermal Zone {zone.parent.name}: {temp:.1f}°C")
        except:
            pass
    
    # Check hwmon sensors
    hwmon_dirs = list(Path("/sys/class/hwmon").glob("hwmon*"))
    for hwmon_dir in hwmon_dirs:
        try:
            name_file = hwmon_dir / "name"
            if name_file.exists():
                name = name_file.read_text().strip()
                temp_files = list(hwmon_dir.glob("temp*_input"))
                for temp_file in temp_files:
                    try:
                        temp = int(temp_file.read_text()) / 1000.0
                        sensors_found.append(f"{name} {temp_file.name}: {temp:.1f}°C")
                    except:
                        pass
        except:
            pass
    
    return sensors_found


def check_gpu_info():
    """Check GPU information"""
    gpu_info = {}
    
    try:
        gpu_path = Path("/sys/class/drm/card0/device")
        
        # GPU utilization
        gpu_busy_file = gpu_path / "gpu_busy_percent"
        if gpu_busy_file.exists():
            gpu_info["utilization"] = gpu_busy_file.read_text().strip() + "%"
        
        # Power states
        pp_dpm_sclk = gpu_path / "pp_dpm_sclk"
        if pp_dpm_sclk.exists():
            sclk_data = pp_dpm_sclk.read_text().strip()
            # Find active state (marked with *)
            for line in sclk_data.split('\n'):
                if '*' in line:
                    gpu_info["gpu_clock"] = line.strip()
                    break
        
        # Memory clock
        pp_dpm_mclk = gpu_path / "pp_dpm_mclk"
        if pp_dpm_mclk.exists():
            mclk_data = pp_dpm_mclk.read_text().strip()
            for line in mclk_data.split('\n'):
                if '*' in line:
                    gpu_info["memory_clock"] = line.strip()
                    break
                    
    except Exception as e:
        gpu_info["error"] = str(e)
    
    return gpu_info


def check_battery_info():
    """Check battery information"""
    battery_info = {}
    
    try:
        battery_path = Path("/sys/class/power_supply/BAT1")
        if not battery_path.exists():
            battery_path = Path("/sys/class/power_supply/BAT0")
        
        if battery_path.exists():
            # Capacity
            capacity_file = battery_path / "capacity"
            if capacity_file.exists():
                battery_info["capacity"] = capacity_file.read_text().strip() + "%"
            
            # Status
            status_file = battery_path / "status" 
            if status_file.exists():
                battery_info["status"] = status_file.read_text().strip()
            
            # Power draw
            power_file = battery_path / "power_now"
            if power_file.exists():
                power_uw = int(power_file.read_text().strip())
                power_w = power_uw / 1_000_000
                battery_info["power_draw"] = f"{power_w:.1f}W"
    
    except Exception as e:
        battery_info["error"] = str(e)
    
    return battery_info


def check_cpu_info():
    """Check CPU information"""
    cpu_info = {}
    
    try:
        # CPU frequency
        freq_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if freq_file.exists():
            freq_khz = int(freq_file.read_text().strip())
            cpu_info["current_freq"] = f"{freq_khz // 1000}MHz"
        
        # CPU governor
        gov_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if gov_file.exists():
            cpu_info["governor"] = gov_file.read_text().strip()
        
        # CPU count
        cpu_info["cores"] = os.cpu_count()
        
    except Exception as e:
        cpu_info["error"] = str(e)
    
    return cpu_info


def check_memory_info():
    """Check memory information"""
    memory_info = {}
    
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
                memory_info["total_gb"] = f"{total_kb / (1024*1024):.1f}GB"
            elif line.startswith('MemAvailable:'):
                avail_kb = int(line.split()[1])
                memory_info["available_gb"] = f"{avail_kb / (1024*1024):.1f}GB"
                
    except Exception as e:
        memory_info["error"] = str(e)
    
    return memory_info


def test_configuration_files():
    """Test configuration files"""
    config_results = {}
    
    # Test OLED config
    oled_config_path = Path("config/steamdeck_oled_config.json")
    if oled_config_path.exists():
        try:
            with open(oled_config_path, 'r') as f:
                oled_config = json.load(f)
            config_results["oled_config"] = {
                "exists": True,
                "version": oled_config.get("version", "unknown"),
                "oled_optimized": oled_config.get("system", {}).get("oled_optimized", False),
                "max_threads": oled_config.get("system", {}).get("max_compilation_threads", 0)
            }
        except Exception as e:
            config_results["oled_config"] = {"exists": True, "error": str(e)}
    else:
        config_results["oled_config"] = {"exists": False}
    
    return config_results


def main():
    """Main validation function"""
    print("🎮 OLED Steam Deck Hardware Validation")
    print("=" * 50)
    
    # Hardware Detection
    print("\n🔍 Hardware Detection:")
    model = detect_steam_deck_model()
    print(f"Steam Deck Model: {model}")
    
    # System Information
    print(f"\n💻 System Information:")
    cpu_info = check_cpu_info()
    for key, value in cpu_info.items():
        print(f"  CPU {key}: {value}")
    
    memory_info = check_memory_info()
    for key, value in memory_info.items():
        print(f"  Memory {key}: {value}")
    
    # Thermal Monitoring
    print(f"\n🌡️  Thermal Sensors:")
    sensors = check_thermal_sensors()
    if sensors:
        for sensor in sensors[:5]:  # Show first 5
            print(f"  {sensor}")
        if len(sensors) > 5:
            print(f"  ... and {len(sensors) - 5} more")
    else:
        print("  No thermal sensors detected")
    
    # GPU Information
    print(f"\n🎮 GPU Information:")
    gpu_info = check_gpu_info()
    if gpu_info:
        for key, value in gpu_info.items():
            print(f"  GPU {key}: {value}")
    else:
        print("  No GPU information available")
    
    # Battery Information
    print(f"\n🔋 Battery Information:")
    battery_info = check_battery_info()
    if battery_info:
        for key, value in battery_info.items():
            print(f"  Battery {key}: {value}")
    else:
        print("  No battery information available")
    
    # Configuration Files
    print(f"\n⚙️  Configuration Files:")
    config_results = test_configuration_files()
    for config_name, config_data in config_results.items():
        if config_data.get("exists", False):
            if "error" in config_data:
                print(f"  {config_name}: EXISTS (Error: {config_data['error']})")
            else:
                version = config_data.get("version", "unknown")
                optimized = config_data.get("oled_optimized", False)
                threads = config_data.get("max_threads", 0)
                print(f"  {config_name}: {version} (OLED: {'✓' if optimized else '✗'}, Threads: {threads})")
        else:
            print(f"  {config_name}: NOT FOUND")
    
    # OLED-Specific Assessment
    print(f"\n🚀 OLED Optimization Assessment:")
    
    assessment_points = []
    
    # Model detection
    if model == "OLED":
        assessment_points.append("✅ OLED Steam Deck detected")
    elif model == "LCD":
        assessment_points.append("⚠️  LCD Steam Deck detected (OLED optimizations available but not optimal)")
    else:
        assessment_points.append("❌ Steam Deck model not detected")
    
    # Thermal sensors
    if sensors:
        assessment_points.append(f"✅ {len(sensors)} thermal sensors available for monitoring")
    else:
        assessment_points.append("❌ No thermal sensors detected")
    
    # GPU access
    if gpu_info and "error" not in gpu_info:
        assessment_points.append("✅ GPU monitoring and control available")
    else:
        assessment_points.append("⚠️  Limited GPU access")
    
    # Configuration
    oled_config = config_results.get("oled_config", {})
    if oled_config.get("exists") and oled_config.get("oled_optimized"):
        assessment_points.append("✅ OLED configuration file loaded with optimizations enabled")
    else:
        assessment_points.append("⚠️  OLED configuration missing or not optimized")
    
    # Memory availability (16GB for Steam Deck)
    total_memory = memory_info.get("total_gb", "0GB")
    if "15." in total_memory or "16." in total_memory:
        assessment_points.append("✅ Full 16GB memory detected")
    else:
        assessment_points.append(f"⚠️  Unexpected memory size: {total_memory}")
    
    for point in assessment_points:
        print(f"  {point}")
    
    # Overall Assessment
    print(f"\n" + "=" * 50)
    
    passed_checks = sum(1 for point in assessment_points if "✅" in point)
    total_checks = len(assessment_points)
    success_rate = passed_checks / total_checks
    
    if success_rate >= 0.8:
        print("🎉 EXCELLENT: System is ready for OLED optimizations!")
        result = "EXCELLENT"
    elif success_rate >= 0.6:
        print("✅ GOOD: System supports OLED optimizations with minor limitations")
        result = "GOOD"
    elif success_rate >= 0.4:
        print("⚠️  FAIR: System has partial OLED optimization support")
        result = "FAIR"
    else:
        print("❌ POOR: System needs configuration for OLED optimizations")
        result = "POOR"
    
    print(f"Assessment: {passed_checks}/{total_checks} checks passed ({success_rate:.1%})")
    print("=" * 50)
    
    # Save results
    results = {
        "timestamp": time.time(),
        "model": model,
        "assessment": result,
        "success_rate": success_rate,
        "checks_passed": f"{passed_checks}/{total_checks}",
        "system_info": {
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "battery": battery_info
        },
        "thermal_sensors": len(sensors),
        "configuration": config_results
    }
    
    results_file = Path("/tmp/oled_hardware_validation.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    
    return success_rate >= 0.6


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        sys.exit(1)