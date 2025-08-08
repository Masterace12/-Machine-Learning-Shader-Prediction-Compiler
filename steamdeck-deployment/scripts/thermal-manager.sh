#!/bin/bash
#
# Thermal Management Script for Shader Prediction Compiler
# Monitors and adjusts service behavior based on thermal state
#

set -euo pipefail

# Thermal zones and thresholds (in millidegrees Celsius)
readonly CPU_THERMAL_ZONE="/sys/class/thermal/thermal_zone0/temp"
readonly GPU_HWMON_PATH="/sys/class/hwmon/hwmon4/temp1_input"
readonly BATTERY_PATH="/sys/class/power_supply/BAT1"

# Temperature thresholds (in Celsius)
readonly TEMP_COOL=60
readonly TEMP_NORMAL=70
readonly TEMP_WARM=80
readonly TEMP_HOT=85
readonly TEMP_CRITICAL=90

# Service name
readonly SERVICE_NAME="shader-predict.service"

# Logging
log_thermal() {
    logger -t "shader-predict-thermal" "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Get current CPU temperature in Celsius
get_cpu_temp() {
    if [[ -f "$CPU_THERMAL_ZONE" ]]; then
        local temp_milli=$(cat "$CPU_THERMAL_ZONE")
        echo $((temp_milli / 1000))
    else
        echo 0
    fi
}

# Get current GPU temperature in Celsius
get_gpu_temp() {
    if [[ -f "$GPU_HWMON_PATH" ]]; then
        local temp_milli=$(cat "$GPU_HWMON_PATH")
        echo $((temp_milli / 1000))
    else
        echo 0
    fi
}

# Get battery level
get_battery_level() {
    if [[ -f "$BATTERY_PATH/capacity" ]]; then
        cat "$BATTERY_PATH/capacity"
    else
        echo 100
    fi
}

# Check if on battery power
is_on_battery() {
    if [[ -f "$BATTERY_PATH/status" ]]; then
        local status=$(cat "$BATTERY_PATH/status")
        [[ "$status" == "Discharging" ]]
    else
        return 1
    fi
}

# Adjust service CPU quota based on thermal state
adjust_cpu_quota() {
    local quota=$1
    systemctl --user set-property "$SERVICE_NAME" CPUQuota="${quota}%" 2>/dev/null || true
    log_thermal "Adjusted CPU quota to ${quota}%"
}

# Adjust service memory limit
adjust_memory_limit() {
    local limit=$1
    systemctl --user set-property "$SERVICE_NAME" MemoryMax="${limit}M" 2>/dev/null || true
    log_thermal "Adjusted memory limit to ${limit}MB"
}

# Stop service if needed
stop_service_if_needed() {
    if systemctl --user is-active "$SERVICE_NAME" &>/dev/null; then
        systemctl --user stop "$SERVICE_NAME"
        log_thermal "Service stopped due to thermal constraints"
    fi
}

# Main thermal management logic
manage_thermal_state() {
    local cpu_temp=$(get_cpu_temp)
    local gpu_temp=$(get_gpu_temp)
    local battery_level=$(get_battery_level)
    local max_temp=$((cpu_temp > gpu_temp ? cpu_temp : gpu_temp))
    
    log_thermal "CPU: ${cpu_temp}°C, GPU: ${gpu_temp}°C, Battery: ${battery_level}%"
    
    # Critical temperature - stop service
    if [[ $max_temp -ge $TEMP_CRITICAL ]]; then
        log_thermal "CRITICAL: Temperature ${max_temp}°C - stopping service"
        stop_service_if_needed
        return
    fi
    
    # Hot temperature - minimal resources
    if [[ $max_temp -ge $TEMP_HOT ]]; then
        log_thermal "HOT: Temperature ${max_temp}°C - minimal mode"
        adjust_cpu_quota 2
        adjust_memory_limit 200
        return
    fi
    
    # Warm temperature - reduced resources
    if [[ $max_temp -ge $TEMP_WARM ]]; then
        log_thermal "WARM: Temperature ${max_temp}°C - reduced mode"
        adjust_cpu_quota 5
        adjust_memory_limit 300
        return
    fi
    
    # Battery considerations
    if is_on_battery; then
        if [[ $battery_level -le 20 ]]; then
            log_thermal "Low battery (${battery_level}%) - ultra-low power mode"
            adjust_cpu_quota 2
            adjust_memory_limit 200
            return
        elif [[ $battery_level -le 50 ]]; then
            log_thermal "Battery mode (${battery_level}%) - power saving"
            adjust_cpu_quota 5
            adjust_memory_limit 350
            return
        fi
    fi
    
    # Normal temperature - standard resources
    if [[ $max_temp -ge $TEMP_NORMAL ]]; then
        log_thermal "NORMAL: Temperature ${max_temp}°C - standard mode"
        adjust_cpu_quota 10
        adjust_memory_limit 400
        return
    fi
    
    # Cool temperature - full resources
    log_thermal "COOL: Temperature ${max_temp}°C - full mode"
    adjust_cpu_quota 15
    adjust_memory_limit 500
}

# Check if game is running
is_game_running() {
    # Check for common game processes or Steam games
    if pgrep -f "steam.*SteamLaunch" &>/dev/null || \
       pgrep -f "wine.*exe" &>/dev/null || \
       pgrep -f "proton" &>/dev/null; then
        return 0
    fi
    return 1
}

# Main execution
main() {
    # Check if service is running
    if ! systemctl --user is-active "$SERVICE_NAME" &>/dev/null; then
        log_thermal "Service not running - skipping thermal check"
        exit 0
    fi
    
    # If game is running, apply stricter limits
    if is_game_running; then
        log_thermal "Game detected - applying gaming mode limits"
        adjust_cpu_quota 5
        adjust_memory_limit 300
    else
        # Normal thermal management
        manage_thermal_state
    fi
    
    # Write current state to file for monitoring
    local state_file="/tmp/shader-predict-thermal-state"
    cat > "$state_file" << EOF
{
    "timestamp": $(date +%s),
    "cpu_temp": $(get_cpu_temp),
    "gpu_temp": $(get_gpu_temp),
    "battery_level": $(get_battery_level),
    "on_battery": $(is_on_battery && echo "true" || echo "false"),
    "game_running": $(is_game_running && echo "true" || echo "false")
}
EOF
}

# Run main function
main "$@"