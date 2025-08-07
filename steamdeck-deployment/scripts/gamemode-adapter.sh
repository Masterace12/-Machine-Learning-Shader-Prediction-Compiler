#!/bin/bash
#
# Game Mode Adapter for Shader Prediction Compiler
# Integrates with Steam's Gaming Mode for optimal performance
#

set -euo pipefail

# Constants
readonly SERVICE_NAME="shader-predict.service"
readonly GAMESCOPE_SOCKET="/tmp/gamescope-stats"
readonly STEAM_PIPE="/tmp/.steam-runtime"

# Logging
log_gamemode() {
    logger -t "shader-predict-gamemode" "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GAMEMODE: $1"
}

# Check if gamescope is running
is_gamescope_active() {
    pgrep -x "gamescope" > /dev/null 2>&1
}

# Check if a game is running
is_game_active() {
    # Check for Steam game launch
    if pgrep -f "steam.*SteamLaunch" > /dev/null 2>&1; then
        return 0
    fi
    
    # Check for Proton/Wine processes
    if pgrep -f "wine.*\.exe" > /dev/null 2>&1 || pgrep -f "proton" > /dev/null 2>&1; then
        return 0
    fi
    
    # Check for native Linux games
    if [[ -f /tmp/gamemode.active ]]; then
        return 0
    fi
    
    return 1
}

# Get current Steam app ID if available
get_steam_app_id() {
    local app_id=""
    
    # Try to get from environment
    if [[ -n "${SteamAppId:-}" ]]; then
        app_id="$SteamAppId"
    # Try to get from Steam runtime
    elif [[ -f "$STEAM_PIPE/appid" ]]; then
        app_id=$(cat "$STEAM_PIPE/appid" 2>/dev/null || echo "")
    # Try to get from process
    else
        local steam_proc=$(pgrep -f "steam.*SteamLaunch" | head -1)
        if [[ -n "$steam_proc" ]]; then
            app_id=$(ps -p "$steam_proc" -o args= | grep -oP 'AppId=\K\d+' || echo "")
        fi
    fi
    
    echo "$app_id"
}

# Adjust service for gaming mode
enter_gaming_mode() {
    log_gamemode "Entering gaming mode"
    
    # Get Steam app ID
    local app_id=$(get_steam_app_id)
    if [[ -n "$app_id" ]]; then
        log_gamemode "Game detected: Steam App ID $app_id"
        
        # Notify service about the game
        systemctl --user set-environment SHADER_PREDICT_GAME_ID="$app_id"
    fi
    
    # Apply ultra-low resource limits
    systemctl --user set-property "$SERVICE_NAME" CPUQuota=3% 2>/dev/null || true
    systemctl --user set-property "$SERVICE_NAME" MemoryMax=200M 2>/dev/null || true
    systemctl --user set-property "$SERVICE_NAME" IOWeight=1 2>/dev/null || true
    
    # Set to lowest priority
    local service_pid=$(systemctl --user show "$SERVICE_NAME" --property MainPID --value)
    if [[ -n "$service_pid" ]] && [[ "$service_pid" != "0" ]]; then
        renice 19 -p "$service_pid" 2>/dev/null || true
        ionice -c 3 -p "$service_pid" 2>/dev/null || true
    fi
    
    # Pause non-essential operations
    systemctl --user kill -s USR2 "$SERVICE_NAME" 2>/dev/null || true
    
    log_gamemode "Gaming mode active - service throttled"
}

# Restore normal operation
exit_gaming_mode() {
    log_gamemode "Exiting gaming mode"
    
    # Clear game ID
    systemctl --user unset-environment SHADER_PREDICT_GAME_ID
    
    # Restore normal resource limits
    systemctl --user set-property "$SERVICE_NAME" CPUQuota=10% 2>/dev/null || true
    systemctl --user set-property "$SERVICE_NAME" MemoryMax=500M 2>/dev/null || true
    systemctl --user set-property "$SERVICE_NAME" IOWeight=10 2>/dev/null || true
    
    # Restore normal priority
    local service_pid=$(systemctl --user show "$SERVICE_NAME" --property MainPID --value)
    if [[ -n "$service_pid" ]] && [[ "$service_pid" != "0" ]]; then
        renice 15 -p "$service_pid" 2>/dev/null || true
        ionice -c 2 -n 7 -p "$service_pid" 2>/dev/null || true
    fi
    
    # Resume normal operations
    systemctl --user kill -s CONT "$SERVICE_NAME" 2>/dev/null || true
    
    log_gamemode "Normal mode restored"
}

# Monitor Gamescope statistics
monitor_gamescope() {
    if [[ -S "$GAMESCOPE_SOCKET" ]]; then
        # Connect to gamescope stats socket
        local stats=$(timeout 1 nc -U "$GAMESCOPE_SOCKET" 2>/dev/null || echo "{}")
        
        # Parse FPS and frame time
        local fps=$(echo "$stats" | grep -oP '"fps":\K\d+' || echo "0")
        local frametime=$(echo "$stats" | grep -oP '"frametime":\K\d+' || echo "0")
        
        # If FPS is low, reduce service activity further
        if [[ "$fps" -gt 0 ]] && [[ "$fps" -lt 30 ]]; then
            log_gamemode "Low FPS detected ($fps) - pausing service"
            systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
        fi
    fi
}

# Main monitoring loop
main() {
    log_gamemode "Game mode adapter starting"
    
    # Initial state
    local in_gaming_mode=false
    local check_interval=5
    
    # Notify systemd we're ready
    systemd-notify --ready 2>/dev/null || true
    
    while true; do
        if is_gamescope_active || is_game_active; then
            if [[ "$in_gaming_mode" == false ]]; then
                enter_gaming_mode
                in_gaming_mode=true
            fi
            
            # Monitor performance while in gaming mode
            monitor_gamescope
        else
            if [[ "$in_gaming_mode" == true ]]; then
                exit_gaming_mode
                in_gaming_mode=false
            fi
        fi
        
        # Send watchdog notification
        systemd-notify WATCHDOG=1 2>/dev/null || true
        
        # Wait before next check
        sleep "$check_interval"
    done
}

# Handle signals
trap 'log_gamemode "Received SIGTERM"; exit 0' TERM
trap 'log_gamemode "Received SIGINT"; exit 0' INT

# Run main loop
main "$@"