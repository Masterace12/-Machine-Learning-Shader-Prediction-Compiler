# Steam Deck Integration Testing Guide

This document provides comprehensive guidance for running and understanding the Steam Deck integration test suite for the ML Shader Prediction Compiler project.

## Overview

The Steam Deck integration test suite provides thorough validation of all Steam Deck-specific functionality including:

- **Hardware Detection & Configuration**: LCD vs OLED model detection, hardware capability assessment
- **Thermal Management**: Temperature monitoring, thermal throttling, emergency protection
- **Power Management**: Battery optimization, AC power detection, power-aware scheduling
- **Steam Integration**: D-Bus integration, gaming mode detection, game launch/termination handling
- **Performance Optimization**: Cache management, background compilation scheduling, resource optimization
- **Complete Workflows**: End-to-end validation of real-world usage scenarios

## Test Structure

```
tests/
├── fixtures/
│   └── steamdeck_fixtures.py          # Mock hardware and test utilities
├── integration/
│   └── test_steamdeck_comprehensive_integration.py  # Complete integration tests
└── unit/
    ├── test_steamdeck_hardware_integration.py       # Hardware detection tests
    ├── test_steamdeck_thermal_optimizer.py          # Thermal management tests
    ├── test_steamdeck_enhanced_integration.py       # Advanced unit tests
    ├── test_steamdeck_cache_optimization.py         # Cache system tests
    └── test_steamdeck_dbus_integration.py           # D-Bus integration tests
```

## Test Categories

### Hardware Detection Tests
- Steam Deck model detection (LCD vs OLED)
- Hardware capability assessment
- Configuration loading for different models
- DMI information parsing
- CPU signature detection

### Thermal Management Tests
- Temperature sensor monitoring
- Thermal state classification
- Throttling behavior validation
- Emergency shutdown protection
- Performance scaling under thermal stress
- OLED vs LCD thermal limit differences

### Power Management Tests
- Battery level detection and optimization
- AC power vs battery operation
- Power draw monitoring and estimation
- Battery time remaining calculation
- Power-aware shader compilation scheduling

### Steam Integration Tests
- D-Bus connection and fallback mechanisms
- Gaming mode detection via multiple methods
- Steam process monitoring
- Game launch/termination event handling
- Steam overlay detection

### Performance Optimization Tests
- Memory-constrained cache management
- Background compilation scheduling
- Resource usage optimization
- Cache warming and eviction strategies
- SteamOS filesystem integration

### Workflow Tests
- Complete gaming session workflows
- System performance under various loads
- Error recovery and resilience
- Configuration persistence and loading

## Running Tests

### Using the Custom Test Runner (Recommended)

The project includes a specialized test runner for Steam Deck functionality:

```bash
# Run all Steam Deck tests with mocked hardware
./run_steamdeck_tests.py

# Run specific test category
./run_steamdeck_tests.py --category hardware
./run_steamdeck_tests.py --category thermal
./run_steamdeck_tests.py --category integration

# Run on real Steam Deck hardware (when available)
./run_steamdeck_tests.py --real-hw

# Run with specific markers
./run_steamdeck_tests.py --markers "steamdeck,thermal,power"

# Run specific test file
./run_steamdeck_tests.py --test-file tests/unit/test_steamdeck_thermal_optimizer.py

# List available test categories
./run_steamdeck_tests.py --list-categories

# Only detect hardware (no tests)
./run_steamdeck_tests.py --detect-only
```

### Using pytest directly

```bash
# Run all Steam Deck tests
pytest -m steamdeck

# Run unit tests only
pytest -m "steamdeck and unit"

# Run integration tests
pytest -m "steamdeck and integration"

# Run thermal management tests
pytest -m "steamdeck and thermal"

# Run with verbose output
pytest -m steamdeck -v

# Run performance benchmarks
pytest -m "steamdeck and benchmark" --benchmark-autosave
```

## Test Markers

The test suite uses pytest markers for flexible test selection:

- `steamdeck`: All Steam Deck-related tests
- `steamdeck_lcd`: LCD model-specific tests
- `steamdeck_oled`: OLED model-specific tests
- `unit`: Unit tests with mocking
- `integration`: Integration tests
- `hardware`: Hardware detection and monitoring
- `thermal`: Thermal management
- `power`: Power management
- `dbus`: D-Bus integration
- `steam`: Steam platform integration
- `cache`: Cache optimization
- `workflow`: Complete workflow tests
- `benchmark`: Performance benchmarks
- `gaming_mode`: Gaming mode functionality

## Mock Hardware System

The test suite includes a comprehensive mock hardware system that simulates Steam Deck hardware without requiring actual Steam Deck hardware:

### MockHardwareState

Simulates various Steam Deck hardware conditions:

```python
# Normal operation
hardware_state = MockHardwareState(
    model=MockSteamDeckModel.OLED_512GB,
    cpu_temperature=65.0,
    battery_capacity=80
)

# Thermal stress
hardware_state = create_thermal_stress_scenario(base_state)

# Low battery
hardware_state = create_battery_critical_scenario(base_state)

# Gaming mode
hardware_state = create_intensive_gaming_scenario(base_state)

# Docked mode
hardware_state = create_dock_scenario(base_state)
```

### Mock Filesystem

Simulates Steam Deck filesystem structure including:
- `/sys/class/thermal/` - Thermal sensors
- `/sys/class/power_supply/` - Battery information
- `/sys/devices/virtual/dmi/id/` - Hardware identification
- `/sys/class/drm/` - Display connections
- `/proc/cpuinfo` and `/proc/meminfo` - System information

### Mock D-Bus Interface

Simulates Steam D-Bus interface for testing:
- Gaming mode activation/deactivation
- Steam application lifecycle
- Signal monitoring and callbacks

## Hardware-Specific Testing

### LCD vs OLED Model Differences

Tests validate different behavior between LCD and OLED models:

- **LCD Models** (Jupiter): 7nm APU, 40Wh battery, conservative thermal limits
- **OLED Models** (Galileo): 6nm APU, 50Wh battery, better thermal performance

### Model-Specific Optimizations

- OLED models support higher sustained performance
- Different thermal limits and optimization profiles
- Enhanced power efficiency on OLED models
- Storage capacity variations (64GB/256GB/512GB/1TB)

## Performance Testing

The test suite includes performance benchmarks and regression detection:

### Benchmarks
- Hardware detection speed
- Thermal monitoring performance  
- State reading latency
- Optimization decision speed
- Memory usage monitoring

### Performance Assertions
- Hardware state reading < 100ms
- Optimization decisions < 500ms
- Initialization < 5 seconds
- Memory usage within reasonable bounds

## Test Environment Setup

### Dependencies

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-benchmark pytest-cov

# Optional: Install Steam Deck specific libraries
pip install psutil  # For system monitoring
```

### Environment Variables

- `STEAMDECK_TEST_MOCK=1`: Force mock hardware mode
- `SteamDeck=1`: Simulate Steam Deck environment
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: Disable plugin autoloading if needed

## Continuous Integration

The tests are designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Steam Deck Tests
  run: |
    python run_steamdeck_tests.py --category unit
    python run_steamdeck_tests.py --category integration
  env:
    STEAMDECK_TEST_MOCK: '1'
```

## Test Results and Reporting

The custom test runner generates comprehensive reports:

```json
{
  "timestamp": 1703123456.789,
  "test_duration_seconds": 45.2,
  "hardware_detection": {
    "is_steam_deck": true,
    "model": "oled",
    "detection_method": "dmi_galileo",
    "confidence": 0.95
  },
  "test_execution": {
    "return_code": 0,
    "success": true
  }
}
```

## Troubleshooting

### Common Issues

1. **ImportError for Steam Deck modules**:
   - Ensure all source modules are in the Python path
   - Check for missing dependencies

2. **Hardware detection failures**:
   - Verify DMI files exist (when testing on real hardware)
   - Check file permissions for `/sys/` access

3. **D-Bus connection errors**:
   - Tests should gracefully fall back to process-based detection
   - Ensure D-Bus service is running (on real systems)

4. **Performance test failures**:
   - Performance tests may be sensitive to system load
   - Run tests on a relatively idle system for accurate results

### Debug Mode

```bash
# Run with maximum verbosity
./run_steamdeck_tests.py --category unit -- -vv --tb=long

# Run single test with debugging
pytest tests/unit/test_steamdeck_hardware_integration.py::TestSteamDeckHardwareMonitor::test_steam_deck_detection_lcd -vv --tb=long
```

## Contributing to Tests

### Adding New Tests

1. **Unit Tests**: Add to appropriate `test_steamdeck_*.py` file
2. **Integration Tests**: Add to `test_steamdeck_comprehensive_integration.py`
3. **Mock Scenarios**: Add to `steamdeck_fixtures.py`

### Test Guidelines

1. **Use appropriate markers**: Mark tests with relevant pytest markers
2. **Mock hardware by default**: Use real hardware only when necessary
3. **Test error conditions**: Include tests for failure scenarios
4. **Performance awareness**: Consider performance implications
5. **Cross-model compatibility**: Test both LCD and OLED models

### Mock Hardware Development

```python
# Create custom hardware scenario
def create_custom_scenario(base_state: MockHardwareState) -> MockHardwareState:
    base_state.cpu_temperature = 95.0  # High temperature
    base_state.thermal_throttling = True
    base_state.fan_speed = 4800  # High fan speed
    return base_state

# Use in tests
hardware_state = create_custom_scenario(
    MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
)

with mock_steamdeck_environment(hardware_state):
    # Test your functionality
    pass
```

## Integration with Main Test Suite

The Steam Deck tests integrate with the main project test suite:

```bash
# Run all tests including Steam Deck tests
pytest

# Run only non-Steam Deck tests
pytest -m "not steamdeck"

# Include Steam Deck tests in coverage reporting
pytest --cov=src --cov-report=html
```

## Hardware Requirements for Real Hardware Testing

When running tests on actual Steam Deck hardware:

- **Steam Deck Console**: LCD or OLED model
- **SteamOS**: Latest stable version recommended
- **Python 3.9+**: Available through Flatpak or development mode
- **File System Access**: Tests require read access to `/sys/` and `/proc/`
- **Optional D-Bus Access**: For full Steam integration testing

## Future Enhancements

Planned improvements to the test suite:

- **Real-time hardware monitoring**: Live hardware state visualization
- **Performance regression tracking**: Historical performance comparison
- **Automated hardware profiling**: Generate hardware-specific profiles
- **Steam Game integration**: Testing with actual Steam games
- **Multi-device testing**: Testing across different Steam Deck units

---

For questions or issues related to Steam Deck testing, please refer to the main project documentation or open an issue with the `steamdeck` and `testing` labels.
