# Steam Deck QA Framework

A comprehensive game compatibility testing system for the shader prediction compiler on Steam Deck. This framework provides automated testing, validation, and reporting for shader cache systems across popular Steam Deck games.

## Features

### 🎮 Game Compatibility Testing
- Automated testing for popular Steam Deck games (Cyberpunk 2077, Elden Ring, Spider-Man, Portal 2, etc.)
- Multiple test scenarios per game (main menu, gameplay, loading, etc.)
- Proton compatibility validation across versions
- Multiplayer functionality testing

### 🛡️ Anti-Cheat System Validation
- Support for EAC, BattlEye, VAC, and Denuvo anti-tamper
- Shader cache compatibility verification
- Runtime compatibility testing
- File integrity validation
- Memory protection testing

### 🗃️ Shader Cache Integrity
- Comprehensive cache file validation
- Format-specific validation (DXVK, SPIR-V, Mesa, Fossilize)
- Checksum verification and cross-reference validation
- Temporal consistency checking
- Corruption detection and analysis

### 📊 Performance Benchmarking
- Real-time FPS monitoring and analysis
- Stutter detection and classification (minor, moderate, severe)
- Frame time consistency measurement
- Resource usage monitoring (CPU, GPU, Memory, VRAM)
- Shader compilation impact analysis
- Performance regression detection

### 🤖 ML Prediction Testing
- Prediction accuracy measurement and analysis
- Confusion matrix calculation (precision, recall, F1-score)
- Confidence calibration assessment
- Feature importance analysis
- Data quality assessment
- Training data export for model improvement

### 🌐 P2P Cache Sharing Validation
- Peer-to-peer shader sharing functionality testing
- Network connectivity validation
- Cache distribution verification
- Bandwidth usage monitoring
- Security validation (encryption, signatures)

### 📈 Comprehensive Reporting
- Executive summary reports for stakeholders
- Detailed technical analysis reports
- Regression analysis reports
- Interactive compatibility matrices
- Performance trend visualizations
- Exportable data formats (HTML, JSON)

## Installation

### Prerequisites
- Python 3.9 or higher
- Steam Deck or compatible Linux environment
- Steam client installed
- Git for version control

### Quick Setup
```bash
git clone <repository-url> steam-deck-qa-framework
cd steam-deck-qa-framework
pip install -r requirements.txt
```

### Configuration
Edit `config/qa_config.json` to customize:
- Steam installation path
- Game selection and settings
- Performance thresholds
- Validation parameters
- Reporting preferences

## Usage

### Full Test Suite
Run comprehensive tests on all configured games:
```bash
python main.py --full
```

### Single Game Testing
Test a specific game:
```bash
python main.py --game cyberpunk_2077
```

### Regression Testing
Compare against a baseline session:
```bash
python main.py --regression baseline_20231201_143022
```

### List Available Games
View all configured games:
```bash
python main.py --list-games
```

### Validate Configuration
Check configuration file validity:
```bash
python main.py --validate-config
```

## Configuration

### Game Configuration
Each game in the test suite can be configured with:
- Steam App ID
- Launch options
- Test scenarios
- Expected shader count
- Anti-cheat system type
- Test duration
- Critical graphics settings

### Performance Thresholds
Configure acceptable performance metrics:
- Target FPS (default: 60)
- Stutter thresholds
- Cache hit rate targets
- Memory usage limits

### Validation Settings
Control validation behavior:
- Shader compilation timeout
- Cache corruption thresholds
- Regression detection sensitivity
- ML accuracy requirements

## Test Scenarios

### Per-Game Scenarios
Different test scenarios are run for each game:

- **Cyberpunk 2077**: main_menu, driving, combat, raytracing
- **Elden Ring**: main_menu, open_world, boss_fight
- **Spider-Man**: swinging, combat, cutscenes
- **Portal 2**: main_menu, puzzle_solving, coop
- **Apex Legends**: main_menu, training, multiplayer
- **Destiny 2**: main_menu, patrol, strikes

### Test Duration
Each scenario runs for a configured duration (typically 3-8 minutes) to gather sufficient performance data and detect compilation patterns.

## Output Files

### Reports Directory (`data/reports/`)
- `comprehensive_report_<session_id>.html` - Full technical report
- `executive_summary_<session_id>.html` - High-level summary
- `compatibility_matrix_<session_id>.png` - Visual compatibility matrix
- `regression_report_<session_id>.html` - Regression analysis (if applicable)

### Data Directory (`data/`)
- `results/` - Raw test results in JSON format
- `telemetry/` - ML metrics and training data
- `logs/` - Detailed execution logs
- `baselines/` - Baseline performance data

### Charts and Visualizations
- Performance comparison charts
- Stutter analysis graphs
- Cache metrics visualizations
- Anti-cheat compatibility status
- ML prediction accuracy plots

## Architecture

### Core Components
- **SteamDeckQAFramework**: Main orchestrator
- **GameCompatibilityTester**: Game testing logic
- **AntiCheatValidator**: Anti-cheat system validation
- **ShaderCacheValidator**: Cache integrity checking
- **PerformanceAnalyzer**: Performance benchmarking
- **MLMetricsCollector**: ML data collection
- **QAReporter**: Report generation

### Testing Flow
1. **Pre-test validation**: Check game installation and system requirements
2. **Game launch**: Start game with monitoring enabled
3. **Scenario execution**: Run test scenarios with data collection
4. **Performance monitoring**: Collect FPS, stutter, and resource data
5. **Cache validation**: Verify shader cache integrity
6. **Anti-cheat testing**: Validate compatibility with security systems
7. **ML metrics**: Collect prediction accuracy data
8. **Report generation**: Create comprehensive reports and visualizations

## Advanced Features

### Regression Testing
Compare current performance against historical baselines to detect:
- Performance regressions
- New compatibility issues
- Fixed problems
- Trend analysis

### ML Model Improvement
Collect data to improve shader prediction models:
- Prediction accuracy metrics
- Feature importance analysis
- Confidence calibration data
- Training data export

### P2P Cache Testing
Validate peer-to-peer shader sharing:
- Network connectivity
- Cache distribution
- Security verification
- Bandwidth monitoring

## Troubleshooting

### Common Issues

**Game launch failures:**
- Verify Steam installation path in config
- Check game is installed and updated
- Ensure Proton version compatibility

**Performance monitoring issues:**
- Install MangoHUD for FPS monitoring
- Verify GPU monitoring tools (radeontop)
- Check system permissions for performance counters

**Anti-cheat compatibility:**
- Ensure anti-cheat services are running
- Check for kernel module dependencies
- Verify file system permissions

### Debug Mode
Enable detailed logging:
```bash
python main.py --full --debug
```

### Log Files
Check logs for detailed error information:
- `data/logs/qa_framework.log` - General framework logs
- `data/logs/qa_session_<session_id>.log` - Session-specific logs

## Contributing

### Development Setup
1. Install development dependencies
2. Run tests: `pytest tests/`
3. Format code: `black src/`
4. Type checking: `mypy src/`

### Adding New Games
1. Add game configuration to `config/qa_config.json`
2. Define test scenarios appropriate for the game
3. Set expected shader counts and performance targets
4. Test configuration with single game testing

### Extending Functionality
The framework is modular and extensible:
- Add new test scenarios in game configuration
- Extend performance metrics in PerformanceAnalyzer
- Add new report types in QAReporter
- Implement additional anti-cheat systems

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For support and questions:
1. Check the troubleshooting section above
2. Review log files for error details
3. Consult the configuration documentation
4. Report issues with detailed system information

## Roadmap

- [ ] Support for additional anti-cheat systems
- [ ] Real-time monitoring dashboard
- [ ] Cloud-based result comparison
- [ ] Automated performance optimization suggestions
- [ ] Integration with CI/CD pipelines
- [ ] Mobile device compatibility testing
- [ ] Advanced ML model training integration