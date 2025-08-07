# Steam Deck QA Framework - Deployment Guide

This guide covers deployment of the Steam Deck QA Framework in production environments, including continuous integration, automated testing pipelines, and enterprise deployment scenarios.

## Overview

The Steam Deck QA Framework can be deployed in several configurations:
- **Local Development**: Single developer testing
- **CI/CD Integration**: Automated testing in build pipelines
- **Continuous Testing**: 24/7 monitoring and regression detection
- **Enterprise Scale**: Multi-node distributed testing

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Steam Deck or compatible Linux environment
- Python 3.9+
- 8GB RAM
- 50GB free disk space
- Steam client installed

**Recommended for Production:**
- 16GB RAM
- 100GB+ SSD storage
- Dedicated GPU monitoring tools
- Network connectivity for P2P testing

### Dependencies

Install system-level dependencies first:
```bash
# Steam Deck (Arch-based)
sudo steamos-readonly disable
sudo pacman -Sy python-pip python-virtualenv git htop glxinfo radeontop mangohud vulkan-tools
sudo steamos-readonly enable

# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv git htop mesa-utils radeontop vulkan-tools

# CentOS/RHEL
sudo dnf install python3-pip python3-virtualenv git htop mesa-utils radeontop vulkan-tools
```

## Installation

### Quick Setup

1. **Clone the repository:**
```bash
git clone <repository-url> steam-deck-qa-framework
cd steam-deck-qa-framework
```

2. **Run setup script:**
```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

3. **Activate environment:**
```bash
source venv/bin/activate
```

4. **Validate installation:**
```bash
python main.py --validate-config
python main.py --list-games
```

### Manual Installation

If the automated setup doesn't work for your environment:

1. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install Python dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Create directory structure:**
```bash
mkdir -p data/{logs,reports,results,telemetry,baselines}
mkdir -p config/templates
```

4. **Configure Steam path in config/qa_config.json:**
```json
{
  "steam_deck": {
    "steam_path": "/path/to/your/steam/installation"
  }
}
```

## Configuration

### Basic Configuration

Edit `config/qa_config.json` to customize the framework:

```json
{
  "steam_deck": {
    "steam_path": "/home/deck/.steam/steam",
    "cache_directory": "/home/deck/.steam/steam/steamapps/shadercache",
    "performance_metrics": {
      "target_fps": 60,
      "acceptable_stutter_threshold": 16.67,
      "cache_hit_target": 0.85
    }
  },
  "test_games": {
    "your_game": {
      "app_id": "123456",
      "launch_options": "-windowed",
      "test_scenarios": ["main_menu", "gameplay"],
      "expected_shaders": 500,
      "anticheat": "eac",
      "test_duration": 300
    }
  }
}
```

### Performance Tuning

For production deployments, tune these settings:

```json
{
  "validation": {
    "shader_timeout": 300,
    "test_duration": 900,
    "regression_threshold": 0.05,
    "memory_leak_detection": true
  },
  "environment": {
    "resource_limits": {
      "max_memory_usage": 2048,
      "max_cpu_usage": 80,
      "max_disk_usage": 10240
    }
  }
}
```

### Security Configuration

For environments with strict security requirements:

```json
{
  "p2p_shader_system": {
    "enabled": false,
    "security": {
      "encryption_enabled": true,
      "signature_verification": true,
      "peer_authentication": true
    }
  },
  "telemetry": {
    "collect_ml_data": false,
    "upload_anonymized_data": false
  }
}
```

## Deployment Scenarios

### 1. Local Development

For individual developers:

```bash
# Run single game test
python main.py --game cyberpunk_2077 --debug

# Run full test suite
python main.py --full

# Generate reports from previous session
python scripts/generate_test_report.py --session 20231201_143022 --type all
```

### 2. CI/CD Integration

#### GitHub Actions

Create `.github/workflows/qa-testing.yml`:

```yaml
name: Steam Deck QA Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  qa-testing:
    runs-on: self-hosted  # Requires Steam Deck runner
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup QA Framework
      run: |
        chmod +x scripts/setup_environment.sh
        ./scripts/setup_environment.sh
    
    - name: Run QA Tests
      run: |
        source venv/bin/activate
        python main.py --full --debug
    
    - name: Upload Reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: qa-reports
        path: data/reports/
    
    - name: Check for Critical Issues
      run: |
        source venv/bin/activate
        python -c "
        import json
        import sys
        import glob
        
        # Find latest results
        results_files = glob.glob('data/results/qa_results_*.json')
        if not results_files:
            sys.exit(0)
        
        latest_file = max(results_files, key=os.path.getctime)
        with open(latest_file) as f:
            results = json.load(f)
        
        critical_issues = results.get('summary', {}).get('critical_issues', [])
        if critical_issues:
            print(f'CRITICAL ISSUES FOUND: {len(critical_issues)}')
            for issue in critical_issues:
                print(f'  - {issue}')
            sys.exit(1)
        "
```

#### Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent { label 'steamdeck' }
    
    stages {
        stage('Setup') {
            steps {
                sh 'chmod +x scripts/setup_environment.sh'
                sh './scripts/setup_environment.sh'
            }
        }
        
        stage('QA Testing') {
            steps {
                sh '''
                    source venv/bin/activate
                    python main.py --full --debug
                '''
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    source venv/bin/activate
                    # Get latest session ID
                    LATEST_SESSION=$(ls -t data/results/qa_results_*.json | head -1 | grep -o '[0-9]\\{8\\}_[0-9]\\{6\\}')
                    python scripts/generate_test_report.py --session $LATEST_SESSION --type all
                '''
                
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'data/reports',
                    reportFiles: '*.html',
                    reportName: 'QA Test Reports'
                ])
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'data/reports/**', fingerprint: true
            archiveArtifacts artifacts: 'data/logs/**', fingerprint: true
        }
        
        failure {
            emailext (
                subject: "Steam Deck QA Tests Failed - Build ${BUILD_NUMBER}",
                body: "QA testing failed. Check the console output and reports for details.",
                to: "${QA_TEAM_EMAIL}"
            )
        }
    }
}
```

### 3. Continuous Testing

For 24/7 monitoring and regression detection:

```bash
# Start continuous testing (runs every 6 hours)
python scripts/run_continuous_testing.py --interval 6 --max-baseline-age 7

# Run as systemd service
sudo tee /etc/systemd/system/steamdeck-qa.service > /dev/null <<EOF
[Unit]
Description=Steam Deck QA Framework Continuous Testing
After=network.target

[Service]
Type=simple
User=deck
WorkingDirectory=/home/deck/steam-deck-qa-framework
ExecStart=/home/deck/steam-deck-qa-framework/venv/bin/python scripts/run_continuous_testing.py --interval 6
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable steamdeck-qa
sudo systemctl start steamdeck-qa
```

### 4. Enterprise Scale Deployment

For large organizations with multiple testing environments:

#### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    mesa-utils \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 qauser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R qauser:qauser /app

USER qauser

# Create data directories
RUN mkdir -p data/{logs,reports,results,telemetry,baselines}

EXPOSE 8080

CMD ["python", "main.py", "--full"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qa-framework:
    build: .
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - steam_cache:/home/deck/.steam/steam/steamapps/shadercache
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - qa-network
    
  qa-web-dashboard:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./data/reports:/usr/share/nginx/html
    networks:
      - qa-network

volumes:
  steam_cache:

networks:
  qa-network:
```

#### Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steamdeck-qa-framework
spec:
  replicas: 1
  selector:
    matchLabels:
      app: steamdeck-qa
  template:
    metadata:
      labels:
        app: steamdeck-qa
    spec:
      containers:
      - name: qa-framework
        image: steamdeck-qa:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: qa-data
          mountPath: /app/data
        - name: qa-config
          mountPath: /app/config
        env:
        - name: QA_MODE
          value: "continuous"
      volumes:
      - name: qa-data
        persistentVolumeClaim:
          claimName: qa-data-pvc
      - name: qa-config
        configMap:
          name: qa-config
```

## Monitoring and Alerting

### Health Checks

Create health check endpoints:

```python
# Add to main.py or create separate health_check.py
import json
from datetime import datetime, timedelta

def health_check():
    """Check system health"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage("data/")
    free_gb = disk_usage.free / (1024**3)
    health_status["checks"]["disk_space"] = {
        "status": "ok" if free_gb > 10 else "warning",
        "free_gb": free_gb
    }
    
    # Check recent test results
    import glob
    recent_results = glob.glob("data/results/qa_results_*.json")
    if recent_results:
        latest_file = max(recent_results, key=os.path.getctime)
        mod_time = datetime.fromtimestamp(os.path.getctime(latest_file))
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        
        health_status["checks"]["recent_tests"] = {
            "status": "ok" if age_hours < 24 else "warning",
            "last_test_hours_ago": age_hours
        }
    
    return health_status
```

### Prometheus Metrics

Export metrics for monitoring:

```python
# Create metrics_exporter.py
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import json
import glob
import time

# Define metrics
qa_test_duration = Histogram('qa_test_duration_seconds', 'Time spent on QA tests')
qa_pass_rate = Gauge('qa_pass_rate', 'Current pass rate of QA tests')
qa_critical_issues = Gauge('qa_critical_issues_total', 'Number of critical issues')
qa_tests_total = Counter('qa_tests_total', 'Total number of QA tests run')

def update_metrics():
    """Update Prometheus metrics from latest test results"""
    try:
        # Find latest results
        results_files = glob.glob("data/results/qa_results_*.json")
        if not results_files:
            return
        
        latest_file = max(results_files, key=os.path.getctime)
        with open(latest_file) as f:
            results = json.load(f)
        
        summary = results.get("summary", {})
        
        # Update metrics
        qa_pass_rate.set(summary.get("pass_rate", 0))
        qa_critical_issues.set(len(summary.get("critical_issues", [])))
        qa_tests_total.inc(summary.get("total_games_tested", 0))
        
    except Exception as e:
        print(f"Error updating metrics: {e}")

if __name__ == "__main__":
    # Start metrics server
    start_http_server(8000)
    
    while True:
        update_metrics()
        time.sleep(60)  # Update every minute
```

### Grafana Dashboard

Create dashboard configuration in `monitoring/grafana-dashboard.json`:

```json
{
  "dashboard": {
    "title": "Steam Deck QA Framework",
    "panels": [
      {
        "title": "Pass Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "qa_pass_rate",
            "legendFormat": "Pass Rate"
          }
        ]
      },
      {
        "title": "Critical Issues",
        "type": "stat",
        "targets": [
          {
            "expr": "qa_critical_issues_total",
            "legendFormat": "Critical Issues"
          }
        ]
      },
      {
        "title": "Test Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "qa_test_duration_seconds",
            "legendFormat": "Test Duration"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

**Steam not found:**
```bash
# Find Steam installation
find /home /usr -name "steam" -type f 2>/dev/null
# Update config/qa_config.json with correct path
```

**Permission errors:**
```bash
# Fix Steam directory permissions
sudo chown -R $(whoami):$(whoami) ~/.steam
# Or run with appropriate user permissions
```

**Performance monitoring issues:**
```bash
# Install MangoHUD
sudo pacman -S mangohud  # Arch/Steam Deck
sudo apt install mangohud  # Ubuntu

# Test MangoHUD
mangohud glxgears
```

**Memory issues:**
```bash
# Monitor memory usage
htop
# Adjust resource limits in config
vim config/qa_config.json
```

### Debug Mode

Enable comprehensive debugging:

```bash
# Run with debug logging
python main.py --full --debug

# Check logs
tail -f data/logs/qa_framework.log

# Verbose system monitoring
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

### Log Analysis

Analyze logs for issues:

```bash
# Check for errors
grep -i error data/logs/qa_framework.log

# Check performance issues
grep -i "stutter\|fps\|performance" data/logs/qa_framework.log

# Check anti-cheat issues
grep -i "anticheat\|eac\|battleye" data/logs/qa_framework.log
```

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Check log files for errors
- Monitor disk space usage
- Verify test completion

**Weekly:**
- Review test results and trends
- Update game configurations if needed
- Check for framework updates

**Monthly:**
- Clean up old log files and results
- Review and update baseline performance data
- Update system dependencies

### Backup Strategy

```bash
# Create backup script
cat > scripts/backup_data.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/steamdeck-qa"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR/$DATE"

# Backup critical data
cp -r data/results "$BACKUP_DIR/$DATE/"
cp -r data/baselines "$BACKUP_DIR/$DATE/"
cp -r config "$BACKUP_DIR/$DATE/"

# Compress backup
tar -czf "$BACKUP_DIR/qa_backup_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "qa_backup_*.tar.gz" -mtime +30 -delete
EOF

chmod +x scripts/backup_data.sh
```

### Updates and Upgrades

```bash
# Update framework code
git pull origin main

# Update Python dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Update system dependencies
sudo pacman -Syu  # Arch/Steam Deck
sudo apt update && sudo apt upgrade  # Ubuntu

# Validate after updates
python main.py --validate-config
```

## Security Considerations

### Access Control

- Run framework with dedicated user account
- Restrict file system permissions
- Use SSH keys for remote access
- Implement audit logging

### Network Security

- Disable P2P features in sensitive environments
- Use VPN for remote monitoring
- Implement firewall rules
- Monitor network traffic

### Data Protection

- Encrypt sensitive configuration data
- Secure backup storage
- Implement data retention policies
- Monitor for data leaks

## Support and Documentation

### Getting Help

1. **Check logs** in `data/logs/` directory
2. **Review configuration** in `config/qa_config.json`
3. **Run validation** with `--validate-config`
4. **Check system requirements** and dependencies
5. **Consult troubleshooting section** above

### Documentation

- `README.md` - General usage guide
- `DEPLOYMENT_GUIDE.md` - This deployment guide
- `config/qa_config.json` - Configuration reference
- `examples/` - Usage examples and advanced scenarios

### Community and Updates

- Check for framework updates regularly
- Report issues with detailed system information
- Contribute improvements and bug fixes
- Share deployment experiences and best practices

---

This deployment guide covers the major aspects of deploying the Steam Deck QA Framework in various environments. Adapt the configurations and procedures to match your specific requirements and infrastructure.