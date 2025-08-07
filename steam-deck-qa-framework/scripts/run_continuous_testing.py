#!/usr/bin/env python3
"""
Continuous Testing Script for Steam Deck QA Framework
Runs automated tests at regular intervals and monitors for regressions
"""

import sys
import os
import asyncio
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.steam_deck_qa_framework import SteamDeckQAFramework

class ContinuousTestRunner:
    """Manages continuous testing and regression detection"""
    
    def __init__(self, config_path: str, interval_hours: int = 6, max_baseline_age_days: int = 7):
        self.config_path = config_path
        self.interval_hours = interval_hours
        self.max_baseline_age_days = max_baseline_age_days
        self.logger = self._setup_logging()
        self.running = False
        
    def _setup_logging(self):
        """Setup logging for continuous testing"""
        os.makedirs("data/logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("data/logs/continuous_testing.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger("ContinuousTestRunner")
    
    async def run_continuous_testing(self):
        """Run continuous testing loop"""
        self.logger.info("Starting continuous testing runner")
        self.running = True
        
        while self.running:
            try:
                # Run test cycle
                await self._run_test_cycle()
                
                # Wait for next interval
                self.logger.info(f"Waiting {self.interval_hours} hours until next test cycle...")
                
                # Sleep in smaller intervals to allow for graceful shutdown
                sleep_time = self.interval_hours * 3600  # Convert to seconds
                sleep_interval = 60  # Check every minute for shutdown signal
                
                while sleep_time > 0 and self.running:
                    await asyncio.sleep(min(sleep_interval, sleep_time))
                    sleep_time -= sleep_interval
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in continuous testing loop: {e}")
                # Wait before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    async def _run_test_cycle(self):
        """Run a single test cycle"""
        cycle_start = datetime.now()
        self.logger.info(f"Starting test cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Initialize framework
            framework = SteamDeckQAFramework(self.config_path)
            
            # Get current baseline
            baseline_session = await self._get_current_baseline()
            
            # Run full test suite
            self.logger.info("Running full test suite...")
            current_results = await framework.run_full_test_suite()
            
            # Check for critical issues
            summary = current_results.get("summary", {})
            critical_issues = summary.get("critical_issues", [])
            
            if critical_issues:
                self.logger.critical(f"CRITICAL ISSUES DETECTED: {len(critical_issues)} issues found")
                for issue in critical_issues:
                    self.logger.critical(f"  - {issue}")
                
                # Send alert
                await self._send_alert("critical", current_results)
            
            # Run regression tests if we have a baseline
            if baseline_session:
                self.logger.info(f"Running regression tests against baseline {baseline_session}")
                regression_results = await framework.run_regression_tests(baseline_session)
                
                if regression_results.get("overall_regression", False):
                    self.logger.warning("REGRESSION DETECTED")
                    await self._send_alert("regression", regression_results)
                else:
                    self.logger.info("No regressions detected")
            
            # Update baseline if needed
            await self._update_baseline_if_needed(current_results)
            
            # Generate summary report
            await self._generate_cycle_summary(current_results, baseline_session)
            
            cycle_duration = datetime.now() - cycle_start
            self.logger.info(f"Test cycle completed in {cycle_duration}")
            
        except Exception as e:
            self.logger.error(f"Test cycle failed: {e}")
            await self._send_alert("error", {"error": str(e)})
    
    async def _get_current_baseline(self) -> str:
        """Get the current baseline session ID"""
        try:
            baseline_dir = Path("data/baselines")
            if not baseline_dir.exists():
                return None
            
            # Find the most recent baseline
            baseline_files = list(baseline_dir.glob("performance_*_baseline.json"))
            if not baseline_files:
                return None
            
            # Get the most recent baseline
            latest_baseline = max(baseline_files, key=os.path.getctime)
            
            # Check if baseline is too old
            baseline_age = datetime.now() - datetime.fromtimestamp(os.path.getctime(latest_baseline))
            if baseline_age.days > self.max_baseline_age_days:
                self.logger.info(f"Baseline is {baseline_age.days} days old, will create new baseline")
                return None
            
            # Extract session ID from filename
            filename = latest_baseline.stem
            session_id = filename.replace("performance_", "").replace("_baseline", "")
            
            self.logger.info(f"Using baseline: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error getting current baseline: {e}")
            return None
    
    async def _update_baseline_if_needed(self, current_results: dict):
        """Update baseline if the current results are significantly better"""
        try:
            summary = current_results.get("summary", {})
            pass_rate = summary.get("pass_rate", 0)
            critical_issues = summary.get("critical_issues", [])
            
            # Only update baseline if results are good
            if pass_rate >= 0.9 and len(critical_issues) == 0:
                session_id = current_results.get("session_id")
                
                # Copy current results as new baseline
                baseline_dir = Path("data/baselines")
                baseline_dir.mkdir(exist_ok=True)
                
                baseline_file = baseline_dir / f"performance_{session_id}_baseline.json"
                
                with open(baseline_file, 'w') as f:
                    json.dump(current_results, f, indent=2)
                
                self.logger.info(f"Created new baseline: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating baseline: {e}")
    
    async def _send_alert(self, alert_type: str, data: dict):
        """Send alert for critical issues or regressions"""
        try:
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "type": alert_type,
                "data": data
            }
            
            # Save alert to file
            alerts_dir = Path("data/alerts")
            alerts_dir.mkdir(exist_ok=True)
            
            alert_file = alerts_dir / f"alert_{alert_type}_{int(time.time())}.json"
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            self.logger.info(f"Alert saved: {alert_file}")
            
            # Here you could add additional alert mechanisms:
            # - Email notifications
            # - Slack/Discord webhooks
            # - Database logging
            # - etc.
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    async def _generate_cycle_summary(self, current_results: dict, baseline_session: str):
        """Generate summary for this test cycle"""
        try:
            cycle_summary = {
                "timestamp": datetime.now().isoformat(),
                "session_id": current_results.get("session_id"),
                "baseline_session": baseline_session,
                "summary": current_results.get("summary", {}),
                "system_info": current_results.get("system_info", {}),
                "cycle_type": "continuous_testing"
            }
            
            # Save cycle summary
            summaries_dir = Path("data/cycle_summaries")
            summaries_dir.mkdir(exist_ok=True)
            
            summary_file = summaries_dir / f"cycle_{current_results.get('session_id')}.json"
            with open(summary_file, 'w') as f:
                json.dump(cycle_summary, f, indent=2)
            
            self.logger.info(f"Cycle summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating cycle summary: {e}")
    
    def stop(self):
        """Stop continuous testing"""
        self.logger.info("Stopping continuous testing...")
        self.running = False

async def run_single_cycle(config_path: str):
    """Run a single test cycle (for testing purposes)"""
    runner = ContinuousTestRunner(config_path, interval_hours=1)
    await runner._run_test_cycle()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Continuous Testing Runner for Steam Deck QA Framework"
    )
    
    parser.add_argument("--config",
                       default="../config/qa_config.json",
                       help="Path to QA configuration file")
    
    parser.add_argument("--interval",
                       type=int,
                       default=6,
                       help="Test interval in hours (default: 6)")
    
    parser.add_argument("--max-baseline-age",
                       type=int,
                       default=7,
                       help="Maximum baseline age in days (default: 7)")
    
    parser.add_argument("--single-cycle",
                       action="store_true",
                       help="Run a single test cycle and exit")
    
    args = parser.parse_args()
    
    try:
        if args.single_cycle:
            asyncio.run(run_single_cycle(args.config))
        else:
            runner = ContinuousTestRunner(
                args.config,
                args.interval,
                args.max_baseline_age
            )
            asyncio.run(runner.run_continuous_testing())
    
    except KeyboardInterrupt:
        print("\n⚠️  Continuous testing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Continuous testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())