#!/usr/bin/env python3
"""
Test Report Generator
Generate comprehensive test reports from existing test data
"""

import sys
import os
import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reporting.qa_reporter import QAReporter

def setup_logging():
    """Setup logging for report generation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("ReportGenerator")

async def generate_report_from_session(session_id: str, config_path: str, report_type: str = "comprehensive"):
    """Generate report from existing session data"""
    logger = setup_logging()
    
    try:
        # Load session results
        results_file = f"data/results/qa_results_{session_id}.json"
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return False
        
        with open(results_file, 'r') as f:
            test_results = json.load(f)
        
        logger.info(f"Loaded results for session {session_id}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize reporter
        reporter = QAReporter(config)
        
        # Generate requested report type
        if report_type == "comprehensive":
            report_path = await reporter.generate_comprehensive_report(test_results, session_id)
            logger.info(f"Comprehensive report generated: {report_path}")
        
        elif report_type == "executive":
            report_path = await reporter.generate_executive_summary(test_results, session_id)
            logger.info(f"Executive summary generated: {report_path}")
        
        elif report_type == "matrix":
            report_path = await reporter.generate_compatibility_matrix(test_results, session_id)
            logger.info(f"Compatibility matrix generated: {report_path}")
        
        elif report_type == "all":
            # Generate all report types
            reports = []
            
            comp_report = await reporter.generate_comprehensive_report(test_results, session_id)
            reports.append(comp_report)
            
            exec_report = await reporter.generate_executive_summary(test_results, session_id)
            reports.append(exec_report)
            
            matrix_report = await reporter.generate_compatibility_matrix(test_results, session_id)
            reports.append(matrix_report)
            
            logger.info(f"All reports generated: {', '.join(reports)}")
        
        else:
            logger.error(f"Unknown report type: {report_type}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False

async def generate_regression_report(current_session: str, baseline_session: str, config_path: str):
    """Generate regression report comparing two sessions"""
    logger = setup_logging()
    
    try:
        # Load session results
        current_file = f"data/results/qa_results_{current_session}.json"
        baseline_file = f"data/results/qa_results_{baseline_session}.json"
        
        if not os.path.exists(current_file):
            logger.error(f"Current results file not found: {current_file}")
            return False
        
        if not os.path.exists(baseline_file):
            logger.error(f"Baseline results file not found: {baseline_file}")
            return False
        
        with open(current_file, 'r') as f:
            current_results = json.load(f)
        
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        logger.info(f"Loaded results: current={current_session}, baseline={baseline_session}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Perform regression analysis (simplified version)
        regression_analysis = {
            "overall_regression": False,
            "new_failures": [],
            "fixed_issues": [],
            "performance_regressions": []
        }
        
        # Compare game results
        current_games = current_results.get("game_results", {})
        baseline_games = baseline_results.get("game_results", {})
        
        for game in baseline_games:
            if game in current_games:
                baseline_status = baseline_games[game].get("status", "failed")
                current_status = current_games[game].get("status", "failed")
                
                if baseline_status == "passed" and current_status == "failed":
                    regression_analysis["new_failures"].append(game)
                    regression_analysis["overall_regression"] = True
                
                elif baseline_status == "failed" and current_status == "passed":
                    regression_analysis["fixed_issues"].append(game)
        
        # Initialize reporter and generate report
        reporter = QAReporter(config)
        report_path = await reporter.generate_regression_report(
            baseline_results, current_results, regression_analysis, current_session
        )
        
        logger.info(f"Regression report generated: {report_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating regression report: {e}")
        return False

def list_available_sessions():
    """List all available session IDs"""
    results_dir = Path("data/results")
    if not results_dir.exists():
        print("No results directory found")
        return
    
    session_files = list(results_dir.glob("qa_results_*.json"))
    if not session_files:
        print("No test session results found")
        return
    
    print("\nAvailable Test Sessions:")
    print("=" * 50)
    
    sessions = []
    for file in session_files:
        try:
            session_id = file.stem.replace("qa_results_", "")
            
            # Load basic info
            with open(file, 'r') as f:
                data = json.load(f)
            
            timestamp = data.get("timestamp", "unknown")
            if timestamp != "unknown":
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            summary = data.get("summary", {})
            total_games = summary.get("total_games_tested", 0)
            pass_rate = summary.get("pass_rate", 0)
            
            sessions.append({
                "id": session_id,
                "timestamp": timestamp,
                "games": total_games,
                "pass_rate": pass_rate
            })
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    
    for session in sessions:
        print(f"Session ID: {session['id']}")
        print(f"  Timestamp: {session['timestamp']}")
        print(f"  Games Tested: {session['games']}")
        print(f"  Pass Rate: {session['pass_rate']:.1%}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate test reports from existing session data"
    )
    
    parser.add_argument("--config",
                       default="../config/qa_config.json",
                       help="Path to QA configuration file")
    
    parser.add_argument("--session",
                       help="Session ID to generate report for")
    
    parser.add_argument("--type",
                       choices=["comprehensive", "executive", "matrix", "all"],
                       default="comprehensive",
                       help="Type of report to generate")
    
    parser.add_argument("--regression",
                       nargs=2,
                       metavar=("CURRENT", "BASELINE"),
                       help="Generate regression report comparing two sessions")
    
    parser.add_argument("--list-sessions",
                       action="store_true",
                       help="List all available test sessions")
    
    args = parser.parse_args()
    
    try:
        if args.list_sessions:
            list_available_sessions()
            return 0
        
        elif args.regression:
            current_session, baseline_session = args.regression
            success = asyncio.run(
                generate_regression_report(current_session, baseline_session, args.config)
            )
            return 0 if success else 1
        
        elif args.session:
            success = asyncio.run(
                generate_report_from_session(args.session, args.config, args.type)
            )
            return 0 if success else 1
        
        else:
            print("No action specified. Use --help for usage information.")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Report generation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())