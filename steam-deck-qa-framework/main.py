#!/usr/bin/env python3
"""
Steam Deck QA Framework - Main Entry Point
Comprehensive game compatibility testing system for shader prediction compiler
"""

import sys
import os
import asyncio
import argparse
import logging
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.steam_deck_qa_framework import SteamDeckQAFramework

def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    os.makedirs("data/logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data/logs/qa_framework.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("SteamDeckQA")

def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                Steam Deck QA Framework                        ║
    ║                                                               ║
    ║        Comprehensive Game Compatibility Testing System        ║
    ║            for Shader Prediction Compiler on Steam Deck      ║
    ║                                                               ║
    ║  Features:                                                    ║
    ║  • Automated game compatibility testing                       ║
    ║  • Anti-cheat system validation                               ║
    ║  • Shader cache integrity verification                        ║
    ║  • Performance benchmarking & stutter analysis               ║
    ║  • ML prediction accuracy testing                             ║
    ║  • P2P cache sharing validation                               ║
    ║  • Regression testing framework                               ║
    ║  • Comprehensive reporting system                             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def run_full_test_suite(config_path: str, debug: bool) -> int:
    """Run the full QA test suite"""
    try:
        logger = setup_logging(debug)
        logger.info("Starting Steam Deck QA Framework full test suite")
        
        # Initialize framework
        framework = SteamDeckQAFramework(config_path)
        
        # Run full test suite
        results = await framework.run_full_test_suite()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUITE COMPLETED")
        print("="*80)
        
        summary = results.get("summary", {})
        print(f"Session ID: {results.get('session_id', 'unknown')}")
        print(f"Total Games Tested: {summary.get('total_games_tested', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Warnings: {summary.get('warnings', 0)}")
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        critical_issues = summary.get('critical_issues', [])
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"  • {issue}")
        
        print(f"\nDetailed reports saved to: data/reports/")
        print(f"Session logs available at: data/logs/qa_session_{framework.session_id}.log")
        
        # Return exit code based on results
        if critical_issues:
            return 2  # Critical issues
        elif summary.get('failed', 0) > 0:
            return 1  # Some failures
        else:
            return 0  # All passed
    
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        print(f"\n❌ Test suite failed: {e}")
        return 3

async def run_regression_tests(config_path: str, baseline_session: str, debug: bool) -> int:
    """Run regression tests against baseline"""
    try:
        logger = setup_logging(debug)
        logger.info(f"Starting regression tests against baseline {baseline_session}")
        
        # Initialize framework
        framework = SteamDeckQAFramework(config_path)
        
        # Run regression tests
        results = await framework.run_regression_tests(baseline_session)
        
        # Print summary
        print("\n" + "="*80)
        print("REGRESSION TESTS COMPLETED")
        print("="*80)
        
        overall_regression = results.get("overall_regression", False)
        new_failures = results.get("new_failures", [])
        performance_regressions = results.get("performance_regressions", [])
        fixed_issues = results.get("fixed_issues", [])
        
        print(f"Overall Regression: {'YES' if overall_regression else 'NO'}")
        print(f"New Failures: {len(new_failures)}")
        print(f"Performance Regressions: {len(performance_regressions)}")
        print(f"Fixed Issues: {len(fixed_issues)}")
        
        if overall_regression:
            print(f"\n🚨 REGRESSION DETECTED!")
            if new_failures:
                print(f"New failures in: {', '.join(new_failures)}")
            if performance_regressions:
                perf_games = [reg["game"] for reg in performance_regressions]
                print(f"Performance regressions in: {', '.join(perf_games)}")
        else:
            print(f"\n✅ No regressions detected - safe to proceed")
            if fixed_issues:
                print(f"Fixed issues in: {', '.join(fixed_issues)}")
        
        print(f"\nRegression report saved to: data/reports/")
        
        return 1 if overall_regression else 0
    
    except Exception as e:
        logger.error(f"Regression tests failed with error: {e}")
        print(f"\n❌ Regression tests failed: {e}")
        return 3

async def run_single_game_test(config_path: str, game_name: str, debug: bool) -> int:
    """Run tests for a single game"""
    try:
        logger = setup_logging(debug)
        logger.info(f"Starting tests for single game: {game_name}")
        
        # Load config to check if game exists
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        test_games = config.get("test_games", {})
        if game_name not in test_games:
            available_games = list(test_games.keys())
            print(f"❌ Game '{game_name}' not found in configuration.")
            print(f"Available games: {', '.join(available_games)}")
            return 1
        
        # Initialize framework
        framework = SteamDeckQAFramework(config_path)
        
        # Test single game
        game_config = test_games[game_name]
        result = await framework._test_single_game(game_name, game_config)
        
        # Print results
        print("\n" + "="*80)
        print(f"SINGLE GAME TEST COMPLETED: {game_name.upper()}")
        print("="*80)
        
        status = result.get("status", "unknown")
        print(f"Status: {status.upper()}")
        
        # Show test results
        compatibility = result.get("compatibility_test", {})
        print(f"Compatibility: {compatibility.get('status', 'unknown')}")
        
        anticheat = result.get("anticheat_validation", {})
        if anticheat:
            print(f"Anti-cheat: {'Compatible' if anticheat.get('compatible', False) else 'Issues detected'}")
        
        cache_validation = result.get("cache_validation", {})
        print(f"Cache: {'Valid' if cache_validation.get('valid', False) else 'Invalid'}")
        
        performance = result.get("performance_analysis", {})
        if performance:
            overall_perf = performance.get("overall_performance", {})
            avg_fps = overall_perf.get("average_fps_all_scenarios", 0)
            print(f"Performance: {avg_fps:.1f} FPS average")
        
        return 0 if status == "passed" else 1
    
    except Exception as e:
        logger.error(f"Single game test failed with error: {e}")
        print(f"\n❌ Single game test failed: {e}")
        return 3

def list_available_games(config_path: str):
    """List all available games for testing"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        test_games = config.get("test_games", {})
        
        print("\n" + "="*80)
        print("AVAILABLE GAMES FOR TESTING")
        print("="*80)
        
        for game_name, game_config in test_games.items():
            app_id = game_config.get("app_id", "unknown")
            anticheat = game_config.get("anticheat", "none")
            scenarios = len(game_config.get("test_scenarios", []))
            
            print(f"{game_name.replace('_', ' ').title()}")
            print(f"  App ID: {app_id}")
            print(f"  Anti-cheat: {anticheat.upper() if anticheat else 'None'}")
            print(f"  Test scenarios: {scenarios}")
            print()
    
    except Exception as e:
        print(f"❌ Failed to load game list: {e}")

def validate_config(config_path: str) -> bool:
    """Validate configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ["steam_deck", "test_games", "validation", "telemetry"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required configuration section: {section}")
                return False
        
        # Check test games
        test_games = config.get("test_games", {})
        if not test_games:
            print("❌ No test games configured")
            return False
        
        for game_name, game_config in test_games.items():
            if "app_id" not in game_config:
                print(f"❌ Missing app_id for game: {game_name}")
                return False
        
        print("✅ Configuration validation passed")
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return False
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Steam Deck QA Framework - Comprehensive game compatibility testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --full                              # Run full test suite
  %(prog)s --regression baseline_20231201      # Run regression tests
  %(prog)s --game cyberpunk_2077              # Test single game
  %(prog)s --list-games                       # List available games
  %(prog)s --validate-config                  # Validate configuration
        """
    )
    
    parser.add_argument("--config", 
                       default="config/qa_config.json",
                       help="Path to QA configuration file")
    
    parser.add_argument("--full", 
                       action="store_true",
                       help="Run full test suite on all configured games")
    
    parser.add_argument("--regression", 
                       metavar="BASELINE_SESSION",
                       help="Run regression tests against baseline session ID")
    
    parser.add_argument("--game", 
                       metavar="GAME_NAME",
                       help="Run tests for a single game")
    
    parser.add_argument("--list-games", 
                       action="store_true",
                       help="List all available games for testing")
    
    parser.add_argument("--validate-config", 
                       action="store_true",
                       help="Validate configuration file")
    
    parser.add_argument("--debug", 
                       action="store_true",
                       help="Enable debug logging")
    
    parser.add_argument("--version", 
                       action="version",
                       version="Steam Deck QA Framework v1.0.0")
    
    args = parser.parse_args()
    
    # Print banner
    if not args.list_games and not args.validate_config:
        print_banner()
    
    # Validate configuration first
    if not validate_config(args.config):
        return 1
    
    try:
        # Handle different modes
        if args.list_games:
            list_available_games(args.config)
            return 0
        
        elif args.validate_config:
            # Already validated above
            return 0
        
        elif args.full:
            return asyncio.run(run_full_test_suite(args.config, args.debug))
        
        elif args.regression:
            return asyncio.run(run_regression_tests(args.config, args.regression, args.debug))
        
        elif args.game:
            return asyncio.run(run_single_game_test(args.config, args.game, args.debug))
        
        else:
            print("No action specified. Use --help for usage information.")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())