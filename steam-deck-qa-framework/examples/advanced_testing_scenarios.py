#!/usr/bin/env python3
"""
Advanced Testing Scenarios for Steam Deck QA Framework
Examples of custom testing scenarios and use cases
"""

import sys
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.steam_deck_qa_framework import SteamDeckQAFramework

def setup_logging():
    """Setup logging for examples"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("AdvancedTesting")

async def stress_test_scenario():
    """Run intensive stress testing on shader compilation system"""
    logger = setup_logging()
    logger.info("Starting stress test scenario")
    
    # Create custom configuration for stress testing
    stress_config = {
        "steam_deck": {
            "steam_path": "/home/deck/.steam/steam",
            "cache_directory": "/home/deck/.steam/steam/steamapps/shadercache",
            "performance_metrics": {
                "target_fps": 60,
                "acceptable_stutter_threshold": 33.33,  # More lenient for stress test
                "cache_hit_target": 0.70  # Lower expectation for stress test
            }
        },
        "test_games": {
            "cyberpunk_2077": {
                "app_id": "1091500",
                "launch_options": "-windowed -novid",
                "test_scenarios": ["main_menu", "driving", "combat", "raytracing"],
                "expected_shaders": 2000,  # Higher expectation for stress test
                "test_duration": 600,  # 10 minutes per scenario
                "anticheat": None
            }
        },
        "validation": {
            "shader_timeout": 600,  # Longer timeout for stress test
            "test_duration": 1800,  # 30 minutes total
            "regression_threshold": 0.10,  # More lenient for stress test
            "cache_validation": True,
            "memory_leak_detection": True
        },
        "telemetry": {
            "collect_ml_data": True,
            "upload_anonymized_data": False,
            "local_storage_days": 30
        }
    }
    
    # Save temporary config
    temp_config_path = "/tmp/stress_test_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(stress_config, f, indent=2)
    
    try:
        # Initialize framework with stress test config
        framework = SteamDeckQAFramework(temp_config_path)
        
        # Run stress test
        results = await framework.run_full_test_suite()
        
        # Analyze results
        summary = results.get("summary", {})
        logger.info(f"Stress test completed:")
        logger.info(f"  Pass rate: {summary.get('pass_rate', 0):.1%}")
        logger.info(f"  Critical issues: {len(summary.get('critical_issues', []))}")
        
        return results
    
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

async def multi_proton_compatibility_test():
    """Test game compatibility across multiple Proton versions"""
    logger = setup_logging()
    logger.info("Starting multi-Proton compatibility test")
    
    proton_versions = ["8.0", "Experimental", "7.0", "6.3"]
    test_results = {}
    
    config_path = "../config/qa_config.json"
    
    for proton_version in proton_versions:
        logger.info(f"Testing with Proton {proton_version}")
        
        try:
            # Create version-specific config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Override Proton version
            config["steam_deck"]["proton_version"] = proton_version
            
            temp_config_path = f"/tmp/proton_{proton_version.replace('.', '_')}_config.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Initialize framework
            framework = SteamDeckQAFramework(temp_config_path)
            
            # Test single game (Cyberpunk 2077) as example
            game_config = config["test_games"]["cyberpunk_2077"]
            result = await framework._test_single_game("cyberpunk_2077", game_config)
            
            test_results[proton_version] = result
            
            logger.info(f"Proton {proton_version}: {result.get('status', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error testing Proton {proton_version}: {e}")
            test_results[proton_version] = {"status": "error", "error": str(e)}
        
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    # Generate compatibility report
    logger.info("\nProton Compatibility Results:")
    logger.info("=" * 40)
    for version, result in test_results.items():
        status = result.get("status", "unknown")
        logger.info(f"Proton {version}: {status.upper()}")
    
    return test_results

async def performance_regression_detection():
    """Advanced performance regression detection scenario"""
    logger = setup_logging()
    logger.info("Starting performance regression detection")
    
    config_path = "../config/qa_config.json"
    framework = SteamDeckQAFramework(config_path)
    
    # Run initial baseline test
    logger.info("Creating performance baseline...")
    baseline_results = await framework.run_full_test_suite()
    baseline_session = baseline_results.get("session_id")
    
    logger.info(f"Baseline created: {baseline_session}")
    
    # Simulate some time passing and potential changes
    await asyncio.sleep(5)
    
    # Run second test
    logger.info("Running comparison test...")
    current_results = await framework.run_full_test_suite()
    
    # Run regression analysis
    logger.info("Analyzing for regressions...")
    regression_results = await framework.run_regression_tests(baseline_session)
    
    # Detailed analysis
    if regression_results.get("overall_regression", False):
        logger.warning("PERFORMANCE REGRESSION DETECTED!")
        
        new_failures = regression_results.get("new_failures", [])
        if new_failures:
            logger.warning(f"New failures: {', '.join(new_failures)}")
        
        perf_regressions = regression_results.get("performance_regressions", [])
        if perf_regressions:
            logger.warning("Performance regressions:")
            for reg in perf_regressions:
                game = reg.get("game", "unknown")
                details = reg.get("details", {})
                logger.warning(f"  {game}: {details}")
    else:
        logger.info("No performance regressions detected")
        
        fixed_issues = regression_results.get("fixed_issues", [])
        if fixed_issues:
            logger.info(f"Fixed issues: {', '.join(fixed_issues)}")
    
    return regression_results

async def anticheat_compatibility_matrix():
    """Generate comprehensive anti-cheat compatibility matrix"""
    logger = setup_logging()
    logger.info("Starting anti-cheat compatibility matrix generation")
    
    # Games with different anti-cheat systems
    anticheat_games = {
        "elden_ring": "eac",
        "apex_legends": "eac", 
        "destiny_2": "battleye",
        "counter_strike_2": "vac",
        "portal_2": "vac"
    }
    
    compatibility_matrix = {}
    config_path = "../config/qa_config.json"
    
    for game, anticheat_type in anticheat_games.items():
        logger.info(f"Testing {game} ({anticheat_type})")
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if game not in config["test_games"]:
                logger.warning(f"Game {game} not found in config")
                continue
            
            # Initialize framework
            framework = SteamDeckQAFramework(config_path)
            
            # Test anti-cheat validation specifically
            from testing.anticheat_validator import AntiCheatValidator
            validator = AntiCheatValidator(config)
            
            app_id = config["test_games"][game]["app_id"]
            result = await validator.validate_anticheat_compatibility(app_id, anticheat_type)
            
            compatibility_matrix[game] = {
                "anticheat_type": anticheat_type,
                "compatible": result.get("compatible", False),
                "issues": result.get("issues_found", []),
                "recommendations": result.get("recommendations", [])
            }
            
            status = "✅" if result.get("compatible", False) else "❌"
            logger.info(f"{game}: {status}")
            
        except Exception as e:
            logger.error(f"Error testing {game}: {e}")
            compatibility_matrix[game] = {
                "anticheat_type": anticheat_type,
                "compatible": False,
                "error": str(e)
            }
    
    # Generate summary report
    logger.info("\nAnti-Cheat Compatibility Matrix:")
    logger.info("=" * 50)
    
    for game, data in compatibility_matrix.items():
        anticheat = data.get("anticheat_type", "unknown")
        compatible = data.get("compatible", False)
        status = "Compatible" if compatible else "Incompatible"
        
        logger.info(f"{game.replace('_', ' ').title()} ({anticheat.upper()}): {status}")
        
        if not compatible and "issues" in data:
            for issue in data["issues"][:3]:  # Show first 3 issues
                logger.info(f"  - {issue}")
    
    return compatibility_matrix

async def ml_model_evaluation_scenario():
    """Comprehensive ML model evaluation and improvement scenario"""
    logger = setup_logging()
    logger.info("Starting ML model evaluation scenario")
    
    config_path = "../config/qa_config.json"
    framework = SteamDeckQAFramework(config_path)
    
    # Test multiple games to collect ML metrics
    test_games = ["cyberpunk_2077", "elden_ring", "spider_man_remastered"]
    ml_results = {}
    
    for game in test_games:
        logger.info(f"Collecting ML metrics for {game}")
        
        try:
            # Load config to get game details
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if game not in config["test_games"]:
                logger.warning(f"Game {game} not found in config")
                continue
            
            # Use ML metrics collector
            from telemetry.ml_metrics_collector import MLMetricsCollector
            collector = MLMetricsCollector(config)
            
            app_id = config["test_games"][game]["app_id"]
            result = await collector.collect_prediction_metrics(app_id)
            
            ml_results[game] = result
            
            # Extract key metrics
            aggregated = result.get("aggregated_metrics")
            if aggregated:
                precision = getattr(aggregated, 'precision', 0)
                recall = getattr(aggregated, 'recall', 0)
                f1_score = getattr(aggregated, 'f1_score', 0)
                
                logger.info(f"{game} ML metrics:")
                logger.info(f"  Precision: {precision:.3f}")
                logger.info(f"  Recall: {recall:.3f}")
                logger.info(f"  F1 Score: {f1_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error collecting ML metrics for {game}: {e}")
    
    # Generate overall ML evaluation
    logger.info("\nML Model Evaluation Summary:")
    logger.info("=" * 40)
    
    total_games = len(ml_results)
    if total_games > 0:
        # Calculate average metrics
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        valid_results = 0
        
        for game, result in ml_results.items():
            aggregated = result.get("aggregated_metrics")
            if aggregated:
                avg_precision += getattr(aggregated, 'precision', 0)
                avg_recall += getattr(aggregated, 'recall', 0)
                avg_f1 += getattr(aggregated, 'f1_score', 0)
                valid_results += 1
        
        if valid_results > 0:
            avg_precision /= valid_results
            avg_recall /= valid_results
            avg_f1 /= valid_results
            
            logger.info(f"Average Precision: {avg_precision:.3f}")
            logger.info(f"Average Recall: {avg_recall:.3f}")
            logger.info(f"Average F1 Score: {avg_f1:.3f}")
            
            # Model performance assessment
            if avg_f1 >= 0.8:
                logger.info("✅ ML model performance is excellent")
            elif avg_f1 >= 0.7:
                logger.info("⚠️ ML model performance is good but could be improved")
            else:
                logger.info("❌ ML model performance needs significant improvement")
    
    return ml_results

async def comprehensive_stress_test():
    """Run comprehensive stress test combining multiple scenarios"""
    logger = setup_logging()
    logger.info("Starting comprehensive stress test")
    
    stress_test_results = {
        "start_time": datetime.now().isoformat(),
        "scenarios": {}
    }
    
    try:
        # 1. Basic stress test
        logger.info("Running basic stress test...")
        basic_result = await stress_test_scenario()
        stress_test_results["scenarios"]["basic_stress"] = basic_result
        
        # 2. Multi-Proton compatibility
        logger.info("Running multi-Proton compatibility test...")
        proton_result = await multi_proton_compatibility_test()
        stress_test_results["scenarios"]["proton_compatibility"] = proton_result
        
        # 3. Performance regression detection
        logger.info("Running performance regression detection...")
        regression_result = await performance_regression_detection()
        stress_test_results["scenarios"]["regression_detection"] = regression_result
        
        # 4. Anti-cheat compatibility
        logger.info("Running anti-cheat compatibility matrix...")
        anticheat_result = await anticheat_compatibility_matrix()
        stress_test_results["scenarios"]["anticheat_compatibility"] = anticheat_result
        
        # 5. ML model evaluation
        logger.info("Running ML model evaluation...")
        ml_result = await ml_model_evaluation_scenario()
        stress_test_results["scenarios"]["ml_evaluation"] = ml_result
        
        stress_test_results["end_time"] = datetime.now().isoformat()
        stress_test_results["duration"] = (
            datetime.fromisoformat(stress_test_results["end_time"]) -
            datetime.fromisoformat(stress_test_results["start_time"])
        ).total_seconds()
        
        logger.info(f"Comprehensive stress test completed in {stress_test_results['duration']:.1f} seconds")
        
        # Save results
        results_file = f"data/results/stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(stress_test_results, f, indent=2, default=str)
        
        logger.info(f"Stress test results saved to: {results_file}")
        
        return stress_test_results
    
    except Exception as e:
        logger.error(f"Comprehensive stress test failed: {e}")
        stress_test_results["error"] = str(e)
        return stress_test_results

def main():
    """Main entry point for advanced testing scenarios"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Testing Scenarios for Steam Deck QA Framework"
    )
    
    parser.add_argument("--scenario",
                       choices=[
                           "stress",
                           "proton",
                           "regression", 
                           "anticheat",
                           "ml-eval",
                           "comprehensive"
                       ],
                       default="comprehensive",
                       help="Testing scenario to run")
    
    args = parser.parse_args()
    
    try:
        if args.scenario == "stress":
            asyncio.run(stress_test_scenario())
        elif args.scenario == "proton":
            asyncio.run(multi_proton_compatibility_test())
        elif args.scenario == "regression":
            asyncio.run(performance_regression_detection())
        elif args.scenario == "anticheat":
            asyncio.run(anticheat_compatibility_matrix())
        elif args.scenario == "ml-eval":
            asyncio.run(ml_model_evaluation_scenario())
        elif args.scenario == "comprehensive":
            asyncio.run(comprehensive_stress_test())
    
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())