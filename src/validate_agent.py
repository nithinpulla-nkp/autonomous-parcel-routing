#!/usr/bin/env python3
"""
Agent validation script to ensure proper learning and avoid common pitfalls.

This script runs comprehensive validation tests on RL agents to verify:
- Actual learning vs random performance
- Convergence and stability
- Exploration behavior
- Generalization across scenarios
- Local optima avoidance
"""

import argparse
from pathlib import Path

from src.apr import WarehouseEnv
from src.apr.agents import create_agent
from src.apr.validation import RLAgentValidator


def main():
    parser = argparse.ArgumentParser(description='Validate RL agent performance')
    parser.add_argument('--algorithm', default='q_learning', 
                       help='Algorithm to validate (q_learning, double_q_learning, sarsa)')
    parser.add_argument('--train-episodes', type=int, default=300,
                       help='Episodes for training during validation')
    parser.add_argument('--test-episodes', type=int, default=50,
                       help='Episodes for testing')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of seeds for generalization testing')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-plots', type=str, default=None,
                       help='Path to save validation plots')
    
    args = parser.parse_args()
    
    print("🔍 RL Agent Validation Framework")
    print("=" * 50)
    print(f"Algorithm: {args.algorithm}")
    print(f"Training episodes: {args.train_episodes}")
    print(f"Test episodes: {args.test_episodes}")
    print(f"Generalization seeds: {args.seeds}")
    print()
    
    # Create environment and agent
    env = WarehouseEnv(seed=42)
    agent = create_agent(
        args.algorithm,
        env.observation_space,
        env.action_space,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3
    )
    
    print(f"Created agent: {agent}")
    print()
    
    # Run validation
    validator = RLAgentValidator(agent, env, verbose=True)
    
    results = validator.full_validation(
        training_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        n_seeds=args.seeds
    )
    
    # Print summary report
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY REPORT")
    print("="*60)
    
    summary = results['summary']
    print(f"Overall Assessment: {summary['overall_assessment']}")
    print()
    
    print("Component Scores:")
    for component, score in summary['scores'].items():
        status_icon = "✅" if score == "PASS" else "⚠️" if score == "WARNING" else "❌"
        print(f"  {status_icon} {component.capitalize()}: {score}")
    print()
    
    if summary['critical_issues']:
        print("❌ Critical Issues:")
        for issue in summary['critical_issues']:
            print(f"  - {issue}")
        print()
    
    if summary['warnings']:
        print("⚠️  Warnings:")
        for warning in summary['warnings']:
            print(f"  - {warning}")
        print()
    
    # Print key metrics
    print("Key Metrics:")
    learning_result = results['learning']
    print(f"  Learning vs Random: {learning_result['improvement']:.1f} reward improvement")
    print(f"  Statistical Significance: p={learning_result['statistical_test']['p_value']:.4f}")
    
    generalization = results['generalization']
    print(f"  Generalization Consistency: {generalization['consistency_score']:.3f}")
    
    exploration = results['exploration']
    print(f"  State Coverage: {exploration['state_coverage']['coverage_percent']:.1f}%")
    print(f"  Action Diversity: {exploration['action_diversity']['diversity_percent']:.1f}%")
    
    local_optima = results['local_optima']
    print(f"  Performance Range Across Runs: {local_optima['performance_range']:.1f}")
    
    # Visualization
    if args.visualize:
        print(f"\n📈 Generating validation visualizations...")
        save_path = Path(args.save_plots) if args.save_plots else None
        validator.visualize_results(save_path)
    
    # Final verdict
    print("\n" + "="*60)
    if summary['overall_assessment'] == 'PASS':
        print("🎉 VALIDATION PASSED: Agent is learning properly!")
    elif summary['overall_assessment'] == 'PASS_WITH_WARNINGS':
        print("⚠️  VALIDATION PASSED WITH WARNINGS: Agent is learning but has some issues.")
    else:
        print("❌ VALIDATION FAILED: Agent is not learning properly!")
    print("="*60)


if __name__ == "__main__":
    main()