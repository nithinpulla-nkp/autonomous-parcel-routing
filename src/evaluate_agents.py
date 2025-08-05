#!/usr/bin/env python3
"""
Agent evaluation script for testing trained RL agents.

This script provides command-line tools for evaluating saved agents,
comparing multiple agents, and generating comprehensive reports.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from src.apr import WarehouseEnv
from src.apr.evaluate import AgentEvaluator
from src.apr.agents import create_agent


def evaluate_single_agent(args) -> None:
    """Evaluate a single trained agent."""
    print(f"🔍 Evaluating {args.algorithm} agent")
    print("=" * 50)
    
    # Create evaluator
    env = WarehouseEnv(seed=42)
    evaluator = AgentEvaluator(env, verbose=True)
    
    try:
        # Load agent
        if args.agent_path:
            agent = evaluator.load_agent(args.agent_path, args.algorithm)
        else:
            # Create fresh agent for testing
            print("No agent path provided, creating fresh agent for demonstration")
            agent = create_agent(
                args.algorithm,
                env.observation_space,
                env.action_space,
                alpha=0.1,
                gamma=0.95,
                epsilon=0.0  # No exploration during evaluation
            )
        
        # Evaluate agent
        seeds = [42, 123, 456] if args.seeds is None else args.seeds
        results = evaluator.evaluate_agent(
            agent,
            num_episodes=args.episodes,
            seeds=seeds,
            render=args.render
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        stats = results['aggregated_results']['overall_statistics']
        print(f"Mean Reward: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
        print(f"Success Rate: {stats['mean_success_rate']:.1%}")
        print(f"Mean Episode Length: {stats['mean_episode_length']:.1f}")
        print(f"State Coverage: {stats['mean_state_coverage']:.1%}")
        
        # Generate visualizations
        if args.visualize:
            print(f"\n📈 Generating visualizations...")
            save_path = Path(args.output_dir) / f"{args.algorithm}_evaluation.png" if args.output_dir else None
            evaluator.visualize_evaluation(results, save_path)
        
        # Save report
        if args.output_dir:
            report_path = Path(args.output_dir) / f"{args.algorithm}_evaluation_report"
            evaluator.save_evaluation_report(results, report_path)
            print(f"📄 Report saved to {report_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)


def compare_agents(args) -> None:
    """Compare multiple trained agents."""
    print(f"🏁 Comparing {len(args.algorithms)} agents")
    print("=" * 50)
    
    # Create evaluator
    env = WarehouseEnv(seed=42)
    evaluator = AgentEvaluator(env, verbose=True)
    
    try:
        # Load/create agents
        agents = {}
        
        for i, algorithm in enumerate(args.algorithms):
            agent_path = args.agent_paths[i] if args.agent_paths and i < len(args.agent_paths) else None
            
            if agent_path:
                agent = evaluator.load_agent(agent_path, algorithm)
            else:
                print(f"Creating fresh {algorithm} agent for demonstration")
                agent = create_agent(
                    algorithm,
                    env.observation_space,
                    env.action_space,
                    alpha=0.1,
                    gamma=0.95,
                    epsilon=0.0
                )
            
            agents[algorithm] = agent
        
        # Compare agents
        seeds = [42, 123, 456] if args.seeds is None else args.seeds
        results = evaluator.compare_agents(
            agents,
            num_episodes=args.episodes,
            seeds=seeds
        )
        
        # Print comparison summary
        print("\n" + "=" * 60)
        print("📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        stats = results['comparison_stats']['agent_statistics']
        rankings = results['comparison_stats']['rankings']
        
        # Performance table
        print(f"{'Agent':<20} {'Reward':<12} {'Success':<10} {'Length':<10} {'Coverage':<10}")
        print("-" * 62)
        for agent_name, agent_stats in stats.items():
            print(f"{agent_name:<20} {agent_stats['mean_reward']:<12.1f} "
                  f"{agent_stats['success_rate']:<10.1%} {agent_stats['episode_length']:<10.1f} "
                  f"{agent_stats['state_coverage']:<10.1%}")
        
        # Rankings
        print(f"\n🏆 Rankings by Reward:")
        for i, (agent_name, _) in enumerate(rankings['by_reward']):
            print(f"  {i+1}. {agent_name}")
        
        print(f"\n🎯 Rankings by Success Rate:")
        for i, (agent_name, _) in enumerate(rankings['by_success_rate']):
            print(f"  {i+1}. {agent_name}")
        
        # Generate visualizations
        if args.visualize:
            print(f"\n📈 Generating comparison visualizations...")
            save_path = Path(args.output_dir) / "agent_comparison.png" if args.output_dir else None
            evaluator.visualize_evaluation(results, save_path)
        
        # Save report
        if args.output_dir:
            report_path = Path(args.output_dir) / "agent_comparison_report"
            evaluator.save_evaluation_report(results, report_path)
            print(f"📄 Report saved to {report_path}")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single agent evaluation
    single_parser = subparsers.add_parser('single', help='Evaluate a single agent')
    single_parser.add_argument('algorithm', help='Algorithm type (q_learning, double_q_learning, sarsa)')
    single_parser.add_argument('--agent-path', type=str, help='Path to saved agent file')
    single_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    single_parser.add_argument('--seeds', nargs='+', type=int, help='Random seeds for evaluation')
    single_parser.add_argument('--render', action='store_true', help='Render first episode')
    single_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    single_parser.add_argument('--output-dir', type=str, help='Directory to save outputs')
    
    # Multi-agent comparison
    compare_parser = subparsers.add_parser('compare', help='Compare multiple agents')
    compare_parser.add_argument('algorithms', nargs='+', help='Algorithm types to compare')
    compare_parser.add_argument('--agent-paths', nargs='+', type=str, help='Paths to saved agent files')
    compare_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    compare_parser.add_argument('--seeds', nargs='+', type=int, help='Random seeds for evaluation')
    compare_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    compare_parser.add_argument('--output-dir', type=str, help='Directory to save outputs')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        evaluate_single_agent(args)
    elif args.command == 'compare':
        compare_agents(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()