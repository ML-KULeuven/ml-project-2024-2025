#!/usr/bin/env python3
# encoding: utf-8
"""
Code used to load an agent and evaluate its performance.

Usage:
    python3 evaluation.py [-h] [--verbose | --quiet] [--load FILE] [--screen]
                          [--episodes NUM] [--seed SEED] [--agents NUM] [--output FILE]
"""

import argparse
import importlib.util
import json
import logging
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pygame

from utils import create_environment

logger = logging.getLogger("ml-project")


def generate_random_seeds(
    num_seeds: int, master_seed: Optional[int] = None
) -> List[int]:
    """
    Generate a list of random seeds from a master seed.

    Args:
        num_seeds: Number of seeds to generate
        master_seed: Optional seed for the random number generator

    Returns:
        List of random seeds
    """
    if master_seed is not None:
        random.seed(master_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]


def save_results_to_jsonl(
    results: Dict[str, Any],
    output_file: str,
    mode: str = "w",
    agent_id: Optional[str] = None,
) -> None:
    """Save evaluation results to a JSONL file."""
    try:
        result_data = results.copy()
        if agent_id:
            result_data["agent"] = agent_id

        with open(output_file, mode, encoding="utf-8") as f:
            f.write(json.dumps(result_data) + "\n")
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")


def evaluate(
    env: Any,
    predict_function: Callable,
    seeds: List[int],
) -> Dict[str, float]:
    """
    Evaluate an agent's performance over multiple episodes.

    Args:
        env: The environment to evaluate in
        predict_function: Function that takes (obs, agent) and returns action
        seeds: List of seeds for environment initialization

    Returns:
        Dictionary containing evaluation metrics
    """
    rewards = {agent: 0 for agent in env.possible_agents}
    episode_lengths = []
    do_terminate = False
    start_time = time.time()

    for i, seed in enumerate(seeds):
        env.reset(seed=seed)
        env.action_space(env.possible_agents[0]).seed(seed)
        step_count = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            step_count += 1

            # Accumulate rewards for all agents
            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                episode_lengths.append(step_count)
                break

            action = predict_function(obs, agent)

            # Handle rendering and user input
            if env.render_mode == "human":
                if handle_pygame_events():
                    do_terminate = True

            if do_terminate:
                break

            env.step(action)

        if do_terminate or (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{len(seeds)} episodes")

        if do_terminate:
            break

    env.close()
    total_time = time.time() - start_time

    # Calculate statistics
    avg_reward = sum(rewards.values()) / len(seeds)
    avg_reward_per_agent = {
        agent: rewards[agent] / len(seeds) for agent in env.possible_agents
    }

    results = {
        "avg_reward": avg_reward,
        "avg_reward_per_agent": avg_reward_per_agent,
        "total_episodes": len(seeds),
        "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
        "evaluation_time": total_time,
        "time_per_episode": total_time / len(seeds) if seeds else 0,
        "used_seeds": seeds,
    }

    print("\nEvaluation Results:")
    print(f"- Total episodes: {results['total_episodes']}")
    print(f"- Avg reward: {results['avg_reward']:.2f}")
    print("- Avg reward per agent:")
    for agent, reward in results["avg_reward_per_agent"].items():
        print(f"  {agent}: {reward:.2f}")
    print(f"- Avg episode length: {results['avg_episode_length']:.1f} steps")
    print(f"- Total evaluation time: {results['evaluation_time']:.2f} seconds")
    print(f"- Time per episode: {results['time_per_episode']:.2f} seconds")

    return results


def handle_pygame_events() -> bool:
    """Handle pygame events and return True if user requested to quit."""
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True
    return False


def load_agent_module(file_path: str) -> Any:
    """Dynamically load an agent module from the given file path."""
    try:
        spec = importlib.util.spec_from_file_location("KAZ_agent", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to load agent from {file_path}: {str(e)}")
        raise


def setup_logging(verbose: int, quiet: int) -> None:
    """Configure logging based on verbosity levels."""
    log_level = max(logging.INFO - 10 * (verbose - quiet), logging.DEBUG)
    logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Load an agent and evaluate its performance in the KAZ environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="Decrease verbosity (can be used multiple times)",
    )
    parser.add_argument(
        "--load", "-l", metavar="FILE", help="Load agent from the given file path"
    )
    parser.add_argument(
        "--screen",
        "-s",
        action="store_true",
        help="Enable visual rendering (human render mode)",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for random number generation",
    )
    parser.add_argument(
        "--agents", type=int, default=1, help="Number of agents in the environment"
    )
    parser.add_argument("--output", "-o", help="JSONL file to save results")
    parser.add_argument(
        "--append",
        "-a",
        action="store_true",
        help="Append to output file instead of overwriting",
    )
    parser.add_argument("--id", "-i", help="Agent ID to include in the output")

    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.quiet)

    # Environment configuration
    render_mode = "human" if args.screen else None
    logger.info(f"Render mode: {render_mode}")
    if render_mode == "human":
        logger.info("Press Q or close window to terminate evaluation early")

    # Generate random seeds for episodes
    seeds = generate_random_seeds(args.episodes, args.seed)
    logger.info(f"Evaluating with {args.episodes} episodes (master seed: {args.seed})")

    # Load agent
    env_settings = {
        "visual_observation": False,
        "frame_stack": None,
        "resize_dim": None,
    }
    try:
        if args.load:
            agent_module = load_agent_module(args.load)
            CustomWrapper = agent_module.CustomWrapper
            CustomPredictFunction = agent_module.CustomPredictFunction
            if hasattr(agent_module, "ENV_SETTINGS"):
                env_settings = {**env_settings, **agent_module.ENV_SETTINGS}
        else:
            from submission_single_example_rllib import (
                CustomPredictFunction,
                CustomWrapper,
            )

    except Exception as e:
        logger.error(f"Failed to load agent: {str(e)}")
        return 1

    # Create and wrap environment
    env = create_environment(
        num_agents=args.agents,
        render_mode=render_mode,
        **env_settings,
    )
    env = CustomWrapper(env)

    # Run evaluation
    results = evaluate(env, CustomPredictFunction(env), seeds=seeds)

    if args.output:
        save_results_to_jsonl(
            results,
            args.output,
            mode="a" if args.append else "w",
            agent_id=args.id if hasattr(args, "id") else None,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
