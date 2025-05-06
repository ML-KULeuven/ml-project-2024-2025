import logging
from typing import Optional

import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils import BaseWrapper

logger = logging.getLogger("ml-project")


def create_environment(
    num_agents: int = 1,
    max_cycles: int = 2500,
    render_mode: Optional[str] = None,
    max_zombies: int = 4,
    visual_observation: bool = False,
    frame_stack: Optional[int] = None,
    resize_dim: Optional[tuple[int, int]] = None,
) -> BaseWrapper:
    """
    Create a configured KAZ environment.

    Args:
        num_agents: Number of archer agents (1 or 2)
        max_cycles: Maximum steps before episode truncation
        render_mode: None, "human", or "rgb_array"
        max_zombies: Maximum number of zombies in the arena
        visual_observation: Whether to use pixel observations
        frame_stack: Number of frames to stack (None for no stacking)
        resize_dim: Tuple (width, height) to resize visual observations

    Returns:
        A configured PettingZoo environment
    """
    # Validate parameters
    if not 1 <= num_agents <= 2:
        raise ValueError(
            "Number of agents must be either 1 (one archer) or 2 (two archers)"
        )

    # Create base environment
    env = knights_archers_zombies_v10.env(
        max_cycles=max_cycles,
        num_archers=num_agents,
        num_knights=0,  # Disable knights for archer-only scenario
        max_zombies=max_zombies,
        vector_state=not visual_observation,
        render_mode=render_mode,
    )

    # Handle agent termination
    env = ss.black_death_v3(env)

    # Configure visual observations if needed
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")

        if resize_dim is not None:
            env = ss.resize_v1(env, x_size=resize_dim[0], y_size=resize_dim[1])

        if frame_stack is not None and frame_stack > 1:
            env = ss.frame_stack_v1(env, frame_stack)

    logger.info(
        f"Created KAZ environment with {num_agents} agents and max {max_zombies} zombies. "
        f"Observation type: {'visual' if visual_observation else 'vector'}"
    )

    return env
