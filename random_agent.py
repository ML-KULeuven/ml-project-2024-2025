import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

ENV_SETTINGS = {
    "visual_observation": False  # Set to True if using visual observations
}


class CustomWrapper(BaseWrapper):
    """
    An example of a custom wrapper that flattens the symbolic vector state of the environment.

    Wrappers are useful to do state pre-processing (e.g. feature engineering) that does not need to be learned by the agent.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction:
    """A random archer agent."""

    def __init__(self, env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        return self.env.action_space(agent).sample()
