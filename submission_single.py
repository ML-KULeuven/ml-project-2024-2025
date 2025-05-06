import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

ENV_SETTINGS = {
    "visual_observation": False,  # Set to True if using visual observations
    # the following parameters are only relevant if visual_observation is True
    "frame_stack": None,  # Number of frames to stack (None for no stacking)
    "resize_dim": None,  # Tuple (width, height) to resize visual observations
}


class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering).
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        pass

    def observe(self, agent: AgentID) -> ObsType | None:
        pass


class CustomPredictFunction:
    """
    Function to use to load the trained model and predict the action.
    """

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, agent, *args, **kwargs):
        pass

