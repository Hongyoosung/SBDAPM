"""
SBDAPM Environment Wrapper for Schola/RLlib Training

Wraps the Unreal Engine environment via Schola gRPC for RLlib compatibility.

Observation: 71 features (FObservationElement)
Action: 16 discrete (ETacticalAction)
"""

from gymnasium import spaces
import numpy as np

try:
    from schola.envs import UnrealEnv
    SCHOLA_AVAILABLE = True
except ImportError:
    SCHOLA_AVAILABLE = False
    print("Warning: schola not installed. Install with: pip install schola[rllib]")


class SBDAPMEnv:
    """
    SBDAPM tactical action environment.

    Connects to Unreal Engine via Schola gRPC and exposes:
    - Observation: 71 float features from FObservationElement
    - Action: 16 discrete tactical actions (ETacticalAction)
    - Reward: Combat events (+10 kill, +5 damage, -5 take damage, -10 die)
    """

    # Action mapping to ETacticalAction enum
    ACTION_NAMES = [
        "Advance",          # 0
        "Retreat",          # 1
        "FlankLeft",        # 2
        "FlankRight",       # 3
        "TakeCover",        # 4
        "SuppressiveFire",  # 5
        "HoldPosition",     # 6
        "AssaultTarget",    # 7
        "DefendPosition",   # 8
        "Regroup",          # 9
        "Heal",             # 10
        "Reload",           # 11
        "CallSupport",      # 12
        "SetAmbush",        # 13
        "AggressivePush",   # 14
        "DefensiveHold",    # 15
    ]

    def __init__(self, host="localhost", port=50051, **kwargs):
        """
        Initialize environment.

        Args:
            host: gRPC server host (UE Schola plugin)
            port: gRPC server port
        """
        self.host = host
        self.port = port

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(71,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(16)

        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 1000
        self.total_reward = 0.0

        # Internal connection (will be set by Schola)
        self._connected = False

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        self.episode_steps = 0
        self.total_reward = 0.0

        # Return initial observation (zeros until connected)
        obs = np.zeros(71, dtype=np.float32)
        info = {"episode_steps": 0}

        return obs, info

    def step(self, action):
        """
        Execute action and return result.

        Args:
            action: int in [0, 15] corresponding to ETacticalAction

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_steps += 1

        # Default values (will be overwritten by Schola callbacks)
        observation = np.zeros(71, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = self.episode_steps >= self.max_episode_steps

        self.total_reward += reward

        info = {
            "episode_steps": self.episode_steps,
            "action_name": self.ACTION_NAMES[action] if 0 <= action < 16 else "Unknown",
            "total_reward": self.total_reward
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Rendering is handled by UE."""
        pass

    def close(self):
        """Close connection."""
        self._connected = False


if SCHOLA_AVAILABLE:
    class SBDAPMScholaEnv(UnrealEnv):
        """
        Full Schola-integrated environment.

        This class inherits from UnrealEnv to get proper gRPC integration.
        Schola handles the observation/action/reward exchange.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Override spaces to match our agent
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(71,),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(16)

            self.episode_steps = 0
            self.max_episode_steps = kwargs.get("max_episode_steps", 1000)

        def reset(self, **kwargs):
            self.episode_steps = 0
            return super().reset(**kwargs)

        def step(self, action):
            self.episode_steps += 1
            obs, reward, done, truncated, info = super().step(action)

            # Check for truncation
            if self.episode_steps >= self.max_episode_steps:
                truncated = True

            info["episode_steps"] = self.episode_steps
            return obs, reward, done, truncated, info
