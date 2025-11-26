"""
SBDAPM Environment Wrapper for Schola/RLlib Training (v3.0)

Wraps the Unreal Engine environment via Schola gRPC for RLlib compatibility.

Observation: 78 features (71 FObservationElement + 7 current objective embedding)
Action: 8-dimensional (FTacticalAction)
  - MoveDirection (2D continuous): [-1, 1] x [-1, 1]
  - MoveSpeed (1D continuous): [0, 1]
  - LookDirection (2D continuous): [-1, 1] x [-1, 1]
  - Fire (discrete): {0, 1}
  - Crouch (discrete): {0, 1}
  - UseAbility (discrete): {0, 1}
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
    SBDAPM atomic action environment (v3.0).

    Connects to Unreal Engine via Schola gRPC and exposes:
    - Observation: 78 float features (71 FObservationElement + 7 objective embedding)
    - Action: 8-dimensional atomic actions (FTacticalAction)
      - Continuous: move_x, move_y, speed, look_x, look_y (5D)
      - Discrete: fire, crouch, use_ability (3D binary)
    - Reward: Hierarchical (individual + coordination + strategic)
    """

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
            shape=(78,),  # 71 observation + 7 objective embedding
            dtype=np.float32
        )
        # Atomic action space: 5D continuous + 3D discrete
        self.action_space = spaces.Dict({
            "move_direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "move_speed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "look_direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "fire": spaces.Discrete(2),
            "crouch": spaces.Discrete(2),
            "use_ability": spaces.Discrete(2)
        })

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
        obs = np.zeros(78, dtype=np.float32)  # 71 + 7 objective embedding
        info = {"episode_steps": 0}

        return obs, info

    def step(self, action):
        """
        Execute action and return result.

        Args:
            action: Dict with keys:
                - move_direction: (2,) array in [-1, 1]
                - move_speed: (1,) array in [0, 1]
                - look_direction: (2,) array in [-1, 1]
                - fire: int in {0, 1}
                - crouch: int in {0, 1}
                - use_ability: int in {0, 1}

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_steps += 1

        # Default values (will be overwritten by Schola callbacks)
        observation = np.zeros(78, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = self.episode_steps >= self.max_episode_steps

        self.total_reward += reward

        info = {
            "episode_steps": self.episode_steps,
            "action": {
                "move": action["move_direction"].tolist() if isinstance(action["move_direction"], np.ndarray) else action["move_direction"],
                "speed": float(action["move_speed"][0]) if isinstance(action["move_speed"], np.ndarray) else action["move_speed"],
                "look": action["look_direction"].tolist() if isinstance(action["look_direction"], np.ndarray) else action["look_direction"],
                "fire": bool(action["fire"]),
                "crouch": bool(action["crouch"]),
                "ability": bool(action["use_ability"])
            },
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
        Full Schola-integrated environment (v3.0).

        This class inherits from UnrealEnv to get proper gRPC integration.
        Schola handles the observation/action/reward exchange.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Override spaces to match our agent (v3.0 atomic actions)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(78,),  # 71 observation + 7 objective embedding
                dtype=np.float32
            )
            self.action_space = spaces.Dict({
                "move_direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "move_speed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "look_direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "fire": spaces.Discrete(2),
                "crouch": spaces.Discrete(2),
                "use_ability": spaces.Discrete(2)
            })

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
