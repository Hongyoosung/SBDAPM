"""
SBDAPM Environment Wrapper for Schola/RLlib Training (v3.0)

Wraps the Unreal Engine environment via Schola gRPC for RLlib compatibility.

Observation: 78 features (71 FObservationElement + 7 current objective embedding)
Action: 8-dimensional Box (continuous, flattened)
  - [0-1]: MoveDirection (continuous): [-1, 1] x [-1, 1]
  - [2]:   MoveSpeed (continuous): [0, 1]
  - [3-4]: LookDirection (continuous): [-1, 1] x [-1, 1]
  - [5]:   Fire (continuous [0,1], interpreted as binary >= 0.5)
  - [6]:   Crouch (continuous [0,1], interpreted as binary >= 0.5)
  - [7]:   UseAbility (continuous [0,1], interpreted as binary >= 0.5)
"""

from gymnasium import spaces
import numpy as np

try:
    from schola.gym.env import GymEnv as UnrealEnv
    SCHOLA_AVAILABLE = True
except ImportError:
    SCHOLA_AVAILABLE = False
    print("Warning: schola not installed. Install with: pip install schola[rllib]")


class SBDAPMEnv:
    """
    SBDAPM atomic action environment (v3.0).

    Connects to Unreal Engine via Schola gRPC and exposes:
    - Observation: 78 float features (71 FObservationElement + 7 objective embedding)
    - Action: 8-dimensional Box (continuous, flattened)
      - [0-1]: move_direction [-1, 1]
      - [2]:   move_speed [0, 1]
      - [3-4]: look_direction [-1, 1]
      - [5-7]: fire, crouch, use_ability [0, 1] (interpreted as binary)
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
        # Flattened 8D continuous action space
        # [0-1]: move_x, move_y in [-1, 1]
        # [2]:   speed in [0, 1]
        # [3-4]: look_x, look_y in [-1, 1]
        # [5-7]: fire, crouch, ability in [0, 1] (binary interpreted)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(8,),
            dtype=np.float32
        )

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
            action: (8,) numpy array in Box space:
                [0-1]: move_direction (x, y)
                [2]:   move_speed
                [3-4]: look_direction (x, y)
                [5]:   fire (0-1, binary interpreted)
                [6]:   crouch (0-1, binary interpreted)
                [7]:   use_ability (0-1, binary interpreted)

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
                "move": [float(action[0]), float(action[1])],
                "speed": float(action[2]),
                "look": [float(action[3]), float(action[4])],
                "fire": action[5] >= 0.5,
                "crouch": action[6] >= 0.5,
                "ability": action[7] >= 0.5
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
    import gymnasium as gym
    
    class SBDAPMScholaEnv(gym.Env):
        """
        Full Schola-integrated environment (v3.0).

        Pure gymnasium.Env implementation with direct gRPC connection.
        Avoids Schola's GymEnv wrapper to prevent RLlib assertion errors.
        """

        def __init__(self, **kwargs):
            super().__init__()
            
            # Extract host/port for connection
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 50051)

            # Define spaces to match our agent (v3.0 flattened 8D Box)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(78,),  # 71 observation + 7 objective embedding
                dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(8,),
                dtype=np.float32
            )

            self.episode_steps = 0
            self.max_episode_steps = kwargs.get("max_episode_steps", 1000)
            
            # Direct gRPC connection (bypass Schola's GymEnv wrapper)
            from schola.core.unreal_connections.editor_connection import UnrealEditorConnection
            self._connection = UnrealEditorConnection(url=host, port=port)
            self._connected = False
            
            print(f"[SBDAPMScholaEnv] Initialized with host={host}, port={port}")
            print(f"[SBDAPMScholaEnv] Type: {type(self).__name__} (pure gymnasium.Env)")

        def reset(self, seed=None, options=None):
            """Reset environment and return initial observation."""
            self.episode_steps = 0
            
            # Connect to UE if not already connected
            if not self._connected:
                try:
                    self._connection.connect()
                    self._connected = True
                    print("[SBDAPMScholaEnv] Connected to Unreal Engine")
                except Exception as e:
                    print(f"[SBDAPMScholaEnv] Connection failed: {e}")
                    # Return dummy observation if connection fails
                    return np.zeros(78, dtype=np.float32), {"episode_steps": 0}
            
            # Request reset from UE via gRPC
            try:
                # Send reset command
                response = self._connection.reset()
                obs = np.array(response.observation, dtype=np.float32) if hasattr(response, 'observation') else np.zeros(78, dtype=np.float32)
                info = {"episode_steps": 0}
                return obs, info
            except Exception as e:
                print(f"[SBDAPMScholaEnv] Reset failed: {e}")
                return np.zeros(78, dtype=np.float32), {"episode_steps": 0}

        def step(self, action):
            """Execute action via direct gRPC call."""
            self.episode_steps += 1
            
            if not self._connected:
                # Return dummy step if not connected
                return (
                    np.zeros(78, dtype=np.float32),
                    0.0,
                    False,
                    self.episode_steps >= self.max_episode_steps,
                    {"episode_steps": self.episode_steps}
                )
            
            try:
                # Send action to UE via gRPC
                response = self._connection.step(action)
                
                obs = np.array(response.observation, dtype=np.float32) if hasattr(response, 'observation') else np.zeros(78, dtype=np.float32)
                reward = float(response.reward) if hasattr(response, 'reward') else 0.0
                terminated = bool(response.done) if hasattr(response, 'done') else False
                truncated = self.episode_steps >= self.max_episode_steps
                
                info = {"episode_steps": self.episode_steps}
                
                return obs, reward, terminated, truncated, info
                
            except Exception as e:
                print(f"[SBDAPMScholaEnv] Step failed: {e}")
                return (
                    np.zeros(78, dtype=np.float32),
                    0.0,
                    True,  # Terminate on error
                    False,
                    {"episode_steps": self.episode_steps, "error": str(e)}
                )
        
        def render(self):
            """Rendering is handled by UE."""
            pass
        
        def close(self):
            """Close gRPC connection."""
            if self._connected:
                try:
                    self._connection.close()
                    self._connected = False
                    print("[SBDAPMScholaEnv] Connection closed")
                except Exception as e:
                    print(f"[SBDAPMScholaEnv] Close failed: {e}")

