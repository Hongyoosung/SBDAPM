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
    from schola.gym.env import GymVectorEnv as UnrealVectorEnv
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
    from schola.core.unreal_connections.editor_connection import UnrealEditorConnection
    import gymnasium as gym

    class SafeUnrealVectorEnv(UnrealVectorEnv):
        """
        Wrapper around UnrealVectorEnv to handle None observations from Schola.
        Fixes TypeError when some agents return None during reset/step.
        """
        def batch_obs(self, obs):
            # Check if any observation is None
            if any(o is None for o in obs):
                # Find a valid observation to use as template
                valid_obs = next((o for o in obs if o is not None), None)
                if valid_obs is not None:
                    # Replace None with valid_obs
                    obs = [o if o is not None else valid_obs for o in obs]
                else:
                    print("[SafeUnrealVectorEnv] Warning: All observations are None! Using zeros.")
                    # Create dummy observation based on single_observation_space
                    dummy = self.single_observation_space.sample()
                    # Zero it out to be safe
                    if isinstance(dummy, np.ndarray):
                        dummy.fill(0)
                    obs = [dummy for _ in obs]
            
            try:
                return super().batch_obs(obs)
            except Exception as e:
                print(f"[SafeUnrealVectorEnv] Error in batch_obs: {e}")
                # Fallback: return stacked zeros
                # This assumes Box space for simplicity, but handles Dict if we can
                try:
                    import gymnasium.experimental.vector.utils as utils
                    return utils.batch_space(self.single_observation_space, len(obs))
                except:
                    print("[SafeUnrealVectorEnv] Critical: Failed to create fallback batch.")
                    raise e

    class SBDAPMScholaEnv(gym.Wrapper):
        """
        Full Schola-integrated environment (v3.0).

        Wraps Schola's GymVectorEnv and adapts it for RLlib single-env training.
        Handles None observations/info and flattens vector env to single env.
        """

        def __init__(self, **kwargs):
            # Extract configuration
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 50051)
            self.max_episode_steps = kwargs.get("max_episode_steps", 1000)

            # Create Schola connection and wrapped env
            connection = UnrealEditorConnection(url=host, port=port)
            # Use SafeUnrealVectorEnv to handle None observations
            schola_env = SafeUnrealVectorEnv(unreal_connection=connection, verbosity=1)

            print(f"[DEBUG] Schola Env Num Envs: {schola_env.num_envs}")
            print(f"[DEBUG] Schola Env Observation Space Type: {type(schola_env.observation_space)}")
            print(f"[DEBUG] Schola Env Action Space Type: {type(schola_env.action_space)}")

            # Inspect action space structure
            if isinstance(schola_env.action_space, spaces.Dict):
                print(f"[DEBUG] Action space is Dict with keys: {list(schola_env.action_space.keys())}")
                first_agent = list(schola_env.action_space.keys())[0]
                print(f"[DEBUG] Action space for '{first_agent}': {schola_env.action_space[first_agent]}")
                # Check if nested
                if isinstance(schola_env.action_space[first_agent], spaces.Dict):
                    print(f"[DEBUG] Nested Dict detected. Sub-keys: {list(schola_env.action_space[first_agent].keys())}")
            else:
                print(f"[DEBUG] Action space: {schola_env.action_space}")

            # Initialize wrapper
            super().__init__(schola_env)

            self.num_agents = schola_env.num_envs

            # Detect if spaces are Dict (agent ID keys)
            self.is_obs_dict = isinstance(schola_env.observation_space, spaces.Dict)
            self.is_action_dict = isinstance(schola_env.action_space, spaces.Dict)

            # Extract agent IDs from Dict spaces
            self.agent_ids = None
            self.action_space_structure = None  # Store structure for action formatting
            if self.is_obs_dict:
                self.agent_ids = list(schola_env.observation_space.keys())
                print(f"[SBDAPMScholaEnv] Detected Dict Observation Space. Agent IDs: {self.agent_ids}")
            if self.is_action_dict:
                action_agent_ids = list(schola_env.action_space.keys())
                if not self.agent_ids:
                    self.agent_ids = action_agent_ids
                print(f"[SBDAPMScholaEnv] Detected Dict Action Space. Agent IDs: {action_agent_ids}")

            # Store action space structure for each agent
            if self.is_action_dict and self.agent_ids:
                first_agent = self.agent_ids[0]
                self.action_space_structure = schola_env.action_space[first_agent]
                print(f"[SBDAPMScholaEnv] Action structure per agent: {self.action_space_structure}")

            # Use first agent ID if Dict spaces
            self.primary_agent_id = self.agent_ids[0] if self.agent_ids else None
            if self.primary_agent_id:
                print(f"[SBDAPMScholaEnv] Using primary agent: {self.primary_agent_id}")

            # Override spaces to single-agent (we'll handle the first agent only)
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

            print(f"[SBDAPMScholaEnv] Initialized with host={host}, port={port}")
            print(f"[SBDAPMScholaEnv] Detected {self.num_agents} agents, using first agent only")
            print(f"[SBDAPMScholaEnv] Type: {type(self).__name__} (Schola GymVectorEnv wrapper)")

        def _format_action_for_schola(self, action):
            """
            Convert flat 8-dim action to Schola's expected format.

            Handles:
            1. Box((N, 8)) - Schola vector env, reshape (8,) to (N, 8) by tiling
            2. Dict - map flat action to dict keys
            3. Box((8,)) - return as-is
            """
            # Handle Box space
            if isinstance(self.action_space_structure, spaces.Box):
                expected_shape = self.action_space_structure.shape
                if len(expected_shape) == 2 and expected_shape[1] == 8:
                    # Vector env expecting (N, 8) matrix
                    # Reshape our (8,) action to (N, 8) by repeating rows
                    num_envs = expected_shape[0]
                    print(f"[SBDAPMScholaEnv] Reshaping action (8,) to ({num_envs}, 8)")
                    return np.tile(action, (num_envs, 1)).astype(np.float32)
                elif expected_shape == (8,):
                    # Correct shape
                    return action.astype(np.float32)
                else:
                    print(f"[SBDAPMScholaEnv] WARNING: Unexpected action shape {expected_shape}, using as-is")
                    return action.astype(np.float32)

            # Handle nested Dict
            if isinstance(self.action_space_structure, spaces.Dict):
                # If nested Dict, we need to map flat action to dict structure
                # Common structure: {'move': Box(2), 'look': Box(2), 'actions': Box(3)}
                # Our action: [move_x, move_y, speed, look_x, look_y, fire, crouch, ability]
                action_dict = {}
                subspace_keys = list(self.action_space_structure.keys())

                # Try to infer mapping from subspace keys
                idx = 0
                for key in subspace_keys:
                    subspace = self.action_space_structure[key]
                    if isinstance(subspace, spaces.Box):
                        dim = subspace.shape[0] if len(subspace.shape) > 0 else 1
                        if dim == 1:
                            action_dict[key] = np.array([action[idx]], dtype=np.float32)
                            idx += 1
                        else:
                            action_dict[key] = action[idx:idx+dim].astype(np.float32)
                            idx += dim
                    else:
                        # Fallback: use sample
                        action_dict[key] = subspace.sample()

                return action_dict

            # Fallback - return as-is
            return action

        def reset(self, seed=None, options=None):
            """Reset environment and return observation for first agent."""
            self.episode_steps = 0

            try:
                print(f"[SBDAPMScholaEnv] Calling Schola reset...")
                # Get vectorized observations
                obs_vec, info_vec = self.env.reset(seed=seed, options=options)

                print(f"[SBDAPMScholaEnv] Reset returned obs type: {type(obs_vec)}")
                print(f"[SBDAPMScholaEnv] Reset returned info type: {type(info_vec)}")

                # Sync agent_ids based on actual observations after reset
                if isinstance(obs_vec, dict):
                    actual_agent_ids = list(obs_vec.keys())
                    if actual_agent_ids != self.agent_ids:
                        print(f"[SBDAPMScholaEnv] Syncing agent IDs: {len(self.agent_ids)} -> {len(actual_agent_ids)}")
                        self.agent_ids = actual_agent_ids
                        self.primary_agent_id = self.agent_ids[0] if self.agent_ids else None

                # Check if obs_vec is a space class (error case)
                if isinstance(obs_vec, (spaces.Dict, spaces.Box, spaces.Space)):
                    print(f"[SBDAPMScholaEnv] ERROR: Reset returned a space class instead of observations!")
                    print(f"[SBDAPMScholaEnv] This likely means the UE environment is not properly initialized.")
                    obs = np.zeros(78, dtype=np.float32)
                # Extract first agent's observation
                elif obs_vec is None:
                    print(f"[SBDAPMScholaEnv] Warning: No observations returned, using zeros")
                    obs = np.zeros(78, dtype=np.float32)
                elif isinstance(obs_vec, dict):
                    # Dict with actual data: extract using primary agent ID
                    print(f"[SBDAPMScholaEnv] obs_vec is dict with keys: {list(obs_vec.keys())[:5]}")
                    if self.primary_agent_id and self.primary_agent_id in obs_vec:
                        obs = obs_vec[self.primary_agent_id]
                        print(f"[SBDAPMScholaEnv] Extracted obs for agent {self.primary_agent_id}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
                    else:
                        print(f"[SBDAPMScholaEnv] Warning: Agent ID {self.primary_agent_id} not found in obs_vec keys: {list(obs_vec.keys())}")
                        obs = np.zeros(78, dtype=np.float32)
                elif hasattr(obs_vec, '__len__') and len(obs_vec) == 0:
                    obs = np.zeros(78, dtype=np.float32)
                elif isinstance(obs_vec, (list, tuple, np.ndarray)) and len(obs_vec) > 0:
                    # Array-like: use numeric index
                    obs = obs_vec[0]
                else:
                    print(f"[SBDAPMScholaEnv] Warning: Unexpected obs_vec type: {type(obs_vec)}")
                    obs = np.zeros(78, dtype=np.float32)

                # Extract first agent's info
                info = {}
                if info_vec is not None:
                    if isinstance(info_vec, dict):
                        # Try to extract for primary agent ID first
                        if self.primary_agent_id and self.primary_agent_id in info_vec:
                            info = info_vec[self.primary_agent_id]
                        else:
                            # Fallback: extract first entry
                            info = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v
                                    for k, v in info_vec.items()}
                    elif isinstance(info_vec, (list, tuple)) and len(info_vec) > 0:
                        info = info_vec[0] if info_vec[0] is not None else {}

            except Exception as e:
                print(f"[SBDAPMScholaEnv] Error during reset: {e}")
                import traceback
                traceback.print_exc()
                obs = np.zeros(78, dtype=np.float32)
                info = {}

            # Validate observation shape
            if hasattr(obs, 'shape') and obs.shape != (78,):
                print(f"[SBDAPMScholaEnv] Warning: Expected obs shape (78,), got {obs.shape}")
                if obs.size == 78:
                    obs = obs.reshape(78)
                else:
                    obs = np.zeros(78, dtype=np.float32)
            elif not hasattr(obs, 'shape'):
                 obs = np.zeros(78, dtype=np.float32)

            return obs, info

        def step(self, action):
            """Execute action for first agent and return result."""
            try:
                # Ensure action is numpy array
                if not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)

                # Format action according to Schola's expected structure
                formatted_action = self._format_action_for_schola(action)
                print(f"[SBDAPMScholaEnv] Formatted action type: {type(formatted_action)}")
                if isinstance(formatted_action, dict):
                    print(f"[SBDAPMScholaEnv] Formatted action keys: {list(formatted_action.keys())}")
                    for k, v in formatted_action.items():
                        print(f"  {k}: {type(v)} {v.shape if hasattr(v, 'shape') else ''}")

                # Handle Dict action space OR if agent_ids detected
                # (Schola may use Dict internally even if exposed space is Box)
                if (self.is_action_dict or self.agent_ids) and self.agent_ids:
                    # Create action dict with all agent IDs using formatted action
                    if isinstance(formatted_action, dict):
                        # Nested dict - deep copy for each agent
                        action_vec = {agent_id: {k: v.copy() if isinstance(v, np.ndarray) else v
                                                 for k, v in formatted_action.items()}
                                     for agent_id in self.agent_ids}
                    else:
                        # Simple array
                        action_vec = {agent_id: formatted_action.copy() for agent_id in self.agent_ids}
                    print(f"[SBDAPMScholaEnv] Action_vec type: dict with {len(action_vec)} agents")
                else:
                    # Expand action to list of arrays (one per environment)
                    # Convert to list to ensure Schola handles it correctly
                    if isinstance(formatted_action, dict):
                        action_vec = [{k: v.copy() if isinstance(v, np.ndarray) else v
                                      for k, v in formatted_action.items()}
                                     for _ in range(self.num_agents)]
                    else:
                        action_vec = [formatted_action.copy() for _ in range(self.num_agents)]
                    print(f"[SBDAPMScholaEnv] Action_vec type: list with {len(action_vec)} actions")

                # Call vectorized step
                obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = self.env.step(action_vec)

                # Extract first agent's data
                if obs_vec is None:
                    obs = np.zeros(78, dtype=np.float32)
                elif isinstance(obs_vec, dict):
                    # Dict space: extract using primary agent ID
                    if self.primary_agent_id and self.primary_agent_id in obs_vec:
                        obs = obs_vec[self.primary_agent_id]
                    else:
                        print(f"[SBDAPMScholaEnv] Warning: Agent ID {self.primary_agent_id} not found in obs_vec keys: {list(obs_vec.keys())}")
                        obs = np.zeros(78, dtype=np.float32)
                elif isinstance(obs_vec, (list, tuple, np.ndarray)) and len(obs_vec) > 0:
                    obs = obs_vec[0]
                else:
                    obs = np.zeros(78, dtype=np.float32)

                # Extract reward/terminated/truncated
                if isinstance(reward_vec, dict) and self.primary_agent_id:
                    reward = float(reward_vec.get(self.primary_agent_id, 0.0))
                elif reward_vec is not None and hasattr(reward_vec, '__len__') and len(reward_vec) > 0:
                    reward = float(reward_vec[0])
                else:
                    reward = 0.0

                if isinstance(terminated_vec, dict) and self.primary_agent_id:
                    terminated = bool(terminated_vec.get(self.primary_agent_id, False))
                elif terminated_vec is not None and hasattr(terminated_vec, '__len__') and len(terminated_vec) > 0:
                    terminated = bool(terminated_vec[0])
                else:
                    terminated = False

                if isinstance(truncated_vec, dict) and self.primary_agent_id:
                    truncated = bool(truncated_vec.get(self.primary_agent_id, False))
                elif truncated_vec is not None and hasattr(truncated_vec, '__len__') and len(truncated_vec) > 0:
                    truncated = bool(truncated_vec[0])
                else:
                    truncated = False

                # Extract first agent's info
                info = {}
                if info_vec is not None:
                    if isinstance(info_vec, dict):
                        # Try to extract for primary agent ID first
                        if self.primary_agent_id and self.primary_agent_id in info_vec:
                            info = info_vec[self.primary_agent_id]
                        else:
                            # Fallback: extract first entry
                            info = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v
                                    for k, v in info_vec.items()}
                    elif isinstance(info_vec, (list, tuple)) and len(info_vec) > 0:
                        info = info_vec[0] if info_vec[0] is not None else {}

            except Exception as e:
                print(f"[SBDAPMScholaEnv] Error during step: {e}")
                import traceback
                traceback.print_exc()
                obs = np.zeros(78, dtype=np.float32)
                reward = 0.0
                terminated = True
                truncated = False
                info = {}

            self.episode_steps += 1

            # Enforce max episode length
            if self.episode_steps >= self.max_episode_steps:
                truncated = True

            # Validate observation shape
            if hasattr(obs, 'shape') and obs.shape != (78,):
                print(f"[SBDAPMScholaEnv] Warning: Expected obs shape (78,), got {obs.shape}")
                if obs.size == 78:
                    obs = obs.reshape(78)
                else:
                    obs = np.zeros(78, dtype=np.float32)
            elif not hasattr(obs, 'shape'):
                 obs = np.zeros(78, dtype=np.float32)

            return obs, reward, terminated, truncated, info

        def render(self):
            """Rendering is handled by UE."""
            return self.env.render() if hasattr(self.env, 'render') else None

        def close(self):
            """Close Schola connection."""
            if hasattr(self.env, 'close'):
                self.env.close()

