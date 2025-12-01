"""
SBDAPM Environment Wrapper for Schola/RLlib Training (v3.1 Multi-Agent)

Multi-agent environment for 4 follower agents with shared PPO policy.

Observation: 78 features per agent (71 FObservationElement + 7 current objective embedding)
Action: 8-dimensional Box per agent (continuous, flattened)
  - [0-1]: MoveDirection (continuous): [-1, 1] x [-1, 1]
  - [2]:   MoveSpeed (continuous): [0, 1]
  - [3-4]: LookDirection (continuous): [-1, 1] x [-1, 1]
  - [5]:   Fire (continuous [0,1], interpreted as binary >= 0.5)
  - [6]:   Crouch (continuous [0,1], interpreted as binary >= 0.5)
  - [7]:   UseAbility (continuous [0,1], interpreted as binary >= 0.5)
"""

from gymnasium import spaces
import numpy as np

# RLlib multi-agent support
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    print("Warning: ray[rllib] not installed")
    # Fallback for non-RLlib usage
    class MultiAgentEnv:
        pass

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
        self.max_episode_steps = 100000  # Very high limit - let UE control episode ending (2min or team elimination)
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

    class SBDAPMScholaEnv(gym.Env):
        """
        Full Schola-integrated environment (v3.0).

        Direct single-agent environment that interfaces with Schola for RLlib training.
        Handles multi-agent Schola environment but exposes single-agent interface.
        """

        def __init__(self, **kwargs):
            # Extract configuration
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 50051)
            self.max_episode_steps = kwargs.get("max_episode_steps", 100000)  # Very high - let UE control episode ending

            # Create Schola connection
            connection = UnrealEditorConnection(url=host, port=port)
            # Use SafeUnrealVectorEnv to handle None observations
            self.schola_env = SafeUnrealVectorEnv(unreal_connection=connection, verbosity=1)

            print(f"[DEBUG] Schola Env Num Envs: {self.schola_env.num_envs}")
            print(f"[DEBUG] Schola Env Observation Space Type: {type(self.schola_env.observation_space)}")
            print(f"[DEBUG] Schola Env Action Space Type: {type(self.schola_env.action_space)}")

            # Inspect action space structure
            if isinstance(self.schola_env.action_space, spaces.Dict):
                print(f"[DEBUG] Action space is Dict with keys: {list(self.schola_env.action_space.keys())}")
                first_agent = list(self.schola_env.action_space.keys())[0]
                print(f"[DEBUG] Action space for '{first_agent}': {self.schola_env.action_space[first_agent]}")

            # Define our single-agent spaces
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

            # Agent tracking
            self.agent_ids = None
            self.primary_agent_id = None
            self.action_space_structure = None
            self.episode_steps = 0

            print(f"[SBDAPMScholaEnv] Initialized with host={host}, port={port}")
            print(f"[SBDAPMScholaEnv] Type: {type(self).__name__} (Single-agent Schola wrapper)")

        def _format_action_for_schola(self, action):
            """
            Convert flat 8-dim action to Schola's expected format.

            Handles:
            1. Box((N, 8)) - Schola vector env, reshape (8,) to (N, 8) by tiling
            2. Dict - map flat action to dict keys
            3. Box((8,)) - return as-is
            """
            # Ensure action is numpy array
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)

            # Handle Box space
            if isinstance(self.action_space_structure, spaces.Box):
                expected_shape = self.action_space_structure.shape
                if len(expected_shape) == 2 and expected_shape[1] == 8:
                    # Vector env expecting (N, 8) matrix
                    # Reshape our (8,) action to (N, 8) by repeating rows
                    num_envs = expected_shape[0]
                    return np.tile(action.reshape(1, -1), (num_envs, 1)).astype(np.float32)
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
                obs_vec, info_vec = self.schola_env.reset(seed=seed, options=options)

                print(f"[SBDAPMScholaEnv] Reset returned obs type: {type(obs_vec)}")
                print(f"[SBDAPMScholaEnv] Reset returned info type: {type(info_vec)}")

                # Sync agent_ids based on actual observations after reset
                if isinstance(obs_vec, dict):
                    actual_agent_ids = list(obs_vec.keys())
                    if actual_agent_ids != self.agent_ids:
                        print(f"[SBDAPMScholaEnv] Syncing agent IDs: {self.agent_ids} -> {actual_agent_ids}")
                        self.agent_ids = actual_agent_ids
                        self.primary_agent_id = self.agent_ids[0] if self.agent_ids else None
                        # Also sync action space structure
                        if isinstance(self.schola_env.action_space, spaces.Dict) and self.primary_agent_id:
                            self.action_space_structure = self.schola_env.action_space[self.primary_agent_id]

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
                # Handle 2D observations (batch dimension) - take first observation
                if len(obs.shape) == 2 and obs.shape[-1] == 78:
                    # print(f"[SBDAPMScholaEnv] Extracting first observation from batch")
                    obs = obs[0]
                elif obs.size == 78:
                    obs = obs.reshape(78)
                else:
                    print(f"[SBDAPMScholaEnv] Cannot reshape, using zeros. obs.size={obs.size}")
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

                # Handle Dict action space - create action dict with all agent IDs
                if isinstance(self.schola_env.action_space, spaces.Dict) and self.agent_ids:
                    # Create action dict with all agent IDs using formatted action
                    action_vec = {agent_id: formatted_action.copy() if isinstance(formatted_action, np.ndarray)
                                  else formatted_action for agent_id in self.agent_ids}
                else:
                    # Fallback: use formatted action directly
                    action_vec = formatted_action

                # Call vectorized step
                obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = self.schola_env.step(action_vec)

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
                # print(f"[SBDAPMScholaEnv] Warning: Expected obs shape (78,), got {obs.shape}")
                # Handle 2D observations (batch dimension) - take first observation
                if len(obs.shape) == 2 and obs.shape[-1] == 78:
                    # print(f"[SBDAPMScholaEnv] Extracting first observation from batch")
                    obs = obs[0]
                elif obs.size == 78:
                    obs = obs.reshape(78)
                else:
                    print(f"[SBDAPMScholaEnv] Cannot reshape, using zeros. obs.size={obs.size}")
                    obs = np.zeros(78, dtype=np.float32)
            elif not hasattr(obs, 'shape'):
                 obs = np.zeros(78, dtype=np.float32)

            return obs, reward, terminated, truncated, info

        def render(self):
            """Rendering is handled by UE."""
            return self.schola_env.render() if hasattr(self.schola_env, 'render') else None

        def close(self):
            """Close Schola connection."""
            if hasattr(self.schola_env, 'close'):
                self.schola_env.close()


    class SBDAPMMultiAgentEnv(MultiAgentEnv):
        """
        Multi-Agent RLlib Environment for SBDAPM (v3.1)

        Proper multi-agent environment where each of the 4 follower agents:
        - Receives independent observations
        - Executes independent actions (from shared policy)
        - Collects independent rewards

        This fixes the bug where all agents were receiving identical actions.
        """

        def __init__(self, **kwargs):
            super().__init__()
            
            # Extract configuration
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 50051)
            self.max_episode_steps = kwargs.get("max_episode_steps", 100000)

            # Create Schola connection
            from schola.core.unreal_connections.editor_connection import UnrealEditorConnection
            connection = UnrealEditorConnection(url=host, port=port)
            self.schola_env = SafeUnrealVectorEnv(unreal_connection=connection, verbosity=1)

            # Define per-agent spaces
            self._obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(78,),
                dtype=np.float32
            )
            self._action_space = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(8,),
                dtype=np.float32
            )

            # Agent tracking
            self._agent_ids = set()
            self._agent_id_list = []  # Ordered list for consistent action ordering
            self.episode_steps = 0

            print(f"[SBDAPMMultiAgentEnv] Initialized (host={host}, port={port})")
            print(f"[DEBUG] Schola action_space type: {type(self.schola_env.action_space)}")
            print(f"[DEBUG] Schola observation_space type: {type(self.schola_env.observation_space)}")
            
            # Check if Schola has id_manager
            if hasattr(self.schola_env, 'id_manager'):
                print(f"[DEBUG] Schola has id_manager with {len(self.schola_env.id_manager.id_list)} agents")
                print(f"[DEBUG] Schola id_list: {self.schola_env.id_manager.id_list[:10]}")  # First 10



        @property
        def observation_space(self):
            """Return observation space for a single agent."""
            return self._obs_space

        @property
        def action_space(self):
            """Return action space for a single agent."""
            return self._action_space

        def reset(self, *, seed=None, options=None):
            """
            Reset environment for all agents.

            Returns:
                obs_dict: {agent_id: observation} for all agents
                info_dict: {agent_id: info} for all agents
            """
            self.episode_steps = 0

            try:
                # Get observations from Schola (dict with agent IDs as keys)
                obs_vec, info_vec = self.schola_env.reset(seed=seed, options=options)

                # Extract agent IDs from observations
                if isinstance(obs_vec, dict):
                    raw_agent_ids = set(obs_vec.keys())
                    print(f"[SBDAPMMultiAgentEnv] Raw agents detected: {sorted(list(raw_agent_ids))}")
                    
                    # Filter out CDO (Class Default Object) and other invalid entries
                    # CDO usually has the class name without numeric suffix, or "Default"
                    filtered_agent_ids = {
                        aid for aid in raw_agent_ids 
                        if "ScholaAgentComponent" != aid  # Exact match for CDO
                        and "Default" not in aid
                        and "Archetype" not in aid
                    }
                    
                    # Sort to ensure deterministic selection
                    sorted_agent_ids = sorted(list(filtered_agent_ids))
                    
                    # Limit to the number of agents Schola expects (from id_manager)
                    if hasattr(self.schola_env, 'id_manager'):
                        expected_count = len(self.schola_env.id_manager.id_list)
                        if len(sorted_agent_ids) > expected_count:
                            print(f"[SBDAPMMultiAgentEnv] Warning: Found {len(sorted_agent_ids)} valid-looking agents, but Schola expects {expected_count}.")
                            print(f"[SBDAPMMultiAgentEnv] Truncating list to match expected count.")
                            # We take the first N agents. This assumes the extra ones are extraneous/ghosts.
                            sorted_agent_ids = sorted_agent_ids[:expected_count]
                    
                    self._agent_ids = set(sorted_agent_ids)
                    self._agent_id_list = sorted_agent_ids
                    
                    print(f"[SBDAPMMultiAgentEnv] Reset: {len(self._agent_ids)} active agents: {self._agent_id_list}")

                    # Validate and reshape observations
                    obs_dict = {}
                    for agent_id in self._agent_ids:
                        if agent_id not in obs_vec:
                             print(f"[SBDAPMMultiAgentEnv] Error: Agent {agent_id} missing from obs_vec!")
                             obs_dict[agent_id] = np.zeros(78, dtype=np.float32)
                             continue

                        obs = obs_vec[agent_id]
                        # Handle batched observations (N, 78) -> take first
                        if hasattr(obs, 'shape') and len(obs.shape) == 2 and obs.shape[-1] == 78:
                            obs = obs[0]
                        elif hasattr(obs, 'shape') and obs.shape != (78,):
                            print(f"[SBDAPMMultiAgentEnv] Warning: Agent {agent_id} obs shape {obs.shape}, using zeros")
                            obs = np.zeros(78, dtype=np.float32)
                        obs_dict[agent_id] = obs.astype(np.float32)

                    # Extract info
                    info_dict = {}
                    if isinstance(info_vec, dict):
                        for agent_id in self._agent_ids:
                            info_dict[agent_id] = info_vec.get(agent_id, {})
                    else:
                        info_dict = {agent_id: {} for agent_id in self._agent_ids}

                    return obs_dict, info_dict

                else:
                    print(f"[SBDAPMMultiAgentEnv] ERROR: Expected dict observations, got {type(obs_vec)}")
                    # Fallback: create dummy agents
                    self._agent_ids = {f"agent_{i}" for i in range(4)}
                    obs_dict = {agent_id: np.zeros(78, dtype=np.float32) for agent_id in self._agent_ids}
                    info_dict = {agent_id: {} for agent_id in self._agent_ids}
                    return obs_dict, info_dict

            except Exception as e:
                print(f"[SBDAPMMultiAgentEnv] Error during reset: {e}")
                import traceback
                traceback.print_exc()
                # Return empty dicts
                return {}, {}

        def step(self, action_dict):
            """
            Execute actions for all agents.

            Args:
                action_dict: {agent_id: action} where action is (8,) numpy array

            Returns:
                obs_dict: {agent_id: observation}
                reward_dict: {agent_id: reward}
                terminated_dict: {agent_id: terminated}
                truncated_dict: {agent_id: truncated}
                info_dict: {agent_id: info}
            """
            try:
                # Validate action_dict has all agents
                if not isinstance(action_dict, dict):
                    print(f"[SBDAPMMultiAgentEnv] ERROR: Expected action dict, got {type(action_dict)}")
                    action_dict = {agent_id: np.zeros(8, dtype=np.float32) for agent_id in self._agent_ids}

                # Debug: Check what RLlib sent vs what we expect
                received_agents = set(action_dict.keys())
                expected_agents = self._agent_ids
                missing_agents = expected_agents - received_agents
                extra_agents = received_agents - expected_agents
                
                if missing_agents:
                    print(f"[SBDAPMMultiAgentEnv] Warning: RLlib didn't send actions for {len(missing_agents)} agents: {list(missing_agents)[:3]}")
                if extra_agents:
                    print(f"[SBDAPMMultiAgentEnv] Warning: RLlib sent actions for {len(extra_agents)} unknown agents: {list(extra_agents)[:3]}")

                # CRITICAL FIX: Schola's GymVectorEnv.unbatch_actions() expects actions
                # in the SAME FORMAT as observations are returned - a batched dict/array
                # that can be iterated using gym.experimental.vector.utils.iterate()
                #
                # The unbatch_actions method converts this to nested dict format using:
                #   it = gym.experimental.vector.utils.iterate(self.action_space, actions)
                #   return self.id_manager.nest_id_list([value for value in it])
                #
                # This means we need to pass actions in the batched format that matches
                # self.schola_env.action_space (the batched version, not single_action_space)
                
                # Convert RLlib's action_dict {agent_id: action} to Schola's expected format
                # Schola uses integer flat IDs (0, 1, 2, 3...) internally via id_manager
                
                # Build a mapping from agent_id (string) to flat_id (int)
                if not hasattr(self, '_agent_id_to_flat_id'):
                    # Create mapping on first step
                    self._agent_id_to_flat_id = {}
                    if hasattr(self.schola_env, 'id_manager'):
                        # Map each agent_id to its flat ID in Schola's id_manager
                        print(f"[DEBUG] Creating agent_id to flat_id mapping...")
                        for flat_id, (env_id, agent_id_int) in enumerate(self.schola_env.id_manager.id_list):
                            # Try to find matching agent_id in our _agent_ids
                            # Schola uses integer agent IDs internally, but exposes string IDs in obs dict
                            # We need to match by position/index
                            if flat_id < len(self._agent_id_list):
                                agent_id_str = self._agent_id_list[flat_id]
                                self._agent_id_to_flat_id[agent_id_str] = flat_id
                                print(f"[DEBUG]   {agent_id_str} -> flat_id {flat_id}")
                
                # Create action array/dict in Schola's expected batched format
                # The format depends on self.schola_env.action_space structure
                if isinstance(self.schola_env.action_space, spaces.Dict):
                    # Batched Dict space: {key: array(num_envs, ...)} for each action component
                    # Convert our per-agent actions to batched format
                    formatted_actions = {}

                    # Get action space keys from Schola
                    schola_action_keys = list(self.schola_env.single_action_space.keys())

                    # Determine batch size (num_envs)
                    batch_size = self.schola_env.num_envs

                    # Initialize batched arrays for ALL Schola keys (including CDO and extras)
                    for key in schola_action_keys:
                        key_space = self.schola_env.single_action_space[key]
                        if isinstance(key_space, spaces.Box):
                            shape = (batch_size,) + key_space.shape
                            formatted_actions[key] = np.zeros(shape, dtype=np.float32)

                    # Fill in actions ONLY for our active agents using flat_id mapping
                    for agent_id in self._agent_ids:
                        action = action_dict.get(agent_id, np.zeros(8, dtype=np.float32))
                        if not isinstance(action, np.ndarray):
                            action = np.array(action, dtype=np.float32)

                        # Get flat_id for this agent
                        flat_id = self._agent_id_to_flat_id.get(agent_id, 0)

                        # Assign action to the correct position
                        if agent_id in formatted_actions and flat_id < batch_size:
                            formatted_actions[agent_id][flat_id] = action
                        else:
                            print(f"[WARNING] Agent {agent_id} not found in action space or flat_id {flat_id} >= batch_size {batch_size}")
                else:
                    # Batched Box space: array(N, action_dim)
                    # Create ordered array using flat IDs
                    num_agents = len(self._agent_ids)
                    action_dim = 8
                    formatted_actions = np.zeros((num_agents, action_dim), dtype=np.float32)
                    
                    for agent_id in self._agent_ids:
                        action = action_dict.get(agent_id, np.zeros(8, dtype=np.float32))
                        if not isinstance(action, np.ndarray):
                            action = np.array(action, dtype=np.float32)
                        
                        # Get flat_id for this agent
                        flat_id = self._agent_id_to_flat_id.get(agent_id, 0)
                        formatted_actions[flat_id] = action.astype(np.float32)
                
                print(f"[DEBUG] Calling Schola step with formatted_actions type: {type(formatted_actions)}")
                if isinstance(formatted_actions, dict):
                    print(f"[DEBUG] formatted_actions keys: {list(formatted_actions.keys())}")
                    for key, val in formatted_actions.items():
                        # Print shape and check for non-zero actions
                        if list(formatted_actions.keys()).index(key) < 6:  # Show first 6 agents
                            non_zero_count = np.count_nonzero(val) if hasattr(val, 'shape') else 0
                            print(f"[DEBUG] {key}: shape {val.shape if hasattr(val, 'shape') else 'N/A'}, non-zero: {non_zero_count}")
                elif isinstance(formatted_actions, np.ndarray):
                    print(f"[DEBUG] formatted_actions array shape: {formatted_actions.shape}")
                
                obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = self.schola_env.step(formatted_actions)



                # Process observations
                obs_dict = {}
                if isinstance(obs_vec, dict):
                    for agent_id in self._agent_ids:
                        obs = obs_vec.get(agent_id, np.zeros(78, dtype=np.float32))
                        # Handle batched observations
                        if hasattr(obs, 'shape') and len(obs.shape) == 2 and obs.shape[-1] == 78:
                            obs = obs[0]
                        elif hasattr(obs, 'shape') and obs.shape != (78,):
                            obs = np.zeros(78, dtype=np.float32)
                        obs_dict[agent_id] = obs.astype(np.float32)
                else:
                    obs_dict = {agent_id: np.zeros(78, dtype=np.float32) for agent_id in self._agent_ids}

                # Process rewards
                reward_dict = {}
                if isinstance(reward_vec, dict):
                    reward_dict = {agent_id: float(reward_vec.get(agent_id, 0.0)) for agent_id in self._agent_ids}
                else:
                    reward_dict = {agent_id: 0.0 for agent_id in self._agent_ids}

                # Process terminated
                terminated_dict = {}
                if isinstance(terminated_vec, dict):
                    terminated_dict = {agent_id: bool(terminated_vec.get(agent_id, False)) for agent_id in self._agent_ids}
                else:
                    terminated_dict = {agent_id: False for agent_id in self._agent_ids}

                # Process truncated
                truncated_dict = {}
                if isinstance(truncated_vec, dict):
                    truncated_dict = {agent_id: bool(truncated_vec.get(agent_id, False)) for agent_id in self._agent_ids}
                else:
                    truncated_dict = {agent_id: False for agent_id in self._agent_ids}

                # Process info
                info_dict = {}
                if isinstance(info_vec, dict):
                    info_dict = {agent_id: info_vec.get(agent_id, {}) for agent_id in self._agent_ids}
                else:
                    info_dict = {agent_id: {} for agent_id in self._agent_ids}

                self.episode_steps += 1

                # Enforce max episode length
                if self.episode_steps >= self.max_episode_steps:
                    truncated_dict = {agent_id: True for agent_id in self._agent_ids}

                # RLlib expects "__all__" key for global termination
                terminated_dict["__all__"] = all(terminated_dict.values())
                truncated_dict["__all__"] = all(truncated_dict.values())

                return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

            except Exception as e:
                print(f"[SBDAPMMultiAgentEnv] Error during step: {e}")
                import traceback
                traceback.print_exc()
                # Return error state
                obs_dict = {agent_id: np.zeros(78, dtype=np.float32) for agent_id in self._agent_ids}
                reward_dict = {agent_id: 0.0 for agent_id in self._agent_ids}
                terminated_dict = {agent_id: True for agent_id in self._agent_ids}
                terminated_dict["__all__"] = True
                truncated_dict = {agent_id: False for agent_id in self._agent_ids}
                truncated_dict["__all__"] = False
                info_dict = {agent_id: {} for agent_id in self._agent_ids}
                return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

        def render(self):
            """Rendering is handled by UE."""
            return self.schola_env.render() if hasattr(self.schola_env, 'render') else None

        def close(self):
            """Close Schola connection."""
            if hasattr(self.schola_env, 'close'):
                self.schola_env.close()
