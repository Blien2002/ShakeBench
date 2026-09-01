"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        # Most APIs between gym and gymnasium are compatible
        import gym
        from gym import spaces
    except ImportError as gym_import_error:
        raise ImportError("GymWrapper requires gymnasium or gym>=0.26; install one to use it.") from gym_import_error

    gym_version = str(getattr(gym, "__version__", "0.0.0"))
    version_parts = []
    for component in gym_version.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        version_parts.append(int(digits))
    parsed_version = tuple((version_parts + [0, 0])[:2])
    if parsed_version < (0, 26):
        version_error = "GymWrapper requires gym>=0.26 when gymnasium is unavailable; " f"found gym=={gym_version}."
        raise ImportError(version_error)

from robosuite.wrappers import Wrapper


class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
        flatten_obs (bool):
            Whether to flatten the observation dictionary into a 1d array. Defaults to True.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None, flatten_obs=True):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            if getattr(self.env, "observation_tier", None) is not None and hasattr(self.env, "policy_observation_keys"):
                keys = list(self.env.policy_observation_keys)
            else:
                keys = []
                # Add object obs if requested
                if self.env.use_object_obs:
                    keys += ["object-state"]
                # Add image obs if requested
                if self.env.use_camera_obs:
                    keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
                # Iterate over all robots to add to state
                for idx in range(len(self.env.robots)):
                    keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys
        if any(str(key).startswith("privileged_") for key in self.keys):
            raise ValueError("GymWrapper refuses privileged observation keys")

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        if getattr(self.env, "observation_tier", None) is not None:
            missing = [key for key in self.keys if key not in obs]
            if missing:
                raise ValueError("GymWrapper requested unavailable State key(s): " + ", ".join(missing))

        # Whether to flatten the observation space
        self.flatten_obs: bool = flatten_obs

        if self.flatten_obs:
            flat_ob = self._flatten_obs(obs)
            self.obs_dim = flat_ob.size
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:

            def get_box_space(sample):
                """Util fn to obtain the space of a single numpy sample data"""
                if np.issubdtype(sample.dtype, np.bool_):
                    return spaces.MultiBinary(sample.shape)
                elif np.issubdtype(sample.dtype, np.integer):
                    low = np.iinfo(sample.dtype).min
                    high = np.iinfo(sample.dtype).max
                elif np.issubdtype(sample.dtype, np.inexact):
                    low = float("-inf")
                    high = float("inf")
                elif np.issubdtype(sample.dtype, np.str_):
                    text = str(sample.reshape(-1)[0]) if sample.size else ""
                    if hasattr(spaces, "Text"):
                        return spaces.Text(max_length=max(1, len(text)))
                    encoded = self._encode_text(sample)
                    return spaces.Box(
                        low=np.zeros(encoded.shape, dtype=np.uint8),
                        high=np.full(encoded.shape, 255, dtype=np.uint8),
                        dtype=np.uint8,
                    )
                else:
                    raise ValueError()
                return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

            self.observation_space = spaces.Dict({key: get_box_space(obs[key]) for key in self.keys})

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                value = np.asarray(obs_dict[key])
                if np.issubdtype(value.dtype, np.str_):
                    value = self._encode_text(value)
                ob_lst.append(value.flatten())
        return np.concatenate(ob_lst) if ob_lst else np.zeros(0, dtype=np.float32)

    @staticmethod
    def _encode_text(value):
        array = np.asarray(value)
        text = str(array.reshape(-1)[0]) if array.size else ""
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32)

    @staticmethod
    def _decode_text(value):
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError("GymWrapper text observations must be scalar strings")
        return str(array.reshape(-1)[0])

    def _filter_obs(self, obs_dict) -> dict:
        """
        Filters keys of interest out of the observation dictionary, returning a filterd dictionary.
        """
        result = {}
        for key in self.keys:
            if key not in obs_dict:
                continue
            value = obs_dict[key]
            value_array = np.asarray(value)
            if np.issubdtype(value_array.dtype, np.str_) and hasattr(spaces, "Text"):
                value = self._decode_text(value)
            elif np.issubdtype(value_array.dtype, np.str_):
                value = self._encode_text(value)
            result[key] = value
        return result

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return observation instead of normal OrderedDict and optionally resets seed

        Returns:
            2-tuple:
                - (np.array) observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return obs, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return obs, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def close(self):
        """
        wrapper for calling underlying env close function
        """
        self.env.close()
