"""Pareto Q-Learning, but instead of preallocating memory for the Q-table, it will use dynamic data structures
to add incrementally the states and actions seen"""
import numbers
import os
import json
import shelve
import time
from datetime import datetime
from typing import Callable, List, Optional, Dict

import gymnasium as gym
import numpy as np
import pickle
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics, log_all_progress_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value
from morl_baselines.common.logger import Logger


class PQLPolicy:
    def __init__(self, target: np.ndarray, applied_actions: List[int], total_reward: np.ndarray, done: bool):
        self.target = target
        self.applied_actions = applied_actions
        self.total_reward = total_reward
        self.done = done


class DynamicPQL(MOAgent):
    """Pareto Q-learning.

    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
            self,
            env,
            ref_point: np.ndarray,
            gamma: float = 0.8,
            initial_epsilon: float = 1.0,
            epsilon_decay_steps: int = 100000,
            final_epsilon: float = 0.1,
            seed: Optional[int] = None,
            project_name: Optional[str] = "MORL-Baselines",
            experiment_name: Optional[str] = "Pareto Q-Learning",
            logger: Optional[Logger] = None,
            log: bool = True,
            is_loaded_checkpoint: bool = False
    ):
        """Initialize the Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The wandb entity used for logging.
            log: Whether to log or not.
        """
        super().__init__(env, seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.ref_point = ref_point

        if type(self.env.action_space) == gym.spaces.Discrete:
            self.num_actions = self.env.action_space.n
        elif type(self.env.action_space) == gym.spaces.MultiDiscrete:
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

        self.num_objectives = self.env.reward_space.shape[0]

        if not is_loaded_checkpoint:
            # Dynamic data structures: use dicts instead of pre-allocated arrays
            # Mapping from state_id (int) -> actual state (tuple)
            self.seen_states = {}
            # Mapping from actual state (tuple) -> state_id (int) for reverse lookup
            self.state_to_id = {}
            # Counter for next state ID
            self.next_state_id = 0
            # Dynamic structures: state_id -> action -> value
            self.counts = {}  # dict[state_id, dict[action, count]]
            self.non_dominated = {}  # dict[state_id, dict[action, set]]
            self.avg_reward = {}  # dict[state_id, dict[action, np.ndarray]]

        if is_loaded_checkpoint:
            self.get_q_set = self.get_q_set_inference
        else:
            self.get_q_set = self.get_q_set_default

        # Logging
        self.log = log
        self.logger = logger

        if self.log and not self.logger:
            self.project_name = project_name
            self.experiment_name = experiment_name
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def register_additional_config(self, conf: Dict = {}) -> None:
        for key, value in conf.items():
            self.logger.write_param(key=key, value=value)

    def _get_state_id(self, state):
        """Get or create a state ID for the given state.

        Args:
            state: environment state (np.ndarray)

        Returns:
            int: The state ID
        """
        state_tuple = tuple(state.astype(int).tolist())
        if state_tuple not in self.state_to_id:
            state_id = self.next_state_id
            self.next_state_id += 1
            self.state_to_id[state_tuple] = state_id
            self.seen_states[state_id] = state_tuple
            # Initialize data structures for this new state
            self.counts[state_id] = {}
            self.non_dominated[state_id] = {}
            self.avg_reward[state_id] = {}
            # Initialize with default values for all actions
            for action in range(self.num_actions):
                self.counts[state_id][action] = 0
                self.non_dominated[state_id][action] = {tuple(np.zeros(self.num_objectives, dtype=np.float32))}
                self.avg_reward[state_id][action] = np.zeros(self.num_objectives, dtype=np.float32)
        return self.state_to_id[state_tuple]

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def get_q_set_default(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state ID.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        if state not in self.non_dominated or action not in self.non_dominated[state]:
            # State/action not seen yet, return default
            return {tuple(np.zeros(self.num_objectives, dtype=np.float32))}
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state][action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def get_q_set_inference(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[str(state)][str(action)]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        # TODO: Not all envs have action_masks function
        if self.np_random.uniform(0, 1) < self.epsilon:
            # return self.np_random.integers(self.num_actions)
            return self.env.action_space.sample(
                mask=self.env.action_masks().astype(np.int8))
        else:
            action_scores = score_func(state)
            action_scores = action_scores * self.env.action_masks()  # set to zero all invalid actions
            return self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

    def train(
            self,
            total_timesteps: int,
            eval_env: gym.Env,
            ref_point: Optional[np.ndarray] = None,
            known_pareto_front: Optional[List[np.ndarray]] = None,
            num_eval_weights_for_eval: int = 50,
            log_every: Optional[int] = 10000,
            log_progress_every: Optional[int] = 10000,
            action_eval: Optional[str] = "hypervolume",
    ):
        """Learn the Pareto front.

        Args:
            total_timesteps (int, optional): The number of episodes to train for.
            eval_env (gym.Env): The environment to evaluate the policies on.
            eval_ref_point (ndarray, optional): The reference point for the hypervolume metric during evaluation. If none, use the same ref point as training.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 1000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            Set: The final Pareto front.
        """
        score_func = self.score_pareto_cardinality
        if ref_point is None:
            ref_point = self.ref_point
        if self.log:
            if not self.logger:
                super().register_additional_config(
                    {
                        "total_timesteps": total_timesteps,
                        "ref_point": ref_point.tolist(),
                        "known_front": known_pareto_front,
                        "num_eval_weights_for_eval": num_eval_weights_for_eval,
                        "log_every": log_every,
                        "action_eval": action_eval,
                    }
                )
            else:
                self.register_additional_config(
                    {
                        "total_timesteps": total_timesteps,
                        "ref_point": ref_point.tolist(),
                        "known_front": known_pareto_front,
                        "num_eval_weights_for_eval": num_eval_weights_for_eval,
                        "log_every": log_every,
                        "action_eval": action_eval,
                    }
                )

        num_episodes = 0
        train_total_episodes = 0
        train_begin_time = time.time()
        iteration_begin_time = time.time()
        step_time = 0
        update_time = 0
        epsilon_decay_time = 0
        time_logging_metrics = -1
        time_selecting_action = 0
        while self.global_step < total_timesteps:
            begin_step = time.time()
            state, _ = self.env.reset()
            step_time += (time.time() - begin_step)
            num_episodes += 1
            train_total_episodes += 1
            state = self._get_state_id(state)
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                begin_time = time.time()
                action = self.select_action(state, score_func)
                time_selecting_action += (time.time() - begin_time)
                begin_step = time.time()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                step_time += (time.time() - begin_step)
                self.global_step += 1
                next_state = self._get_state_id(next_state)

                begin_time = time.time()
                self.counts[state][action] += 1
                if not (terminated or truncated):
                    self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self.avg_reward[state][action] += (reward - self.avg_reward[state][action]) / self.counts[state][action]
                update_time += (time.time() - begin_time)
                state = next_state

                if self.log and self.global_step % log_progress_every == 0:
                    begin_time = time.time()
                    log_all_progress_metrics(
                        global_step=self.global_step,
                        num_pf_solutions=len(self.get_local_pcs(0)),
                        num_episodes=num_episodes,
                        train_total_episodes=train_total_episodes,
                        iteration_time=time.time() - iteration_begin_time,
                        elapsed_time=time.time() - train_begin_time,
                        step_time=step_time,
                        update_time=update_time,
                        eval_time=0,
                        time_logging_metrics=time_logging_metrics,
                        time_selecting_action=time_selecting_action,
                        epsilon_decay_time=epsilon_decay_time,
                        custom_logger=self.logger
                    )
                    time_logging_metrics = time.time() - begin_time
                    self.logger.dump(step=self.global_step)
                    num_episodes = 0
                    step_time = 0
                    iteration_begin_time = time.time()
                    update_time = 0
                    time_selecting_action = 0
                    epsilon_decay_time = 0

            begin_time = time.time()
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )
            epsilon_decay_time += (time.time() - begin_time)

        return self.get_local_pcs(state=0)

    def track_policy(self, vec, env: gym.Env, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
        """
        target = np.array(vec)
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            state_id = self.state_to_id.get(state)
            if state_id is None:
                # State not seen, break
                break

            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state_id, action]
                non_dominated_set = self.non_dominated[state_id][action]

                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew

    def get_policy_from_state(self, vec, current_state, previous_reward: np.ndarray, env: gym.Env, tol=1e-3) -> tuple:
        """
        Get the set of actions used in the tracked policy along with the total reward obtained
        :param vec: array_like with the return vector to track.
        :param current_state: Current state in the environment.
        :param previous_reward: Cumulative reward obtained until the current state
        :param env: The environment to track the policy in.
        :param tol: The tolerance for the return vector. (Default value = 1e-3)
        :return: list of actions taken and total
        """
        target = np.array(vec)
        state = current_state
        terminated = False
        truncated = False
        total_rew = previous_reward
        current_gamma = 1.0
        actions_list = []

        while not (terminated or truncated):
            state_id = self.state_to_id.get(state)
            if state_id is None:
                # State not seen, break
                break

            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state_id, action]
                non_dominated_set = self.non_dominated[str(state)][str(action)]

                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            actions_list.append(closest_action)
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return actions_list, total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.

        Args:
            state (int): The state to get a local PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)

    def get_local_pcs_from_state(self, state):
        """
        Get the pareto coverage set from an array-like state
        :param state: The state to get a local PCS for. (array-like)
        :return: A set of pareto optimal vectors
        """
        state_id = self.state_to_id.get(state)
        if state_id is None:
            # State not seen, return empty set
            return set()
        return self.get_local_pcs(state_id)

    def save(self, path: str) -> None:
        """
        Save model checkpoint
        :param path: dir where to save the config of the algorithm
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Convert dynamic structures back to original format for compatibility
        num_seen_states = len(self.seen_states)
        if num_seen_states == 0:
            # No states seen yet, create empty structures
            counts_array = np.zeros((1, self.num_actions), dtype=np.int32)
            non_dominated_list = [
                [{tuple(np.zeros(self.num_objectives, dtype=np.float32))} for _ in range(self.num_actions)]
                for _ in range(1)
            ]
            avg_reward_array = np.zeros((1, self.num_actions, self.num_objectives), dtype=np.float32)
        else:
            # Create arrays with size num_states (original size) but only fill seen states
            counts_array = np.zeros((num_seen_states, self.num_actions), dtype=np.int32)
            non_dominated_list = [
                [{tuple(np.zeros(self.num_objectives, dtype=np.float32))} for _ in range(self.num_actions)]
                for _ in range(num_seen_states)
            ]
            avg_reward_array = np.zeros((num_seen_states, self.num_actions, self.num_objectives), dtype=np.float32)

            # Fill arrays with data from seen states
            # Use state_id directly as array index (state_ids should be sequential starting from 0)
            for state_id, state_tuple in self.seen_states.items():
                for action in range(self.num_actions):
                    if state_id in self.counts and action in self.counts[state_id]:
                        counts_array[state_id, action] = self.counts[state_id][action]
                    if state_id in self.non_dominated and action in self.non_dominated[state_id]:
                        non_dominated_list[state_id][action] = self.non_dominated[state_id][action]
                    if state_id in self.avg_reward and action in self.avg_reward[state_id]:
                        avg_reward_array[state_id, action] = self.avg_reward[state_id][action]

        # Save params in json file
        params_path = path + "/pql_params.json"
        pql_params = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "ref_point": self.ref_point.tolist(),
            "num_actions": int(self.num_actions),
            "num_objectives": int(self.num_objectives),
            "num_states": num_seen_states,
            "next_state_id": self.next_state_id
        }
        dump_json_file(path=params_path, data=pql_params)

        # Save seen_states mapping
        seen_states_data = {str(k): list(v) if isinstance(v, (tuple, np.ndarray)) else [v]
                            for k, v in self.seen_states.items()}
        dump_json_file(path=path + "/seen_states.json", data=seen_states_data)

        # Save counts, non_dominated and avg_reward tables in original format
        np.save(file=path + "/counts", arr=counts_array)
        dump_non_dominated_in_shelve(path=path + "/non_dominated.shelf", non_dominated=non_dominated_list)
        np.save(file=path + "/avg_reward", arr=avg_reward_array.reshape(avg_reward_array.shape[0], -1))

    @classmethod
    def load(cls, checkpoint_path: str, env, new_logger: Optional[Logger] = None):
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for model in {checkpoint_path} not found!")

        # Load params dict, counts, non_dominated and avg_reward tables
        pql_params = load_json_file(path=checkpoint_path + "/pql_params.json")
        counts_array = np.load(file=checkpoint_path + "/counts.npy", mmap_mode="r")
        non_dominated = shelve.open(checkpoint_path + "/non_dominated.shelf", 'r')
        avg_reward_array = np.load(file=checkpoint_path + "/avg_reward.npy", mmap_mode="r")
        avg_reward_array = avg_reward_array.reshape(pql_params["num_states"], pql_params["num_actions"],
                                                    pql_params["num_objectives"])

        # Create instance of the algorithm with loaded params
        model = DynamicPQL(
            env=env,
            ref_point=np.array(pql_params["ref_point"]),
            gamma=pql_params["gamma"],
            initial_epsilon=pql_params["initial_epsilon"],
            epsilon_decay_steps=pql_params["epsilon_decay_steps"],
            final_epsilon=pql_params["final_epsilon"],
            logger=new_logger,
            log=new_logger is not None,  # Log only if new_logger is provided
            is_loaded_checkpoint=True
        )

        # Load state mappings if available, otherwise reconstruct
        seen_states_path = checkpoint_path + "/seen_states.json"
        if os.path.exists(seen_states_path):
            seen_states_data = load_json_file(path=seen_states_path)
            # Reconstruct seen_states and state_to_id mappings
            model.seen_states = {}
            model.state_to_id = {}
            for state_id_str, state_list in seen_states_data.items():
                state_id = int(state_id_str)
                state_tuple = tuple(state_list)
                model.seen_states[state_id] = state_tuple
                model.state_to_id[state_tuple] = state_id
            model.next_state_id = pql_params.get("next_state_id", len(model.seen_states))
        else:
            # Fallback: cannot reconstruct state tuples without seen_states.json
            # This should not happen if save() was called properly
            raise ValueError("seen_states.json not found. Cannot reconstruct state mappings.")

        # Keep original array structures (don't convert to dicts)
        # The arrays use state_id as index, which matches our state_to_id mapping
        model.counts = counts_array
        model.non_dominated = non_dominated
        model.avg_reward = avg_reward_array

        return model

    def close_all_files(self):
        # Only close if it's a shelf (when loaded), not if it's a list or dict
        if hasattr(self.non_dominated, 'close'):
            self.non_dominated.close()
        if isinstance(self.counts, np.ndarray):
            del self.counts
        if isinstance(self.avg_reward, np.ndarray):
            del self.avg_reward


# TODO: move somewhere else
def load_pickle(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def dump_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def dump_non_dominated_in_shelve(path, non_dominated):
    with shelve.open(path, 'n') as shelf:
        num_states = len(non_dominated)
        num_actions = len(non_dominated[0])
        for state in range(num_states):
            print(f"{datetime.now()} - Processing state {state}")
            if str(state) not in shelf:
                shelf[str(state)] = {}
            state_q_sets = shelf[str(state)]
            for action in range(num_actions):
                state_q_sets[str(action)] = non_dominated[state][action]
            shelf[str(state)] = state_q_sets


def load_json_file(path):
    with open(path, encoding="utf-8") as json_data:
        return json.load(json_data)


def dump_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f)
