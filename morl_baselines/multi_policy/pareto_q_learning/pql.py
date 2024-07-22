"""Pareto Q-Learning."""
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


class PQL(MOAgent):
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

        if type(self.env.observation_space) == gym.spaces.Discrete:
            self.env_shape = (self.env.observation_space.n,)
        elif type(self.env.observation_space) == gym.spaces.MultiDiscrete:
            self.env_shape = self.env.observation_space.nvec
        elif (
                type(self.env.observation_space) == gym.spaces.Box
                and self.env.observation_space.is_bounded(manner="both")
                and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.reward_space.shape[0]

        if not is_loaded_checkpoint:
            self.counts = np.zeros((self.num_states, self.num_actions), dtype=np.int32)
            self.non_dominated = [
                [{tuple(np.zeros(self.num_objectives, dtype=np.float32))} for _ in range(self.num_actions)] for _ in
                range(self.num_states)
            ]
            self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives), dtype=np.float32)

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

    def score_hypervolume(self, state: int):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def get_q_set_default(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
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
        if self.np_random.uniform(0, 1) < self.epsilon:
            # return self.np_random.integers(self.num_actions)
            print("Sampling valid action")

            return self.env.action_space.sample(
                mask=self.env.action_masks().astype(np.int8))  # TODO: això només funciona amb nxg
        else:
            action_scores = score_func(state)
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
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")
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
            state = int(np.ravel_multi_index(state, self.env_shape))
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
                next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                begin_time = time.time()
                self.counts[state, action] += 1
                if not (terminated or truncated):
                    self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]
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

                if self.log and self.global_step % log_every == 0:
                    begin_time = time.time()
                    # pf = self._eval_all_policies(eval_env)
                    pf = list(self.get_local_pcs(0))
                    eval_time = time.time() - begin_time

                    log_all_multi_policy_metrics(
                        current_front=pf,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                        custom_logger=self.logger
                    )
                    self.logger.dump(step=self.global_step)

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

    def register_additional_config(self, conf: Dict = {}) -> None:
        for key, value in conf.items():
            self.logger.write_param(key=key, value=value)

    def _eval_all_policies(self, env: gym.Env) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))

        return pf

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
            state = np.ravel_multi_index(state, self.env_shape)
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                non_dominated_set = self.non_dominated[state][action]

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
            state = np.ravel_multi_index(state, self.env_shape)
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
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

    def get_all_policies_from_state(self, tracked_policies: List[PQLPolicy], env: gym.Env, tol=1e-3,
                                    desired_sectors: List[str] = None):
        # Check if there's any undone policy
        while any(map(lambda policy: not policy.done, tracked_policies)):
            undone_policies = [policy for policy in tracked_policies if not policy.done]
            for policy in undone_policies:
                # Reset environment and apply the policy's actions
                episode_reward = np.zeros(self.num_objectives)
                state, _ = env.reset()
                for action in policy.applied_actions:
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                # For each found action create a new PQLPolicy
                state = np.ravel_multi_index(state, self.env_shape)
                for action in range(self.num_actions):
                    im_rew = self.avg_reward[state, action]
                    non_dominated_set = self.non_dominated[str(state)][str(action)]

                    for q in non_dominated_set:
                        q = np.array(q)
                        dist = np.sum(np.abs(self.gamma * q + im_rew - policy.target))
                        if dist < tol and (not desired_sectors or env.action_manager.get_policy_from_action(action)[
                            0]['sector'] in desired_sectors):
                            # Found action corresponding to q vector
                            _, action_reward, terminated, truncated, _ = env.step(action)
                            new_policy = PQLPolicy(target=q, applied_actions=policy.applied_actions + [action],
                                                   total_reward=episode_reward + action_reward,
                                                   done=terminated or truncated)
                            tracked_policies.append(new_policy)
                            # Reset environment and apply the policy's actions
                            episode_reward = np.zeros(self.num_objectives)
                            state, _ = env.reset()
                            for action in policy.applied_actions:
                                state, reward, terminated, truncated, _ = env.step(action)
                                episode_reward += reward
                            state = np.ravel_multi_index(state, self.env_shape)

                # Pop tracked policy from list of tracked policies
                tracked_policies.remove(policy)

        # Return final results
        return tracked_policies

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
        int_state = np.ravel_multi_index(state, self.env_shape)
        return self.get_local_pcs(int_state)

    def save(self, path: str) -> None:
        """
        Save model checkpoint
        :param path: dir where to save the config of the algorithm
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Save params in json file
        params_path = path + "/pql_params.json"
        pql_params = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "ref_point": self.ref_point.tolist(),
            "num_states": int(self.num_states),
            "num_actions": int(self.num_actions),
            "num_objectives": int(self.num_objectives)
        }
        dump_json_file(path=params_path, data=pql_params)

        # Save counts, non_dominated and avg_reward tables
        np.save(file=path + "/counts", arr=self.counts)
        dump_non_dominated_in_shelve(path=path + "/non_dominated.shelf", non_dominated=self.non_dominated)
        np.save(file=path + "/avg_reward", arr=self.avg_reward.reshape(self.avg_reward.shape[0], -1))

    @classmethod
    def load(cls, checkpoint_path: str, env, new_logger: Optional[Logger] = None):
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for model in {checkpoint_path} not found!")

        # Load params dict, counts, non_dominated and avg_reward tables
        pql_params = load_json_file(path=checkpoint_path + "/pql_params.json")
        counts = np.load(file=checkpoint_path + "/counts.npy", mmap_mode="r")
        non_dominated = shelve.open(checkpoint_path + "/non_dominated.shelf", 'r')
        avg_reward = np.load(file=checkpoint_path + "/avg_reward.npy", mmap_mode="r")
        avg_reward = avg_reward.reshape(pql_params["num_states"], pql_params["num_actions"],
                                        pql_params["num_objectives"])

        # Create instance of the algorithm with loaded params
        model = PQL(
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
        model.counts = counts
        model.non_dominated = non_dominated
        model.avg_reward = avg_reward
        model.num_objectives = pql_params['num_objectives']

        return model

    def close_all_files(self):
        self.non_dominated.close()


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
