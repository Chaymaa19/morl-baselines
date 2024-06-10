import mo_gymnasium as mo_gym
import numpy as np
import sys
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from memory_profiler import profile


def get_size(obj, seen=None):
    """
    Recursively finds size of objects.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important to mark as seen *before* entering recursion to handle
    # self-referential objects properly.
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)
    return size


def to_kilobytes(size_bytes):
    """
    Convert size in bytes to megabytes or gigabytes.
    """
    kilobytes = size_bytes / 1024
    return kilobytes


def to_megabytes(size_bytes):
    """
    Convert size in bytes to megabytes or gigabytes.
    """
    megabytes = size_bytes / (1024 * 1024)
    return megabytes


def to_gigabytes(size_bytes):
    """
    Convert size in bytes to megabytes or gigabytes.
    """
    gigabytes = size_bytes / (1024 * 1024 * 1024)
    return gigabytes


@profile
def main():
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    ref_point = np.array([0, -25])

    # agent = PQL(
    #     env,
    #     ref_point,
    #     gamma=0.99,
    #     initial_epsilon=1.0,
    #     epsilon_decay_steps=50000,
    #     final_epsilon=0.2,
    #     seed=1,
    #     project_name="MORL-Baselines",
    #     experiment_name="Pareto Q-Learning",
    #     log=True,
    # )
    #
    # pf = agent.train(
    #     total_timesteps=1000,
    #     log_every=10000,
    #     action_eval="hypervolume",
    #     known_pareto_front=env.pareto_front(gamma=0.99),
    #     ref_point=ref_point,
    #     eval_env=env,
    # )
    folder = "shelve-test/"

    # agent.save(folder)

    agent = PQL.load(folder, env)
    print(str(to_kilobytes(get_size(agent))) + "KB")


if __name__ == "__main__":
    main()
