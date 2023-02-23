from ray.rllib.agents import ppo
from cuas.envs.cuas_env_multi_agent_v1 import CuasEnvMultiAgentV1
from ray.rllib.agents.callbacks import MultiCallbacks
from cuas.utils.callbacks import TrainCallback, FillInActions
from gym.spaces import Dict
import numpy as np


def get_agent_config(config, ppo_config=None):
    """Return train ppo agent
    See link for common paramenters:
    https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters

    Args:
        config (_type_): _description_
    """

    if ppo_config is None:
        env_config = config["env_config"]
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config["train_config"])
    else:
        ppo_config.update(config["test_config"])
        env_config = ppo_config["env_config"]
        env_config.update(config["test_env_config"])

    temp_env = CuasEnvMultiAgentV1(env_config)

    # only update here what can't be updated from the configuration file.
    callback_list = [TrainCallback]
    if config["centralized_observer"]:
        observer_space = Dict(
            {
                "own_obs": temp_env.observation_space[0]["observations"],
                "all_obs": Dict(
                    {
                        agent_id: temp_env.observation_space[0]["observations"]
                        for agent_id in range(temp_env.num_agents)
                    }
                ),
                "all_actions": temp_env.action_space,
            }
        )
        callback_list.append(FillInActions)
    else:
        observer_space = temp_env.observation_space[0]

    training_callback = MultiCallbacks(callback_list)
    ppo_config.update(
        {
            "callbacks": training_callback,
            "env": config["env_name"],
            "env_config": env_config,
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": {
                    "pursuer": (
                        None,
                        observer_space,
                        temp_env.action_space[0],
                        {},
                    )
                },
                # Always use "shared" policy.
                # "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "pursuer"),
                "policy_mapping_fn": lambda _: "pursuer",
            },
            # See https://docs.ray.io/en/latest/rllib/rllib-models.html#default-behaviors for common model configs
            # "model": {
            #     "custom_model": "TorchFixModel",
            #     "custom_model_config": {},
            # },
        }
    )

    if config["centralized_observer"]:
        ppo_config["multiagent"]["observation_fn"] = central_critic_observer

    ppo_config["model"]["custom_model_config"].update(
        {
            "max_num_agents": temp_env.max_num_agents,
            "max_num_evaders": temp_env.max_num_evaders,
            "max_num_obstacles": temp_env.max_num_obstacles,
            "num_obstacle_states": temp_env.obs_num_states,
            "num_agent_states": temp_env.agent_num_states,
            "num_evader_other_agent_states": temp_env.evader_other_agent_num_states,
            "num_agents": temp_env.num_agents,
            "num_obstacles": temp_env.num_obstacles,
            "num_evaders": temp_env.num_evaders,
        }
    )

    return ppo_config


def central_critic_observer(agent_obs, **kwargs):
    """Rewrites the agent obs to include opponent data for training."""

    num_agents = len(agent_obs)
    new_obs = {
        i: {
            "own_obs": agent_obs[i]["observations"],
            "all_obs": {
                agent_id: agent_obs[agent_id]["observations"]
                for agent_id in range(num_agents)
            },
            "all_actions": {
                agent_id: np.array([0, 0]) for agent_id in range(num_agents)
            },  # filled in by FillInActions
        }
        for i in range(len(agent_obs))
    }

    return new_obs
