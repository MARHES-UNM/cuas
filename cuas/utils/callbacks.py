from typing import Dict, Tuple

import numpy as np
from gym.spaces import Box
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class TrainCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))

        episode.user_data["target_collisions"] = []
        # episode.hist_data["target_collisions"] = []

        episode.user_data["agent_collisions"] = []
        # episode.hist_data["agent_collisions"] = []

        episode.user_data["obstacle_collisions"] = []
        # episode.hist_data["obstacle_collisions"] = []

        episode.user_data["evader_captured"] = []
        # episode.hist_data["evader_captured"] = []

        episode.user_data["target_breached"] = []
        # episode.hist_data["target_breached"] = []
        # episode.user_data["pole_angles"] = []
        # episode.hist_data["pole_angles"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        agent_ids = episode.get_agents()
        cum_target_collisions = 0
        cum_agent_collisions = 0
        cum_obstacle_collisions = 0
        evader_captured = []
        target_breached = []

        for agent_id in agent_ids:
            last_info = episode.last_info_for(agent_id)
            cum_target_collisions += last_info["target_collision"]
            cum_agent_collisions += last_info["agent_collision"]
            cum_obstacle_collisions += last_info["obstacle_collision"]
            evader_captured.append(last_info["evader_captured"])
            target_breached.append(last_info["target_breached"])

        evader_captured = max(evader_captured)
        target_breached = max(target_breached)

        # pole_angle = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        # episode.user_data["pole_angles"].append(pole_angle)
        episode.user_data["target_collisions"].append(cum_target_collisions)
        episode.user_data["agent_collisions"].append(cum_agent_collisions)
        episode.user_data["obstacle_collisions"].append(cum_obstacle_collisions)
        episode.user_data["evader_captured"].append(evader_captured)
        episode.user_data["target_breached"].append(target_breached)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print(
        #     "episode {} (env-idx={}) ended with length {} and pole "
        #     "angles {}".format(
        #         episode.episode_id, env_index, episode.length, pole_angle
        #     )
        # )
        # episode.custom_metrics["pole_angle"] = pole_angle
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
        target_collisions = np.sum(episode.user_data["target_collisions"])
        episode.custom_metrics["target_collisions"] = target_collisions
        # episode.hist_data["target_collisions"] = episode.user_data["target_collisions"]

        agent_collisions = np.sum(episode.user_data["agent_collisions"])
        episode.custom_metrics["agent_collisions"] = agent_collisions
        # episode.hist_data["agent_collisions"] = episode.user_data["agent_collisions"]

        obstacle_collisions = np.sum(episode.user_data["obstacle_collisions"])
        episode.custom_metrics["obstacle_collisions"] = obstacle_collisions
        # episode.hist_data["obstacle_collisions"] = episode.user_data[
        # "obstacle_collisions"
        # ]

        evader_captured = np.max(episode.user_data["evader_captured"])
        episode.custom_metrics["evader_captured"] = evader_captured
        # episode.hist_data["evader_captured"] = episode.user_data["evader_captured"]

        target_breached = np.max(episode.user_data["target_breached"])
        episode.custom_metrics["target_breached"] = target_breached
        # episode.hist_data["target_breached"] = episode.user_data["target_breached"]

    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print(
    #         "trainer.train() result: {} -> {} episodes".format(
    #             trainer, result["episodes_this_iter"]
    #         )
    #     )
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ):
    #     pass
    #     # result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     # print(
    #     #     "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #     #         policy, result["sum_actions_in_train_batch"]
    #     #     )
    #     # )

    # def on_postprocess_trajectory(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     episode: Episode,
    #     agent_id: str,
    #     policy_id: str,
    #     policies: Dict[str, Policy],
    #     postprocessed_batch: SampleBatch,
    #     original_batches: Dict[str, Tuple[Policy, SampleBatch]],
    #     **kwargs
    # ):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches.
    Keep track of the single agent action space. It is declared here but should be changed if declared elsewhere in the env.
    """

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        action_space_shape = 2
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(low=-1, high=1, dtype=np.float32, shape=(action_space_shape,))
        )

        # get all the actions, clip the actions just in case
        all_actions = []
        num_agents = len(original_batches)
        for agent_id in range(num_agents):
            _, single_agent_batch = original_batches[agent_id]
            single_agent_action = np.array(
                [
                    action_encoder.transform(np.clip(a, -1, 1))
                    # action_encoder.transform(a)
                    for a in single_agent_batch[SampleBatch.ACTIONS]
                ]
            )

            all_actions.append(single_agent_action)


        all_actions = np.array(all_actions)
        num_agent_actions = num_agents * action_space_shape
        all_actions = all_actions.reshape(-1, num_agent_actions)
        # print(f"action shape: {all_actions.shape}")
        # print(f"to_update shape: {to_update.shape}")
        to_update[:, -num_agent_actions:] = all_actions
