import argparse
import json
import logging
import pathlib
import pickle
import subprocess
from datetime import datetime
import numpy as np
import cv2

import ray.rllib.agents.ppo as ppo
import torch
from ray import tune
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
)
from ray.tune import ExperimentAnalysis
import ray

from cuas.envs.cuas_env_multi_agent_v1 import CuasEnvMultiAgentV1
from cuas.models.deepset_model import DeepsetModel
from cuas.models.fix_model import TorchFixModel
from cuas.models.lstm_model import LstmModel

# from cuas.models.fix_cent_model import TorchFixCentModel
from cuas.utils.config import Config
from cuas.utils.callbacks import TrainCallback, FillInActions
from cuas.utils.agent_utils import get_agent_config
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch


import os

# # TODO: automatically set number of GPUS based on environmet
# # https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/examples/rllib.py

# RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
# num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0
# num_gpus_per_worker = (
#     (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1)
# )


# from cuas.utils.train_callback import TrainCallback
from ray.rllib.utils.numpy import convert_to_numpy

formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PATH = pathlib.Path(__file__).parent.absolute().resolve()

# Register custom models
ModelCatalog.register_custom_model("TorchFixModel", TorchFixModel)
# ModelCatalog.register_custom_model("CentralizedFixModel", TorchFixCentModel)
ModelCatalog.register_custom_model("DeepsetModel", DeepsetModel)
ModelCatalog.register_custom_model("LstmModel", LstmModel)


# https://github.com/ray-project/ray/blob/ray-1.13.0/rllib/examples/custom_torch_policy.py
# def custom_ppo_loss(
#     self,
#     model: ModelV2,
#     dist_class: Type[ActionDistribution],
#     train_batch: SampleBatch,
# ) -> Union[TensorType, List[TensorType]]:
def custom_ppo_loss(self, model, dist_class, train_batch):
    """Constructs the loss for Proximal Policy Objective.
    https://github.com/ray-project/ray/blob/ray-1.13.0/rllib/examples/custom_torch_policy.py
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.
    Returns:
        The PPO loss tensor given the input batch.
    """

    # print("in my custom loss")
    logits, state = model(train_batch)
    # print(f"logits: {logits}")
    curr_action_dist = dist_class(logits, model)
    # print(f"action_dstribution: {curr_action_dist}")

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

    # safe_action = train_batch[SampleBatch.ACTIONS]
    batch_info = train_batch[SampleBatch.INFOS]
    batch_info_size = len(batch_info)
    if (
        self.config["sgd_minibatch_size"] == batch_info_size
        and self.config["model"]["custom_model_config"]["modify_action"]
    ):
        original_action = train_batch[SampleBatch.ACTIONS]
        device = original_action.get_device()
        safe_action = (
            torch.from_numpy(np.array([info["agent_action"] for info in batch_info]))
            .float()
            .to(device)
        )
        # print(safe_action)
        # print(f"train_batch: {train_batch[SampleBatch.ACTIONS]}")
    else:
        safe_action = train_batch[SampleBatch.ACTIONS]
    logp_ratio = torch.exp(
        curr_action_dist.logp(safe_action) - train_batch[SampleBatch.ACTION_LOGP]
    )

    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if self.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    # Compute a value function loss.
    if self.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    # # Compute safe_layer loss.
    # # if self.config["model"]["custom_model_config"]["use_safe_action"]:
    # safe_a_bar_out, safe_a_bar_target = model.safe_layer_function()
    # safe_a_bar_loss = torch.pow(safe_a_bar_out - safe_a_bar_target, 2.0)
    # safe_a_bar_loss = torch.sum(safe_a_bar_loss, dim=1)

    # mean_safe_a_bar_loss = reduce_mean_valid(safe_a_bar_loss)
    # # else:
    # # safe_a_bar_loss = 0.0
    # # mean_safe_a_bar_loss = 0.0

    total_loss = reduce_mean_valid(
        -surrogate_loss
        + self.config["vf_loss_coeff"] * vf_loss_clipped
        - self.entropy_coeff * curr_entropy
        # + safe_a_bar_loss
    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    if self.config["kl_coeff"] > 0.0:
        total_loss += self.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss
    # model.tower_stats["mean_safe_a_bar_loss"] = mean_safe_a_bar_loss

    return total_loss


class MyCustomPPOLoss(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def loss(
        self,
        model,
        dist_class,
        train_batch,
    ):
        return custom_ppo_loss(self, model, dist_class, train_batch)

    def extra_grad_info(self, train_batch):
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
                # "safe_a_bar_loss": torch.mean(
                #     torch.stack(self.get_tower_stats("mean_safe_a_bar_loss"))
                # ),
            }
        )


# Create new Trainer using above defined policy
class MyTrainer(ppo.PPOTrainer):
    def get_default_policy_class(self, config):
        return MyCustomPPOLoss


def experiment(config):
    """
    https://github.com/ray-project/ray/blob/ray-1.13.0/rllib/examples/custom_experiment.py

    Args:
        config (_type_): _description_
    """

    num_iterations = int(config["num_iterations"])
    max_num_episodes = int(config["max_num_episodes"])
    tune_run = config.get("tune_run", False)
    analysis = config.get("analysis", None)
    restore_checkpoint = config.get("restore_checkpoint", False)
    update_environment = config.get("update_environment", False)

    checkpoint = None
    if not analysis is None:
        trial = config["trial"]

        checkpoint = analysis.get_best_checkpoint(
            trial=trial, metric="episode_reward_mean", mode="max"
        )

        ppo_config = trial.config

    if restore_checkpoint:
        checkpoint = config["checkpoint"]
        checkpoint_path = pathlib.Path(config["checkpoint"])

        # get the config
        with open(checkpoint_path.parent.parent.absolute() / "params.pkl", "rb") as f:
            ppo_config = pickle.load(f)

    else:
        ppo_config = get_agent_config(config)

    ppo_config.update(
        {
            # this must be set to 0 to work correctly
            "num_workers": 0,
            "num_gpus": 0,
            "num_gpus_per_worker": 0,
            "num_envs_per_worker": 1,
        }
    )

    logger.debug("Creating agent")
    logger.debug(f"ppo config: {ppo_config}")
    logger.debug(f"config: {config}")
    # Allows updating the config for the environment for evaluation.
    if update_environment:
        ppo_config = get_agent_config(config, ppo_config)
        # ppo_config["env_config"] = config["env_config"]

    ppo_agent = ppo.PPOTrainer(config=ppo_config, env=ppo_config["env"])
    if checkpoint is not None:
        ppo_agent.restore(checkpoint)

    # take a look at the graph
    policy = ppo_agent.get_policy("pursuer")

    print(policy.model)
    # ppo_config["env_config"]["constraint_k"] = .01

    logger.debug(f"Using seed: {config['env_config']['seed']}")

    logger.debug("Creating evaluation environment")
    env = ppo_agent.workers.local_worker().env

    obs, dones = env.reset(), {i.id: False for i in env.agents}
    dones["__all__"] = False

    results = {
        "agent_collisions": 0,
        "episode_reward": 0,
        "episode_reward_per_agent": 0,
        "evader_captured": 0,
        "num_episodes": 1,
        "obstacle_collisions": 0,
        "target_breached": 0,
        "target_collisions": 0,
        "timesteps_total": 0,
    }

    logger.debug("running experiment")

    # create video
    if config["video"] and config["render"]:
        video_name = config["video_file_name"]
        logger.debug(f"video name: {video_name}")
        _fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            video_name, _fourcc, 30, (env.screen_width, env.screen_height)
        )

        img = env.render(mode="rgb_array")

    num_episodes = 0
    iteration = 0
    while iteration < num_iterations and num_episodes < max_num_episodes:
        # for _ in range(num_iterations):

        actions = {}
        for agent in env.agents:
            if config["centralized_observer"]:

                # agent_obs = np.hstack([obs[agent.id]["observations"],
                #                        np.array([obs[agent_id]["observations"] for agent_id in range(env.num_agents)]).flatten(), np.array([np.array([0, 0]) for agent_id in range(env.num_agents)]).flatten()])
                agent_obs = {
                    "own_obs": obs[agent.id]["observations"],
                    # not used for evaluation
                    "all_obs": {
                        agent_id: obs[agent_id]["observations"]
                        for agent_id in range(env.num_agents)
                    },
                    # this is not used for evaluation
                    "all_actions": {
                        agent_id: np.array([0, 0]) for agent_id in range(env.num_agents)
                    },
                }
            else:
                agent_obs = obs[agent.id]
            actions[agent.id] = ppo_agent.compute_single_action(
                observation=agent_obs, policy_id="pursuer"
            )

        obs, rewards, dones, infos = env.step(actions)
        results["episode_reward"] += sum([v for k, v in rewards.items()])
        results["episode_reward_per_agent"] += (
            sum([v for k, v in rewards.items()]) / env.num_agents
        )
        results["agent_collisions"] += sum(
            [v["agent_collision"] for k, v in infos.items()]
        )
        results["target_collisions"] += sum(
            [v["target_collision"] for k, v in infos.items()]
        )
        results["obstacle_collisions"] += sum(
            [v["obstacle_collision"] for k, v in infos.items()]
        )
        results["target_breached"] += max(
            [v["target_breached"] for k, v in infos.items()]
        )
        results["evader_captured"] += max(
            [v["evader_captured"] for k, v in infos.items()]
        )
        results["timesteps_total"] += 1

        if config["video"] and config["render"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_writer.write(img)

        if config["render"]:
            img = env.render(mode="rgb_array")

        if dones["__all__"]:
            if tune_run:
                tune.report(**results)
            obs, dones = env.reset(), {agent.id: False for agent in env.agents}
            dones["__all__"] = False
            results["episode_reward"] = 0
            results["episode_reward_per_agent"] = 0
            num_episodes += 1
            results["num_episodes"] = num_episodes
        iteration += 1


def parse_arguments():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_config",
        help="Load configuration for experiment.",
        default=f"{PATH}/configs/sim_config.cfg",
    )
    parser.add_argument("--algo", default="ppo", type=str, help="algorithm for testing")
    parser.add_argument("--name", type=str, help="experiment name", default="debug")
    parser.add_argument("--log_dir", type=str)

    subparsers = parser.add_subparsers(dest="command")
    train_sub = subparsers.add_parser("train")
    train_sub.add_argument("--duration", type=int, default=5 * 24 * 60 * 60)
    train_sub.add_argument("--gpu", type=float, default=0.50)
    train_sub.add_argument("--cpu", type=int, default=4)
    train_sub.add_argument("--num_timesteps", type=int, default=int(30e6))
    train_sub.add_argument("--restore", type=str, default=None)
    train_sub.add_argument("--resume", action="store_true", default=False)
    train_sub.add_argument("--num_envs_per_worker", type=int, default=10)
    train_sub.set_defaults(func=train)

    test_sub = subparsers.add_parser("test")
    checkpoint_or_experiment = test_sub.add_mutually_exclusive_group()
    checkpoint_or_experiment.add_argument("--checkpoint", type=str)
    checkpoint_or_experiment.add_argument("--experiment", type=str)
    test_sub.add_argument("--cpu", type=int, default=8)
    test_sub.add_argument("--tune_run", action="store_true", default=False)
    test_sub.add_argument("--video", action="store_true", default=False)
    test_sub.add_argument("--max_num_episodes", type=int, default=10000)
    test_sub.add_argument("--no_render", action="store_true", default=False)
    test_sub.set_defaults(func=test)

    args = parser.parse_args()

    return args


def train(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    # trainer = ppo.PPOTrainer
    trainer = MyTrainer

    # All original config items must be updated here before creating the trainer config.
    args.config["env_config"]["observation_type"] = tune.grid_search(["local"])
    args.config["env_config"]["beta"] = 0.0900
    args.config["env_config"]["use_safe_action"] = tune.grid_search([False, True])


    ppo_config = get_agent_config(args.config)
    ppo_config.update(
        {
            # "num_gpus": args.gpu,
            "num_gpus": args.gpu,
            "num_workers": args.cpu,
            "num_envs_per_worker": args.num_envs_per_worker,
        }
    )

    ppo_config["model"]["custom_model"] = tune.grid_search(["DeepsetModel"])

    ray.init(num_gpus=1)
    metric_to_optimize = "custom_metrics/evader_captured_mean"
    results = tune.run(
        trainer,
        stop={
            "timesteps_total": args.num_timesteps,
            # "training_iteration": 100,
            "time_total_s": args.duration,
        },
        config=ppo_config,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=args.log_dir,
        name=args.name,
        restore=args.restore,
        resume=args.resume,
    )


def test(args):
    if args.video:
        args.config["video"] = True
        image_folder_path = f"results/images/{args.log_dir}/"
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)
        args.config["video_file_name"] = f"{image_folder_path}/cuas_video.mp4"
    else:
        args.config["video"] = False

    args.config["render"] = not args.no_render
    args.config["tune_run"] = args.tune_run
    args.config["max_num_episodes"] = args.max_num_episodes

    if args.experiment:
        args.config["analysis"] = ExperimentAnalysis(args.experiment)
        if not args.tune_run:
            args.config["trial"] = args.config["analysis"].trials[1]
        else:
            args.config["trial"] = tune.grid_search(
                [trial for trial in args.config["analysis"].trials]
            )
    elif args.checkpoint:
        args.config["checkpoint"] = args.checkpoint
        args.config["restore_checkpoint"] = True
    else:
        args.config["restore_checkpoint"] = False


    args.config["test_config"] = {
        "seed": tune.grid_search([5000, 0, 173, 93, 507, 1001])
        # "seed": 20
    }

    args.config["restore_checkpoint"] = True
    args.config["update_environment"] = True
    args.config["env_config"]["observation_type"] = "local"
    args.config["test_env_config"] = {
        "observation_radius": 20,
        # "observation_radius": tune.grid_search([10, 20, 30, 40]),
        "num_obstacles": tune.grid_search([2, 6, 10]),
        "num_pursuers": tune.grid_search([2, 4, 10]),
    }

    args.config["train_config"]["model"]["custom_model"] = "LstmModel"

    if args.tune_run:
        ray.init(num_cpus=args.cpu)
        results = tune.run(
            experiment,
            stop={
                "num_episodes": args.max_num_episodes,
            },
            config=args.config,
            # checkpoint_freq=10,
            checkpoint_at_end=True,
            local_dir=args.log_dir,
            name=args.name,
            resources_per_trial={"cpu": 1, "gpu": 0},
            resume="AUTO",
        )
    else:
        experiment(args.config)


def main():
    args = parse_arguments()

    if args.load_config:
        with open(args.load_config, "rt") as f:
            args.config = json.load(f)

    logger.debug(f"config: {args.config}")
    config = Config(args.config)

    env_config = config.env_config

    env_suffix = (
        f"{env_config.num_pursuers}v{env_config.num_evaders}o{env_config.num_obstacles}"
    )
    env_name = f"cuas_multi_agent-v1_{env_suffix}"

    branch_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )

    dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    if not args.log_dir:
        args.log_dir = (
            f"./results/{args.algo}/cuas_{env_suffix}_{dir_timestamp}_{branch_hash}"
        )

    # Must register by passing env_config if wanting to grid search over environment variables
    args.config["env_name"] = env_name
    tune.register_env(env_name, lambda env_config: CuasEnvMultiAgentV1(env_config))

    # https://stackoverflow.com/questions/27529610/call-function-based-on-argparse
    args.func(args)


if __name__ == "__main__":
    main()
