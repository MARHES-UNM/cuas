import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import numpy as np
import gym
from qpsolvers import solve_qp


class BaseModel(TorchModelV2, nn.Module):
    """"""

    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )
        nn.Module.__init__(self)

        self.orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(self.orig_space, gym.spaces.Dict)
            and "observations" in self.orig_space.spaces
        )
        self.hidden_layer_size = model_config["custom_model_config"][
            "hidden_layer_size"
        ]

        self.num_agent_states = model_config["custom_model_config"]["num_agent_states"]
        self.num_obstacle_states = model_config["custom_model_config"][
            "num_obstacle_states"
        ]
        self.num_evader_other_agent_states = model_config["custom_model_config"][
            "num_evader_other_agent_states"
        ]

        # get number of entities in environment
        self.num_evaders = model_config["custom_model_config"]["num_evaders"]
        self.num_obstacles = model_config["custom_model_config"]["num_obstacles"]
        self.num_agents = model_config["custom_model_config"]["num_agents"]

        # max number of entities in environment
        self.max_num_obstacles = model_config["custom_model_config"][
            "max_num_obstacles"
        ]
        self.max_num_agents = model_config["custom_model_config"]["max_num_agents"]
        self.max_num_evaders = model_config["custom_model_config"]["max_num_evaders"]

        self.use_safe_action = model_config["custom_model_config"].get(
            "use_safe_action", False
        )
        self.model_batch_size = model_config["custom_model_config"].get(
            "model_batch_size", 4096
        )

        self.norm_low = np.array([-1.0, -1.0])
        self.norm_high = np.array([1.0, 1.0])
        self.low = np.array([0, -np.pi * 2])
        self.high = np.array([10, np.pi * 2])

    @torch.no_grad()
    def unscale_action(self, action):
        action = self.low + (self.high - self.low) * (
            (action - self.norm_low) / (self.norm_high - self.norm_low)
        )
        # action = np.clip(action, self.low, self.high)
        return action

    @torch.no_grad()
    def scale_action(self, action):
        action = (self.norm_high - self.norm_low) * (
            (action - self.low) / (self.high - self.low)
        ) + self.norm_low

        return action

    @torch.no_grad()
    def proj_safe_actions(self, input_dict, logits):
        if not (self.use_safe_action and logits.size(dim=0) != self.model_batch_size):
            return logits

        for i in range(logits.size(dim=0)):
            logits[i, :2] = self.proj_safe_action(
                logits[i, :2],
                input_dict["obs"]["action_g"][i],
                input_dict["obs"]["action_h"][i],
                input_dict["obs"]["action_r"][i],
            )

        return logits

    @torch.no_grad()
    def proj_safe_action(self, a_rl, G, h, R):

        a_rl = a_rl.cpu().numpy()
        G = G.cpu().numpy()
        h = h.cpu().numpy()
        R = R.cpu().numpy()

        a_rl = self.unscale_action(a_rl)
        a_si_rl = np.dot(R, a_rl.T)

        P = np.eye(2)
        q = -np.dot(P.T, a_si_rl)

        # try:
        a_si_qp = solve_qp(
            P.astype(np.float64),
            q.astype(np.float64),
            G.astype(np.float64),
            h.astype(np.float64),
            None,
            None,
            None,
            None,
            solver="quadprog",
        )
        # except Exception as e:
        #     print(f"error running solver: {e}")
        #     a_si_qp = a_si_rl

        # just return original action
        if a_si_qp is None:
            print("infeasible solver")
            a_si_qp = a_si_rl

        # convert to unicycle
        u_ni_qp = np.dot(np.linalg.pinv(R), a_si_qp.T)
        u_ni_qp = self.scale_action(u_ni_qp)

        return torch.from_numpy(u_ni_qp)

    def forward(self, input_dict, state, seq_lens):
        raise NotImplementedError

    def value_function(self):
        raise NotImplementedError
