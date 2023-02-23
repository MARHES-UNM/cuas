import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import numpy as np
import gym


class TorchFixCentModel(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, gym.spaces.Dict)
            and "own_obs" in orig_space.spaces
            and "all_obs" in orig_space.spaces
            # and "all_actions" in orig_space.spaces
        )

        hidden_layer_size = 256
        self.actor_network = nn.Sequential(
            nn.Linear(orig_space["own_obs"].shape[0], hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Linear(hidden_layer_size, num_outputs),
        )
        self.value_network = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Linear(hidden_layer_size, 1),
        )

        self._input = None

    def forward(self, input_dict, state, seq_lens):

        self._input = input_dict["obs_flat"]
        action_out = self.actor_network(input_dict["obs"]["own_obs"])

        return action_out, state

    def value_function(self):
        assert self._input is not None, "must call forward first"
        value_out = self.value_network(self._input)
        return torch.reshape(value_out, [-1])
