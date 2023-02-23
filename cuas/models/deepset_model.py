import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from cuas.models.base_model import BaseModel


class DeepsetModel(BaseModel):
    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        BaseModel.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )
        self.pooling_type = model_config["custom_model_config"]["pooling_type"]

        if self.pooling_type == "sum":
            self.pooling_func = torch.sum
        elif self.pooling_type == "mean":
            self.pooling_func = torch.mean
        elif self.pooling_type == "max":
            self.pooling_func = torch.amax

        # size of tensor [batch, input, output]
        hidden_layer_size = 256
        self.phi_evader = nn.Sequential(
            nn.Linear(self.num_evader_other_agent_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_evader = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )

        self.phi_agents = nn.Sequential(
            nn.Linear(self.num_evader_other_agent_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_agents = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )

        self.phi_obs = nn.Sequential(
            nn.Linear(self.num_obstacle_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_obs = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )

        # concatenate the agent, evader, other_agents and obstacles
        self.last_state = nn.Sequential(
            nn.Linear(
                self.num_agent_states
                + hidden_layer_size
                + hidden_layer_size
                + hidden_layer_size,
                hidden_layer_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.policy_fn = nn.Linear(hidden_layer_size, num_outputs)
        self.value_fn = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_dict, state, seq_lens):

        index = 0
        end_index = self.num_agent_states
        main_agent = input_dict["obs"]["observations"][:, index:end_index]

        # get evader states
        index = end_index
        end_index += self.num_evader_other_agent_states * self.max_num_evaders
        evader = input_dict["obs"]["observations"][:, index:end_index]

        # evader weights
        evader = torch.reshape(
            evader, (-1, self.max_num_evaders, self.num_evader_other_agent_states)
        )
        # only get active evaders in environment
        evader = evader[:, : self.num_evaders, :]

        # other agents states
        index = end_index
        end_index += self.num_evader_other_agent_states * (self.max_num_agents - 1)
        other_agents = input_dict["obs"]["observations"][:, index:end_index]
        other_agents = torch.reshape(
            other_agents,
            (-1, self.max_num_agents - 1, self.num_evader_other_agent_states),
        )

        # other agent weights
        # only get active agents in environment
        other_agents = other_agents[:, : self.num_agents - 1, :]

        # obstacle states
        index = end_index
        end_index += self.num_obstacle_states * self.max_num_obstacles
        obstacles = input_dict["obs"]["observations"][:, index:end_index]

        obstacles = torch.reshape(
            obstacles, (-1, self.max_num_obstacles, self.num_obstacle_states)
        )

        # obstacle weights
        # only get active obstacles in environment, just in case there's no obstacles, add a dummy obstacle
        self.num_obstacles = 1 if self.num_obstacles == 0 else self.num_obstacles
        obstacles = obstacles[:, : self.num_obstacles, :]

        # evaders deepset weights
        x_evaders = self.phi_evader(evader)
        x_evaders = self.pooling_func(x_evaders, dim=1)
        x_evaders = self.rho_evader(x_evaders)

        # other agent weights deepset weights
        x_agents = self.phi_agents(other_agents)
        x_agents = self.pooling_func(x_agents, dim=1)
        x_agents = self.rho_agents(x_agents)

        # obstacles deepset weights
        x_obs = self.phi_obs(obstacles)
        x_obs = self.pooling_func(x_obs, dim=1)
        x_obs = self.rho_obs(x_obs)

        x = torch.cat((main_agent, x_evaders, x_agents, x_obs), dim=1)
        x = self.last_state(x)

        # Save for value function
        self._value_out = self.value_fn(x)

        logits = self.policy_fn(x)
        logits = self.proj_safe_actions(input_dict, logits)
        return logits, state

    def value_function(self):
        return self._value_out.flatten()
