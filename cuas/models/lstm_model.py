import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from cuas.models.base_model import BaseModel


# TODO: Fix LSTM model
# TODO: debug: https://stackoverflow.com/questions/51125933/lstm-layer-returns-nan-when-fed-by-its-own-output-in-pytorch
class LstmModel(BaseModel):
    """
    # https://cnvrg.io/pytorch-lstm/

    Args:
        BaseModel (_type_): _description_
    """

    def __init__(
        self, obs_space, act_space, num_outputs, model_config, *args, **kwargs
    ):
        BaseModel.__init__(
            self, obs_space, act_space, num_outputs, model_config, *args, **kwargs
        )

        self.evader_lstm = nn.LSTM(
            self.num_evader_other_agent_states,
            self.hidden_layer_size,
            num_layers=1,
            batch_first=True,
        )
        self.evader_relu = nn.ReLU()

        self.rho_evader = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size), nn.ReLU()
        )
        self.other_agents_lstm = nn.LSTM(
            self.num_evader_other_agent_states,
            self.hidden_layer_size,
            num_layers=1,
            batch_first=True,
        )
        self.other_agents_relu = nn.ReLU()

        self.rho_other_agents = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size), nn.ReLU()
        )
        self.obstacles_lstm = nn.LSTM(
            self.num_obstacle_states,
            self.hidden_layer_size,
            num_layers=1,
            batch_first=True,
        )

        self.obstacles_relu = nn.ReLU()

        self.rho_obstacles = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size), nn.ReLU()
        )
        # concatenate the agent, evader, other_agents and obstacles
        self.last_state = nn.Sequential(
            nn.Linear(self.num_agent_states + self.hidden_layer_size + 256 + 256, 256),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, 256),
        )
        self.policy_fn = nn.Linear(self.hidden_layer_size, num_outputs)
        self.value_fn = nn.Linear(self.hidden_layer_size, 1)

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

        # ideally we should create dummay variables for h_0 and c_0 and pass them in
        # accordingly but the default values for LSTMS are zeros so no need to for now
        # # initiate dummy variables
        # h_0_evader = Variable(torch.zeros(1, evader.size(dim=0), self.hidden_layer_size))
        # c_0_evader = Variable(torch.zeros(1, evader.size(dim=0), self.hidden_layer_size))
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        out_evader_lstm, (h_n_evader, c_n_evader) = self.evader_lstm(evader)
        # we use use hn instead of out because this is a single layer and hn = out. Not the case for multilayer lstms
        # https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        x_evaders = h_n_evader.view(-1, self.hidden_layer_size)
        x_evaders = self.evader_relu(x_evaders)
        x_evaders = self.rho_evader(x_evaders)

        out_other_agents_evader, (
            h_n_other_agents,
            c_n_other_agents,
        ) = self.other_agents_lstm(other_agents)
        x_agents = h_n_other_agents.view(-1, self.hidden_layer_size)
        x_agents = self.other_agents_relu(x_agents)
        x_agents = self.rho_other_agents(x_agents)

        out_obs, (h_n_obstacles, c_n_obstacles) = self.obstacles_lstm(obstacles)
        x_obs = h_n_obstacles.view(-1, self.hidden_layer_size)
        x_obs = self.obstacles_relu(x_obs)
        x_obs = self.rho_obstacles(x_obs)

        x = torch.cat((main_agent, x_evaders, x_agents, x_obs), dim=1)
        x = self.last_state(x)

        # Save for value function
        self._value_out = self.value_fn(x)

        logits = self.policy_fn(x)
        logits = self.proj_safe_actions(input_dict, logits)
        return logits, state

    def value_function(self):
        return self._value_out.flatten()
