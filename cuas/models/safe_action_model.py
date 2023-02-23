import torch
from torch import nn


class SafeActionModel(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super(SafeActionModel, self).__init__()

        hidden_layer_size = 32
        self.layer1 = nn.Linear(in_dim, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, out_dim)
        self._layers = [self.layer1, self.layer2]

        self.init_weights()

    def init_weights(self):
        for layer in self._layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        x = nn.ReLU()(self.layer1(input))
        out = self.layer2(x)

        return out

    def load_model(self, model_dir, device="cpu"):
        model_state, _ = torch.load(model_dir, map_location=torch.device(device))
        self.load_state_dict(model_state)
