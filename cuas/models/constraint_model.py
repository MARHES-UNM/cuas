from torch.nn import Module
from torch import nn


class ConstraintModel(Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super(ConstraintModel, self).__init__()

        self.layer1 = nn.Linear(in_dim, 10)
        self.layer2 = nn.Linear(10, out_dim)
        self._layers = [self.layer1, self.layer2]

        self.init_weights()

    def init_weights(self):
        for layer in self._layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        x = nn.ReLU()(self.layer1(input))
        out = self.layer2(x)

        return out
