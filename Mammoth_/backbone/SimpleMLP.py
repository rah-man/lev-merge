import torch
import torch.nn as nn

from backbone import MammothBackbone, num_flat_features, xavier

class SimpleMLP(MammothBackbone):
    """
    Network composed of one hidden layers for classification with 1000 units.
    Designed for generic use.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size=1000) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.in_features = input_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)

        self._features = nn.Identity()
        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))

        feats = self._features(x)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)

        raise NotImplementedError("Unknown return type")

if __name__ == "__main__":
    network = SingleMLP(768, 100)
    print(network)