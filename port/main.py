import torch

from port.utils import postprocess, preprocess


class SourceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.conv2 = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x, y, a=1, b=0.0):
        x = preprocess(x)
        y = preprocess(y)

        out1 = self.conv1(x + y)

        out1 = a * out1 + b

        out2 = self.conv2(x - y)
        out1 = out1 * out2

        out1 = postprocess(out1)

        return out1, out2
