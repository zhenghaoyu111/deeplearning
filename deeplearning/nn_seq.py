import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        return x

model = Model()
input = torch.ones((64,3,32,32))
output = model(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(model, input)
writer.close()