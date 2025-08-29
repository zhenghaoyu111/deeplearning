import torch
from torch import nn


class green(nn.Module):
    def __init__(self):
        super(green, self).__init__()
        
    def forward(self, input):
        output = input + 1
        return output

sample1 = green()
x = torch.tensor(1.0)
output = sample1(x)
print(output)