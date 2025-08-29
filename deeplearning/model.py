# 创建网络模型
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = Model()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)