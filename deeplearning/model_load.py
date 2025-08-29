import torch
import torchvision.models.vgg

# 方案1：添加所有必需的安全全局变量
torch.serialization.add_safe_globals([
    torchvision.models.vgg.VGG,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.pooling.AdaptiveAvgPool2d,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.dropout.Dropout,
    torch.nn.modules.flatten.Flatten
])
model = torch.load('vgg16_method1.pth')

print(model)

vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16)

