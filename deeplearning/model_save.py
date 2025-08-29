import torchvision
import torch

# 更新：使用weights参数替代pretrained参数
vgg16 = torchvision.models.vgg16(weights=None)  # 不加载预训练权重


torch.save(vgg16.state_dict(),'vgg16_method2.pth')