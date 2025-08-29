import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 测试Fashion MNIST数据集是否能正确加载
print("测试Fashion MNIST数据集加载...")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

try:
    # 尝试加载数据集
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=False, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=False, transform=transform
    )
    
    print(f"✅ 数据集加载成功！")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_iter = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 测试数据加载
    train_batch = next(iter(train_iter))
    images, labels = train_batch
    print(f"✅ 数据加载器工作正常！")
    print(f"训练批次形状: 图像 {images.shape}, 标签 {labels.shape}")
    
    # Fashion MNIST类别名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print(f"数据集类别: {class_names}")
    print(f"第一个样本的标签: {labels[0].item()} ({class_names[labels[0]]})")
    
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    print("请检查数据文件是否在正确位置：./data/FashionMNIST/raw/")
