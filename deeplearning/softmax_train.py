import torch
from torch import nn
from d2l import torch as d2l
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 256

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # Fashion MNIST图像归一化
    transforms.Normalize((0.5,), (0.5,))
])

# 使用本地下载的Fashion MNIST数据集 (数据集现在在 ./FashionMNIST/ 目录)
train_dataset = torchvision.datasets.FashionMNIST(
    root='./', train=True, download=False, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./', train=False, download=False, transform=transform
)

# 创建数据加载器
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10

# 简单的训练循环
print("开始训练...")
for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    train_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_iter):
        trainer.zero_grad()
        output = net(data)
        l = loss(output, target)
        l.backward()
        trainer.step()
        
        train_loss += l.item()
        pred = output.argmax(dim=1)
        train_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {l.item():.4f}')
    
    # 计算训练精度
    train_acc = 100. * train_correct / total_samples
    avg_train_loss = train_loss / len(train_iter)
    
    # 测试精度
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_iter:
            output = net(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
    
    test_acc = 100. * test_correct / test_total
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

print("训练完成！")