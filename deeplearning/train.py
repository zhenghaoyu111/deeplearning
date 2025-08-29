import torch
from torch.utils.data import DataLoader
from model import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


train_data = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


    
model = Model()

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("./logs_train") 
# 训练
for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 测试步骤开始
    with torch.no_grad():
        total_test_loss = 0
        total_accuracy = 0
        for data in test_dataloader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()
