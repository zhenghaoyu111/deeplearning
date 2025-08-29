import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 加载数据集
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)  # 取第一张图片
    step += 1
    
writer.close()


