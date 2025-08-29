from torch.utils.tensorboard.writer import SummaryWriter
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset",transform=dataset_transform,train = True,download = True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",transform=dataset_transform,train = False,download = True)

#print(test_set[0])
#print(test_set.classes)

#img,target = test_set[0]
#print(img)
#print(target)
#print(test_set.classes[target])

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)