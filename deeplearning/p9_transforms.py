from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

img_path = "/Users/zhenghaoyu/Downloads/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# 如何使用transforms进行图片的变换

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("tensor_img",tensor_img)
writer.close()