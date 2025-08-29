from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
# 这是一个测试修改
img_path = "/Users/zhenghaoyu/Downloads/data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image("test",img_array,2,dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()