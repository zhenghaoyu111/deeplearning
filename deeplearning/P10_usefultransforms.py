from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img = Image.open("/Users/zhenghaoyu/Downloads/data/train/ants_image/0013035.jpg")

writer = SummaryWriter("logs")

#ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image("Reszie",img_resize,0)
print(img_resize.size)

writer.close()

