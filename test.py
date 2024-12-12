import torch
from PIL import Image
from torchvision import datasets, transforms, utils as tvutils
from train import calc_psnr
import numpy as np
import os


device = torch.device('mps')

model = torch.load("./pixel_loss_2.pth", map_location = device)


transform = transforms.Compose([
    transforms.Resize(288),           
    transforms.CenterCrop(288),      
    transforms.ToTensor()
])

blur = transforms.Compose([    
    transforms.Resize(72),       
    transforms.GaussianBlur(1),
    transforms.Resize(72)
])

total_psnr = 0

data = Image.open("mai1.jpeg")
data = transform(data)

img = np.array(data).transpose([1, 2, 0])
img = (img*255.0).clip(0, 255).astype("uint8")
img = Image.fromarray(img)
img.save("test_image.jpg")

for path in os.listdir("test"):
    data = Image.open("./test/" + path)
    data = transform(data)

    img = np.array(data).transpose([1, 2, 0])
    img = (img*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save("./test1/" + path)

    data = data.unsqueeze(0)
    blurred_data = blur(data)

    img = blurred_data[0]
    img = np.array(img).transpose([1, 2, 0])
    img = (img*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save("./blurred_test/" + path)

    blurred_data = blurred_data.to(device)
    data = data.to(device)

    with torch.no_grad():
        preds = model(blurred_data)
    # print(blurred_data.shape)
    # print(preds.shape)

    psnr = calc_psnr(preds, data)
    total_psnr += psnr
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.to(torch.device('cpu'))
    preds = preds.squeeze(0)
    output = np.array(preds).transpose([1, 2, 0])
    output = (output*255.0).clip(0, 255).astype("uint8")
    output = Image.fromarray(output)
    output.save("./test_output/" + path)

print(total_psnr / 50)
