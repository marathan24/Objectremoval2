import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import cv2

lab1 = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
file1=st.file_uploader("Choose a file", type =['jpg','jpeg','jfif','png'])
img=Image.open(file1)
st.image(img)
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
inp = transform(img).unsqueeze(0)
new_img = trans(img)
T1 = transforms.ToTensor()
im = T1(new_img)
#T = transforms.ToPILImage()
#im_PIL = T(im)
im_numpy = im.numpy()
out = lab1(inp)['out']
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

from torchvision.transforms.transforms import ToPILImage
rgb = decode_segmap(om)

T2 = transforms.Compose([ ToPILImage() ,transforms.Resize(256), transforms.CenterCrop(224)])
T = transforms.ToPILImage()
seg = T2(rgb)
ori_img = cv2.imread("file1")
seg_img = cv2.imread("rgb")
#plt.figure(figsize=[12,12])
#plt.subplot(121);plt.imshow(ori_img,cmap='gray')
#plt.subplot(122);plt.imshow(rgb,cmap='gray')
desired_width = 224
desired_height = 224
dim = (desired_width, desired_height)

# Resize background image to sae size as logo image
resized_ori_img = cv2.resize(ori_img, dsize=dim, interpolation=cv2.INTER_AREA)
resized_rgb = cv2.resize(rgb, dsize=dim, interpolation=cv2.INTER_AREA)

result = cv2.bitwise_and(resized_ori_img, resized_rgb , mask = None)
#st.image(result)
placeholders=st.beta_columns(2)
placeholders[0].image(img)
placeholders[1].image(result)
