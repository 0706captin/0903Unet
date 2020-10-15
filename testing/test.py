import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd

import torch
import numpy as np


#%%

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%%

img_integeration = Image.open("/home/huangjq/PyCharmCode/4_project/1_UNet/B4_attUnetv4/results/images/0822v1/k0/input/133-LN1.JPG")
mask_integeration = Image.open("/home/huangjq/PyCharmCode/4_project/1_UNet/B4_attUnetv4/testing/draw.JPG")
mask_integeration = mask_integeration.convert('1')

print("img_integeration.size: ", img_integeration.size, ",  mask_integeration.size: ",  mask_integeration.size)
img_integeration_numpy = np.array(img_integeration)
mask_integeration_numpy = np.array(mask_integeration)
print("img_integeration_numpy.shape: ", img_integeration_numpy.shape,", mask_integeration_numpy.shape: ",  mask_integeration_numpy.shape)
print("img.mode: ", img_integeration.mode,", mask.mode: ",  mask_integeration.mode)
print(mask_integeration_numpy)

# plt.imshow(img_integeration)
plt.pause(0.1)
# plt.imshow(mask_integeration)
plt.pause(0.1)
# plt.imshow(img_integeration_numpy)
plt.pause(0.1)
# plt.imshow(mask_integeration_numpy)
plt.pause(0.1)


mask = np.expand_dims(mask_integeration_numpy, axis=2)
print("原来灰度图shape:", mask_integeration_numpy.shape, ", 扩展通道后shape: ", mask.shape)
print("灰度图最大值: ", np.max(mask))
merge1 = img_integeration_numpy * (mask > 0)
print("合成图shape: ", merge1.shape)
print("检查3个通道的是否都泛洪merge到了")
print(merge1[:, :, 0])
print(merge1[:, :, 1])
print(merge1[:, :, 2])

plt.imshow(merge1)
plt.pause(0.1)



plt.show()

#%%




