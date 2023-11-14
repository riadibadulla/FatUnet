import copy

import scipy.ndimage
from PIL import Image
import numpy as np
# from fatunetmodel import UNet
from fatunetmodel import UNet_v3
import torch
from utils import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter

transform = A.Compose(
    [A.Resize(160, 160),
     A.Normalize(mean=(0.6379), std=(0.0855)),
     ToTensorV2(),
     ]
)

image = Image.open("Hela_4c_160/Test_large_image/HeLa_8000_0330.tiff").convert('L')
# image = np.pad(image, 64, mode='constant')
mask = Image.open("Hela_4c_160/Test_large_image/mask330.png")

model = UNet_v3().to(DEVICE)
model.eval()
# load_checkpoint(torch.load("8epochs.pth.tar"), model)
load_checkpoint(torch.load("helabest.tar"), model)
patch_size = 160
predicted_final_mask = np.zeros_like(image)

# image = np.array(image)

image = scipy.ndimage.convolve(image,np.array([[0.0625, 0.125, 0.0625],
                                              [0.125, 0.25, 0.125],
                                              [0.0625, 0.125, 0.0625]]),
                               mode="nearest")
# Calculate the number of patches in each dimension
num_patches_x = image.shape[1] // patch_size
num_patches_y = image.shape[0] // patch_size

# Set the model to evaluation mode
model.eval()
plt.imshow(image, cmap='Greys')
plt.show()
# Iterate over all patches
with torch.no_grad():
    for i in tqdm(range(num_patches_y)):
        for j in range(num_patches_x):
            y1 = i * patch_size
            y2 = y1 + patch_size
            x1 = j * patch_size
            x2 = x1 + patch_size
            if (x1>=5661 and x1<=7032) and (y1>6824):
                continue
            patch = image[y1:y2, x1:x2]
            patch = transform(image=patch)['image']
            with torch.no_grad():
                predicted_mask  = torch.sigmoid(model(patch.unsqueeze(0).cuda()))
                predicted_mask  = (predicted_mask > 0.5).float().cpu().numpy()
            #Assign the patch mask to the corresponding region in the segmented mask
            predicted_final_mask[y1:y2, x1:x2] = copy.copy(predicted_mask)
# Convert the segmented mask tensor to a NumPy array
print(mask.size)
print(predicted_final_mask.size)
quit()
intersection = (predicted_final_mask * mask).sum().item()
union = (predicted_final_mask+mask-predicted_final_mask * mask).sum().item()
iou = intersection/(union+1e-8)
print(iou)