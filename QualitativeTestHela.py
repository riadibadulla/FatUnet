import copy

import scipy.ndimage
from PIL import Image
import numpy as np
from fatunetmodel import UNet
import torch
from utils import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter

transform = A.Compose(
    [A.Resize(160, 160),
     A.Normalize(mean=(0.6379), std=(0.0855)),
     ToTensorV2(),
     ]
)

# image = Image.open("Hela_4c_160/Test_large_image/ROI_1416-1932-171.tiff")
image = Image.open("Hela_4c_160/Test_large_image/ROI_1656-6756-329.tiff")
image.seek(119)
image = image.convert('L')
# image = np.array(image.convert('L'))
# image = np.pad(image, 40, mode='constant')
# image = np.array(Image.open("Hela_4c_160/Test_large_image/Prefix_3VBSED_roi_00_slice_0253.tif").convert('L'))
# image = np.pad(image, 10, mode='constant')

model = UNet().to(DEVICE)
model.eval()
load_checkpoint(torch.load("20epochs.pth.tar"), model)
patch_size = 160
mask = np.zeros_like(image)

image = np.array(image)

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
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            y1 = i * patch_size
            y2 = y1 + patch_size
            x1 = j * patch_size
            x2 = x1 + patch_size
            patch = image[y1:y2, x1:x2]
            patch = transform(image=patch)['image']
            with torch.no_grad():
                predicted_mask  = torch.sigmoid(model(patch.unsqueeze(0).cuda()))
                predicted_mask  = (predicted_mask > 0.5).float().cpu().numpy()
            #Assign the patch mask to the corresponding region in the segmented mask
            mask[y1:y2, x1:x2] = copy.copy(predicted_mask)
# Convert the segmented mask tensor to a NumPy array
plt.imshow(mask, cmap='Greys')
plt.show()
final_image = torchvision.utils.draw_segmentation_masks(torch.tensor(image).unsqueeze(0).repeat(3, 1, 1).cpu(),torch.tensor(mask.astype(bool)).cpu(),alpha=0.3,colors="orange")
plt.imshow(final_image.permute(1, 2, 0))
plt.show()
