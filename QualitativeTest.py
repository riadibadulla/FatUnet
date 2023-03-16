from PIL import Image
import numpy as np
from fatunetmodel import UNet
import torch
from utils import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [
     A.Normalize(mean=(0.6379), std=(0.0855)),
     ToTensorV2(),
     ]
)

# image = Image.open("Hela_4c_160/Test_large_image/ROI_1416-1932-171.tiff")
image = Image.open("Hela_4c_160/Test_large_image/ROI_1656-6756-329.tiff")
image.seek(119)
image = np.array(image.convert('L'))
image = np.pad(image, 40, mode='constant')
# image = np.array(Image.open("Hela_4c_160/Test_large_image/Prefix_3VBSED_roi_00_slice_0253.tif").convert('L'))
# image = np.pad(image, 64, mode='constant')

model = UNet().to(DEVICE)
load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
segmented_mask = np.zeros_like(image)
patch_size = 160
stride = 160

# Calculate the number of patches in each dimension
num_patches_x = (image.shape[0] - patch_size) // stride + 1
num_patches_y = (image.shape[1] - patch_size) // stride + 1

image_tensor = torch.from_numpy(image).unsqueeze(0)

segmented_mask_tensor = torch.from_numpy(segmented_mask).unsqueeze(0).float()

# Set the model to evaluation mode
model.eval()
plt.imshow(image, cmap='Greys')
plt.show()
# Iterate over all patches
with torch.no_grad():
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Extract the patch
            patch = image_tensor[..., i * stride:i * stride + patch_size, j * stride:j * stride + patch_size].squeeze(0)
            # Apply the segmentation model to the patch
            # patch_mask = model(patch).squeeze().cpu().numpy()
            patch = transform(image=patch.cpu().numpy())["image"].to(DEVICE).unsqueeze(0)
            patch_mask = torch.sigmoid(model(patch))
            patch_mask = (patch_mask > 0.05).float().squeeze().cpu().numpy()
            if np.amax(patch_mask)==1.0:
                print("hoora")
            #Assign the patch mask to the corresponding region in the segmented mask
            segmented_mask_tensor[..., i * stride:i * stride + patch_size,
            j * stride:j * stride + patch_size] = torch.from_numpy(patch_mask).unsqueeze(0)
            break
# Convert the segmented mask tensor to a NumPy array
segmented_mask = segmented_mask_tensor.squeeze().numpy()
plt.imshow(segmented_mask, cmap='Greys')
plt.show()
torchvision.utils.save_image(segmented_mask_tensor.unsqueeze(0), f"done.png")