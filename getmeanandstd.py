import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from fatunetmodel import fat_UNet_non_refined
from HelaDataset import HelaDataset
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logs/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using: {DEVICE}")

LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 300
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = False

def main():
    train_transform = A.Compose(
        [A.Resize(160, 160),
         A.augmentations.transforms.ToFloat(),
         ToTensorV2(),
         ]
    )

    dataset = HelaDataset(root="./", transforms=train_transform, split="all", random_seed=2023)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False
    )


    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)


    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(mean)
    print(std)

if __name__ == "__main__":
    main()
