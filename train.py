import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from fatunetmodel import UNet
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logs/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using: {DEVICE}")

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = True

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch=0):
    loop = tqdm(loader, desc="Epoch"+str(epoch+1))

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        writer.add_scalar("train loss", loss.item())
        writer.flush()


def main():
    #for pets
    # train_transform = A.Compose(
    #     [A.Resize(160, 160),
    #      A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    #      A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    #      A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    #      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #      ToTensorV2(),
    #      ]
    # )
    #
    # val_transforms = A.Compose(
    #     [A.Resize(160, 160),
    #      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #      ToTensorV2()
    #      ]
    # )

    #for cells
    train_transform = A.Compose(
        [A.Resize(160, 160),
         # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
         # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
         A.Normalize(mean=(0.6379), std=(0.0855)),
         ToTensorV2(),
         ]
    )
    val_transforms = train_transform

    test_transforms = val_transforms
    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader, test_loader = get_loaders(
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
        "hela"
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        acc, dice_score, iou = check_accuracy(val_loader, model, device=DEVICE)
        writer.add_scalar("val_accuracy", acc)
        writer.add_scalar("dice_score", dice_score)
        writer.add_scalar("IoU", iou)
        writer.flush()


    print("EVALUATION:")
    check_accuracy(test_loader, model, device=DEVICE)
    save_predictions_as_imgs(test_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()
