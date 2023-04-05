import matplotlib.pyplot as plt
import torch
import torchvision
from OxfordPetDataset import OxfordIIITPet
from torch.utils.data import DataLoader
from HelaDataset import HelaDataset
from tqdm import tqdm
def save_checkpoint(state,epoch=0, filename="my_checkpoint.pth.tar"):
    #print("=> Saving checkpoint")
    torch.save(state, str(epoch)+filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    batch_size,
    train_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
    dataset="pets"
):
    if dataset == "pets":
        train_ds = OxfordIIITPet(root="./datasetOxford/", split="trainval", target_types=["segmentation"],
                                 transforms=train_transform, download=True)
        test_ds = OxfordIIITPet(root="./datasetOxford/", split="test", target_types="segmentation", transforms=test_transform, download=True)
        train_ds, val_ds = torch.utils.data.random_split(train_ds,
                                                         lengths=[int(len(train_ds) * 0.9), int(len(train_ds) * 0.1)])
    elif dataset == "hela":
        train_ds = HelaDataset(root="./", transforms=train_transform, split="train")
        test_ds = HelaDataset(root="./", transforms=test_transform, split="test")
        # train_ds, val_ds = torch.utils.data.random_split(train_ds,
        #                                                  lengths=[int(len(train_ds) * 0.9), int(len(train_ds) * 0.1)],
        #                                                  generator=torch.Generator().manual_seed(41))
        val_ds = HelaDataset(root="./", transforms=test_transform, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    IoU = 0
    iou = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            IoU += ((preds * y).sum())/((preds+y-preds * y).sum()+ 1e-8)

            intersection = (preds * y).sum()+ 1e-8
            union = torch.logical_or(preds, y).sum()+ 1e-8
            iou += intersection / union


    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice: {dice_score/len(loader)}")
    print(f"IoU:{IoU/len(loader)}")
    print(f"new IoU:{iou / len(loader)}")
    model.train()
    return num_correct/num_pixels*100, dice_score/len(loader), IoU/len(loader)



def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")


    model.train()
