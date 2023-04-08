"""
In this file, we go through the test set of Hela dataset, and visualise the Accuracy, Dice and IoU per each slice.
"""

import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from fatunetmodel import fat_UNet_non_refined
from fatunetmodel import UNet_v3
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logs/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using: {DEVICE}")

LEARNING_RATE = 1e-4
#Make sure that batch is exaclty equal to the number of patches in the slice
# BATCH_SIZE = 529
BATCH_SIZE = 23
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = True
NUMBER_OF_PATCHES_IN_SLICE = 529

def get_metrics(model, device="cuda"):
    transform = A.Compose(
        [A.Resize(160, 160),
         A.Normalize(mean=(0.6379), std=(0.0855)),
         ToTensorV2(),
         ]
    )
    model.eval()
    dice_score_list = []
    IoU_list = []
    acc_list = []
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    IoU = 0
    acc = 0
    with torch.no_grad():
        slice_wise_batch = 1
        for slice in range(119,301,2):
            IoU = []
            num_correct = 0
            num_pixels = 0
            dice_score = 0
            loader = get_manual_loader_hela(BATCH_SIZE, transform, NUM_WORKERS, PIN_MEMORY,slice)
            for x, y in tqdm(loader):

                x = x.to(device)
                y = y.to(device)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()+ 1e-8) / (
                    (preds + y).sum() + 1e-8
                )
                IoU.append((((preds * y).sum())/((preds+y-preds * y).sum()+ 1e-8)).item())
                # intersection = (preds * y).sum()+ 1e-8
                # union = torch.logical_or(preds, y).sum()+ 1e-8
                # iou += intersection / union
            acc_list.append((num_correct/num_pixels*100).item())
            dice_score_list.append((dice_score/23).item())
            IoU_list.append(sum(IoU)/len(IoU))
            print(f"Slice:{slice}: iou:{sum(IoU)/len(IoU)}")

    return acc_list, dice_score_list, IoU_list

def visualise_the_metric(metric,name_of_metric,title):
    # x_axis = [x for x in range(1, 301, 2)]
    # val = [x for x in range(5,301,20)]
    # test = [x for x in range(1, 300, 10)]
    # x_axis = [x for x in x_axis if x not in test]
    # x_axis = [x for x in x_axis if x not in val]
    x_axis = [i for i in range(1,301,2)]
    y_axis = metric
    plt.title("epoch: " + title)
    plt.xlabel("Slice")
    plt.ylabel(name_of_metric)
    plt.ylim(0, 1)
    plt.plot(x_axis,y_axis,'bo-')
    filename = "results/v3/train_epoch_"+title+".png"
    plt.savefig(filename,dpi=500)
    plt.show()

def main():


    model = UNet_v3().to(DEVICE)

    #29,21,8
    #Load the weights you need
    for i in range(19,20):
        load_checkpoint(torch.load(str(i)+"my_checkpoint.pth.tar"), model)
        acc, dice, iou = get_metrics(model, device=DEVICE)
        visualise_the_metric(iou, "IoU", str(i+1))




if __name__ == "__main__":
    main()
