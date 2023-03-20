import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from OxfordPetDataset import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
transforms = A.Compose(
    [A.Resize(160, 160),
     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ToTensorV2()
     ])

dataset = OxfordIIITPet(root="./datasetOxford/", split="test", target_types="segmentation", transforms=transforms, download=True)
dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        shuffle=True)

for image,mask in dataloader:
    for batch_image in image:
        print(image.size())
        plt.imshow(batch_image.permute(1,2,0))
        plt.show()
        masked = draw_bounding_boxes(batch_image,mask[0][0]>0)
        plt.imshow(masked)
        plt.show()
        break
    break