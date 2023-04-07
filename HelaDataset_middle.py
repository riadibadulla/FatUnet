import os
import os.path
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import glob
class HelaDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            split='train'
    ):
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder =  "Hela_4c_160/"
        self._images_folder = self._base_folder + "trainingImages_4c_160/"
        self._segs_folder = self._base_folder + "trainingLabels_4c_160/"
        self.split = split
        self._allimages = glob.glob(self._images_folder+"*")
        self._segs = glob.glob(self._segs_folder+"*")
        self._images = []
        self.test_slices = ["Slice_"+"{:03d}".format(i) for i in range(3,303,10)]
        self.val_slices = ["Slice_"+"{:03d}".format(i) for i in range(5,301,20)]
        self.train_slices = ["Slice_"+"{:03d}".format(i) for i in range(97,183,2) if i%100%10!=1 and i%100%10!=5]
        if split == "manual_test":
            for image in self._allimages:
                if "Slice_119" in image:
                    self._images.append(image)
        if split == "all_images":
            for image in self._allimages:
                self._images.append(image)
        if split == "test":
            for image in self._allimages:
                for slice_name in self.test_slices:
                    if slice_name in image:
                        self._images.append(image)
                        break
        if split == "val":
            for image in self._allimages:
                for slice_name in self.val_slices:
                    if slice_name in image:
                        self._images.append(image)
                        break
        if split == "train":
            for image in self._allimages:
                for slice_name in self.train_slices:
                    if slice_name in image:
                        self._images.append(image)
                        break

    def __len__(self) -> int:
        return len(self._images)

    def preprocess_mask(self, mask):
        #Taking only nucleus
        mask = mask.astype(np.float32)
        mask[(mask == 1.0) | (mask == 3.0) | (mask == 4.0) | (mask == 5.0)] = 0.0
        mask[mask == 2.0] = 1.0
        return mask

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        image_filename = os.path.join(self._images[idx])
        image = np.array(Image.open(image_filename).convert('L'))
        mask_file_name = self._segs_folder + os.path.basename(self._images[idx])
        mask_file_name = mask_file_name.replace("Data","Label")
        mask = self.preprocess_mask(np.array(Image.open(os.path.join(mask_file_name)).convert("L"), dtype=np.float32))
        if self.transforms:
            data = self.transforms(image=image, mask=mask)
            image = data["image"]
            mask = np.expand_dims(data["mask"], axis=0)
        return image, mask
