# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import Resize
import json

# from tqdm import tqdm

DATASET_PATH = "drive_data"


class CustomCrop(object):
    def __init__(self, start_row, end_row):
        self.start_row = start_row
        self.end_row = end_row

    def __call__(self, img, *args):
        # Crop the image from start_row to end_row vertically
        return (F.crop(img, self.start_row, 0, self.end_row - self.start_row, img.width),) + args


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            coordinates, *rest = args
            flipped_coordinates = np.array([coordinates[0], -coordinates[1]])
            args = (flipped_coordinates,) + tuple(rest)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args
        # image = F.to_tensor(image) # , center, depth):  # *args):
        # center = torch.tensor(center, dtype=torch.float32)
        # depth = torch.tensor(depth, dtype=torch.float32)
        # return image, center, depth


class ResizeImage(object):
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.resize = Resize(self.target_shape)

    def __call__(self, image, *args):
        image = self.resize(image)
        return (image,) + args


# class ToHeatmap(object):
#    def __init__(self, radius=2):
#        self.radius = radius
#
#    def __call__(self, image, *dets):
#        peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)
#        return image, peak, size


def open_json(json_path):
    with open(json_path, "rb") as f:
        d = json.load(f)
    return d


def none_to_nan(jsonfloat, target_shape=1):
    if jsonfloat == None and target_shape == 1:
        return np.nan
    elif jsonfloat == None:
        return [np.nan] * target_shape
    else:
        return jsonfloat


def index_to_normalized(array_size, row_index, col_index):
    """
    Converts array indices to normalized coordinates.

    Parameters:
    - array_size: A tuple (height, width) representing the size of the numpy array.
    - row_index: The row index of the element in the array.
    - col_index: The column index of the element in the array.

    Returns:
    - A tuple (normalized_x, normalized_y) representing the normalized coordinates.
    """

    height, width = array_size
    normalized_x = (col_index / (width)) * 2 - 1
    normalized_y = (row_index / (height)) * 2 - 1
    return (normalized_y, normalized_x)


class SuperTuxDataset(Dataset):
    def __init__(
        self,
        dataset_path=DATASET_PATH,
        transform=ToTensor(),
        keys=["gt_puck_center", "gt_puck_depth", "puck_center", "puck_pixels"],
        include_nans=False,
        cut_parts=False,
    ):
        from PIL import Image
        from glob import glob
        from os import path

        self.keys = keys
        self.data = []
        for f in tqdm(glob(path.join(dataset_path, "*.json"))[:1000]):
            i = Image.open(f.replace("_info.json", "_image.png"))
            i.load()
            dati = open_json(f)

            puck_center = dati[self.keys[2]]
            if puck_center is not None:
                y_center = puck_center[0]
                start_y = 92
                end_y = 284
                padding = 4
                is_outside_cut = True if (start_y + padding) < y_center < (start_y - padding) else False
            puck_flag1 = True if dati[self.keys[0]] is not None else False
            puck_flag2 = True if dati[self.keys[2]] is not None else False
            puck_flag3 = dati[self.keys[3]] > 72
            if puck_flag1 == True or (puck_flag2 == True and puck_flag3 == True):
                puck_flag = True
            else:
                puck_flag = False

            if cut_parts and is_outside_cut:
                puck_flag = False

            if puck_flag == False and include_nans == False:
                continue

            puck_flag = np.array(puck_flag).astype(np.float32).reshape(1)
            center = none_to_nan(dati[self.keys[0]], target_shape=2)
            center = np.array(index_to_normalized((300, 400), center[0], center[1])).astype(np.float32)
            depth = np.array(none_to_nan(dati[self.keys[1]], target_shape=1)).reshape(1).astype(np.float32)

            self.data.append((i, center, depth, puck_flag))
        self.transform = transform
        print("Num images in training set: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=ToTensor(), num_workers=0, batch_size=128, include_nans=False):
    dataset = SuperTuxDataset(dataset_path, transform=transform, include_nans=include_nans)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == "__main__":

    import torch
    from torchvision import datasets, transforms
    from PIL import Image
    import os
    import numpy as np
    from tqdm import tqdm

    # Directory containing the PNG images
    image_directory = "/Users/andreas/Movies/train_set"

    # Transform to convert images to PyTorch tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    def calculate_mean_std(directory):
        tensor_list = []
        for filename in tqdm(os.listdir(directory)):
            if filename.endswith(".png"):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                tensor_list.append(tensor)

        # Stack all tensors to shape (N, C, H, W) where N is number of images
        all_tensors = torch.stack(tensor_list, dim=0)

        # Calculate the mean and std along the (N, H, W) dimensions
        mean = torch.mean(all_tensors, dim=[0, 2, 3])
        std = torch.std(all_tensors, dim=[0, 2, 3])

        return mean, std

    mean, std = calculate_mean_std(image_directory)
    print("Mean: ", mean)
    print("Std: ", std)
