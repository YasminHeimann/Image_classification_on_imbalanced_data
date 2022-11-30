# create dataset
from torch.utils.data import Dataset
import pickle
import numpy as np


class MyDataset(Dataset):
    """ex1 dataset."""

    def __init__(self, the_list, transform=None):
        self.the_list = the_list
        self.transform = transform

    def __len__(self):
        return len(self.the_list)

    def __getitem__(self, idx):
        item = self.the_list[idx]
        if self.transform:
            # fixed a bug in the supplied dataset class
            item = [self.transform(item[0]), item[1]]
        return item


def get_dataset_as_array(path='./data/dataset.pickle'):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# useful for using data-loaders
def get_dataset_as_torch_dataset(path='./data/dataset.pickle'):
    dataset_as_array = get_dataset_as_array(path)
    dataset = MyDataset(dataset_as_array)
    return dataset


# for visualizations
def un_normalize_image(img):
    img = img / 2 + 0.5
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def label_names():
    return {0: 'car', 1: 'truck', 2: 'cat'}

