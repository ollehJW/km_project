import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import json
from skimage import io
import os

transform_dict = {
    'train': transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class ConstructDataset(Dataset):
    """
    Construct pytorch Dataset from file list.

    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    phase : str
        train phase. (Default: 'train')

    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, img_list, json_list, phase = 'train'):
        self.img_list = img_list
        self.json_list = json_list
        self.phase = phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        width = np.array(image).shape[0]
        height = np.array(image).shape[1]
        image = transform_dict[self.phase](image)
        json_file = open(self.json_list[index], encoding = 'utf-8')
        json_file = json.load(json_file)
        LeftUp = json_file['region']['LeftUp']
        RightUp = json_file['region']['RightUp']
        LeftDown = json_file['region']['LeftDown']
        RightDown = json_file['region']['RightDown']
        LeftUp[0] = LeftUp[0] * 448 / height 
        LeftUp[1] = LeftUp[1] * 224 / width 
        RightUp[0] = RightUp[0] * 448 / height 
        RightUp[1] = RightUp[1] * 224 / width 
        LeftDown[0] = LeftDown[0] * 448 / height 
        LeftDown[1] = LeftDown[1] * 224 / width 
        RightDown[0] = RightDown[0] * 448 / height 
        RightDown[1] = RightDown[1] * 224 / width 
        target = LeftUp + RightUp + LeftDown + RightDown
        target = np.array(target)
        return {'image': image, 'target': target}


class DatasetGenerator(object):
    """
    Construct pytorch DataLoader from file list.

    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    batch_size : int
        batch size. (Default: 16)
    phase : str
        train phase. (Default: 'train')
    train_valid_split : bool
        whether to split data with train and validation. (Default: False)  
    valid_ratio : float
        validation ratio. (Default: 0.2) 
    stratify : list  
        Target to be used for stratified extraction (Default: None) 
    random_seed : int
        random seed number (Default: 1004)  

    Returns
    --------
    pytorch DataLoader
    """
    def __init__(self, img_list, json_list, batch_size = 16, phase = 'train'):
        self.img_list = img_list
        self.json_list = json_list
        self.batch_size = batch_size
        self.phase = phase

    def dataloader(self):

        if self.phase == 'train':
            train_dataset = ConstructDataset(self.img_list, self.json_list, phase = 'train')
            return dict({'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)})
        
        else:
            test_dataset = ConstructDataset(self.img_list, self.json_list, phase = self.phase)
            return dict({'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)})


class ConstructClassDataset(Dataset):
    """
    Construct pytorch Dataset from file list.
    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    phase : str
        train phase. (Default: 'train')
    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, file_list, target_list, phase = 'train'):
        self.file_list = file_list
        self.target_list = target_list
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = Image.open(self.file_list[index])
        target_class = self.target_list[index]
        image = transform_dict[self.phase](image)
        return {'image': image, 'target': target_class}


class ClassDatasetGenerator(object):
    """
    Construct pytorch DataLoader from file list.
    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    batch_size : int
        batch size. (Default: 16)
    phase : str
        train phase. (Default: 'train')
    Returns
    --------
    pytorch DataLoader
    """
    def __init__(self, file_list, target_list, batch_size = 16, phase = 'train'):
        self.file_list = file_list
        self.target_list = target_list
        self.batch_size = batch_size
        self.phase = phase

    def dataloader(self):

        if self.phase == 'train':
            train_dataset = ConstructClassDataset(self.file_list, self.target_list, phase = 'train')
            return dict({'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)})
        
        else:
            test_dataset = ConstructClassDataset(self.file_list, self.target_list, phase = self.phase)
            return dict({'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)})


class ConstructInferencePointDataset(Dataset):
    """
    Construct pytorch Dataset from file list.
    Parameters
    ----------
    file_list : list
        image file list
    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        original_image = Image.open(self.file_list[index])
        width = np.array(original_image).shape[0]
        height = np.array(original_image).shape[1]
        image = transform_dict['test'](original_image)
        return {'image': image, 'width': width, 'height': height, 'original_image': np.array(original_image), 'fname': os.path.basename(self.file_list[index])}


class PointInferenceDatasetGenerator(object):
    """
    Construct pytorch DataLoader from file list.
    Parameters
    ----------
    file_list : list
        image file list
    Returns
    --------
    pytorch DataLoader
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def dataloader(self):

        test_dataset = ConstructInferencePointDataset(self.file_list)
        return dict({'test': DataLoader(test_dataset, batch_size=1, shuffle=False)})

class ConstructInferenceClassDataset(Dataset):
    """
    Construct pytorch Dataset from file list.
    Parameters
    ----------
    file_list : list
        image file list
    phase : str
        train phase. (Default: 'train')
    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = Image.open(self.file_list[index])
        image = transform_dict['test'](image)
        return {'image': image, 'fname': self.file_list[index]}


class ClassInferenceDatasetGenerator(object):
    """
    Construct pytorch DataLoader from file list.
    Parameters
    ----------
    file_list : list
        image file list
    Returns
    --------
    pytorch DataLoader
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def dataloader(self):

        test_dataset = ConstructInferenceClassDataset(self.file_list)
        return dict({'test': DataLoader(test_dataset, batch_size=1, shuffle=False)})