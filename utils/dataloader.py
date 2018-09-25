import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import numpy as np
import PIL
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import torchvision


class MarcoDataset(Dataset):
    
    def __init__(self, image_dir):
        self.info = pd.read_csv(os.path.join(image_dir,"info.csv"))
        #self.image_dir = image_dir
    
    def __getitem__(self,index):
        #self.info_row = self.info.iloc[index]
        self.image_path = self.info['image_path'][index]
        self.image = Image.open(self.image_path)
        self.image = np.array(self.image.resize((599,599)))
        #self.shape = self.image.shape
        #self.image = self.image.reshape(self.shape[2],self.shape[0],self.shape[1])
        self.label = self.info['label_id'][index]
        return self.image, self.label
    
    def __len__(self):
        return len(self.info)
