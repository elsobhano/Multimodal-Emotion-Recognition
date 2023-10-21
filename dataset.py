import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torchvision
import torchvision.transforms as transforms
import numpy as np

from components.data_ingestion import create_dataset_directory, files_to_dataframe
from transformers import BertTokenizer

class MSCTDDataset(torch.utils.data.Dataset):
    """
    Uses dataframe to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(self, root = './data', dataset_name = 'train', img_transform=None, 
                 text_transform=None, max_length=15 ,random_state=42):
        
        """
        Args:
        
        text_add : .txt of chats.
        sentiment_add : .txt of emotions for each line.
        index_add : .txt of related images for each line.
        img_dir : path to iamges folder
        
        """
        self.data_pathes = create_dataset_directory(root, dataset_name)
        self.dataframe = files_to_dataframe(self.data_pathes[0], self.data_pathes[1], self.data_pathes[2])
        self.img_dir = self.data_pathes[3]
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.max_length = max_length
        
    def __len__(self):
        
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = self.dataframe.loc[idx, 'text']
        label = self.dataframe.loc[idx, 'label']
        label = np.array(label)
        img_path = f'{self.img_dir}/{idx}.jpg'
        img = torchvision.io.read_image(path=img_path)
        
        if self.img_transform:
            img = self.img_transform(img)
            
        if self.text_transform:
            tokenized_text = self.text_transform(text, padding='max_length',
                                    max_length=self.max_length, truncation=True, return_tensors="pt")
        
        sample = {
            'text': tokenized_text,
            'img': img,
            'label': label
        }
        return sample

class MSCTDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage):

        img_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
        text_transform = BertTokenizer.from_pretrained('bert-base-cased')

        self.train_dataset =  MSCTDDataset(root='./data1', dataset_name='train',
                            img_transform=img_transform, text_transform=text_transform)
        self.dev_dataset =  MSCTDDataset(root='./data1', dataset_name='dev',
                            img_transform=img_transform, text_transform=text_transform)
        self.test_dataset =  MSCTDDataset(root='./data1', dataset_name='test',
                            img_transform=img_transform, text_transform=text_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )