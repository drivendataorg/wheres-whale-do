from torch.utils.data import Dataset,DataLoader
import cv2
import torch
class TrainDataset(Dataset):
    
    def __init__(self, csv, transforms=None, transforms2 = None):
        self.csv = csv.reset_index()
        self.augmentations = transforms
        self.augmentations2 = transforms2

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread('../data/' + row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image1 = augmented['image']
        
        return image1,torch.tensor(row.label)