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
        
        image2 = cv2.imread('../data/' + row.path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        if image2.shape[0] < image2.shape[1]:
            image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image1 = augmented['image']
            augmented2 = self.augmentations(image=image2)                          
            image2 = augmented2['image']  
        
        return image1,image2,torch.tensor(row.label)