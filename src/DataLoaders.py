import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils



class ImageDataset(Dataset):
    def __init__(self, root, transform1=None,transform2=None,max_samples=None):
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.files = sorted(os.listdir(root))
        if max_samples:
            self.files = self.files[:min(max_samples, len(self.files))]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        img = Image.open(os.path.join(self.root, self.files[idx]))
        img = img.convert('RGB')
        if self.transform1:
            img1 = self.transform1(img)
        if self.transform2:
            img2= self.transform2(img)
        return img2, img1
    