import os, torchvision
from torch.utils.data import Dataset
from matplotlib import use
import matplotlib.pyplot as plt
from torchvision.transforms import *
import torchvision.transforms.functional as TF
from random import random, randint



class FaceDataset(Dataset):

    def __init__(self, type='train'):

        super(FaceDataset, self).__init__()
        self.type = type
        self.path = self.getDatasetPath()
  
    def __len__(self):
        return len(os.listdir(f'{self.path}/ori'))  

    def __getitem__(self, index):

        filename = f'{str(index).zfill(5)}.png'
        ori_img = torchvision.io.read_image(f'{self.path}/ori/{filename}')
        dgd_img = torchvision.io.read_image(f'{self.path}/dgd/{filename}')

        ori_img, dgd_img = self.pair_transform(ori_img, dgd_img)

        return ori_img, dgd_img

    def pair_transform(self, img1, img2):

        # To PIL image
        ToPILImage =  TF.to_pil_image
        img1 = ToPILImage(img1)
        img2 = ToPILImage(img2)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(768,768))
        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)

        # Random Horizontal Flip
        if random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # Resize
        Resize = transforms.Resize(size=(512,512))
        img1 = Resize(img1)
        img2 = Resize(img2)

        # To tensor
        ToTensor = TF.to_tensor
        img1 = ToTensor(img1)
        img2 = ToTensor(img2)

        # Normalization
        Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        img1 = Normalize(img1)
        img2 = Normalize(img2)

        return img1, img2

    def imshow(self, row=4, id=0):

        use('WebAgg')
        images = [(self[i][id]+1)/2 for i in range(row**2)]
        grid_img = torchvision.utils.make_grid(images, nrow=row)
        plt.figure(figsize=(10,10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

    def getDatasetPath(self):

        def getPath():
            return os.path.dirname(os.path.abspath(__file__))      

        path = f'{getPath()}/{self.type}_set'

        try:
            if not os.path.isdir(path):
                raise ValueError(f"Error: {path} doesn't exist.")
            return path

        except ValueError as e:
            print(e)
            exit()


if __name__ == '__main__':

    dataset = FaceDataset('train')
    dataset.imshow(5, 0)
    dataset.imshow(5, 1)
        