import os, torchvision
from torch.utils.data import Dataset
from matplotlib import use
import matplotlib.pyplot as plt
from torchvision.transforms import *



class FaceDataset(Dataset):

    def __init__(self, type='train'):

        super(FaceDataset, self).__init__()
        self.type = type
        self.path = self.getDatasetPath()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(768),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(15),
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        ])

    def __getitem__(self, index):

        filename = f'{str(index).zfill(5)}.png'
        ori_img = torchvision.io.read_image(f'{self.path}/{filename}')
        
        from copy import deepcopy
        dgd_img = deepcopy(ori_img)

        ori_img, dgd_img = self.transform(ori_img, dgd_img)


        return ori_img, dgd_img

    def __len__(self):
        return len(os.listdir(self.path))

    def imshow(self, row=4):

        use('WebAgg')
        images = [(self[i][1]+1)/2 for i in range(row**2)]
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
    dataset.imshow(5)
        