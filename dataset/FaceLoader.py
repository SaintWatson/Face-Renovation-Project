from FaceDataset import FaceDataset
from torch.utils.data import DataLoader


def getFaceLoader(type='train'):

    dataset = FaceDataset(type)

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 12,
        shuffle = (type=='train'),
        num_workers = 2
    )

    return dataloader
