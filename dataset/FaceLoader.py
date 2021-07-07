from dataset.FaceDataset import FaceDataset
from torch.utils.data import DataLoader


def getFaceLoader(opt, type='train'):

    dataset = FaceDataset(type)

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = opt.batch_size,
        shuffle = (type=='train'),
        num_workers = opt.num_workers
    )

    return dataloader
