from options.Config import *
from dataset.FaceLoader import getFaceLoader

train_loader = getFaceLoader(DataLoaderConfig(), 'train')
valid_loader = getFaceLoader(DataLoaderConfig(), 'valid')
