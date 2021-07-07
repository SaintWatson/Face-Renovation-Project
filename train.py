from options.Config import *
from dataset.FaceLoader import getFaceLoader
from network.Generator import FaceGenerator
from network.Discriminator import FaceDiscriminator

train_loader = getFaceLoader(DataLoaderConfig(), 'train')
valid_loader = getFaceLoader(DataLoaderConfig(), 'valid')

G = FaceGenerator(GeneratorConfig())
D = FaceDiscriminator()


