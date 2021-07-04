import os, sys
from shutil import rmtree, copyfile
import numpy as np
from tqdm import tqdm

def getPath():
    return os.path.dirname(os.path.abspath(__file__))

def getDatesetPath():
    return  getPath() + '/images1024x1024'

def sample(train_size, valid_size, init=True):

    if not (0 <= train_size <= 70000 and 0 <= valid_size <= 70000):
        raise ValueError('ValueError: sample: Size must be integer between 0~70000')

    train_set_path = f'{getPath()}/train_set'
    valid_set_path = f'{getPath()}/valid_set'

    def initSampler():
        for dir in [train_set_path, valid_set_path]:
            if os.path.isdir(dir):
                rmtree(dir)
        os.mkdir(train_set_path)
        os.mkdir(valid_set_path)

    def index2path(index):
        filename = str(index).zfill(5) + '.png'
        filename = filename[0:2] + "000/" + filename
        return f'{getDatesetPath()}/{filename}'

    if init:    
        initSampler()

    full_index = np.random.permutation(70000)
    train_index = full_index[:train_size]
    valid_index = full_index[-valid_size:]


    for (dir, index) in [(train_set_path, train_index), (valid_set_path, valid_index)]:
        for i in tqdm(index, desc=dir[-9:]):
            file_path = index2path(i)
            copyfile(file_path, f'{dir}/{i}.png')
        
if __name__ == '__main__':

    try:
        if len(sys.argv) != 3:
            raise ValueError('Usage: python3 dataset-slicer.py <train_set_size> <valid_set_size>')

        train_size = int(sys.argv[1])
        valid_size = int(sys.argv[2])

        if not (0 <= train_size <= 70000 and 0 <= valid_size <= 70000):
            raise ValueError('Error: Size must be integer between 0~70000')

        if not os.path.isdir(getDatesetPath()): 
            raise ValueError("Error: Dataset directory doesn't exist.")
        
        sample(train_size, valid_size)


    except ValueError as e:
        print(e)
        exit()
