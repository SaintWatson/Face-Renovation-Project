import os
import cv2
import numpy as np
from random import randint
from tqdm import tqdm
from shutil import rmtree

def getPath():
    return os.path.dirname(os.path.abspath(__file__))

def getDatasetPath(datasetName):
    return f'{getPath()}/{datasetName}'

def downSampling(img):

    ori_w, ori_h = img.shape[0:2]
    ratio = randint(4,8)

    new_w, new_h = ori_w // ratio, ori_h//ratio
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.resize(img, (ori_w, ori_h))

    return img

def addNoise(img):

    def Guassian_noise(img):
        std = np.random.rand() * 2
        return np.random.normal(0, std, img.size)

    def Laplace_noise(img):
        scale = np.random.rand() * 2
        return np.random.laplace(0, scale, img.size)


    functions = [Guassian_noise, Laplace_noise]
    f = functions[randint(0,1)]

    noise = f(img).reshape(img.shape).astype('uint8')

    return cv2.add(img, noise)
    
def blurring(img):

    def rand_odd():

        r = randint(3, 7)
        return 2*r + 1

    def Median_blur(img):
        return cv2.medianBlur(img, rand_odd())

    def Gaussian_blur(img):
        ksize = rand_odd()
        kernal = (ksize, ksize)
        sigma = np.random.rand() * 10

        return cv2.GaussianBlur(img, kernal, sigma)

    functions = [Median_blur, Gaussian_blur]

    f = functions[randint(0,1)]
    
    return f(img)

def compress(img):
    quality = randint(50, 85)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    img_string = encimg.tobytes()
    npimg_string = np.frombuffer(img_string, dtype=np.uint8)
    return cv2.imdecode(npimg_string, 1)

def degrade(img):

    function_pool = [downSampling, addNoise, blurring, compress]
    for i in np.random.permutation(4):
        
        degrade_function = function_pool[i]
        img = degrade_function(img)

    return img

def savePair(ori_imgs, dgd_imgs):

    if len(ori_imgs) != len(dgd_imgs):
        return
    L = len(ori_imgs)

    from matplotlib import use
    import matplotlib.pyplot as plt
    # use('WebAgg')

    def cvt(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(L,2)

    for row in range(L):
        ax[row, 0].imshow(cvt(ori_imgs[row]))
        ax[row, 1].imshow(cvt(dgd_imgs[row]))

    plt.savefig('file.png')

if __name__ == '__main__':
    
    src_paths = ['train_set/ori', 'valid_set/ori']
    dst_paths = ['train_set/dgd', 'valid_set/dgd']

    try:
        for dir in src_paths:
            if not os.path.isdir(getDatasetPath(dir)):
                raise ValueError(f'Error: essential directory "{dir}" doesn\'t exist.')

        for dir in dst_paths:
            if os.path.isdir(getDatasetPath(dir)):
                rmtree(getDatasetPath(dir))
                os.mkdir(getDatasetPath(dir))
                

    except ValueError as e:
        print(e)
        exit()

    srcFiles = []
    for path_id in [0,1]:
        path = getDatasetPath(src_paths[path_id])
        for file in os.listdir(path):
            srcFiles.append(f'{path}/{file}')

    for img_path in tqdm(srcFiles, desc='Degrading: '):

        ori_img = cv2.imread(img_path)
        dgd_img = degrade(ori_img)


        dst_path = img_path.replace("ori","dgd")
        cv2.imwrite(dst_path, dgd_img)

