'''
Function:
    load the images for training
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import glob
import torchvision
from PIL import Image
from torch.utils.data import Dataset


'''load images'''
class ImageDataset(Dataset):
    def __init__(self, rootdir, imagesize, img_norm_info, **kwargs):
        assert rootdir.endswith('*')
        self.rootdir = rootdir
        self.imagesize = imagesize
        self.img_norm_info = img_norm_info
        self.imagepaths = glob.glob(rootdir)
    '''get item'''
    def __getitem__(self, index):
        image = Image.open(self.imagepaths[index])
        return ImageDataset.preprocess(image, self.imagesize, self.img_norm_info)
    '''calculate length'''
    def __len__(self):
        return len(self.imagepaths)
    '''preprocess image'''
    @staticmethod
    def preprocess(image, imagesize, img_norm_info):
        means_norm, stds_norm = img_norm_info.get('means'), img_norm_info.get('stds')
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(imagesize),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=means_norm, std=stds_norm)])
        return transform(image)