import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import cv2
def load_as_float(path, height, width):
    img=cv2.resize(cv2.imread(path), (width,height), interpolation = cv2.INTER_AREA)  
    return img.astype(np.float32)

def resize_intrinsics(intrinsics, target_height, target_width, img_height, img_width):
    downscale_height = target_height/img_height
    downscale_width = target_width/img_width

    intrinsics_scaled = np.concatenate((intrinsics[0]*downscale_width,intrinsics[1]*downscale_height, intrinsics[2]), axis=0).reshape(3,3)
    return intrinsics_scaled

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root,height=480, width=640, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.height= height
        self.width = width
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) ==0:
                imgs = sorted(scene.files('*.png'))
            dummy_img =cv2.imread(imgs[0])
            intrinsics = resize_intrinsics(intrinsics,self.height,self.width,*dummy_img.shape[0:2])
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'],self.height,self.width)
        ref_imgs = [load_as_float(ref_img,self.height,self.width) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
