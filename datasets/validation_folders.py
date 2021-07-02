import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import cv2

def crawl_folders(folders_list):
        imgs = []
        depth = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.jpg'))
            current_depth = []
            for img in current_imgs:
                d = img.dirname()/(img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                depth.append(d)
            imgs.extend(current_imgs)
            depth.extend(current_depth)
        return imgs, depth


def load_as_float(path, height, width):
    img=cv2.resize(cv2.imread(path), (width,height), interpolation = cv2.INTER_AREA)  
    return img.astype(np.float32)

def resize_intrinsics(intrinsics, target_height, target_width, img_height, img_width):
    downscale_height = target_height/img_height
    downscale_width = target_width/img_width

    intrinsics_scaled = np.concatenate((intrinsics[0]*downscale_width,intrinsics[1]*downscale_height, intrinsics[2]), axis=0).reshape(3,3)
    return intrinsics_scaled

class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, width=640, height=480, transform=None):
        self.root = Path(root)
        self.width = width
        self.height= height
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.imgs, self.depth = crawl_folders(self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        img = load_as_float(self.imgs[index])
        depth = np.load(self.depth[index]).astype(np.float32)
        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth

    def __len__(self):
        return len(self.imgs)

class CustomValidationSet(data.Dataset):
    def __init__(self, root,width=640, height=480, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.height= height
        self.width = width
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
   
        for scene in self.scenes:
            
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) ==0:
                imgs = sorted(scene.files('*.png'))
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            dummy_img =cv2.imread(imgs[0])
            intrinsics = resize_intrinsics(intrinsics,self.height,self.width,*dummy_img.shape[0:2])
            if len(imgs) < sequence_length:
                continue
            depths = []
            for img in imgs:
                d = img.dirname()/(img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                depths.append(d)

            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'depth': depths[i]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'],self.height,self.width)
        tgt_depth = np.resize(np.load(sample['depth']).astype(np.float32),(self.height,self.width))
        ref_imgs = [load_as_float(ref_img,self.height,self.width) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), tgt_depth

    def __len__(self):
        return len(self.samples)



class ValidationSetWithPose(data.Dataset):
    """A sequence validation data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_1/cam.txt
        root/scene_1/pose.txt
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .
    """

    def __init__(self, root, seed=None, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            poses = np.genfromtxt(scene/'poses.txt').reshape((-1, 3, 4))
            poses_4D = np.zeros((poses.shape[0], 4, 4)).astype(np.float32)
            poses_4D[:, :3] = poses
            poses_4D[:, 3, 3] = 1
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            assert(len(imgs) == poses.shape[0])
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                tgt_img = imgs[i]
                d = tgt_img.dirname()/(tgt_img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                sample = {'intrinsics': intrinsics, 'tgt': tgt_img, 'ref_imgs': [], 'poses': [], 'depth': d}
                first_pose = poses_4D[i - demi_length]
                sample['poses'] = (np.linalg.inv(first_pose) @ poses_4D[i - demi_length: i + demi_length + 1])[:, :3]
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sample['poses'] = np.stack(sample['poses'])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        depth = np.load(sample['depth']).astype(np.float32)
        poses = sample['poses']
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, _ = self.transform([tgt_img] + ref_imgs, None)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        return tgt_img, ref_imgs, depth, poses

    def __len__(self):
        return len(self.samples)
