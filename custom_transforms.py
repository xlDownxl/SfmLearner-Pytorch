from __future__ import division
import torch
import random
import numpy as np
from skimage.transform import resize
import cv2
'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            #print("horizontal")
            #print(output_intrinsics)
            output_intrinsics[0,2] = w - output_intrinsics[0,2] #flip focal point of x
            #print(output_intrinsics)
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics

class RandomVerticalFlip(object):
    """Randomly vertically flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics, ):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
        
            output_images = [np.copy(np.flipud(im)) for im in images]
            w = output_images[0].shape[0]
            #print("vertical")
            #print(output_intrinsics)
            output_intrinsics[1,2] = w - output_intrinsics[1,2]  #flip focal point of y
            #print(output_intrinsics)
            #cv2.imwrite("aug/image_processed"+str(random.random())+".jpg", output_images[0])
        else:
            output_images = images
            output_intrinsics = intrinsics
            #cv2.imwrite("aug/image_processed_NOT"+str(random.random())+".jpg", output_images[0])
            
        return output_images, output_intrinsics

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [resize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, output_intrinsics

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def change_saturation(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = s*value
    s[s > 255] = 255
    s[s < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img

class ColorJitter(object):
    """TODO"""

    def __call__(self, images, intrinsics):

        brightness = random.gauss(0,10)
        #saturation = random.uniform(-1.5,1.5)
        #print(brightness)
        
        output_images = [change_brightness(img, value=brightness) for img in images]
        #output_images = [change_saturation(img, value=saturation) for img in images]
        #cv2.imwrite("aug/image_processed"+str(brightness)+".jpg", output_images[0])
      
        return output_images, intrinsics

class RandomGauss(object):
    """TODO"""

    def __call__(self, images, intrinsics):
        if random.getrandbits(1)==1:
            sigma = random.uniform(0,1)       
            output_images = [cv2.GaussianBlur(img,(5,5),sigma) for img in images]
            #cv2.imwrite("aug/image_processed"+str(sigma)+".jpg", output_images[0])
        else:
            output_images = images
      
        return output_images, intrinsics