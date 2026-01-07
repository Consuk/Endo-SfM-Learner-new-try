from __future__ import division
import torch
import random
import numpy as np
from PIL import Image
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

class Resize(object):
    def __init__(self, size):  # size = (h, w)
        self.h, self.w = size

    def __call__(self, images, intrinsics):
        # images: list of np arrays HxWxC float32 [0,1] o uint8
        in_h, in_w = images[0].shape[:2]

        resized = []
        for im in images:
            # asegurar float32
            if im.dtype != np.float32:
                im = im.astype(np.float32)
            # cv2 expects (w,h)
            im_r = cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            resized.append(im_r)

        # ajusta intrinsics si existe
        if intrinsics is not None:
            K = np.array(intrinsics, dtype=np.float32).copy()
            sx = self.w / float(in_w)
            sy = self.h / float(in_h)
            K[0, :] *= sx
            K[1, :] *= sy
            return resized, K

        return resized, intrinsics

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix
    to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor.
    """

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # HWC -> CHW (transpose gives a view)
            im = np.transpose(im, (2, 0, 1))

            # make contiguous
            im = np.ascontiguousarray(im)

            # IMPORTANT: break numpy <-> torch memory sharing
            t = torch.from_numpy(im).float().div_(255.0).clone()

            tensors.append(t)

        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics
