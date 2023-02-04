#!/home/dongyi/anaconda3/envs/paddle_env/bin/python3
from __future__ import division

import collections
import math
import numbers
import random
import types
import warnings

import cv2
import numpy as np

import inference.functional as F

__all__ = [
    "Compose", "ComposeWithPoint", "Resize", "Scale",
    "CenterCrop", "CropCenterSquare", "Pad", "Lambda", "RandomApply", "RandomChoice",
    "RandomOrder", "RandomCrop", "Crop", "RandomHorizontalFlip", "RandomHorizontalFlipWithPoint", "RandomHorizontalFlipWithPoints",
    "RandomVerticalFlip", "RandomVerticalFlipWithPoint","RandomVerticalFlipWithPoints", "RandomHVFlip", "RandomHVFlipWithPoints",
    "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "PaddedSquare","PaddedSquareWithPoints", "RandomPad", "RandomPadWithPoints","RandomPadWithoutPoints"
     "ColorJitter","ColorJitterWithPoints", "BackgroundReplacement","BackgroundReplacementWithPoints",
     "ColorThresholdSegmentation","ColorThresholdSegmentationWithPoints","GrayThresholdSegmentation","GrayThresholdSegmentationWithPoints",
     "RandomRotation", "RandomRotationWithPoint","RandomRotationWithPoints", "RandomAffine",
    "Grayscale", "RandomGrayscale"
]

_cv2_pad_to_str = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}
_cv2_interpolation_to_str = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}
_cv2_interpolation_from_str = {
    v: k
    for k, v in _cv2_interpolation_to_str.items()
}


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ComposeWithPoint(object):
    """Composes several transforms together for img + a single point or multiple points
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, x, y, confidence = 1):  # 3 inputs without given format
        for t in self.transforms:
            img,x,y = t(img, x, y, confidence)
        return img, x, y

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize(object):
    """Resize the input numpy ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        # assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = size
        elif isinstance(size, collections.abc.Iterable) and len(size) == 2:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _cv2_interpolation_from_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of the transforms.Scale transform is deprecated, " +
            "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CropCenterSquare(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        img_h, img_w = img.shape[:2]
        h, w = min(img_h, img_w), min(img_h, img_w)
        return F.center_crop(img, (h, w))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):
    """Pad the given numpy ndarray on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple, list))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding,
                      collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError(
                "Padding must be an int or a 2, or 4 element tuple, not a " +
                "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be padded.
        Returns:
            numpy ndarray: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """
    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


class RandomCrop(object):
    """Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop. 
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[0:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.shape[1], 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.shape[0]), self.fill,
                        self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(
            self.size, self.padding)


class Crop(object):   # c,h,w
    """Crop images, given top-left corner (i,j).bash_history.
    Args:
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    def __init__(self, i, j, h, w):
        self.i=i
        self.j=j
        self.h=h
        self.w=w

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return img[:, self.i:self.i + self.h, self.j:self.j + self.w]  # c,h,w
    
    def __repr__(self):
        return self.__class__.__name__ + 'i={0}, j={1}, h={2}, w={3}'.format(
            self.i, self.j, self.h, self.w
        )
    


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """random
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """

        if random.random() < self.p:
            return F.flip(img, axis='x')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHorizontalFlipWithPoint(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,x, y, w=1, h=1):
        """random
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """

        if random.random() < self.p:
            return F.flip_with_point(img,x,y, 'x',w,h)
        return img,x,y

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHorizontalFlipWithPoints(object):
    """Horizontally flip the given PIL Image randomly with multiples points and a given probability.
    Args:
        p (float): probability of the image being horizontally or non flipped. Default value is 0.5,0.50 for shoe data
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,x,y,confidence, w=1, h=1):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.p:  # 0-0.5
            return F.flip_with_points(img,x,y, 'x',confidence, w,h)    # Horizontally Flip with points
        else:                                # 0.5-0.1
            return img,x,y                       # non-flip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipWithPoint(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,x,y,w=1, h=1):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.flip_with_point(img,x,y, 'y',w,h)
        return img,x,y

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlipWithPoints(object):
    """Vertically flip the given PIL Image randomly with multiples points and a given probability.
    Args:
        p (float): probability of the image being vertically or non flipped. Default value is 0.5,0.50 for shoe data
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,x,y,confidence, w=1, h=1):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.p:  # 0-0.5
            return F.flip_with_points(img,x,y, 'y',confidence, w,h)    # Vertically Flip with points
        else:                                # 0.5-0.1
            return img,x,y                       # non-flip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.flip(img, axis='y')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHVFlip(object):
    """Horizontally or Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being horizontally, vertically or non flipped. Default value is 0.25,0.25,0.50
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() <= self.p:  # 0-0.25
            return F.flip(img, axis='y')    # Vertical Flip
        elif random.random() > (1- self.p):  # 0.75 -1 
            return F.flip(img, axis='x')    # Horizontal Flip
        else:                                # 0.25-0.75
            return img                      # non-flip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHVFlipWithPoints(object):
    """Horizontally or Vertically flip the given PIL Image randomly with multiples points and a given probability.
    Args:
        p (float): probability of the image being horizontally, vertically or non flipped. Default value is 0.25,0.25,0.50 for shoe data
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img,x,y,confidence, w=1, h=1):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() <= self.p:  # 0-0.25
            return F.flip_with_points(img,x,y, 'y',confidence, w,h)    # Vertically Flip with points
        elif random.random() > (1- self.p):  # 0.75 -1 
            return F.flip_with_points(img,x,y, 'x',confidence, w,h)    # Horizontally Flip with points
        else:                                # 0.25-0.75
            return img,x,y                       # non-flip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomResizedCrop(object):
    """Crop the given numpy ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: cv2.INTER_CUBIC
    """
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=cv2.INTER_LINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _cv2_interpolation_from_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomSizedCrop(RandomResizedCrop):
    """
    Note: This transform is deprecated in favor of RandomResizedCrop.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of the transforms.RandomSizedCrop transform is deprecated, "
            + "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)

class PaddedSquare(object):
    """
    Padding to square the image. The default method is edge padding.
    """
    def __init__(self, padding_mode= 'constant'):
        self.padding_mode= padding_mode

    def __call__(self,img):
        """
        Args: 
            img (numpy ndarray): Image to be padding to square image
        Returns: 
            the squared image with original sizes of objects
        """
        h,w,c= img.shape
        # print(img.shape)
        side_len = max(h,w)
        if h<w:
            pad_top = int((side_len - h)/2)
            pad_bottom =  side_len - pad_top - h
            pad_left=0
            pad_right=0
        elif h>w:
            pad_top = 0
            pad_bottom = 0
            pad_left = int((side_len - w)/2)
            pad_right  = side_len - pad_left - w
        else: # h==w
            return img
        
        # fill = (int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2]))   # img is r,g,b, while fill= (b,g,r)
        b1, g1, r1 = int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2])
        b2, g2, r2 = int(img[0,-1,0]),int(img[0,-1,1]),int(img[0,-1,2])
        b3, g3, r3 = int(img[-1,0,0]),int(img[-1,0,1]),int(img[-1,0,2])
        b4, g4, r4 = int(img[-1,-1,0]),int(img[-1,-1,1]),int(img[-1,-1,2])
        fill = (int((b1+b2+b3+b4)/4), int((g1+g2+g3+g4)/4), int((r1+r2+r3+r4)/4))
        return cv2.copyMakeBorder(img, 
                                top=pad_top, 
                                bottom=pad_bottom, 
                                left=pad_left, 
                                right=pad_right, 
                                borderType=_cv2_pad_to_str[self.padding_mode],
                                value=fill)    # when padding_mode == 'constant'

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={})'.format(self.padding_mode)

class PaddedSquareWithPoints(object):
    """
    Padding to square the image. The default method is edge padding.
    """
    def __init__(self, padding_mode= 'constant'):
        self.padding_mode= padding_mode

    def __call__(self,img,x,y,confidence):
        """
        Args: 
            img (numpy ndarray): Image to be padding to square image
        Returns: 
            the squared image with original sizes of objects
        """
        h,w,c= img.shape
        # print(img.shape)
        side_len = max(h,w)
        if h<w:
            pad_top = int((side_len - h)/2)
            pad_bottom =  side_len - pad_top - h
            pad_left=0
            pad_right=0
            for i in range(len(x)):
                if confidence[i] >= 0.5:
                    # keep x
                    y[i] =  (pad_top + y[i]*h)/side_len 
                # else keep the coordinates of the points with confidence < 0.5
        elif h>w:
            pad_top = 0
            pad_bottom = 0
            pad_left = int((side_len - w)/2)
            pad_right  = side_len - pad_left - w
            for i in range(len(x)):
                if confidence[i] >= 0.5:
                    x[i] =  (pad_left + x[i]*w)/side_len 
                    # keep y
                # else keep the coordinates of the points with confidence < 0.5
        else: # h==w
            return img,x,y
        
        # fill = (int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2]))   # img is r,g,b, while fill= (b,g,r)
        b1, g1, r1 = int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2])
        b2, g2, r2 = int(img[0,-1,0]),int(img[0,-1,1]),int(img[0,-1,2])
        b3, g3, r3 = int(img[-1,0,0]),int(img[-1,0,1]),int(img[-1,0,2])
        b4, g4, r4 = int(img[-1,-1,0]),int(img[-1,-1,1]),int(img[-1,-1,2])
        fill = (int((b1+b2+b3+b4)/4), int((g1+g2+g3+g4)/4), int((r1+r2+r3+r4)/4))
        return cv2.copyMakeBorder(img, 
                                top=pad_top, 
                                bottom=pad_bottom, 
                                left=pad_left, 
                                right=pad_right, 
                                borderType=_cv2_pad_to_str[self.padding_mode],
                                value=fill),x,y    # when padding_mode == 'constant'

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={})'.format(self.padding_mode)

class RandomPad(object):
    """
    Randomly expand the border of imgs.
    """
    def __init__(self, padding_mode= 'constant', pad_thresh_l=0):  
        """
        Args:
            pad_thresh_l: >=0, add int(pad_thresh_l*(h,w)) rows/cols on the border 
        """
        self.padding_mode= padding_mode
        self.pad_thresh_l = pad_thresh_l

    def __call__(self,img):
        """
        Args: 
            img (numpy ndarray): Image to be padding to square image
        Returns: 
            the squared image with original sizes of objects
        """
        if self.pad_thresh_l<0:
            print("pad_thresh_l should be no less than 0")
            return img
        h,w,c= img.shape
        # side_len = max(h,w)
        pad_random= self.pad_thresh_l*random.random()
        h2 = int(h*pad_random)
        w2 = int(w*pad_random)

        pad_top = random.randint(0,h2)
        pad_bottom = h2 - pad_top
        pad_left = random.randint(0,w2)
        pad_right  = w2 - pad_left
        
        # fill = (int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2]))   # img is r,g,b, while fill= (b,g,r)
        b1, g1, r1 = int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2])
        b2, g2, r2 = int(img[0,-1,0]),int(img[0,-1,1]),int(img[0,-1,2])
        b3, g3, r3 = int(img[-1,0,0]),int(img[-1,0,1]),int(img[-1,0,2])
        b4, g4, r4 = int(img[-1,-1,0]),int(img[-1,-1,1]),int(img[-1,-1,2])
        fill = (int((b1+b2+b3+b4)/4), int((g1+g2+g3+g4)/4), int((r1+r2+r3+r4)/4))
        return cv2.copyMakeBorder(img, 
                                top=pad_top, 
                                bottom=pad_bottom, 
                                left=pad_left, 
                                right=pad_right, 
                                borderType=_cv2_pad_to_str[self.padding_mode],
                                value=fill)    # when padding_mode == 'constant'

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={})'.format(self.padding_mode)

class RandomPadWithoutPoints(object):  # when inference in RandomPadWithPoints cases
    """
    Randomly expand the border of imgs.
    """
    def __init__(self, padding_mode= 'constant', pad_thresh_l=0.416, pad_thresh_h=0.512): # (√2-1) (0.414)
        """
        Args:
            pad_thresh_l: >=0, add int(pad_thresh_l*(h,w)) rows/cols on the border 
            pad_thresh_h should be larger than pad_thresh_l
        """
        self.padding_mode= padding_mode
        self.pad_thresh_l = pad_thresh_l
        self.pad_thresh_h = pad_thresh_h

    def __call__(self,img):
        """
        Args: 
            img (numpy ndarray): Image to be padding to square image
        Returns: 
            the squared image with original sizes of objects
        """
        if (self.pad_thresh_l<0) or (self.pad_thresh_l > self.pad_thresh_h) : 
            print("pad_thresh_l should be no less than 0, and no larger pad_thresh_h")
            return img

        h,w,c= img.shape
        # print(img.shape)
        # side_len = max(h,w)
        pad_random= random.uniform(self.pad_thresh_l, self.pad_thresh_h)
        h2 = int(h*pad_random)
        w2 = int(w*pad_random)
        h1 = int(h*self.pad_thresh_l)
        w1 = int(w*self.pad_thresh_l)

        
        # assign edges with values randomly but keep their sum at h2, w2
        pad_top = random.randint(h1//2,h2//2)
        pad_bottom = h2 - pad_top
        pad_left = random.randint(w1//2,w2//2)
        pad_right  = w2 - pad_left

        # fill = (int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2]))   # img is r,g,b, while fill= (b,g,r)
        b1, g1, r1 = int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2])
        b2, g2, r2 = int(img[0,-1,0]),int(img[0,-1,1]),int(img[0,-1,2])
        b3, g3, r3 = int(img[-1,0,0]),int(img[-1,0,1]),int(img[-1,0,2])
        b4, g4, r4 = int(img[-1,-1,0]),int(img[-1,-1,1]),int(img[-1,-1,2])
        fill = (int((b1+b2+b3+b4)/4), int((g1+g2+g3+g4)/4), int((r1+r2+r3+r4)/4))
        return cv2.copyMakeBorder(img, 
                                top=pad_top, 
                                bottom=pad_bottom, 
                                left=pad_left, 
                                right=pad_right, 
                                borderType=_cv2_pad_to_str[self.padding_mode],
                                value=fill)   # when padding_mode == 'constant'

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={})'.format(self.padding_mode)


class RandomPadWithPoints(object):
    """
    Randomly expand the border of imgs.
    """
    def __init__(self, padding_mode= 'constant', pad_thresh_l=0.416, pad_thresh_h=0.512): # (√2-1) (0.414)
        """
        Args:
            pad_thresh_l: >=0, add int(pad_thresh_l*(h,w)) rows/cols on the border 
            pad_thresh_h should be larger than pad_thresh_l
        """
        self.padding_mode= padding_mode
        self.pad_thresh_l = pad_thresh_l
        self.pad_thresh_h = pad_thresh_h

    def __call__(self,img,x,y,confidence):
        """
        Args: 
            img (numpy ndarray): Image to be padding to square image
        Returns: 
            the squared image with original sizes of objects
        """
        if (self.pad_thresh_l<0) or (self.pad_thresh_l > self.pad_thresh_h) : 
            print("pad_thresh_l should be no less than 0, and no larger pad_thresh_h")
            return img,x,y

        h,w,c= img.shape
        # print(img.shape)
        # side_len = max(h,w)
        pad_random= random.uniform(self.pad_thresh_l, self.pad_thresh_h)
        h2 = int(h*pad_random)
        w2 = int(w*pad_random)
        h1 = int(h*self.pad_thresh_l)
        w1 = int(w*self.pad_thresh_l)

        
        # assign edges with values randomly but keep their sum at h2, w2
        pad_top = random.randint(h1//2,h2//2)
        pad_bottom = h2 - pad_top
        pad_left = random.randint(w1//2,w2//2)
        pad_right  = w2 - pad_left

        # print(pad_left)
        # print(pad_right)

        for i in range(len(x)):
            if confidence[i] >= 0.5:
                x[i] =  (pad_left + x[i]*w)/(w+w2) 
                y[i] =  (pad_top + y[i]*h)/(h+h2)
            # else keep the coordinates of the points with confidence < 0.5

        # fill = (int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2]))   # img is r,g,b, while fill= (b,g,r)
        b1, g1, r1 = int(img[0,0,0]),int(img[0,0,1]),int(img[0,0,2])
        b2, g2, r2 = int(img[0,-1,0]),int(img[0,-1,1]),int(img[0,-1,2])
        b3, g3, r3 = int(img[-1,0,0]),int(img[-1,0,1]),int(img[-1,0,2])
        b4, g4, r4 = int(img[-1,-1,0]),int(img[-1,-1,1]),int(img[-1,-1,2])
        fill = (int((b1+b2+b3+b4)/4), int((g1+g2+g3+g4)/4), int((r1+r2+r3+r4)/4))
        return cv2.copyMakeBorder(img, 
                                top=pad_top, 
                                bottom=pad_bottom, 
                                left=pad_left, 
                                right=pad_right, 
                                borderType=_cv2_pad_to_str[self.padding_mode],
                                value=fill), x,y    # when padding_mode == 'constant'

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={})'.format(self.padding_mode)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue,
                                     'hue',
                                     center=0,
                                     bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn(
                'Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn(
                'Hue jitter enabled. Will slow down loading immensely.')

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(
                    name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(
                    lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(
                    lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class ColorJitterWithPoints(object): 
    """Randomly change the brightness, contrast and saturation of an image. For regression task
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue,
                                     'hue',
                                     center=0,
                                     bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn(
                'Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn(
                'Hue jitter enabled. Will slow down loading immensely.')

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(
                    name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(
                    lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(
                    lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img,x,y, confidence):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img),x,y  # x,y are not changed

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class BackgroundReplacement(object):
    """
    Replace (averaged) background with random color (b,g,r)
    """
    def __init__(self, default=0,r=36,g=146,b=145):
        self.default = default
        self.b=b
        self.g=g
        self.r=r

    def __call__(self,img):
        """
        Args: 
            img (numpy ndarray): Image to be replaced its background.
        Returns: 
            numpy ndarray: Orginal object with random background color.
        """
        r, g, b = cv2.split(img)
        b1, g1, r1 = int(b[0,0]),int(g[0,0]),int(r[0,0])
        b2, g2, r2 = int(b[0,-1]),int(g[0,-1]),int(r[0,-1])
        b3, g3, r3 = int(b[-1,0]),int(g[-1,0]),int(r[-1,0])
        b4, g4, r4 = int(b[-1,-1]),int(g[-1,-1]),int(r[-1,-1])
        fill = (round((b1+b2+b3+b4)/4), round((g1+g2+g3+g4)/4), round((r1+r2+r3+r4)/4))       

        if self.default == True:
            rand_b, rand_g, rand_r = self.b,self.g,self.r 
        else:
            rand_b, rand_g, rand_r = random.randint(0,255), random.randint(0,255), random.randint(0,255) # random color 

        r[r==fill[2]]= rand_r   
        g[g==fill[1]]= rand_g 
        b[b==fill[0]]= rand_b 
        img=cv2.merge([r,g,b])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(default={})'.format(self.default)

class BackgroundReplacementWithPoints(object):
    """
    Replace (averaged) background with random color (b,g,r)
    """
    def __init__(self, default=0, r=36,g=146,b=145):
        self.default = default
        self.b=b
        self.g=g
        self.r=r

    def __call__(self,img,x,y, confidence):
        """
        Args: 
            img (numpy ndarray): Image to be replaced its background.
        Returns: 
            numpy ndarray: Orginal object with random background color.
        """
        r, g, b = cv2.split(img)
        b1, g1, r1 = int(b[0,0]),int(g[0,0]),int(r[0,0])
        b2, g2, r2 = int(b[0,-1]),int(g[0,-1]),int(r[0,-1])
        b3, g3, r3 = int(b[-1,0]),int(g[-1,0]),int(r[-1,0])
        b4, g4, r4 = int(b[-1,-1]),int(g[-1,-1]),int(r[-1,-1])
        fill = (round((b1+b2+b3+b4)/4), round((g1+g2+g3+g4)/4), round((r1+r2+r3+r4)/4))       

        if self.default == True:
            rand_b, rand_g, rand_r =  self.b,self.g,self.r
        else:
            rand_b, rand_g, rand_r = random.randint(0,255), random.randint(0,255), random.randint(0,255) # random color

        r[r==fill[2]]= rand_r   
        g[g==fill[1]]= rand_g 
        b[b==fill[0]]= rand_b 
        img=cv2.merge([r,g,b])
        return img,x,y  # x,y are not changed

    def __repr__(self):
        return self.__class__.__name__ + '(default={})'.format(self.default)

class ColorThresholdSegmentation(object):
    """
    Image Segmentation by color threshold/masking (hsv_low. hsv_high)
    Args:
        color_mode specify the median of hsv color threshold, it could be "corner", "constant", "most_frequent"
        hue_de, sat_de, val_de are deviations for the median of color threshold when run cv2.inRange
        (hue_de, sat_de, val_de) =(13,107,140), (18,85,140), ()  for ShoePackaging green,white,red background respectively
    """
    def __init__(self, color_mode='corner',hue_de=13,sat_de=107,val_de=110):
        self.color_mode = color_mode
        self.hue_de = hue_de
        self.sat_de = sat_de
        self.val_de = val_de

    def __call__(self,img):
        """
        Args: 
            img (numpy ndarray): Image to be replaced its background.
        Returns: 
            numpy ndarray: Orginal object with random background color.
        """
        # img= img.transpose(1, 2, 0) # C, H, W → H, W, C
        # h,w,c =img.shape
        img_hsv =  cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        # Acquire the median of hsv color threshold
        if self.color_mode == "corner":
            # Method 1: use hue, saturation, value(brightness) of corner(0,0)
            # c_hue, c_sat, c_val = int(img_hsv[0,0,0]), int(img_hsv[0,0,1]), int(img_hsv[0,0,2]) 
            
            # Method 2: use the average hue, saturation, value(brightness) of 4 corners referring to class BackgroundReplacement()
            h, s, v = cv2.split(img_hsv)
            v1, s1, h1 = int(v[0,0]),int(s[0,0]),int(h[0,0])
            v2, s2, h2 = int(v[0,-1]),int(s[0,-1]),int(h[0,-1])
            v3, s3, h3 = int(v[-1,0]),int(s[-1,0]),int(h[-1,0])
            v4, s4, h4 = int(v[-1,-1]),int(s[-1,-1]),int(h[-1,-1])
            c_hue, c_sat, c_val = round((h1+h2+h3+h4)/4),round((s1+s2+s3+s4)/4),round((v1+v2+v3+v4)/4)
        elif self.color_mode == "constant": # replace constant pixel values, take black as an example
        	c_hue, c_sat, c_val = int(0), int(0), int(0)
        else:
            pass # c_hue, c_sat, c_val = int(128), int(128), int(128)
        #elif self.color_mode == "most_frequent"： #TODO

        mask =  cv2.inRange(img_hsv, (c_hue-self.hue_de, c_sat-self.sat_de , c_val-self.val_de),(c_hue+self.hue_de, c_sat+self.sat_de , c_val+self.val_de)) # cv2.inRange(img, (int, int, int), ())
        mask= cv2.medianBlur(mask, 9) # 7
        mask = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(color_mode={})'.format(self.color_mode)

class ColorThresholdSegmentationWithPoints(object):
    """
    Image Segmentation by color threshold/masking (hsv_low. hsv_high) with points
    Args:
        color_mode specify the median of hsv color threshold, it could be "corner", "constant", "most_frequent"
        hue_de, sat_de, val_de are deviations for the median of color threshold when run cv2.inRange
        (hue_de, sat_de, val_de) =(13,107,140), (18,85,140), ()  for ShoePackaging green,white,red background respectively
    """
    def __init__(self, color_mode='corner',hue_de=13,sat_de=107,val_de=110):
        self.color_mode = color_mode
        self.hue_de = hue_de
        self.sat_de = sat_de
        self.val_de = val_de

    def __call__(self,img,x,y,confidence):
        """
        Args: 
            img (numpy ndarray): Image to be replaced its background.
        Returns: 
            numpy ndarray: Orginal object with random background color.
        """
        # img= img.transpose(1, 2, 0) # C, H, W → H, W, C
        # h,w,c =img.shape
        img_hsv =  cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        # Acquire the median of hsv color threshold
        if self.color_mode == "corner":
            # Method 1: use hue, saturation, value(brightness) of corner(0,0)
            # c_hue, c_sat, c_val = int(img_hsv[0,0,0]), int(img_hsv[0,0,1]), int(img_hsv[0,0,2]) 

            # Method 2: use the average hue, saturation, value(brightness) of 4 corners referring to class BackgroundReplacement()
            h, s, v = cv2.split(img_hsv)
            v1, s1, h1 = int(v[0,0]),int(s[0,0]),int(h[0,0])
            v2, s2, h2 = int(v[0,-1]),int(s[0,-1]),int(h[0,-1])
            v3, s3, h3 = int(v[-1,0]),int(s[-1,0]),int(h[-1,0])
            v4, s4, h4 = int(v[-1,-1]),int(s[-1,-1]),int(h[-1,-1])
            c_hue, c_sat, c_val = round((h1+h2+h3+h4)/4),round((s1+s2+s3+s4)/4),round((v1+v2+v3+v4)/4)
        elif self.color_mode == "constant": # replace constant pixel values, take black as an example
        	c_hue, c_sat, c_val = int(0), int(0), int(0)
        else:
            pass
        #elif self.color_mode == "most_frequent"： #TODO

        mask =  cv2.inRange(img_hsv, (c_hue-self.hue_de, c_sat-self.sat_de , c_val-self.val_de),(c_hue+self.hue_de, c_sat+self.sat_de , c_val+self.val_de)) # cv2.inRange(img, (int, int, int), ())
        mask= cv2.medianBlur(mask, 9) # 7
        mask = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)

        return img,x,y

    def __repr__(self):
        return self.__class__.__name__ + '(color_mode={})'.format(self.color_mode)

class RandomRotation(object):
    """Rotate the image by angle, and positive value means anti-clockwise rotation
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (numpy ndarray): Image to be rotated.
        Returns:
            numpy ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(
            self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomRotationWithPoint(object):
    """Rotate the image with a single point by angle, and positive value means anti-clockwise rotation
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        x,y: the extra point to random rotation
    """
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
       
    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, x, y, centerx=0.5, centery=0.5):
        """
            img (numpy ndarray): Image to be rotated.
        Returns:
            numpy ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate_with_point(img, angle, x, y, self.resample, self.expand, self.center, centerx, centery)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(
            self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class RandomRotationWithPoints(object):
    """Rotate the image with multiple points by angle, and positive value means anti-clockwise rotation
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        x,y: the extra point to random rotation
    """
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
       
    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1]) # get a random angle from the uniform distrubution of (-degrees, degrees)

        return angle

    def __call__(self, img, x, y, confidence, centerx=0.5, centery=0.5):
        """
            img (numpy ndarray): Image to be rotated.
        Returns:
            numpy ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate_with_points(img, angle, x, y, confidence, self.resample, self.expand, self.center, centerx, centery)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(
            self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image.
    """
    def __init__(self,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 interpolation=cv2.INTER_LINEAR,
                 fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        "translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError(
                        "If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        # self.resample = resample
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (numpy ndarray): Image to be transformed.
        Returns:
            numpy ndarray: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale,
                              self.shear, (img.shape[1], img.shape[0]))
        return F.affine(img,
                        *ret,
                        interpolation=self.interpolation,
                        fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _cv2_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class Grayscale(object):
    """Convert image to grayscale.
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    Returns:
        numpy ndarray: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be converted to grayscale.
        Returns:
            numpy ndarray: Randomly grayscaled image.
        """
        return F.to_grayscale(img,
                              num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(
            self.num_output_channels)


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        numpy ndarray: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be converted to grayscale.
        Returns:
            numpy ndarray: Randomly grayscaled image.
        """
        num_output_channels = 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)
