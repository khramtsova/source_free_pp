
import random

import numpy as np
import re
from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import random
from dataclasses import dataclass
from typing import Union
from torchvision import transforms


@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    shear: MinMax = MinMax(.0, .3)
    translate: MinMax = MinMax(0, 10)  # different from uniaug: MinMax(0,14.4)
    rotate: MinMax = MinMax(0, 30)
    solarize: MinMax = MinMax(0, 256)
    posterize: MinMax = MinMax(0, 4)  # different from uniaug: MinMax(4,8)
    enhancer: MinMax = MinMax(.1, 1.9)
    cutout: MinMax = MinMax(.0, .2)



def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def __repr__(self):
        return '<' + self.name + '>'

    def pil_transformer(self, probability, level):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)


def _rotate_impl(pil_img, level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, min_max_vals.rotate.max)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)



def _posterize_impl(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, min_max_vals.posterize.max - min_max_vals.posterize.min)
    return ImageOps.posterize(pil_img, min_max_vals.posterize.max - level)


def _shear_x_impl(pil_img, level):
    """Applies PIL ShearX to `pil_img`.
  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))



def _shear_y_impl(pil_img, level):
    """Applies PIL ShearY to `pil_img`.
  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def _translate_x_impl(pil_img, level):
    """Applies PIL TranslateX to `pil_img`.
  Translate the image in the horizontal direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def _translate_y_impl(pil_img, level):
    """Applies PIL TranslateY to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))



def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    level = int_parameter(level, 10)
    w = pil_img.width
    h = pil_img.height
    cropped = pil_img.crop((level, level, w - level, h - level))
    resized = cropped.resize((w, h), interpolation)
    return resized



def _solarize_impl(pil_img, level):
    """Applies PIL Solarize to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, min_max_vals.solarize.max)
    return ImageOps.solarize(pil_img, 256 - level)


def _enhancer_impl(enhancer, minimum=None, maximum=None):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        mini = min_max_vals.enhancer.min if minimum is None else minimum
        maxi = min_max_vals.enhancer.max if maximum is None else maximum
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


def _mirrored_enhancer_impl(enhancer, minimum=None, maximum=None):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        mini = min_max_vals.enhancer.min if minimum is None else minimum
        maxi = min_max_vals.enhancer.max if maximum is None else maximum
        assert mini == 0., "This enhancer is used with a strength space that is mirrored around one."
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        if random.random() < .5:
            v = -v
        return enhancer(pil_img).enhance(1. + v)

    return impl


def CutoutDefault(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v <= 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def get_transform_dict(augmentation_space, num_strengths, custom_augmentation_space_augs):
    global min_max_vals, PARAMETER_MAX
    assert num_strengths > 0
    PARAMETER_MAX = num_strengths - 1
    min_max_vals = MinMaxVals()

    cutout = TransformT('Cutout',
                        lambda img, l: CutoutDefault(img, int_parameter(l, img.size[0] * min_max_vals.cutout.max)))

    mirrored_color = TransformT('Color', _mirrored_enhancer_impl(ImageEnhance.Color))
    mirrored_contrast = TransformT('Contrast', _mirrored_enhancer_impl(ImageEnhance.Contrast))
    mirrored_brightness = TransformT('Brightness', _mirrored_enhancer_impl(
        ImageEnhance.Brightness))
    mirrored_sharpness = TransformT('Sharpness', _mirrored_enhancer_impl(ImageEnhance.Sharpness))

    color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
    ohl_color = TransformT('Color', _enhancer_impl(ImageEnhance.Color, .3, .9))
    contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
    brightness = TransformT('Brightness', _enhancer_impl(
        ImageEnhance.Brightness))
    sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
    contour = TransformT(
        'Contour', lambda pil_img, level: pil_img.filter(ImageFilter.CONTOUR))
    detail = TransformT(
        'Detail', lambda pil_img, level: pil_img.filter(ImageFilter.DETAIL))
    edge_enhance = TransformT(
        'EdgeEnhance', lambda pil_img, level: pil_img.filter(ImageFilter.EDGE_ENHANCE))
    sharpen = TransformT(
        'Sharpen', (lambda pil_img, level: pil_img.filter(ImageFilter.SHARPEN)))
    max_ = TransformT(
        'Max', lambda pil_img, level: pil_img.filter(ImageFilter.MaxFilter))
    min_ = TransformT(
        'Min', lambda pil_img, level: pil_img.filter(ImageFilter.MinFilter))
    median = TransformT(
        'Median', lambda pil_img, level: pil_img.filter(ImageFilter.MedianFilter))
    gaussian = TransformT(
        'Gaussian', lambda pil_img, level: pil_img.filter(ImageFilter.GaussianBlur))

    crop_bilinear = TransformT('CropBilinear', _crop_impl)

    solarize = TransformT('Solarize', _solarize_impl)

    translate_y = TransformT('TranslateY', _translate_y_impl)

    translate_x = TransformT('TranslateX', _translate_x_impl)

    identity = TransformT('identity', lambda pil_img, level: pil_img)
    flip_lr = TransformT(
        'FlipLR',
        lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    flip_ud = TransformT(
        'FlipUD',
        lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
    # pylint:disable=g-long-lambda
    auto_contrast = TransformT(
        'AutoContrast',
        lambda pil_img, level: ImageOps.autocontrast(
            pil_img))
    equalize = TransformT(
        'Equalize',
        lambda pil_img, level: ImageOps.equalize(
            pil_img))
    invert = TransformT(
        'Invert',
        lambda pil_img, level: ImageOps.invert(
            pil_img))
    # pylint:enable=g-long-lambda
    blur = TransformT(
        'Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
    smooth = TransformT(
        'Smooth',
        lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
    rotate = TransformT('Rotate', _rotate_impl)
    posterize = TransformT('Posterize', _posterize_impl)
    shear_x = TransformT('ShearX', _shear_x_impl)
    shear_y = TransformT('ShearY', _shear_y_impl)
    custom_augmentation_space_augs_mapping = {
        'identity': identity,
        'auto_contrast': auto_contrast,
        'equalize': equalize,
        'rotate': rotate,
        'solarize': solarize,
        'color': color,
        'posterize': posterize,
        'contrast': contrast,
        'brightness': brightness,
        'sharpness': sharpness,
        'shear_x': shear_x,
        'shear_y': shear_y,
        'translate_x': translate_x,
        'translate_y': translate_y,
        # sample_pairing,
        'blur': blur,
        'invert': invert,
        'flip_lr': flip_lr,
        'flip_ud': flip_ud,
        'cutout': cutout,
        'crop_bilinear': crop_bilinear,
        'contour': contour,
        'detail': detail,
        'edge_enhance': edge_enhance,
        'sharpen': sharpen,
        'max_': max_,
        'min_': min_,
        'median': median,
        'gaussian': gaussian
    }