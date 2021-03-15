import numpy as np


def rgb2grey(rgb):
    """
    Converts color image pixels to greyscale

    Keyword arguments:
    rbg -- a color image

    source: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """

    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# should not accelerate while turning
# source: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
available_actions = [[0, 0, 0],  # no action
                     [-1, 0, 0],  # left
                     [-1, 0, 1],  # left+break
                     [1, 0, 0],  # right
                     [1, 0, 1],  # right+break
                     [0, 1, 0],  # accelerate
                     [0, 0, 1]]  # break
