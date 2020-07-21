import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


''' NOTE: there are two different coordinates considered below
    1) "Pixel Coordinates": scene, subimages, and bounding boxes
    are collections of pixels. In this system, the origin of the 
    bounding box is the upper left corner.
    
    2) "Normalized Coordinates: scene, subimages, and bounding boxes
    are objects within space [0,1]x[0,1]. In this case, the origin
    of a bounding box is its center
'''

class BoundingBox:
    ''' Box which demarcates an object within a larger scene S = [0,1]x[0,1]
        (bx, by): centerpoint of the box
        (bh, bw): height and width of the box
    '''
    def __init__(self, bx, by, bh, bw):
        self.center = np.asarray([bx, by])
        self.dims = np.asarray([bh, bw])

class BoxedScene:
    ''' X: image which represents the larger scene of interest
        bb_s: list of bounding boxes
        y_s: list of classes for object inside corresponding bb
    '''
    def __init__(self, X, bb_s, y_s):
        self.X = X
        self.bb_s = bb_s
        self.y_s = y_s


class ScatteredMNIST(BoxedScene):
    def __init__(self, big_im_shape, images, y_s):
        super().__init__(np.zeros(*big_im_shape), [], y_s)
        self.X_shape = big_im_shape
        self.X, self.bb_s = self.scatter_images(images)

    def scatter_images(self, images):
        X = np.zeros(*self.new_im_shape)
        bounding_boxes = []

        for image in images:
            img_shape = np.shape(image)
            random_corner = self.get_random_center(img_shape)

        return X

    def get_random_corner(self, small_im_shape):
        ''' Returns random pixel interpreted as top left corner of
            the image
        '''
        buffer = small_im_shape-1
        c_x = np.random.randint(0, self.X_shape[0]-buffer[0])
        c_y = np.random.randint(0, self.X_shape[1]-buffer[1])

        return [c_x, c_y]

def box_overlap_test(bb_1, bb_2):

    return