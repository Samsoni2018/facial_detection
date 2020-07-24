import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


# -----------------------------------------------------------------------
# SYNTHETIC DATASET UTILITIES
# -----------------------------------------------------------------------

''' NOTE: there are two different coordinates considered below
    1) "Pixel Coordinates": scene, subimages, and bounding boxes
    are collections of pixels. In this system, the origin of the 
    bounding box is the upper left corner.
    
    2) "Normalized Coordinates: scene, subimages, and bounding boxes
    are objects within space [0,1]x[0,1]. Origin at top left corner
    of the scene. In this case, the origin of a bounding box is its center.
    
    NOTE: [x,y] image coordinates are s.t. x is oriented vertically and 
    y is oriented horizontally (based on indexing of 2d matrices in python)
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
        self.X_shape = np.shape(X)
        self.bb_s = bb_s
        self.y_s = y_s

    def show_with_boxes(self):
        ''' Plots image 'self.X' with bounding boxes
        '''
        fig, ax = plt.subplots()
        ax.imshow(self.X)
        for bb in self.bb_s:
            self.plot_bbox(bb, ax)
        plt.show()

    def plot_bbox(self, bbox, ax):
        ''' Plots bounding box 'bbox' to axis ax
        '''
        corner, subimg_dims = self.norm_bb_to_subimage(bbox)
        rect = patches.Rectangle(tuple(np.flip(corner)-[0.5, 0.5]), *(np.flip(subimg_dims)),
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    def subimage_to_norm_bb(self, corner, subimage_shape):
        ''' Given image in scene , gives bounding box in normalized scene
            coordinates S = [0,1] x [0,1]. Image provided with top left
            corner pixel as well as shape of image in pixels
        '''
        normalized_corner = pixel_to_normalized_coord(self.X_shape, *corner)
        normalized_dim = np.divide(subimage_shape, self.X_shape)
        bb_center = normalized_corner + normalized_dim/2
        bb = BoundingBox(*bb_center, *normalized_dim)

        return bb

    def norm_bb_to_subimage(self, bb):
        ''' Inverse of above function
        '''
        corner = np.multiply(bb.center - bb.dims/2, self.X_shape)
        subimage_shape = np.multiply(bb.dims, self.X_shape)

        return corner, subimage_shape


class ScatteredScene(BoxedScene):
    ''' Scene with images scattered throughout. Bounding boxes of scattered images
        are saved along with the class of each.
    '''
    def __init__(self, scene_shape, subimages, y_s):
        super().__init__(np.zeros(scene_shape), [], y_s)
        self.bb_s = self.scatter_subimages(subimages)
        self.subimages = subimages

    def scatter_subimages(self, subimages, max_iter=10):
        ''' Place subimages within larger scene such that none overlap. Returns
            bounding boxes of the scattered images. Modifieds 'self.X' to reflect
            scattered images.
        '''
        bounding_boxes = []
        overlap = True

        for subimage in subimages:
            subimage_shape = np.asarray(np.shape(subimage))

            # Attempt to place non-overlapping 'image' for up to 'max_iter' trials
            for _ in range(0, max_iter):
                # Scatter subimage randomly within the scene
                corner = self.get_random_corner(subimage_shape)
                bb_candidate = self.subimage_to_norm_bb(corner, subimage_shape)
                if bounding_boxes != []:
                    # Check for overlap with all other bboxes
                    for bb in bounding_boxes:
                        overlap = box_overlap_test(bb_candidate, bb)
                        if overlap:
                            break
                else:
                    # First bbox doesn't overlap with anything
                    overlap = False
                if not overlap:
                    # Successfully found non-overlapping bb
                    bounding_boxes.append(bb_candidate)
                    self.insert_subimage(corner, subimage)
                    break

        # assert len(bounding_boxes) == len(subimages)   # O.w. the process failed
        return bounding_boxes

    def get_random_corner(self, subimage_shape):
        ''' Returns random pixel interpreted as top left corner of
            the image. Pixel chosen such that small image can fit
            inside larger scene.
        '''
        buffer = subimage_shape-1
        c_x = np.random.randint(0, self.X_shape[0]-buffer[0])
        c_y = np.random.randint(0, self.X_shape[1]-buffer[1])

        return [c_x, c_y]

    def insert_subimage(self, corner, subimage):
        ''' Inserts subimage into scene 'self.X'. Corner defines upper
            left corner of the subimage.
        '''
        subimage_shape = np.shape(subimage)
        self.X[corner[0]:corner[0]+subimage_shape[0],
        corner[1]:corner[1]+subimage_shape[1]] = subimage

def box_overlap_test(bb_1, bb_2):
    ''' Checks to see if two bounding boxes overlap
    '''
    center_diff_mags = np.abs(bb_1.center - bb_2.center)
    thresholds = (bb_1.dims + bb_2.dims)/2
    bool_test = np.all([center_diff_mags < thresholds])

    return bool_test


def pixel_to_normalized_coord(scene_shape, p_x, p_y):
    ''' Returns top left of pixel (p_x, p_y) in normalized coordinate
        system
    '''
    c_x = p_x / scene_shape[0]
    c_y = p_y / scene_shape[1]

    return [c_x, c_y]

# -----------------------------------------------------------------------
# SAMSONI ROTATION AND SCALING FUNCTIONS
# -----------------------------------------------------------------------
def img_rotation(img):
    return


def img_scaling(img):
    return

# -----------------------------------------------------------------------
# CREATE SYNTHETIC DATASET
# -----------------------------------------------------------------------

# -------------------------------
# LOAD + BATCH MNIST DATA
# -------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# -------------------------------
# CREATE OD DATASET
# -------------------------------
images_per_scene = 2
scene_shape = [100, 100]
trainset_size = 10
# testset_size = 1000

# ----- Training set ------------------------
synthetic_train = []
for _ in range(0, trainset_size):
    random_indices = np.random.randint(0, np.shape(x_train)[0], images_per_scene)
    images = [x_train[index] for index in random_indices]
    labels = [y_train[index] for index in random_indices]
    # =========== HERE IS WHERE YOU CAN ADD ROTATIONS AND SCALING =================

    # =============================================================================
    scene = ScatteredScene(scene_shape, images, labels)
    scene.show_with_boxes()
    synthetic_train.append(scene)

# =========== HERE WE NEED TO SAVE THE DATA TO FILE ===========================

# =============================================================================


# # ----- Test set ------------------------
# synthetic_test = []
# for _ in range(0, testset_size):
#     random_indices = np.random.randint(0, np.shape(x_test)[0], images_per_scene)
#     images = [x_test[index] for index in random_indices]
#     labels = [y_test[index] for index in random_indices]
#     scene = ScatteredScene(scene_shape, images, labels)
#     synthetic_test.append(scene)


