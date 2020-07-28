'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

import image_segmentation.acs_simplified  as acs_simplified
import image_segmentation.acs_basic  as acs_basic
import image_segmentation.acs_gvf  as acs_gvf

from collections import OrderedDict

# Segmentation methods
ACTIVE_CONTOURS_LEVEL_SET_BASIC = 1
ACTIVE_CONTOURS_LEVEL_SET_SIMPLIFIED = 2
ACTIVE_CONTOURS_LEVEL_SET_GVF = 3

methods_dict = OrderedDict()
methods_dict['Basic ACS'] = ACTIVE_CONTOURS_LEVEL_SET_BASIC
methods_dict['Simplified ACS'] = ACTIVE_CONTOURS_LEVEL_SET_SIMPLIFIED
methods_dict['GVF ACS'] = ACTIVE_CONTOURS_LEVEL_SET_GVF

class SegmentationHub(object):
    '''
    The aim of this class is to create an intermediate layer between the
    MainWindow and the segmentation method.

    This class allows to hide which segmentation method we use, and it can be
    selected changing the name of the segmentation method passed on construction.
    '''
    def __init__(self, seg_method):
        '''
        Constructor. The segmentation method is chosen here.

        Args:
            seg_method: Indicates which segmentation method wants to be used.
            Up to know, the possible options (not all of them implemented yet)
            are: ACTIVE_CONTOURS_LEVEL_SET, SNAKE_LEVEL_SET.
        '''
        self.lenEweight = 0.5
        self.shaEweight = 0
        self.seg_method = seg_method
        self.box_side = 150

    def get_segmentation_object(self, method):
        segm_obj = None
        if method == ACTIVE_CONTOURS_LEVEL_SET_SIMPLIFIED:
            print("SegmentationHub: Using ACS Simplified implementation")
            segm_obj = acs_simplified.ActiveContourSegmentation(
                max_its = 200,
                lengthEweight = self.lenEweight,
                shapeEweight = self.shaEweight)

        elif method == ACTIVE_CONTOURS_LEVEL_SET_BASIC:
            print("SegmentationHub: Using ACS Basic implementation")
            segm_obj = acs_basic.ActiveContourSegmentation(
                max_its = 200)

        elif method == ACTIVE_CONTOURS_LEVEL_SET_GVF:
            print("SegmentationHub: Using ACS Basic algorithm with GVF implementation")
            segm_obj = acs_gvf.ActiveContourSegmentation(
                max_its = 80)

        else:
            raise ValueError("Invalid segmentation method selected")

        return segm_obj

    def get_bounding_box(self, image_shape, centroid_xy):
        '''
        Create a bounding box around the center point selected by the user
        The idea is to create a bounding box specified by the top left
        and bottom right corners of the box of size self.box_side.

        This will help us to crop the image and with that, reduce the
        amount of computation the algorithm needs to do.

        Args:
            image_shape: List with [rows, cols] values of the image that
                will be cropped
            centroid_xy: List with the [X,Y] coordinate of the center of the points
                entered by the user

        Returns:
            List with 4 elements. Top left corner X and Y coordinates and the
            Bottom right corner X and Y coordinates
        '''
        origin_rows = int(centroid_xy[1] - (self.box_side / 2.0))
        origin_cols = int(centroid_xy[0] - (self.box_side / 2.0))
        ending_rows = int(origin_rows + self.box_side)
        ending_cols = int(origin_cols + self.box_side)

        if origin_rows < 0:
            origin_rows = 0

        if origin_cols < 0:
            origin_cols = 0

        if ending_rows > image_shape[0] - 1:
            ending_rows = image_shape[0] - 1

        if ending_cols > image_shape[1] - 1:
            ending_cols = image_shape[1] - 1

        return [origin_rows, origin_cols, ending_rows, ending_cols]

    def start_segmentation(self, image_set, image_idx, initial_mask, centroid=None):
        '''
        Run the segmentation method. If the segmentation method is valid and
        implemented, then this function will run the selected segmentation
        for the provided image.

        Args:
            image_set: List of the images to segment.
            image_idx: Image from which the X, Y coordinates belongs to
            initial_mask: Mask of the same size of the input image. The
                initial contour of this mask must contain ones in the interior
                and zeros in the background
            centroid: List with the [X,Y] coordinate of the center of the points
                entered by the user. This argument will be used to crop the image
                around this center. If this argument is not provided, then the entire
                image is used.

        Returns:
           Contour found.
        '''
        assert(image_set.shape[0:2] == initial_mask.shape)

        if not centroid is None and len(centroid):
            bounding_box = self.get_bounding_box(image_set.shape[0:2], centroid)
        else:
            bounding_box = [0, 0, image_set.shape[0], image_set.shape[1]]

        segm_obj = self.get_segmentation_object(self.seg_method)
        if segm_obj is None:
            print("Selection of segmentation method is not valid")
            return False, []

        img = image_set[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3], image_idx, 0]
        initial_mask = initial_mask[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]

        seg_mask, cont, area = segm_obj.run_segm(
            image=img,
            initial_mask=initial_mask)

        if not cont is None and len(cont):
            cont[:,0] = cont[:,0] + bounding_box[1]
            cont[:,1] = cont[:,1] + bounding_box[0]

        return True, cont