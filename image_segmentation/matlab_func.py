'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

import skimage
import skimage.morphology
from skimage import measure
from matplotlib.path import Path

import scipy
import scipy.ndimage

import numpy as np

def ismember(A, B):
    '''
    Inspired in https://stackoverflow.com/questions/25923027/matlab-ismember-function-in-python
    and https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1
    It returns a vector of the shape of A. Each element in the output will
    contain a 1 (true) if the corresponding element in A is found in B. It will
    contain 0 (False) otherwise

    Args:
        A: Array with the elements to test against B
        B: Reference array
    Returns:
        Array with the shape of A.
    '''
    A = np.array(A)
    B = np.array(B)
    return np.array([1 if x in B else 0 for x in np.nditer(A)]).reshape(A.shape)

def bin_conn_comp_prop(image, conn, prop):
    '''
    It will return a mesuare of a property of image, which should be a binary
    image. It will create an image with the label connected regions of the
    given image. Two pixels are connected when they are neighbors and have the
    same value. The connectivity limit depends on the dimensions of the image.
    If it is 2D, then conn parameter can be 4 (up, down, left, right),
    or 8 (up, down, left, right, and the 4 diagonals).
    For a more full reference to how connected labels can be done, refer to
    the following link.
        https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    Then, using the skimage module, we will extract the properties of the
    labeled region, and finally, we will extract only the requested property.

    Args:
        image: Binary image from which we want to extract the properties.
        conn: Desired connectivity for the labeling.
        prop: Property to be extracted. It can be writting in either, lower
            case, upper case, or a mix.
              For now, we only request Area. For a full reference about the
            possible properties that can be extracted, refer to
                https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

    Returns:
        List with all the values of the requested property. In the case of
        area property, if there are several closed sections (holes) then
        the output list will contain a list where each element represents
        the area of each hole
    '''
    # Connectivity = 1 considers pixels connected horizontally and vertically,
    # not diagonally.
    # Connectivity = 2 means that it also includes pixels connected diagonally
    label_img = skimage.measure.label(image, connectivity=conn / 4)
    props = skimage.measure.regionprops(label_img)
    # inspired in https://stackoverflow.com/questions/2682012/how-to-call-same-method-for-a-list-of-objects
    return list(map(lambda x : x[prop.lower()], props))

def labelmatrix(img):
    '''
    It creates an image with the label connected regions of the
    given image. Two pixels are connected when they are neighbors and have the
    same value. The connectivity limit depends on the dimensions of the image.
    If it is 2D, then conn parameter can be 4 (up, down, left, right),
    or 8 (up, down, left, right, and the 4 diagonals). This function will fix
    the connectivity to the dimensions of the matrix.
    For a more full reference to how connected labels can be done, refer to
    the following link.
        https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

    Args:
        img: Binary image to be labeled

    Returns:
        Labeled image
    '''
    return skimage.measure.label(img, connectivity=img.ndim)

def get_bin_image_area(BW):
    '''
    Compute the biggest labeled area that it is not background

    Args:
        BW: Binary image from which are going to do a connex labeling,
            and determine the area of the biggest label.

    Returns:
        Maximum area found
    '''
    BW2 = imfill(BW, 'holes')
    # It replaced bwconncomp(BW2, 4) and regionprops(cc, 'Area')
    area = bin_conn_comp_prop(BW2, 4, 'Area')
    max_area = 0
    if area:
        # If we have one area element only, its index will be 0,
        # but the connected label area will have a label 1, so we
        # add one to the area index to make them equal, since zero
        # means edges for connex labeling
        max_area = np.max(area)
    return max_area

def imfill(image, option=''):
    '''
    Fill the holes in binary objects. If the input image has a set of 1s that
    surrounds a set of 0s, then those 0s will be converted into 1s, filling
    the hole in the image. For a further reference of how it works, refer to:
        https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.morphology.binary_fill_holes.html

    Args:
        image: Binary object with holes
        option: Not used. Kept because the original function in Matlab (more
            powerful one) has this option too. This python version only has
            the option holes.

    Returns:
        Transformation of the initial image input where holes have been filled.
        The output image will have values 0 and 255 only
    '''
    # scipy give us a matrix of True False values, so we cast them into
    # uint8 (0 or 1) and then we multiply to have values between 0 and 255
    # BW = np.array(scipy.ndimage.morphology.binary_fill_holes(image)).astype(np.uint8, order='C', casting='unsafe')
    import cv2
    img = np.array(image, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    BW = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    BW = cv2.dilate(BW, kernel)
    BW = cv2.erode(BW, kernel)
    BW = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    BW = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    BW = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    BW *= 255
    return BW

def bwconvhull(bin_image):
    '''
    Compute the convex hull image of a binary image.
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    For a more complete reference, refer to:
        https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.convex_hull_image

    Args:
        bin_image: Binary image

    Returns:
        Binary image with pixels in convex hull set to 1, and the rest to 0.
    '''
    return skimage.morphology.convex_hull_image(bin_image).astype(np.uint8, order='C', casting='unsafe')

def bwdist(BW):
    '''
    Exact euclidean distance transform.

    Args:
        BW: Binary image from which compute the transform

    Returns:
        Image representing the eucidean distance transformation of the input
        image
    '''
    bin_mask = np.array(BW == 0).astype(np.uint8, order='C', casting='unsafe')
    return scipy.ndimage.morphology.distance_transform_edt(bin_mask)

def im2double(input_img):
    '''
    Convert a integer valued image into a doubled valued image
    '''
    return np.array(input_img).astype(np.double, order='C', casting='unsafe')

def eps():
    '''
    The smallest representable positive number such that 1.0 + eps != 1.0.
    Type of eps is an appropriate floating point type. This Episilon
    is given for a np.double type number.
    '''
    return np.finfo(np.double).eps

def ind2sub(array_shape, ind):
    '''
    Convert linear indices to subscripts

    Args:
        array_shape: Vector with two elements, where the first element
            specifies the number of rows and the second one specifies the
            number of columns.
        ind: Vector with the indices to be converted

    Returns:
        Arrays row and col containing the equivalent row and column subscripts
        corresponding to the linear indices ind for a matrix of size array_shape.
    '''
    # Taken from https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python
    ind = np.array(ind)
    rows = (ind / array_shape[1]).astype('int')
    cols = (ind % array_shape[1]).astype('int')
    return rows, cols

def sub2ind(array_shape, rows, cols):
    '''
    Convert subscripts to linear indices

    Args:
        array_shape: Vector with two elements, where the first element
            specifies the number of rows and the second one specifies the
            number of columns.
        rows: Vector with the list of rows to be converted
        cols: Vector with the list of columns to be converted

    Returns:
        Linear indices ind corresponding to the row and column subscripts
            in row and col for a matrix of size array_shape
    '''
    ind = np.array(rows * array_shape[1] + cols, dtype=np.uint32)
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def create_random_mask(rows, columns, polygon_points):
    '''
    Create a mask based only in the provided points that belongs to a polygon boundary.

    Args:
        rows: Amount of rows of the output image
        columns: Amount of columns of the output image
        polygon_points: List of 2D points that belongs to the polygon boundaries

    Returns:
        Image of rows x columns size where the pixels in the interior of the polygon
        have a value of 1, and the outside, 0
    '''
    # Inspired by https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    mask = np.zeros([rows, columns])
    if len(polygon_points):
        x, y = np.meshgrid(np.arange(columns), np.arange(rows)) # make a canvas with coordinates
        x = x.flatten()
        y = y.flatten()
        points = np.vstack((x,y)).T

        p = Path(polygon_points) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(rows, columns)

    return mask

def create_circular_mask(rows, columns, center_x, center_y, radious):
    '''
    Create a synthetic mask image that represents a circle with center
    (center_x, center_y) and radious given.

    Args:
        rows: Amount of rows of the output image
        columns: Amount of columns of the output image
        center_x: X coordinate of the center of the circle
        center_y: Y coordinate of the center of the circle

    Returns:
        [rows, columns] binary image with value 255 inside the circle and
        0s outside it
    '''
    assert(radious > 0)
    mask = np.zeros([rows, columns])
    # The parametric equation of a circle is:
    #       delta_X = X - X0 = radious * cosine(theta)
    #       delta_Y = Y - Y0 = radious * sine(theta)
    # Since range goes from 0 until X - 1, in order to get the case
    # of asin(1), we do the range from 0 until radious + 1
    delta_y = np.array(range(radious + 1))
    theta = np.arcsin(delta_y / radious)

    delta_x = np.array((center_x + radious * np.cos(theta)), dtype=np.uint32) - center_x

    x_idx = np.array([], dtype=np.uint32)
    y_idx = np.array([], dtype=np.uint32)
    for i in range(len(delta_y)):
        # In mostly cases, we need to dupplicate the entries except in two
        # cases: when we are over the Y axis and when we are on the X axis
        # Those cases corresponds to delta_x or delta_y being zero
        if delta_x[i]:
            tmp_x = np.arange(center_x - delta_x[i], center_x + delta_x[i] + 1)
            pos_tmp_y = center_y + np.ones(len(tmp_x), dtype=np.uint32) * delta_y[i]
            neg_tmp_y = center_y - np.ones(len(tmp_x), dtype=np.uint32) * delta_y[i]

        else:
            tmp_x = np.array([center_x])
            pos_tmp_y = np.array([center_y + delta_y[i]])
            neg_tmp_y = np.array([center_y - delta_y[i]])

        if delta_y[i]:
            # Since we have +delta_y and -delta_y, and the X values are the same, then
            # we dupplicate the X array elements and we create the Y array based on the neg and pos arrays
            tmp_x = np.concatenate((tmp_x, tmp_x), axis=None)
            tmp_y = np.concatenate((pos_tmp_y, neg_tmp_y), axis=None)
        else:
            # Just to avoid confusion, we put tmp_x = tmp_x,
            # because if delta_y is zero, then tmp_x does not need to
            # be duplicated
            tmp_x = tmp_x
            tmp_y = pos_tmp_y

        # We add the new values to the final arrays
        x_idx = np.concatenate((x_idx, tmp_x), axis=None)
        y_idx = np.concatenate((y_idx, tmp_y), axis=None)

    # For the mask filled with zeros, we fill only the interior of circles
    # with 1s
    mask[y_idx, x_idx] = 1
    return mask

def get_contours(input):
    '''
    Get an array of the contours of the holes contained in the given image.

    For a better reference about how the contours are detected, refer to the
    following link:
        https://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html#find-contours
    Args:
        input: Input image

    Returns:
        3D array. The first index is to choose one contour, the other two
        indexes contains the coordinates of the point of the chosen contour.
    '''
    raw = measure.find_contours(input, level=0.5)
    contours = []
    for cap in raw:
        # determine contours
        x = [x[1] for x in cap]
        y = [x[0] for x in cap]
        contours.append(list(zip(x, y)))

    return np.array(contours)

def conv2d(image, kernel):
    '''
    Make the convolution of the image with the given kernel. This function
    is made for 2D signals, not for 1D signals.

    The algorithm implemented is called Add-shift-multiply. It does not reduce
    the image size once applied

    Args:
        image: Input image
        kernel: Input kernel (filter)

    Returns:
        Result of doing the convolution of image and the given kernel
    '''
    np_img = np.array(image, dtype=np.float)
    if len(np_img.shape) == 2:
        np_img = np_img.reshape(np_img.shape + (1,))
    np_h = np.array(kernel)

    img_rows = np_img.shape[0]
    img_cols = np_img.shape[1]
    bands = np_img.shape[2]

    h_rows = np_h.shape[0]
    h_cols = np_h.shape[1]
    A = np.zeros([img_rows + h_rows - 1, img_cols + h_cols - 1, bands])

    for m in range(h_rows):
        for n in range(h_cols):
            A[m:m+img_rows,n:n+img_cols,:] += (np_img * np_h[m,n])

    center_x = int(h_rows / 2.0)
    center_y = int(h_cols / 2.0)

    out_img = np.array(A[center_x:center_x + img_rows, center_y:center_y + img_cols, :], dtype=np.int)
    return out_img.reshape(image.shape)