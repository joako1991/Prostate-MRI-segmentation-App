# Original MATLAB version:
# LV segmentation from 2D cardiac MRI
# M. R. Avendi, 2014-2015
# Modified and improved to Python by Joaquin Rodriguez: 03-2020

# Note:
# To save the mask as a MATLAB matrix
# import scipy.io
# scipy.io.savemat('./matrix.mat', mdict={'arr': init_mask})

import numpy as np

import image_segmentation.matlab_func as matlab

import copy

class ActiveContourSegmentation(object):
    '''
    Active contours segmentation algorithm that implements the paper
    "STACS: new active contour scheme for cardiac MR image segmentation"
    in a simplified way. Equations are not precisely implemented, but
    it works with less precision, and it is faster than implementing all
    the considerations of the paper
    '''
    def __init__(self, max_its, lengthEweight = 0.5, shapeEweight = 0):
        # maximum iterations
        self.max_iter = max_its
        # Weight of length energy
        self.lengthEweight = lengthEweight
        # Weight of shape engery
        self.shapeEweight = shapeEweight
        # Every 10 iterations, we check if the contour had changed more than a delta
        self.frequency_check = 15
        # The minimum change is 1 (pixel changed). So any error less than 1 is the same
        # as an error of zero
        self.min_error = 3

    def run_segm(self, image, center_x = -1, center_y = -1, radious = -1, initial_mask = None):
        '''
        Segment an image

        If both, initial_mask and the coordinates of the circular initial mask are not valid,
        an exception is thrown. The priority is the initial mask. If it is not empty, it will be
        used, and the other variables will be ignored. If the initial mask is empty or
        None, a circular initialization will be done, using the variables center_x, center_y
        and radious. If any of those variables is not valid, the exception ValueError will be thrown.

        Args:
            image: Image to be segmented
            center_x (Optional): X Pixel coordinate of the center of a circular initial mask.
                It will be used only if initial_mask is empty
            center_y (Optional): Y Pixel coordinate of the center of a circular initial mask.
                It will be used only if initial_mask is empty
            radious (Optional): Radious in pixels, of the circular initial mask.
                It will be used only if initial_mask is empty
            initial_mask (Optional): Initial mask to be used. It should have 0s in the background area
                and 1s in the object area.

        Returns:
            Final mask of the segmentation, the contour of that mask and its area, measured in pixels.
            If more than one contour is detected from the mask, or if the mask has only background values,
            the output mask will be filled by zeros, and the contour variable will be empty.
        '''
        output_contour = np.array([])

        # Check given data
        if image is None or not image.shape:
            raise ValueError("Invalid image provided")

        # Initialization
        initial_mask = np.array(initial_mask)
        if not initial_mask is None and initial_mask.any():
            init_mask = initial_mask
        elif (center_x > 0 and center_y > 0 and radious > 0) and \
            ((center_x - radious) >= 0) and \
            ((center_x - radious) >= 0) and \
            ((center_x + radious) < image.shape[1]) and \
            ((center_y + radious) < image.shape[0]):
            ## Init mask is a circle filled of 1s inside and 0s inside
            init_mask = matlab.create_circular_mask(image.shape[0], image.shape[1], center_x, center_y, radious)
        else:
            raise ValueError("No initial mask provided nor valid parameters to build one")

        # run segmentation
        [auto_seg1, phi] = self.active_cont_segm(image, init_mask)

        # clean segmentation, remove islands and small contours
        auto_seg2 = self.clean_segs(auto_seg1)

        if auto_seg2.any():
            img = auto_seg2 * (255.0 / np.max(auto_seg2))
            contours = matlab.get_contours(img)

            if contours.shape[0]:
                if contours.shape[0] == 1:
                    output_contour = contours[0]
                else:
                    print("Segmentation failed. More than 1 contour founded")
                    auto_seg2 = np.zeros(auto_seg2.shape)
        else:
            print("Segmentation failed. The segmentation mask is filled with zeros")

        contour_area = matlab.get_bin_image_area(auto_seg2)
        return auto_seg2, output_contour, contour_area

    def active_cont_segm(self, I, init_mask):
        '''
        Segmentation algorithm. It will evolve the initial mask
        towards the border of the closest object.

        Args:
            I: 2D Input image to segment (gray-level)
            init_mask: Initial mask, with zeros in the background and 1s
                in the interior of the object

        Returns:
            Segmentation mask, and final level-set function
        '''
        # We ensure that the image is 2D graylevel, and we convert it into double.
        I = np.array(I).astype(np.double, order='C', casting='unsafe')
        assert(len(I.shape) == 2)

        # Create a signed distance map (SDF) from mask
        phi = self.mask2phi(init_mask)
        phi_prior = phi

        # Main loop
        # Note: no automatic convergence test
        previous_mask = phi
        for its in range(self.max_iter):
            # Get the curve's narrow band
            idx = np.where((phi <= 1.2) & (phi >= -1.2))

            # Find interior and exterior mean
            upts = np.array(np.where(phi <= 0.0))
            # interior points
            vpts = np.array(np.where(phi > 0.0))

            # exterior points mean (c2)
            u = np.sum(I[upts[0], upts[1]]) / (upts.shape[1] + matlab.eps())

            # interior points mean (c1)
            v = np.sum(I[vpts[0], vpts[1]]) / (vpts.shape[1] + matlab.eps())

            # exterior mean
            # TODO: Check why is a difference and not a sum (is lambda_1 = 1 and lambda_2 = -1?)
            F = np.power(I[idx[0], idx[1]] - u, 2) - np.power(I[idx[0], idx[1]] - v, 2)

            # region-based force from image information
            # force from curvature penalty
            curvature = self.get_curvature(phi, idx)

            # Derivative of energy function for prior shape
            dEdl = (-1) * (phi[idx[0], idx[1]] - phi_prior[idx[0], idx[1]])

            # Note that dPhi/dt = - dE / dPhi
            # Gradient descent to minimize energy
            if len(F):
                # TODO: Check better the performance. Dividing by the maximum value of F is reasonable
                # to do since it has really huge values, but it reduces its contribution to dphidt, which is not so good.
                # (It is like adding an scaling factor lambda too little). Setting the scaling factor to 1 diminishes
                # that influence
                max_avg_val = 1
                # max_avg_val = np.max(abs(F))
                dphidt = (F / max_avg_val) + (self.lengthEweight * curvature) + (self.shapeEweight * dEdl)
                # Maintain the CFL condition
                dt = 0.45 / (np.max(dphidt) + matlab.eps())

                # Evolve the curve
                phi[idx[0], idx[1]] += dt * dphidt
            # Keep SDF smooth
            phi = self.sussman(phi, 0.5)

            # We check how much the contour changed between the actual one and the last
            # check
            if its and (its % self.frequency_check) == 0:
                prev_seg = np.uint8(previous_mask <= 0)
                seg = np.uint8(phi <= 0)
                error = np.sum(abs(seg - prev_seg))
                if error < self.min_error:
                    break
                # we create a copy of the current mask
                previous_mask = phi + 0

        # Make mask from Signed Distance Function
        # Get mask from levelset
        seg = np.uint8(phi <= 0)
        return seg, phi

    ## Helper functions
    # Converts a mask to a SDF (Signed Distance Function)
    def mask2phi(self, init_a):
        return matlab.bwdist(init_a) - matlab.bwdist(1 - init_a) + matlab.im2double(init_a) - 0.5

    # Compute curvature along SDF
    def get_curvature(self, phi, idx):
        dimx = phi.shape[1]
        dimy = phi.shape[0]

        # Get subscripts
        y = idx[0]
        x = idx[1]

        # Get subscripts of neighbors
        ym1 = y - 1
        xm1 = x - 1
        yp1 = y + 1
        xp1 = x + 1

        # Bounds checking
        # If the indexes are negative or greater than the array shape,
        # then we bound them to the limits
        # ym1 : Y minus 1
        # xm1 : X minus 1
        # yp1 : Y plus 1
        # xp1 : X plus 1
        ym1[ym1 < 0] = 0
        xm1[xm1 < 0] = 0
        yp1[yp1 > (dimy - 1)] = dimy - 1
        xp1[xp1 > (dimx - 1)] = dimx - 1

        # Get central derivatives of SDF at x,y
        phi_x  = phi[y, xp1] - phi[y, xm1]
        phi_y  = phi[yp1, x] - phi[ym1, x]
        phi_xx = phi[y, xm1] - 2 * phi[y, x] + phi[y, xp1]
        phi_yy = phi[ym1, x] - 2 * phi[y, x] + phi[yp1, x]
        phi_xy = (-0.25 * phi[ym1, xm1]) - (0.25 * phi[yp1, xp1]) + (0.25 * phi[ym1, xp1]) + (0.25 * phi[yp1, xm1])
        phi_x2 = np.power(phi_x, 2)
        phi_y2 = np.power(phi_y, 2)

        # Compute curvature (Kappa)
        # TODO: Check for zero division
        tmp_1 = np.multiply( \
                    phi_x2, phi_yy) + \
                    np.multiply(phi_y2, phi_xx) - \
                    (2 * np.multiply(np.multiply(phi_x, phi_y), phi_xy))

        tmp_2 = phi_x2 + phi_y2 + matlab.eps()
        curvature = np.divide(tmp_1, tmp_2)
        return curvature

    # Level-set re-initialization by the sussman method
    def sussman(self, D, dt):
        # D is phi_n
        # Forward/backward differences
        # Backward
        a = D - self.shiftR(D)
        # Forward
        b = self.shiftL(D) - D
        # Backward
        c = D - self.shiftD(D)
        # Forward
        d = self.shiftU(D) - D

        # a+ and a-
        # We need to do a copy of each array. If not, python will
        # do a shallow copy, which is basically a pointer, so
        # d_n and d_p are the same, and without the copy, the arrays
        # will be after these lines equal to zero
        a_p = copy.deepcopy(a)
        a_n = copy.deepcopy(a)
        b_p = copy.deepcopy(b)
        b_n = copy.deepcopy(b)
        c_p = copy.deepcopy(c)
        c_n = copy.deepcopy(c)
        d_p = copy.deepcopy(d)
        d_n = copy.deepcopy(d)
        a_p[a < 0] = 0
        a_n[a > 0] = 0
        b_p[b < 0] = 0
        b_n[b > 0] = 0
        c_p[c < 0] = 0
        c_n[c > 0] = 0
        d_p[d < 0] = 0
        d_n[d > 0] = 0

        dD = np.zeros(D.shape)
        D_neg_ind = np.where(D < 0)
        D_pos_ind = np.where(D > 0)

        # np.max returns the maximum value among all of the possibilities. np.maximum gives the max value element wise
        # between two vectors
        dD[D_pos_ind[0], D_pos_ind[1]] = np.sqrt(
            np.maximum(np.power(a_p[D_pos_ind[0], D_pos_ind[1]], 2), np.power(b_n[D_pos_ind[0], D_pos_ind[1]], 2)) + \
            np.maximum(np.power(c_p[D_pos_ind[0], D_pos_ind[1]], 2), np.power(d_n[D_pos_ind[0], D_pos_ind[1]], 2))) - 1

        dD[D_neg_ind[0], D_neg_ind[1]] = np.sqrt(
            np.maximum(np.power(a_n[D_neg_ind[0], D_neg_ind[1]], 2), np.power(b_p[D_neg_ind[0], D_neg_ind[1]], 2)) + \
            np.maximum(np.power(c_n[D_neg_ind[0], D_neg_ind[1]], 2), np.power(d_p[D_neg_ind[0], D_neg_ind[1]], 2))) - 1
        return D - np.multiply(np.multiply(dt, self.sussman_sign(D)), dD)

    # Down shift function
    def shiftD(self, M):
        return np.transpose(self.shiftR(np.transpose(M)))

    # Left shift function
    def shiftL(self, M):
        return np.hstack((M[:, 1:M.shape[1]], M[:, M.shape[1] - 1].reshape(M.shape[0], 1)))

    # Right shift function
    def shiftR(self, M):
        return np.hstack((M[:, 0].reshape(M.shape[0],1), M[:, 0:(M.shape[1] - 1)]))

    # Up shift function
    def shiftU(self, M):
        return np.transpose(self.shiftL(np.transpose(M)))

    def sussman_sign(self, D):
        square_root = np.sqrt(np.power(D, 2) + 1)
        return np.divide(D, square_root)

    # Clean the segmentation result. If there are several images, this function cleans all of them
    def clean_segs(self, t_yLVh):
        t_yLVh = np.array(t_yLVh)
        t_yLV_clean = np.zeros(t_yLVh.shape)
        if len(t_yLVh.shape) == 3:
            for k in range(t_yLVh.shape[2]):
                t_yLV_clean[:, :, k] = self.clean_image(t_yLVh[:, :, k], t_yLV_clean[:, :, k])
        else:
            t_yLV_clean[:, :] = self.clean_image(t_yLVh[:, :], t_yLV_clean[:, :])

        return t_yLV_clean

    # Clean a single segmentation image
    def clean_image(self, BW, acum):
        BW2 = matlab.imfill(BW, 'holes')
        # It replaced bwconncomp(BW2, 4) and regionprops(cc, 'Area')
        area = matlab.bin_conn_comp_prop(BW2, 4, 'Area')
        if area:
            # If we have one area element only, its index will be 0,
            # but the connected label area will have a label 1, so we
            # add one to the area index to make them equal, since zero
            # means edges for connex labeling
            max_area_idx = np.argmax(area) + 1

            temp = matlab.ismember(matlab.labelmatrix(BW2), max_area_idx)
            holes_filled = matlab.imfill(temp, 'holes')
            acum = matlab.bwconvhull(holes_filled)
        return acum