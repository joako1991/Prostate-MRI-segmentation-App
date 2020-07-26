'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

# Note:
# To save the mask as a MATLAB matrix
# import scipy.io
# scipy.io.savemat('./matrix.mat', mdict={'arr': init_mask})
# Interactive mode for live plotting
# import matplotlib.pyplot as plt
# ax = plt.figure()
# # We set the axis limits
# plt.axis([0, I.shape[1],0,I.shape[0]])
# plt.ion()
# plt.show()
# plt.clf()
# plt.cla()
# # We chose what information we plot
# plt.imshow(I, cmap='gray')
# plt.scatter(idx2[1, :], idx2[0, :], s=1, color='y')
# plt.draw()
# plt.pause(0.001)
import numpy as np
import copy

import os
dir_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(1, dir_path)
import matlab_func as matlab
from gvf import GradientVectorFlow

from ellipse import LsqEllipse

class ActiveContourSegmentation(object):
    '''
    Active contours segmentation algorithm that implements the paper
    "STACS: new active contour scheme for cardiac MR image segmentation"
    with the addition of a Gradient Vector Flow component.
    '''
    def __init__(self, max_its):
        # maximum iterations
        self.max_iter = max_its

        # Every 10 iterations, we check if the contour had changed more than a delta
        self.frequency_check = 15
        # The minimum change is 1 (pixel changed). So any error less than 1 is the same
        # as an error of zero
        self.min_error = 3

        self.set_default_values()

    def set_default_values(self):
        self.starting_lambda_1 = 1.0
        self.starting_lambda_2 = 1.0
        self.starting_lambda_3 = 0.0
        self.starting_beta = 0.0

        self.ending_lambda_1 = 0.5
        self.ending_lambda_2 = 0.5
        self.ending_lambda_3 = 0.0
        self.ending_beta = 0.0

        # Weight of the model matching function (Lambda_1)
        self.lambda_1 = 0

        # Lambda_2: Edgemap weight
        self.lambda_2 = 0

        # Weight of shape energy (lambda_3)
        self.lambda_3 = 0

        # Weight of length energy (Lambda_4)
        # Produces a lot of inestabilities, and almost no benefits, so we remove it
        self.lambda_4 = 0.8 # Good value

        # GVF contribution
        self.beta = 1.5

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
        self.set_default_values()
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
        # We ensure that the image is 2D graylevel
        assert(len(I.shape) == 2)

        # Create a signed distance map (SDF) from mask
        phi = self.mask2phi(init_mask)

        # Main loop
        # Note: no automatic convergence test
        previous_mask = phi
        dt = 0.45

        # We compute the edge map, since it does not change with the time
        # edge_map = - mod(grad(G * image))
        from scipy import ndimage
        filtered_image = ndimage.median_filter(I, 3)

        sigma = np.std(filtered_image)
        kernel = self.get_gaussian_kernel(3, sigma)
        filtered_img = np.array(matlab.conv2d(I, kernel), dtype=np.float)
        grad_img = np.gradient(filtered_img)
        edge_map = (-1) * np.linalg.norm(grad_img, axis=0)
        edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map))
        edge_map = 1 - edge_map
        edge_map[edge_map < 0.25] = 0

        # Compute Gradient Vector flow
        gvf_field = self.get_gvf(edge_map)
        near_area = 1.2

        for its in range(self.max_iter):
            self.update_weights(its)

            # Get the curve's narrow band
            # Since the object is defined by the values where phi >= 0 and the background
            # by the values where phi < 0, idx will have points around the edge where
            # phi == 0
            idx = np.array(np.where((phi <= near_area) & (phi >= -near_area)))

            # Equation 40 from Paper STACS: (M1 - M2) * lambda_1.
            # normalized_grad = grad(phi) / norm(grad(phi))
            normalized_grad = self.compute_gradient_phi(phi)
            # Curvature = divergence(grad(phi) / mod(grad(phi)))
            curvature = self.get_real_curvature(normalized_grad)
            shrink_curvature = curvature[idx[0], idx[1]]
            # model_match = M_1 - M_2
            model_match = self.compute_model_matching_functional(I, idx, phi, self.lambda_1)

            # GVF component
            gvf_factor = self.beta * self.compute_gvf_contribution(phi, curvature, gvf_field)[idx[0], idx[1]]

            # Potential function P (Eq 31)
            potential_func_P = self.lambda_2 * edge_map + \
                self.lambda_3 * np.power(self.get_distance_func(phi, idx), 2) + \
                self.lambda_4

            if len(potential_func_P) > 1:
                grad_pot_func_P = np.array(np.gradient(potential_func_P))
                second_term = self.dot_product_vector_fields(grad_pot_func_P, normalized_grad)[idx[0], idx[1]]
            else:
                second_term = np.zeros(potential_func_P.shape)

            # Main equation: Equation 40
            try:
                dphidt = \
                    (model_match) - \
                    (second_term) - \
                    (np.multiply(potential_func_P[idx[0], idx[1]], shrink_curvature)) + \
                    gvf_factor
            except TypeError:
                import ipdb; ipdb.set_trace()

            # Maintain the CFL condition
            if len(dphidt):
                dt = 0.45 / (np.max(dphidt) + matlab.eps())
            else:
                dt = 0.45 / matlab.eps()

            # Evolve the curve
            phi[idx[0], idx[1]] += dt * dphidt

            # Keep SDF smooth, and avoid the level set to get flat. Taken from the paper of M. Sussman
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

    def update_weights(self, its):
        '''
        Update the lambda weights. This function must be
        executed on each iteration of the algorithm

        Args:
            its: Iteration index. The first call, its will be zero, the second iteration
                its will be 1, the third iteration it will be 2, and so on.
        '''
        self.lambda_1 = self.starting_lambda_1 - ((its / self.max_iter) * (self.starting_lambda_1 - self.ending_lambda_1))
        self.lambda_2 = self.ending_lambda_2 + 0.5 * (self.starting_lambda_2 - self.ending_lambda_2) * (1 + np.cos(its * np.pi / self.max_iter))
        self.lambda_3 = self.starting_lambda_3 + (self.ending_lambda_3 - self.starting_lambda_3) / np.cosh(10.0 * ((its / self.max_iter) - 1))

        # TODO: UPDATE BETA TOO
        # We do not change lambda_4
        # self.lambda_4

    def get_gvf(self, edge_map):
        '''
        Get the normalized Gradient Vector Flow vector field.

        Args:
            edge_map: Edge map of the image for which we want to compute the GVF.

        Returns:
            2D array of the image size, i.e., output.shape = [2, rows, cols].
            For the first indexation level, the element 0 are the vector components
            in the rows direction, and the second one are the vector components in the
            columns direction
        '''
        grad_vector_flow = GradientVectorFlow(0.2, 100)
        u,v = grad_vector_flow.compute_gvf(edge_map)
        mag = np.sqrt(np.power(u,2) + np.power(v,2))
        px = np.divide(v, mag + 1e-10)
        py = np.divide(u, mag + 1e-10)
        return np.array([px, py])

    def compute_gvf_contribution(self, phi, curvature, gvf):
        '''
        Compute the GVF contribution for the optimization problem.
        The Alpha value has been hardcoded in this function to 1

        The implemented equation is:
            output = (Alpha * curvature) - (GVF_map * grad(phi) / (mod(grad(phi)))) * (mod(grad(phi)))

        Args:
            phi: Signed Distance Function of the kth iteration.
            curvature: Curvature of the actual contour, defined as divergence(grad(phi) / (mod(grad(phi))))
            gvf: Gradient Vector Flow map

        Returns:
            2D array of the image size, where each pixel contains the value of
            the GVF contribution
        '''
        self.alpha = 1.0
        grad = np.gradient(phi)
        mag = np.sqrt(np.power(grad[0],2) + np.power(grad[1],2))
        normalized_grad = self.compute_gradient_phi(phi)

        first_factor = self.alpha * np.array(curvature)

        sec_factor = self.dot_product_vector_fields(gvf, normalized_grad)

        return np.multiply(first_factor - sec_factor, mag)

    def dot_product_vector_fields(self, vector_1, vector_2):
        '''
        Compute the dot product between two vector fields
        Each vector is of the shape [2, rows, columns]. The first index
        is for X and Y of the vector field. Rows and columns are the size of
        the input image

        Args:
            vector_1: First input of the dot product
            vector_2: Second input of the dot product

        Returns:
            2D image that corresponds to the dot product of the two vector fields
        '''
        # TODO: FINISHED
        vector_1 = np.array(vector_1)
        vector_2 = np.array(vector_2)
        assert(len(vector_1.shape) == 3 and vector_1.shape[0] == 2)
        assert(len(vector_2.shape) == 3 and vector_2.shape[0] == 2)

        res = np.multiply(vector_1[0], vector_2[0]) + np.multiply(vector_1[1], vector_2[1])
        return res

    def get_distance_func(self, phi, index):
        '''
        Compute the ellipse parameters and the distance of each
        pixel position to this ellipse (Shape prior)

        Args:
            phi: Signed distance function at the kth iteration
            index: List of indexes of the points that are close to the level-set zero
                of the phi function. This will provide a set of points from
                which to compute the ellipse

        Returns:
            2D array of the size of the original image, where each pixel value represents
            the distance from that pixel position to the estimated ellipse. This distance
            is normalized so all the distances values are between 0 and 1.
        '''
        # We need at least 6 points to have an unique solution. If we don't, then
        # we skip this distance function
        if (index[0].shape[0]) < 6:
            return np.zeros(phi.shape)
        index = np.array(index, dtype=np.float)

        index = np.swapaxes(index, 0, 1)
        # The Ellipse module requires to have as first index X and the second
        # index Y, which is the inverse of our code
        index[:, [0,1]] = index[:, [1,0]]

        reg = LsqEllipse().fit(index)
        a,b,c,d,e,f = reg.coefficients

        xv, yv = np.meshgrid(range(phi.shape[1]), range(phi.shape[0]))
        xv_2 = np.power(xv,2)
        yv_2 = np.power(yv,2)
        xv_yv = np.multiply(xv, yv)
        distances = a * xv_2 + b * xv_yv + c * yv_2 + d * xv + e * yv + f

        distances /= (np.max(np.absolute(distances)) + 1)
        return distances

    def get_real_curvature(self, grad):
        '''
        Get the curvature of the level-set function, defined as
            Divergence(gradient / mod(gradient))

        Args:
            grad: Gradient of the Level set function Phi

        Returns:
            2D array of the size of the original image, where each pixel value
            contains the curvature value
        '''
        # TODO: FINISHED AND OPTIMIZED
        assert(len(grad.shape) == 3 and grad.shape[0] == 2)
        grad = np.array(grad)

        grad_x = grad[1]
        der_grad_x = np.gradient(grad_x)[1]
        grad_y = grad[0]
        der_grad_y = np.gradient(grad_y)[0]

        return np.array(der_grad_x + der_grad_y)

    def get_gaussian_kernel(self, kernel_size, sigma):
        '''
        Create a 2D kernel for a normalized Gaussian filter.

        Args:
            kernel_size: Size of the kernel. If kernel_size is 3,
                the output filter would be 3x3
            sigma: Standard deviation of the Gaussian.

        Returns:
            NumPy array of kernel_size * kernel_size, whose elements are the
            components of a normalized Gaussian filter of parameter sigma.
        '''
        # TODO Finished
        X = np.array(np.meshgrid(range(kernel_size), range(kernel_size), indexing='ij'))
        X = X.reshape(2, kernel_size * kernel_size)

        kernel_index = X + int((-0.5) * kernel_size)
        sigma_inv_root = sigma ** (-0.5)
        kernel = sigma_inv_root * np.exp((-0.25 / sigma) * (np.power(kernel_index[0], 2) + np.power(kernel_index[1], 2)))
        return kernel.reshape([kernel_size, kernel_size])

    def compute_gradient_phi(self, phi):
        '''
        Compute the normalized gradient of the level-set function phi

        Args:
            phi: Level set function at the kth iteration

        Returns:
            Vector field of the size of the original image, so its shape
            is [2, rows, cols]
        '''
        # TODO: FINISHED
        grad = np.gradient(phi)
        # Correct implementation
        mod_grad = np.sqrt(np.power(grad[0], 2) + np.power(grad[1], 2))
        mod_grad[mod_grad == 0] = 1
        grad[0] = np.divide(grad[0], mod_grad)
        grad[1] = np.divide(grad[1], mod_grad)
        # # Cheap implementation
        # max_val = np.amax(np.absolute(grad))
        # if max_val != 0:
        #     grad /= max_val
        return np.array(grad)

    def compute_model_matching_functional(self, image, index_range, phi, weight):
        '''
        Compute the model matching contribution to the snake.

        It computes the model M1 and M2, and then its difference. Additionally,
        the result is already multiplied by the weight constant

        The implemented equation is:
            m_i = mean i
            var_i = variance i
            M_i = 0.5 * ln(2*pi*var_i) + (image - m_i)^2 / (2*var_i))
            for i = 1,2

            Output = weight * (M_2 - M_1)

        Args:
            image: Input image from which to compute the models
            index_range: List of indexes of the input image that corresponds to
                an area closer to the level-set zero of the phi function.
            phi: Level-set function at the kth iterations
            weight: Model matching factor contribution

        Returns:
            2D array with the original image size, where each pixel contains
                the value of the model matching contribution.
        '''
        # TODO: FINISHED
        # Find interior and exterior mean
        assert(len(np.array(image).shape) == 2)
        upts = np.array(np.where(phi < 0.0))
        # interior points
        vpts = np.array(np.where(phi >= 0.0))

        M_1 = np.array([])
        M_2 = np.array([])
        if vpts.shape[1]:
            # interior points mean (c1)
            m_1 = np.mean(image[vpts[0], vpts[1]], axis=0)
            var_1 = np.var(image[vpts[0], vpts[1]], axis=0)
            if var_1 != 0:
                M_1 = 0.5 * np.log(2*np.pi*var_1) + (np.power(image[index_range[0], index_range[1]] - m_1, 2) / (2*var_1))
            else:
                M_1 = np.power(image[index_range[0], index_range[1]] - m_1, 2)

        if upts.shape[1]:
            # exterior points mean (c2)
            m_2 = np.mean(image[upts[0], upts[1]], axis=0)
            var_2 = np.var(image[upts[0], upts[1]], axis=0)
            if var_1 != 0:
                M_2 = 0.5 * np.log(2*np.pi*var_2) + (np.power(image[index_range[0], index_range[1]] - m_2, 2) / (2*var_2))
            else:
                M_2 = np.power(image[index_range[0], index_range[1]] - m_2, 2)

        # The right equation comes from Active Contours Without edges paper
        # which is M_2 - M_1 (Discretization and linearization of (9))
        F = []
        if len(M_1) and len(M_2):
            F = M_2 - M_1
        elif len(M_1):
            F = (-1) * M_1
        elif len(M_2):
            F =  M_2

        return F

    ## Helper functions
    # Converts a mask to a SDF (Signed Distance Function)
    def mask2phi(self, init_a):
        return matlab.bwdist(init_a) - matlab.bwdist(1 - init_a) + matlab.im2double(init_a) - 0.5

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

        # np.max returns the maximum value among all of the possibilities.
        # np.maximum gives the max value element wise between two vectors
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
            acum = holes_filled
            # acum = matlab.bwconvhull(holes_filled)
        return acum