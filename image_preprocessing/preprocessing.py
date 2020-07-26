'''
This file contains a class that are mainly used to do a pre processing step of the
image to be analyzed. The idea of this preprocessing is to make the segmentation step
easier / simpler. It might not be neccesary to do the preprocessing, but as a first
idea, needs to be considered.

The implemented class contains a flag in the constructor that enables or disables the
preprocessing step, so in order to disable, we only change a True by a False on construction
and that's it.
'''

import copy
import os

import numpy as np
import cv2
from scipy import ndimage
from skimage.filters import threshold_otsu

class ImagePreprocessing(object):
    '''
    Class that will make some treatments to the provided image.
    It can be enabled or disabled. If disabled, the output image will
    be directly the input.
    '''
    def __init__(self, is_preprocessing_enabled):
        '''
        Constructor

        Args:
            is_preprocessing_enabled: Boolean flag. If it is True, the module will
                process the image when preprocess_image method is called.
        '''
        self.enabled = is_preprocessing_enabled

        # We load a reference MRI image for the histogram transfer
        dirpath = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(dirpath, 'reference_mri.png')
        self.reference_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if self.reference_img is None:
            raise IOError("Cannot open the reference image located at {fp}".format(fp=filepath))

    def is_module_enabled(self):
        return self.enabled

    def set_enable(self, val):
        self.enabled = val

    def preprocess_image(self, img, processes):
        '''
        Apply some Image processing to the input in order to generate an easy
        to segment image.

        Args:
            img: Input image to process
            processes: List of tuples that contains the desired processes that want
                to be applied to the input image. The first element of the tuple must
                be the name of the processes to be applied and the second one,
                the argument of such process. The supported processes are:
                    ('smoothing', size_of_window - odd integer)
                    ('remove_bits', amount_of_bits_to_remove - Integer between 0 and 7)
                    ('histogram_eq', 0)
                    ('transfer_histogram', 0)
                    ('median_filter', window_size)
                    ('edge_detector', 0)
                    ('thresholding', 0)
                    ('gaussian_smoothing', 0)
                    ('edge_reinforcement', edge_contribution)

        Returns:
            Processed image
        '''
        if self.enabled:
            final_img = np.array(img, dtype=np.uint8)
            if len(final_img.shape) == 2:
                final_img = final_img.reshape(final_img.shape + (1,))

            for proc in processes:
                if proc[0] == 'smoothing':
                    final_img = self.smooth_filtering(final_img, proc[1])

                elif proc[0] == 'gaussian_smoothing':
                    final_img = self.gaussian_filter(final_img)

                elif proc[0] == 'remove_bits':
                    final_img = self.remove_smaller_bits(final_img, proc[1])

                elif proc[0] == 'histogram_eq':
                    final_img = self.histeq(final_img)

                elif proc[0] == 'transfer_histogram':
                    for i in range(final_img.shape[2]):
                        final_img = self.transfer_one_band_histogram(self.reference_img, final_img, band=i)

                elif proc[0] == 'median_filter':
                    final_img = self.median_filtering(final_img, proc[1])

                elif proc[0] == 'edge_detector':
                    final_img = self.edge_detection(final_img)

                elif proc[0] == 'edge_reinforcement':
                    final_img = self.edge_reinforcement(final_img, proc[1])

                elif proc[0] == 'thresholding':
                    final_img = self.image_thresholding(final_img)

            # Final reshaping
            final_img = final_img.reshape(img.shape)
            return final_img
        else:
            return copy.deepcopy(img)

    def image_thresholding(self, img):
        '''
        Convert image into binary image. It uses the Otsu threshold to binarize

        Args:
            img: Input image to binarize

        Returns:
            Binarized image. Values below the Otsu threshold are set to zero,
            and higher than threshold are set to 255
        '''
        out = copy.deepcopy(img)
        threshold = threshold_otsu(out)
        out[out < threshold] = 0
        # out[out >= threshold] = 255
        return out

    def edge_reinforcement(self, image, edge_weight):
        '''
        Increase the edges intensity. This function will do the edge detection
        and then make the sum of the resulting edges and the original image.

        Args:
            image: Input image
            edge_weight: Float. Value that indicates how important should be the edge
                in contrast with the original image value.

        Returns:
            Image with the edges reinforced
        '''
        edges = self.edge_detection(image)
        th_edges = self.image_thresholding(edges)
        denoised_img = self.gaussian_filter(image)
        output_img = (edge_weight * np.array(th_edges, dtype=np.float)) + np.array(denoised_img, dtype=np.float)
        # output_img[output_img > 255] = 255
        # return np.array(output_img, dtype=np.uint8)
        factor = 255.0 / np.max(output_img)
        return np.array(output_img * factor, dtype=np.int16)

    def edge_detection(self, img):
        '''
        Generate an image that contains only edges

        Args:
            img: input image from which we want to detect edges

        Returns:
            Image with edges only
        '''
        g_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float)
        g_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float)

        f_x = np.array(self.apply_convolution(img, g_x), dtype=np.float)
        f_y = np.array(self.apply_convolution(img, g_y), dtype=np.float)
        sobel_filtered = np.sqrt(np.power(f_x, 2) + np.power(f_y, 2))
        max_val = np.max(sobel_filtered) if np.max(sobel_filtered) > 0 else 1
        return np.array(sobel_filtered * (255.0 / max_val), dtype=np.uint8)

    def median_filtering(self, image, kernel_size):
        '''
        Apply median filtering to an image

        Args:
            image: Input image to filter
            kernel_size: Size of the filter to be applied. It is going to
                be same size of rows and columns. This number must be odd

        Returns:
            Filtered image
        '''
        if kernel_size % 2 == 0:
            raise ValueError("The kernel size must be an odd number. Provided: {n}".format(n=kernel_size))
        return ndimage.median_filter(image, 3)

    def remove_smaller_bits(self, input_img, bits_to_remove):
        '''
        Truncate to zero the smaller bits of the image values. This is a
        basic noise reduction technique.

        Args:
            input_img: Noise image to filter
            bits_to_remove: Amount of bits to remove from each image value.
                If this value is zero, then this function has no effect.
                The maximum allowed value is 7.

        Returns:
            Filtered image of the same size as the input image
        '''
        if bits_to_remove < 0 or bits_to_remove > 7:
            raise ValueError("Amount of bits to remove too big. Choose an integer number between 1 and 7")
        if type(bits_to_remove) != int:
            raise TypeError("bits_to_remove should be an integer. Type given {}".format(type(bits_to_remove)))

        mask = 255 - (np.power(2, bits_to_remove) - 1)
        return np.bitwise_and(input_img, mask)

    def zero_padding(self, img, filter_size):
        '''
        Pad with zeros the sides of the image before doing a standard
        convolution algorithm.

        NOTE: For add-shift-multiply algorithm for convolution, zero-padding
        is not needed

        Args:
            img: Input image
            filter_size: List with the size of the filter (rows and columns)

        Returns:
            Augmented size input image zero padded on top, bottom,
            right and left sides
        '''
        starting_coord_x = int(filter_size[1] / 2)
        starting_coord_y = int(filter_size[0] / 2)
        if len(img.shape) == 2:
            img = img.reshape(img.shape + (1,))
        output = np.zeros([img.shape[0] + 2 * starting_coord_y, img.shape[1] + 2 * starting_coord_x, img.shape[2]])

        for i in range(img.shape[2]):
            output[starting_coord_y:img.shape[0]+starting_coord_y,starting_coord_x:img.shape[1]+starting_coord_x,i] = img[:,:,i]

        if output.shape[2] == 1:
            output = output.reshape([output.shape[0], output.shape[1]])
        return np.array(output, dtype=np.uint8)

    def smooth_filtering(self, input_img, kernel_size):
        '''
        Apply a low pass filter to the given image.

        Args:
            input_img: 2D array that represents the image to filter
            kernel_size: Integer that represents the amount of columns
                and rows that the kernel will have

        Returns:
            Filtered image
        '''
        if kernel_size % 2 == 0:
            raise ValueError("The kernel size must be an odd number. Provided: {n}".format(n=kernel_size))
        factor = 1.0 / (kernel_size * kernel_size)
        kernel_shape = np.array((kernel_size, kernel_size))
        kernel = np.ones(kernel_shape) * factor
        output_img = self.apply_convolution(input_img, kernel)
        return output_img

    def gaussian_filter(self, image):
        '''
        Apply a 3x3 Gaussian filter to the image.

        Args:
            image: Input image to be filtered

        Returns:
            Filtered image
        '''
        kernel = np.array([[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]]) / 16.0
        output_img = self.apply_convolution(image, kernel)
        return output_img

    def apply_convolution(self, image, kernel):
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

        return np.array(A[center_x:center_x + img_rows, center_y:center_y + img_cols, :], dtype=np.int)

    def get_histogram(self, image_matrix):
        '''
        Compute histograms of an image's matrix

        Args:
            image_matrix: Matrix of the image from which we are going to extract
                the histograms.

        Returns:
            List of histograms. If the image is grayscale, this list has only one element,
            If the image is RGB, then the list contains 3 elements, one per band
        '''
        amount_of_levels = 256
        if len(image_matrix.shape) == 2:
            image_matrix = image_matrix.reshape(image_matrix.shape + (1,))
        histograms_bgr = []
        image_matrix = np.array(image_matrix, dtype=np.uint8)
        ## calcHist returns a list of lists, where each list is the histogram value, so in order to have only one
        # dimentional array, we need to concatenate the values
        ## Taken from https://answers.opencv.org/question/33909/calchist-on-bgr-in-python/
        for i in range(image_matrix.shape[2]):
            histograms_bgr.append(np.concatenate(cv2.calcHist([image_matrix],[i],None,[amount_of_levels],[0,amount_of_levels])))

        return histograms_bgr

    def compute_image_cumulative_distribution_funct(self, histogram, normalized=True):
        '''
        Get the CDF function given an histogram

        Args:
            histogram: List with the histogram values of the image of which we want to
                create the CDF

            normalized: Boolean. If true, the CDF returned list will contain float values
                contained in the interval [0.0, 1.0] (probabilities). If False, each value would
                be a integer within the interval [0, amount_of_pixels]

        Returns:
            List with the CDF values
        '''
        amount_of_elements = len(histogram)
        cdf = []
        acumulator = 0
        for i in range(amount_of_elements):
            acumulator += histogram[i]
            cdf.append(acumulator)

        if normalized:
            return np.array(cdf) / float(acumulator)
        else:
            return np.array(cdf)

    def get_image_cdf(self, hist_arrays):
        '''
        Get the CDF function of the given list of histograms (grayscale or RGB)
        '''
        cdf_array = []
        for hist in hist_arrays:
            cdf_array.append(self.compute_image_cumulative_distribution_funct(hist))

        return cdf_array

    def histeq(self, image_matrix):
        '''
        Equalize image's histogram (make it flat)

        Args:
            image_matrix: NumPy array like matrix with the image to correct

        Returns:
            NumPy array like matrix that corresponds to the image with the
                contrast corrected through histogram's equalization
        '''
        hist_float = self.get_histogram(image_matrix)
        image_cdf = self.get_image_cdf(hist_float)[0] * 255.0

        K = np.array(image_cdf[image_matrix], dtype=np.uint8)
        print("Original image size: {s}".format(s=image_matrix.shape))
        print("K shape: {}".format(K.shape))
        return K

    def transfer_histogram(self, ref_image_matrix, dis_image_matrix, hist_ref, hist_discolored, rows, cols):
        '''
        Take the histogram of a reference image and transfer it to another image.
        This function will work with only one band at time, so hist_ref and
        hist_discolored should be lists of numbers

        TODO: Make this function actually work with one array and not a list of lists.
        TODO: Analyze how to remove the images as arguments from this function
        Args:
            ref_image_matrix: NumPy Array like matrix with the information of the image
                taken as reference, from which we are going to extract the histogram.

            dis_image_matrix: NumPy Array like matrix with the information of the image
                that is going to receive the histogram from the reference image.

            hist_ref: Histogram of the reference image (it should be only one band of an
                RGB image)

            hist_discolored: Histogram of the image we want to change (it should be only one band of an
                RGB image)

            rows: Amount of rows from the image to be changed

            cols: Amount of columns from the image to be changed

        Returns:
            NumPy array like matrix of the modified image
        '''
        K = np.zeros([rows, cols])

        ref_image_matrix = np.array(ref_image_matrix)
        dis_image_matrix = np.array(dis_image_matrix)

        int_hist_discolored = []
        int_hist_ref = []
        for hist in hist_discolored:
            int_hist_discolored.append(np.array(hist, dtype=np.uint32))

        for hist in hist_ref:
            int_hist_ref.append(np.array(hist, dtype=np.uint32))

        ## m_d = Minimum intensity in the discolored image
        m_d = np.min(dis_image_matrix)
        ## M_d: Maximum intensity in the discolored image
        M_d = np.max(dis_image_matrix)
        ## m_d = Minimum intensity in the reference image
        m_r = np.min(ref_image_matrix)
        ## M_d: Maximum intensity in the reference image
        M_r = np.max(ref_image_matrix)

        P_d = self.get_image_cdf(int_hist_discolored)[0]
        P_r = self.get_image_cdf(int_hist_ref)[0]
        g_r = m_r

        for g_d in np.arange(m_d, M_d, 1):
            while g_r < M_r and P_d[g_d+1] < 1.0 and P_r[g_r+1] < P_d[g_d+1]:
                g_r = g_r + 1
            ## This function returns us a matrix where the first row is the row
            # and the second row is the column of an element in dis_image_matrix
            # that has the same value as g_d
            K[np.where(dis_image_matrix == g_d)] += g_r

        K = np.array(K, dtype=np.uint8)
        return K

    def transfer_one_band_histogram(self, img_ref, img_dis, band=0):
        '''
        Transfer a specific band histogram from one image to another one

        Args:
            img_ref: Image from which we are going to extract the histogram(s).
            img_dis: Image at which we are going to transfer the extracted histogram
            band: Band to transfer / to extract from the reference image. If the images
                are grayscale, then we don't use this parameter
        '''
        ## We compute both image histograms from the selected band
        dis_img_band = img_dis
        ref_img_band = img_ref

        if len(img_dis.shape) == 3:
            dis_img_band = img_dis[:, :, band]

        if len(img_ref.shape) == 3:
            ref_img_band = img_ref[:, :, band]

        hist_ref = self.get_histogram(ref_img_band)
        hist_dis = self.get_histogram(dis_img_band)

        ## We create a new image based on img_discolored, and that has the same
        # histogram as img_reference
        new_histogram = self.transfer_histogram(ref_img_band,
            dis_img_band,
            hist_ref,
            hist_dis,
            img_dis.shape[0],
            img_dis.shape[1])

        if len(img_dis.shape) == 3:
            img_dis[:, :, band] = new_histogram
            return img_dis
        else:
            return new_histogram

if __name__ == '__main__':
    dirpath = os.path.dirname(os.path.realpath(__file__))
    # filepath = os.path.join(dirpath, 'cameraman.tif')
    filepath = os.path.join(dirpath, 'lena.png')
    img = cv2.imread(filepath)
    preproc = ImagePreprocessing(True)

    preprocessing = [
        ('edge_detector', 0),
        ('thresholding', 0),
        ('transfer_histogram', 0),
        ('smoothing', 7),
        ('median_filter', 3),
        ('remove_bits', 6),
        ('histogram_eq', 0),
    ]

    for proc in preprocessing:
        out = np.array(preproc.preprocess_image(img, [proc]), dtype=np.uint8)
        cv2.imshow('Original', img)
        cv2.imshow(proc[0], out)
        cv2.waitKey()
        cv2.destroyAllWindows()