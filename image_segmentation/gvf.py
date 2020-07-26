import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

class GradientVectorFlow(object):
    '''
    Gradient vector flow module. Original version in MATLAB from
    Chenyang Xu and Jerry L. Prince

    http://iacl.ece.jhu.edu/static/gvf/

    It computes the GVF vector fieldV = [u,v] based on an edge map.

    An example of how to compute and how to print it is included at the end of this file.
    '''
    def __init__(self, mu, iters):
        self.mu = mu
        self.iters = iters

    def compute_gvf(self, edgemap):
        '''
        Get the components u and v of the gradient vector flow
        vector field. These components are not normalized.

        Using the edge-map as a reference, the component u corresponds to the
        columns of the edge-map and v to the rows.

        Args:
            edgemap: Edge-map of the image from which we want to compute the GVF.
                It can be a binary image or not, and it can be normalized or not.

        Returns:
            Two arrays, u and v, which represent the GVF components.
        '''
        edgemap = np.array(edgemap)
        fmax = np.max(edgemap)
        fmin = np.min(edgemap)
        # We normalize the edge map to be between 0 and 1
        normalized_em = (edgemap - fmin) / (fmax - fmin)
        # X are columns, Y are rows
        # Take care of boundary condition
        normalized_em = self.bound_mirror_expand(normalized_em)
        # Compute the gradient
        grad_edge_map = np.gradient(normalized_em)
        # Initialize the GVF
        fx = grad_edge_map[1]
        fy = grad_edge_map[0]

        u = np.array(fx)
        v = np.array(fy)
        # Squared magnitude of the gradient field
        square_mag_em = np.power(fx,2) + np.power(fy,2)

        for i in range(self.iters):
            u = self.bound_mirror_ensure(u)
            u = u + 4 * self.mu * (laplace(u, mode='wrap') / 4.0) - np.multiply(square_mag_em, u - fx)

            v = self.bound_mirror_ensure(v)
            v = v + 4 * self.mu * (laplace(v, mode='wrap') / 4.0) - np.multiply(square_mag_em, v - fy)

        u = self.bound_mirror_shrink(u)
        v = self.bound_mirror_shrink(v)

        return u, v

    def bound_mirror_expand(self, A):
        '''
        Expand the matrix using mirror boundary condition

        for example
            A = [
                [1, 2, 3, 11],
                [4, 5, 6, 12],
                [7, 8, 9, 13]
                ]
            B = BoundMirrorExpand(A) will yield
                B = [
                    [5, 4, 5, 6, 12, 6],
                    [2, 1, 2, 3, 11, 3],
                    [5, 4, 5, 6, 12, 6],
                    [8, 7, 8, 9, 13, 9],
                    [5, 4, 5, 6, 12, 6]
                ]
        Chenyang Xu and Jerry L. Prince, 9/9/1999
        http://iacl.ece.jhu.edu/projects/gvf
        '''
        A = np.array(A)
        # Yi are rows and Xi are columns
        m = A.shape[0]
        n = A.shape[1]

        B = np.zeros((m+2, n+2))
        B[1:m+1,1:n+1] = A

        # Mirror the corners
        B[np.ix_([0,m+1],[0,n+1])] = B[np.ix_([2,m-1],[2,n-1])]
        # Mirror left and right boundaries
        B[np.ix_([0,m+1],range(1,n+1))] = B[np.ix_([2,m-1],range(1,n+1))]
        # Mirror top and bottom boundary
        B[np.ix_(range(1,m+1),[0,n+1])] = B[np.ix_(range(1,m+1),[2,n-1])]

        return B

    def bound_mirror_ensure(self, A):
        '''
        Ensure mirror boundary condition
        The number of rows and columns of A must be greater than 2
        for example (X means value that is not of interest)

        A = [
            X  X  X  X  X   X
            X  1  2  3  11  X
            X  4  5  6  12  X
            X  7  8  9  13  X
            X  X  X  X  X   X
            ]
        B = BoundMirrorEnsure(A) will yield
        B = [
            [5, 4, 5, 6, 12, 6],
            [2, 1, 2, 3, 11, 3],
            [5, 4, 5, 6, 12, 6],
            [8, 7, 8, 9, 13, 9],
            [5, 4, 5, 6, 12, 6]
            ]
        Chenyang Xu and Jerry L. Prince, 9/9/1999
        http://iacl.ece.jhu.edu/projects/gvf
        '''
        A = np.array(A)
        m = A.shape[0]
        n = A.shape[1]

        if m < 3 or n < 3:
            raise ValueError("Either the number of rows or columns is smaller than 3")

        B = A
        # Mirror the corners
        B[np.ix_([0,m-1],[0,n-1])] = B[np.ix_([2,m-3],[2,n-3])]
        # Mirror left and right boundaries
        B[np.ix_([0,m-1],range(1,n-1))] = B[np.ix_([2,m-3],range(1,n-1))]
        # Mirror top and bottom boundary
        B[np.ix_(range(1,m-1),[0,n-1])] = B[np.ix_(range(1,m-1),[2,n-3])]

        return B

    def bound_mirror_shrink(self, A):
        '''
        Shrink the matrix to remove the padded mirror boundaries
        for example
        A = [
            [5, 4, 5, 6, 12, 6],
            [2, 1, 2, 3, 11, 3],
            [5, 4, 5, 6, 12, 6],
            [8, 7, 8, 9, 13, 9],
            [5, 4, 5, 6, 12, 6]
            ]

        B = BoundMirrorShrink(A) will yield
        B = [
            [1, 2, 3, 11],
            [4, 5, 6, 12],
            [7, 8, 9, 13]
            ]
        Chenyang Xu and Jerry L. Prince, 9/9/1999
        http://iacl.ece.jhu.edu/projects/gvf
        '''
        A = np.array(A)
        m = A.shape[0]
        n = A.shape[1]
        B = A[1:m-1, 1:n-1]

        return B

def get_gaussian_kernel(kernel_size, sigma):
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

if __name__ == '__main__':
    # TEST THE GVF
    import cv2
    import os
    dir_path = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.insert(1, dir_path)
    import matlab_func as matlab

    # Example of usage of GVF
    # Open image
    import os
    filepath = os.path.join(dir_path, 'U64.pgm')
    I = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)

    # Normal way to compute the edge-map
    # # Compute edge map
    # sigma = np.std(I)
    # kernel = get_gaussian_kernel(3, sigma)

    # filtered_img = np.array(matlab.conv2d(I, kernel), dtype=np.float)
    # grad_img = np.gradient(filtered_img)
    # edge_map = (-1) * np.linalg.norm(grad_img, axis=0)
    # # We normalize the edge map to be between 0 and 1
    # edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map))

    # Simplified edge-map for the U64 image
    edge_map = 1 - I/255

    grad_vector_flow = GradientVectorFlow(0.2, 80)
    u,v = grad_vector_flow.compute_gvf(edge_map)

    # We normalize the GVF field
    # mag is the module, element-wise --> same size as u and v
    mag = np.sqrt(np.power(u,2) + np.power(v,2))
    px = np.divide(u, mag + 1e-10)
    py = np.divide(v, mag + 1e-10)

    import matplotlib.pyplot as plt
    plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(np.array(I, dtype=np.uint8), cmap='gray')
    # # plt.imshow(I, cmap='gray')
    # grad = np.gradient(edge_map)
    # mod = np.power(grad[0], 2) + np.power(grad[1], 2)
    # grad = np.divide(grad, mod)
    # gx = grad[1]
    # gy = grad[0]
    # plt.quiver(gx, gy)

    # plt.subplot(2,1,2)
    plt.imshow(np.array(I, dtype=np.uint8), cmap='gray')
    plt.quiver(px, -py)
    plt.grid()
    plt.show()

# import cv2; cv2.namedWindow('hello', cv2.WINDOW_NORMAL); cv2.imshow('hello', I); cv2.waitKey(); cv2.destroyAllWindows() 