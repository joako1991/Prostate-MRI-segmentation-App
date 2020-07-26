# Taken from https://matplotlib.org/3.1.1/gallery/user_interfaces/embedding_in_qt_sgskip.html
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import \
    QHBoxLayout, \
    QWidget

class Model3DWidget(QWidget):
    '''
    Widget that shows a 3D plot of the organs in the MRi image.
    The 3D model is show the segmented contour, repeated several times in
    the Z axis. Further improvements like adding textures is left for
    future implementations
    '''
    def __init__(self, parent=None):
        super(Model3DWidget,self).__init__(parent)
        layout = QHBoxLayout()
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.axes = self.canvas.figure.gca(projection='3d')
        self.axes.set_title("3D representation of the prostate")
        # Amount of times that the contours are repeated
        self.amount_of_slices = 50
        layout.addWidget(self.canvas)
        # We create a list of colors, in the same order than in the
        # image widget. Then the plots here will have the same
        # color code than in that widget
        self.list_of_colors = [
            'y', \
            'c', \
            'b', \
            'r', \
            'g'
        ]

    def show_3d_model(self, contours, spatial_resolution, slice_size, image_shape):
        '''
        Show the 3D plot as a pop up window, using MatPLotLib for PyQt.
        The axis will be represented in real units.

        Args:
            contours: List with all the contours to be plotted
            spatial_resolution: List with two float numbers. The first
                one corresponds to the spatial resolution in X axis in mm.
                The second element corresponds to the Y axis.
            slice_size: Size of the organs in Z axis
            image_shape: Tuple with amount of rows and columns of the image.
        '''
        factor = float(slice_size) / self.amount_of_slices
        self.axes.clear()
        for i in range(self.amount_of_slices):
            for j in range(len(contours)):
                if contours[j].any():
                    z = [i * factor] * contours[j].shape[0]
                    self.axes.plot(contours[j][:,1] * spatial_resolution[0], contours[j][:,0] * spatial_resolution[1], z, c=self.list_of_colors[j])

        axis_limit = max(image_shape[1] * spatial_resolution[0], image_shape[0] * spatial_resolution[1])
        self.axes.set_xlim(0, axis_limit)
        self.axes.set_ylim(0, axis_limit)
        self.axes.set_xlabel("X axis (mm)")
        self.axes.set_ylabel("Y axis (mm)")
        self.axes.set_zlabel("Z axis (mm)")
        self.axes.grid()
        self.canvas.figure.canvas.show()