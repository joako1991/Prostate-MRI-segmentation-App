'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

'''
The idea of this module is to focus on showing the image only of the loaded nifty file.
It will be a subclass of QWidget that reimplements the paintEvent method, so we can set
the image we want
'''

from PyQt5.QtWidgets import \
    QWidget

from PyQt5.QtGui import \
    QImage, \
    qRgb, \
    QPixmap, \
    QPainter, \
    QPolygonF, \
    QPen, \
    QPainterPath

from PyQt5.QtCore import \
    pyqtSignal, \
    QPointF, \
    Qt, \
    QRectF

import copy
import numpy as np

# Image possible modes
CONTOUR_CIRCULAR = 0
CONTOUR_MANUAL = 1
NORMAL_IMAGE = 2

snake_init_list = [
    'Circular',
    'Manual',
]

class ImageWidget(QWidget):
    '''
    This class is aimed to show a MR images at one depth. It can be only
    one image (3D MRI file), or several images (4D MRI file).
    It also counts with the capability of showing the contours of the left
    ventricle, when it is provided through the right function.
    '''
    pix_coordinates_event = pyqtSignal([int, int])

    def __init__(self, radious = 20, parent=None):
        super(ImageWidget, self).__init__(parent)
        assert(radious >= 0)

        self.scaling_factor = 1.5

        self.image = None
        self.contour_set = []
        self.current_image = None
        self.circle_radious = radious
        self.circle_center = None
        self.manual_polygon = QPolygonF()
        self.last_contour = []
        self.initial_contours = None
        self.view_mode = NORMAL_IMAGE
        self.contours_colors = [
            Qt.yellow,
            Qt.cyan,
            Qt.blue,
            Qt.red,
            Qt.green,
            Qt.white,
            Qt.black,
            Qt.darkRed,
            Qt.darkGreen,
            Qt.magenta,
            Qt.darkYellow,
            Qt.gray,
        ]

        # We create a grayscale colortable for showing the grayscale value in the image.
        # This indexed image implementation is inspired on
        # https://forum.qt.io/topic/102688/numpy-grayscale-array-2d-to-rgb-qimage-or-qpixmap-in-either-red-green-or-blue/2
        self.grayscale_colortable = np.array([qRgb(i, i, i) for i in range(256)])

        self.image_mode(self.view_mode)

    def image_mode(self, image_mode):
        '''
        Change the image mode between the normal view, the view where
        a circle is shown with the mouse, or the mode where the user
        is going to enter a polygon manually

        Args:
            image_mode: Desired mode to be set. The possible values are:
                    NORMAL_IMAGE
                    CONTOUR_CIRCULAR
                    CONTOUR_MANUAL
                If an invalid value is entered, a ValueError exception is thrown
        '''
        self.view_mode = image_mode
        if image_mode == NORMAL_IMAGE:
            self.setMouseTracking(False)
            self.manual_polygon = QPolygonF()
        elif image_mode == CONTOUR_CIRCULAR:
            self.setMouseTracking(True)
            self.manual_polygon = QPolygonF()
            self.last_contour = []
        elif image_mode == CONTOUR_MANUAL:
            self.setMouseTracking(False)
            self.manual_polygon = QPolygonF()
            self.last_contour = []
        else:
            raise ValueError("Non valid mode provided: {}".format(image_mode))

        self.repaint()

    def set_image(self, data):
        '''
        It updates the internal image buffer. If the input image_array has
        information, we will remap the image_array so it is between the range 0
        and 255, and we do a deep copy of it. Then, we show the widget.

        Args:
            image_array: New image to be set
        '''
        if not data is None and data.shape:
            # We copy the data into the internal buffer
            self.image = copy.deepcopy(data)
            self.show()
            self.refresh_image()

    def set_initial_contours(self, new_contours_dictionary):
        '''
        Setter that will take the information for the initial contours
        to be shown in the image. This function also converts the list
        of points for the manual mode into QPolygonF, so the paintEvent
        function does not need to do it every painting

        After this function is executed, a new painting is going to be triggered
        Args:
            new_contours_dictionary: Dictionary in which the key
                is the zone that this contours represents. Then inside this key
                there is another dictionary, which key can be 'type', for the type
                of contour that it containts, and data if it is manual type, or
                center_x, center_y and radious if the type is circular
        '''
        self.initial_contours = copy.deepcopy(new_contours_dictionary)
        for key, value in new_contours_dictionary.items():
            if value and value['type'] == snake_init_list[CONTOUR_MANUAL]:
                contour = np.array(value['data'])
                polygon = QPolygonF()
                if not contour is None and len(contour.shape):
                    for i in np.arange(contour.shape[0]):
                        point = contour[i]
                        polygon.append(QPointF(point[0]*self.scaling_factor, point[1]*self.scaling_factor))
                self.initial_contours[key]['data'] = polygon

        self.repaint()

    def refresh_image(self):
        '''
        Refresh the image that is being shown. If it is a single depth
        information, the image information is always the same. If it is a
        video, then we read a new index image from the stored data.
        '''
        if not self.image is None and self.image.shape:
            assert (np.max(self.image) <= 255)
            # We select one of the slices
            image8 = self.image.astype(np.uint8, order='C', casting='unsafe')

            height = image8.shape[0]
            width = image8.shape[1]

            # We create a QImage as an indexed image to show the grayscale
            # values. Because it is in indexed format, we set its color table
            # too to be grayscale
            qimage = QImage(
                image8,
                width,
                height,
                width,
                QImage.Format_Indexed8)
            qimage.setColorTable(self.grayscale_colortable)

            # We scale the image
            self.current_image = qimage.scaledToWidth(
                width * self.scaling_factor)

            # We limit the QWidget size to be the equal to the Image size
            self.setFixedSize(
                self.current_image.width(),
                self.current_image.height())

            self.repaint()

    def mouseMoveEvent(self, event):
        '''
        Overloaded function. It will be called every time the user moves the
        mouse over the image widget
        '''
        # Here we get a QPoint
        self.circle_center = event.pos()
        self.repaint()

    def set_circle_radious(self, radious):
        '''
        Change the circle radious

        Args:
            radious: Circle radious in pixels
        '''
        assert(radious >= 0)
        self.circle_radious = radious
        self.repaint()

    def get_last_contour(self):
        return copy.deepcopy(self.last_contour)

    def mousePressEvent(self, event):
        '''
        Event executed each time we click on the widget. Since the widget size
        is constrained to be equal to the image size, the X and Y information
        is directly related to the pixel value in the image matrix, up to
        a scale factor, which we know.

        This function also emits a signal carrying the image position and the
        pixel coordinates in the original image coordinates (row and column
        of the pixel in the image's matrix)

        Args:
            Event: Object that comes with this overloaded function. It contains
                information about where the click was made.
        '''
        # Because all the coordinates are scaled by the same factor, the real
        # pixel value is the one that every coordinate is divided by the
        # scaling factor
        x_coord = int((event.pos().x() / self.scaling_factor) + 0.5)
        y_coord = int((event.pos().y() / self.scaling_factor) + 0.5)

        if self.view_mode == CONTOUR_CIRCULAR:
            print("Pixel received: {x} {y}".format(x=x_coord, y=y_coord))
            self.pix_coordinates_event.emit(x_coord, y_coord)

        if self.view_mode == CONTOUR_MANUAL:
            self.last_contour.append([x_coord, y_coord])
            pt = QPointF(event.pos().x(), event.pos().y())
            self.manual_polygon.append(pt)
            self.repaint()

    def reset_contours(self):
        '''
        Clear the internal contours buffers
        '''
        self.contour_set.clear()
        self.manual_polygon = QPolygonF()
        self.last_contour = []
        self.repaint()

    def get_image(self):
        return self.image

    def set_contours_list(self, contour):
        '''
        Set the contours, result of the segmentation of the left ventricle.
        The amount of contours must be the same as the amount of images stored
        in this widget, and they should not be None nor empty.

        Even if it is only the contour for a sole image, it should be stored
        in a list with the format:
            [Image_idx, [Array of 2D points]]

        After the contours has been set, the images will be refreshed

        Args:
            contour: List of lists, with the contours of the segmented
                left ventricle
        '''
        contour = np.array(contour)
        if not contour is None and len(contour.shape):
            self.contour_set.clear()
            # We copy the data into the internal buffer
            for i in np.arange(contour.shape[0]):
                polygon = QPolygonF()
                for j in range(len(contour[i])):
                    point = contour[i][j]
                    polygon.append(QPointF(point[0]*self.scaling_factor, point[1]*self.scaling_factor))
                self.contour_set.append(polygon)
            self.refresh_image()

    def paintEvent(self, event):
        '''
        Overloaded paintEvent. It will draw:
        -) The MRI image
        -) The contour of the left ventricle, if it has been found
        -) A circle that indicates if the initial contour selected by the user
            is of this type
        -) Random contour draw by the user
        '''
        painter = QPainter(self)
        line_width = 2
        painter.setPen(QPen(Qt.yellow, line_width, Qt.SolidLine))

        if not self.current_image is None:
            painter.drawPixmap(0,0, self.current_image.width(), self.current_image.height(), QPixmap(self.current_image))

        # We plot the segmentation contours resulting from the algorithm
        for i in range(len(self.contour_set)):
            contour = self.contour_set[i]
            painter.setPen(QPen(self.contours_colors[i], line_width, Qt.SolidLine))
            painter.drawPolygon(contour)

        # We plot the polygon that we have until now if we are in manual mode
        if self.view_mode == CONTOUR_MANUAL:
            painter_path = QPainterPath()
            painter_path.addPolygon(self.manual_polygon)
            painter.drawPath(painter_path)
        # We plot a circle at each mouse position if we are in circular mode
        elif self.view_mode == CONTOUR_CIRCULAR and not self.circle_center is None:
            painter.drawEllipse(QRectF(
                self.circle_center.x() - self.circle_radious * self.scaling_factor,
                self.circle_center.y() - self.circle_radious * self.scaling_factor,
                self.circle_radious * self.scaling_factor * 2.0,
                self.circle_radious * self.scaling_factor * 2.0))

        # We plot the initial contours
        if not self.initial_contours is None and any(self.initial_contours.values()):
            list_of_contours = list(self.initial_contours.values())
            color_counter = 0
            for i in range(len(list_of_contours)):
                cont = list_of_contours[i]
                if cont:
                    if cont['type'] == snake_init_list[CONTOUR_CIRCULAR]:
                        x = cont['center_x']
                        y = cont['center_y']
                        rad = cont['radious']
                        painter.setPen(QPen(self.contours_colors[color_counter], line_width, Qt.SolidLine))
                        painter.drawEllipse(QRectF(
                            (x - rad) * self.scaling_factor,
                            (y - rad) * self.scaling_factor,
                            rad * 2.0 * self.scaling_factor,
                            rad * 2.0 * self.scaling_factor))
                    elif cont['type'] == snake_init_list[CONTOUR_MANUAL]:
                        path = QPainterPath()
                        path.addPolygon(cont['data'])
                        painter.setPen(QPen(self.contours_colors[color_counter], line_width, Qt.SolidLine))
                        painter.drawPath(path)

                    color_counter += 1