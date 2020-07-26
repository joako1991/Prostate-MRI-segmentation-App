'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

import os
filepath = os.path.realpath(__file__)
dirname = os.path.dirname(filepath)
root_path = os.path.join(dirname, '../')
image_segmentation_path = os.path.join(root_path, 'image_segmentation')

import sys
sys.path.insert(1, image_segmentation_path)
from matlab_func import create_circular_mask, create_random_mask

from PyQt5.QtWidgets import \
    QHBoxLayout, \
    QVBoxLayout, \
    QPushButton, \
    QWidget, \
    QLabel, \
    QGroupBox, \
    QRadioButton, \
    QButtonGroup, \
    QSpinBox

from PyQt5.QtCore import \
    pyqtSignal

import numpy as np
from collections import OrderedDict
import copy

class SnakeInitWidget(QWidget):
    '''
    This class will enclose all the information needed
    for creating the initial contour of each segmentation part.

    It contains 5 buttons:
        -) Zone TZ, to capture the initial contour for the zone TZ
        -) Zone PZ, to capture the initial contour for the zone PZ
        -) Zone CZ, to capture the initial contour for the zone CZ
        -) Tumeur, to capture the initial contour for the tumeur
        -) OK: useful for the case of entering a manual contour, and
            it will stop the acquisition of points

    There are two possible ways to select the initial contour:
    Circular shape or random, manual shape. Each area can be selected
    as the user wants. They can be all of them circles, all of them manual,
    or a mix.
    '''
    request_zone = pyqtSignal([int])
    new_radious = pyqtSignal([int])
    retrieve_points = pyqtSignal([np.ndarray])
    contours_changed = pyqtSignal()

    def __init__(self, snake_init_list, parent = None):
        '''
        Constructor

        Args:
            snake_init_list: List with the type of ways in which the user
                can enter the initial snake. For instance, this list can be a
                2 elements list, where the first element is an string 'Circular',
                and the other one 'Manual'. These values will be used to show
                the Radio buttons labels.
        '''
        super(SnakeInitWidget, self).__init__(parent)

        self.layout = QVBoxLayout()
        assert(len(snake_init_list))
        self.init_types = snake_init_list

        self.zone_tz_key = 'Zone TZ'
        self.zone_pz_key = 'Zone PZ'
        self.zone_cz_key = 'Zone CZ'
        self.zone_tumeur_key = 'Tumeur'
        self.circular_key ='Circular'
        self.circular_index = 0
        self.manual_key = 'Manual'
        self.manual_index = 1

        assert(snake_init_list[self.circular_index] == self.circular_key and snake_init_list[self.manual_index] == self.manual_key)

        self.zones_dict = OrderedDict()
        self.clear_contours()

        self.requested_area = None
        self.image_loaded = False
        self.initialize_widget()

    def set_image_loaded(self, val):
        self.image_loaded = val

    def initialize_widget(self):
        '''
        Initialize widget layout
        '''
        # Snake init group
        self.add_push_buttons()
        self.add_radio_buttons()
        self.add_spinbox()
        self.setLayout(self.layout)

        self.set_buttons_enabled(True)

    def clear_contours(self):
        '''
        Clear all the stored contours
        '''
        self.zones_dict[self.zone_tz_key] = None
        self.zones_dict[self.zone_pz_key] = None
        self.zones_dict[self.zone_cz_key] = None
        self.zones_dict[self.zone_tumeur_key] = None
        self.contours_changed.emit()

    def add_spinbox(self):
        '''
        Add an spin box that will indicate the radious of the circle.
        '''
        self.radious_spinbox = QSpinBox()
        self.radious_spinbox.valueChanged.connect(self.on_update_radious)
        self.radious_spinbox.setMinimum(0)
        self.radious_spinbox.setMaximum(30)
        self.radious_spinbox.setValue(20)
        local_layout = QHBoxLayout()
        text_label = QLabel("Radious: ")
        local_layout.addWidget(text_label)
        local_layout.addWidget(self.radious_spinbox)
        my_tmp_widget = QWidget()
        my_tmp_widget.setLayout(local_layout)

        for button in self.radio_button_snk_init:
            button.toggled.connect(self.snake_type_button_changed)

        self.layout.addWidget(my_tmp_widget)

    def add_push_buttons(self):
        '''
        Add the buttons to select zones to the widget layout
        '''
        snake_init_layout = QVBoxLayout()
        snake_init_groupbox = QGroupBox('Snake Initialization')

        self.zone_tz = QPushButton(self.zone_tz_key)
        self.zone_pz = QPushButton(self.zone_pz_key)
        self.zone_cz = QPushButton(self.zone_cz_key)
        self.tumeur = QPushButton (self.zone_tumeur_key)

        self.ok_button = QPushButton('OK')

        snake_init_layout.addWidget(self.zone_tz)
        snake_init_layout.addWidget(self.zone_pz)
        snake_init_layout.addWidget(self.zone_cz)
        snake_init_layout.addWidget(self.tumeur)
        snake_init_layout.addWidget(self.ok_button)

        snake_init_groupbox.setLayout(snake_init_layout)
        self.layout.addWidget(snake_init_groupbox)

        # Connections
        self.zone_pz.clicked.connect(self.on_select_zone_pz)
        self.zone_cz.clicked.connect(self.on_select_zone_cz)
        self.zone_tz.clicked.connect(self.on_select_zone_tz)
        self.tumeur.clicked.connect(self.on_select_tumour)

    def set_data_receptor_callback(self, callback):
        '''
        Set a callback to the OK button. It is useful when the initial
        snake enter mode is manual.
        Then this function will be called when the user decides that the
        entered snake is finished

        Args:
            callback: Function that will be called as callback.
                It should not receive any argument
        '''
        self.ok_button.clicked.connect(callback)

    def set_last_requested_contour(self, contour):
        '''
        Function that must be called once the contour is ready for
        the selected area of the prostate

        Args:
            contour: Dictionary with the contour information. It must have this shape.
                -) manual contour:
                    my_dict['type'] --> String that indicated which type of contour is
                    my_dict['data'] --> List of lists. Each list must contain the
                        X,Y coordinates of the selected point by the user.

                -) Circular contour
                    my_dict['type'] --> String that indicated which type of contour is
                    my_dict['center_x'] --> X coordinate of the center of the circle
                    my_dict['center_y'] --> Y coordinate of the center of the circle
                    The radious will be set in this function
        '''
        if 'center_x' in contour.keys():
            contour['radious'] = self.radious
        self.zones_dict[self.requested_area] = copy.deepcopy(contour)
        self.set_buttons_enabled(True)
        self.contours_changed.emit()

    def are_all_initial_contours_present(self):
        return all(self.zones_dict.values())

    def get_initial_contours_polygons(self):
        return self.zones_dict

    def restore_zones_dict(self, prev_dict):
        self.zones_dict = copy.deepcopy(prev_dict)

    def get_initial_contours_centroids(self):
        '''
        Get a dictionary with the centroids of the set of points entered by the user
        as initial mask. If a circular mask has been entered, the centroid
        is the center of the circle. If a manual polygon has been entered,
        the centroid is the average of the maximum and minimum value of each
        axis coordinate.

        Returns:
            Dictionary where each key contains a list of two elements, [X,Y]
            that represents the centroid of the initial snake. X corresponds to
            a column value and Y to a row value. The keys are the same as the ones
            in zones_dict
        '''
        centers_dict = OrderedDict()
        for key, val in self.zones_dict.items():
            if val:
                if self.circular_key == val['type']:
                    # Circular initial contour
                    centers_dict[key] = [val['center_x'], val['center_y']]
                elif self.manual_key == val['type']:
                    min_x = min(np.array(val['data'])[:,0])
                    min_y = min(np.array(val['data'])[:,1])
                    max_x = max(np.array(val['data'])[:,0])
                    max_y = max(np.array(val['data'])[:,1])
                    centers_dict[key] = [int((min_x + max_x) / 2),int((min_y + max_y) / 2)]
                else:
                    raise ValueError("Invalid type entered: {}".format(val['type']))
        return centers_dict

    def get_initial_contours_masks(self, image_shape):
        '''
        Convert polygons into masks. Since the segmentation methods receives
        a binary image where the interior of the initial snake has values of 1
        and the rest, zero, this function will create a similar dictionary to
        the polygons of the initial contours, but with these masks, instead
        of the data about the polygon.

        Args:
            image_shape: Tuple with the amount of rows and columns of the
                output mask

        Returns:
            Dictionary where each key is the zone type and the value is
            the mask itself
        '''
        masks_dict = OrderedDict()
        for key, val in self.zones_dict.items():
            if val:
                if self.circular_key == val['type']:
                    # Circular initial contour
                    masks_dict[key] = create_circular_mask(
                        image_shape[0],
                        image_shape[1],
                        val['center_x'],
                        val['center_y'],
                        val['radious'])
                elif self.manual_key == val['type']:
                    masks_dict[key] = create_random_mask(
                        image_shape[0],
                        image_shape[1],
                        val['data'])
                else:
                    raise ValueError("Invalid type entered: {}".format(val['type']))
        return masks_dict

    def add_radio_buttons(self):
        '''
        Add radio buttons to select the type of initial contour.
        Up to now, only circular and manual contours are allowed
        '''
        buttons_layout = QVBoxLayout()

        snake_init_groupbox = QGroupBox('Initial contour')
        self.snk_init_group = QButtonGroup(self)
        self.radio_button_snk_init = []
        for i in range(len(self.init_types)):
            snk_type = self.init_types[i]
            button = QRadioButton(snk_type)
            buttons_layout.addWidget(button)
            self.snk_init_group.addButton(button, i)
            self.radio_button_snk_init.append(button)

        self.radio_button_snk_init[0].setChecked(True)
        snake_init_groupbox.setLayout(buttons_layout)
        self.layout.addWidget(snake_init_groupbox)

    def snake_type_button_changed(self):
        '''
        Slot executed when the type of initial contour selected by the user had changed.
        It will disable the radious button then the initial contour is manual,
        and enabled when it is circular.
        '''
        if self.radio_button_snk_init[self.circular_index].isChecked():
            self.radious_spinbox.setEnabled(True)
        else:
            self.radious_spinbox.setEnabled(False)

    def set_buttons_enabled(self, enabled):
        '''
        Disable all buttons at one, and the OK button follows the negative
        logic (If all the buttons are disabled, the OK button is enabled)

        Args:
            enabled: If true, all the buttons will be enabled
        '''
        self.zone_tz.setEnabled(enabled)
        self.zone_pz.setEnabled(enabled)
        self.zone_cz.setEnabled(enabled)
        self.tumeur.setEnabled(enabled)
        for button in self.radio_button_snk_init:
            button.setEnabled(enabled)
        self.ok_button.setEnabled(not enabled)
        # If we disable, then the radious is disabled, but when enabled,
        # we need to check which radio button is selected
        if enabled:
            self.snake_type_button_changed()
        else:
            self.radious_spinbox.setEnabled(False)

    def on_select_zone_tz(self):
        '''
        Slot executed when the user wants to create an initial contour for the zone TZ.
        This function will have effects only if an image has been loaded
        '''
        if self.image_loaded:
            self.zones_dict[self.zone_tz_key] = None
            self.contours_changed.emit()

            self.set_buttons_enabled(False)
            self.requested_area = self.zone_tz_key
            self.request_zone.emit(self.snk_init_group.checkedId())

    def on_select_zone_pz(self):
        '''
        Slot executed when the user wants to create an initial contour for the zone PZ.
        This function will have effects only if an image has been loaded previously
        '''
        if self.image_loaded:
            self.zones_dict[self.zone_pz_key] = None
            self.contours_changed.emit()
            self.set_buttons_enabled(False)
            self.requested_area = self.zone_pz_key
            self.request_zone.emit(self.snk_init_group.checkedId())

    def on_select_zone_cz(self):
        '''
        Slot executed when the user wants to create an initial contour for the zone CZ.
        This function will have effects only if an image has been loaded previously
        '''
        if self.image_loaded:
            self.zones_dict[self.zone_cz_key] = None
            self.contours_changed.emit()
            self.set_buttons_enabled(False)
            self.requested_area = self.zone_cz_key
            self.request_zone.emit(self.snk_init_group.checkedId())

    def on_select_tumour(self):
        '''
        Slot executed when the user wants to create an initial contour for the tumeur.
        This function will have effects only if an image has been loaded previously
        '''
        if self.image_loaded:
            self.zones_dict[self.zone_tumeur_key] = None
            self.contours_changed.emit()
            self.set_buttons_enabled(False)
            self.requested_area = self.zone_tumeur_key
            self.request_zone.emit(self.snk_init_group.checkedId())

    def on_cancel(self):
        '''
        Action to be taken when the main window says the action must be cancelled
        '''
        self.set_buttons_enabled(True)

    def on_update_radious(self, value):
        '''
        Slot executed each time the radious is updated
        '''
        self.radious = value
        self.new_radious.emit(value)