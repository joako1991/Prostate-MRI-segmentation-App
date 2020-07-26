'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''
import copy

from PyQt5.QtWidgets import \
    QMainWindow, \
    QHBoxLayout, \
    QVBoxLayout, \
    QPushButton, \
    QWidget, \
    QLabel, \
    QScrollArea, \
    QSplitter, \
    QGroupBox, \
    QRadioButton, \
    QButtonGroup, \
    QShortcut, \
    QFileDialog, \
    QSpinBox, \
    QMessageBox

from PyQt5.QtGui import \
    QKeySequence

from PyQt5.QtCore import \
    Qt, \
    QCoreApplication

from image_widget.image_widget import ImageWidget, NORMAL_IMAGE, CONTOUR_CIRCULAR, CONTOUR_MANUAL, snake_init_list
from snake_init_widget.snake_init_widget import SnakeInitWidget
from dicom_format.dicom import DicomFormat
import image_segmentation.segmentation_hub as sh
from user_data_widget.user_data_widget import UserDataWidget
from copyright import Copyright
from model_3d_widget.model_3d_widget import Model3DWidget

def is_number(string):
    '''
    Check if an string is a number

    Args:
        string: String to be checked

    Returns:
        True if the string can be successfully converted into float
    '''
    try:
        float(string)
    except ValueError:
        return False
    else:
        return True

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        '''
        Constructor. It will initialize the MainWindow layouts and its widgets
        '''
        super(MainWindow, self).__init__(parent)
        # The main layout will be splitted in 2 vertical layouts: Left and right.
        # Then each layout contains several widgets.
        self.splitter = QSplitter()
        self.left_layout = QVBoxLayout()
        self.central_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.left_image = None
        self.right_image = None
        self.is_image_loaded = False
        self.contours_dict = {}
        self.initial_contours_dict = {}
        # We choose which segmentation method we are going to use
        self.segm_module = None
        self.is_segm_running = False

        self.initialize()
        self.setWindowTitle("Medical images segmentation app")

    def initialize(self):
        # Parser that will be in charge of processing the opened files
        self.dicom_parser = DicomFormat()
        self.dicom_parser.read_finished.connect(self.on_dicom_read_finished)
        self.model_3d = Model3DWidget()

        # The initialization order matters. First we need to create the image widget
        # and then we add the buttons
        self.add_left_image_widget()
        self.add_buttons()
        self.add_radio_buttons()
        self.add_right_image_widget()
        self.add_user_info_area()
        self.add_spinbox()
        self.add_copyright()

        ## All the widget should be added to the main layout before this line
        self.set_main_window_layouts()

    def set_main_window_layouts(self):
        # We create the layouts that are going to host the left and right layouts
        main_scroll = QScrollArea()
        left_widget = QWidget()
        central_widget = QWidget()
        right_widget = QWidget()

        left_widget.setLayout(self.left_layout)
        self.central_layout.addStretch()
        central_widget.setLayout(self.central_layout)
        right_widget.setLayout(self.right_layout)

        # We add the left, central and right layouts to the main one
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(central_widget)
        self.splitter.addWidget(right_widget)
        main_scroll.setWidget(self.splitter)
        main_scroll.setWidgetResizable(True)

        self.setCentralWidget(main_scroll)

    def resizeEvent(self, event):
        # These values are set as a proportion.
        # The only requirement is that they must be greater than the minimum size hint
        self.splitter.setSizes([1000000,500000,1000000])

    def add_copyright(self):
        widget = Copyright()
        self.central_layout.addStretch()
        self.central_layout.addWidget(widget)

    def add_radio_buttons(self):
        '''
        Add the radio buttons that will be used to allow the user to select
        a segmentation method

        # inspired in https://stackoverflow.com/questions/14798058/grouping-radio-buttons-in-pyqt
        '''
        buttons_layout = QVBoxLayout()
        self.buttons_groupbox = QGroupBox('Segmentation method')
        self.bt_group = QButtonGroup(self)
        self.radio_buttons_list = []
        for key, value in sh.methods_dict.items():
            button = QRadioButton(key)
            buttons_layout.addWidget(button)
            self.bt_group.addButton(button, value)
            self.radio_buttons_list.append(button)

        self.radio_buttons_list[2].setChecked(True)
        self.buttons_groupbox.setLayout(buttons_layout)
        self.central_layout.addWidget(self.buttons_groupbox)

    def add_spinbox(self):
        '''
        Add an spin box that will indicate which image we are selecting.
        Changing the value of this spinbox will update both, patient information
        and shown image
        '''
        self.slide_spinbox = QSpinBox()
        self.slide_spinbox.valueChanged.connect(self.on_new_slide_selected)
        self.slide_spinbox.setEnabled(False)
        self.slide_spinbox.setMinimum(0)
        layout = QHBoxLayout()
        text_label = QLabel("Instance number: ")
        layout.addWidget(text_label)
        layout.addWidget(self.slide_spinbox)
        my_tmp_widget = QWidget()
        my_tmp_widget.setLayout(layout)
        self.central_layout.addWidget(my_tmp_widget)

    def add_buttons(self):
        '''
        Add the buttons to the main window. They will
        be added in the central part of the main layout
        '''
        # Snake init module
        self.snake_init_widget = SnakeInitWidget(snake_init_list)
        self.snake_init_widget.set_data_receptor_callback(self.request_end_of_contour)

        # Segmentation group
        seg_group_layout = QVBoxLayout()
        seg_groupbox = QGroupBox('Segmentation')
        self.run_button = QPushButton('RUN')
        self.cancel_button = QPushButton('Cancel')
        self.show_3d = QPushButton('Show 3D model')
        seg_group_layout.addWidget(self.run_button)
        seg_group_layout.addWidget(self.show_3d)
        seg_group_layout.addWidget(self.cancel_button)
        seg_groupbox.setLayout(seg_group_layout)

        # File management group
        file_management_layout = QVBoxLayout()
        file_groupbox = QGroupBox('File')
        self.open_button_file = QPushButton('Open file...')
        self.open_button_folder = QPushButton('Open Folder...')
        self.close_button = QPushButton('Close file...')
        self.anonymize = QPushButton('Anonymize')
        file_management_layout.addWidget(self.open_button_file)
        file_management_layout.addWidget(self.open_button_folder)
        file_management_layout.addWidget(self.close_button)
        file_management_layout.addWidget(self.anonymize)
        file_groupbox.setLayout(file_management_layout)

        self.central_layout.addWidget(self.snake_init_widget)
        self.central_layout.addWidget(seg_groupbox)
        self.central_layout.addWidget(file_groupbox)

        # Shortcuts
        self.run_button.setToolTip("Run segmentation of left ventricle (Enter)")
        shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        shortcut.activated.connect(self.on_run_segmentation)

        self.cancel_button.setToolTip("Cancel segmentation process (Esc)")
        shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut.activated.connect(self.on_cancel)

        self.open_button_file.setToolTip("Open a new DICOM file (Ctrl + O)")
        shortcut = QShortcut(QKeySequence("Ctrl+o"), self)
        shortcut.activated.connect(self.on_click_load_file)

        self.open_button_folder.setToolTip("Open a folder with DICOM files (Ctrl + Shift + O)")
        shortcut = QShortcut(QKeySequence("Ctrl+Shift+o"), self)
        shortcut.activated.connect(self.on_click_load_folder)

        # Connections
        self.run_button.clicked.connect(self.on_run_segmentation)
        self.cancel_button.clicked.connect(self.on_cancel)
        self.open_button_file.clicked.connect(self.on_click_load_file)
        self.open_button_folder.clicked.connect(self.on_click_load_folder)
        self.close_button.clicked.connect(self.on_click_close_image)
        self.anonymize.clicked.connect(self.on_anonymize)

        self.show_3d.clicked.connect(self.on_show_3d_model)
        self.snake_init_widget.request_zone.connect(self.left_image.image_mode)
        self.snake_init_widget.new_radious.connect(self.left_image.set_circle_radious)
        self.snake_init_widget.contours_changed.connect(self.on_initial_contours_changed)

    def add_left_image_widget(self):
        '''
        Add the widget where we are going to show the image.
        It will be added in the left side of the main window.
        The image widget is located in an Scrollable area, so it does not
        matter the size of the screen, we can always see all the images
        from the opened MRI file
        '''
        self.message_label = QLabel('No image loaded...')
        self.left_layout.addWidget(self.message_label)
        self.left_image = ImageWidget()
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidget(self.left_image)
        self.left_scroll.setWidgetResizable(True)
        self.left_layout.addWidget(self.left_scroll)

        self.left_image.pix_coordinates_event.connect(self.on_pixel_coords_received)

    def add_right_image_widget(self):
        '''
        Add the widget where we are going to show the results of the segmentation.
        It will be added in the right side of the main window.
        The image widget is located in an Scrollable area, so it does not
        matter the size of the screen, we can always see all the images
        from the opened MRI file
        '''
        self.right_image = ImageWidget()
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidget(self.right_image)
        self.right_scroll.setWidgetResizable(True)
        self.right_layout.addWidget(self.right_scroll)

    def add_user_info_area(self):
        '''
        Add a widget where patient information is going to be shown.
        '''
        self.user_data = UserDataWidget()
        self.left_layout.addWidget(self.user_data)

    def change_patient_data(self, index):
        '''
        Update the shown patient data

        Args:
            index: Index from the patient's array that indicated
                which patient we want to read
        '''
        patient_data = self.dicom_parser.get_patient_data()
        assert(index < len(patient_data))
        if len(patient_data):
            self.user_data.set_user_data(patient_data[index])

    def set_buttons_enabled(self, enabled):
        '''
        Enable or disable all the buttons at once, except for the cancel
        button.
        We don't change the cancel button since it will be the only way to
        abort the processing when the segmentation is running

        Args:
            enabled: Boolean. If True, this function will enable all the buttons,
                and it will disable them if this argument if False.
        '''
        self.run_button.setEnabled(enabled)
        self.open_button_file.setEnabled(enabled)
        self.open_button_folder.setEnabled(enabled)
        self.close_button.setEnabled(enabled)
        self.buttons_groupbox.setEnabled(enabled)
        self.anonymize.setEnabled(enabled)
        self.show_3d.setEnabled(enabled)
        self.slide_spinbox.setEnabled(enabled)
        self.snake_init_widget.setEnabled(enabled)

    # Slots
    def on_cancel(self):
        '''
        Slot to be executed when the cancel button is pressed.
        If an image is loaded, this function will enable the other buttons,
        restore the normal visualization of the loaded image and it will
        cancel the segmentation procedure
        '''
        if self.is_image_loaded:
            self.is_segm_running = False
            self.left_image.image_mode(image_mode=NORMAL_IMAGE)
        self.snake_init_widget.on_cancel()

    def on_run_segmentation(self):
        '''
        Slot executed when the RUN button is pressed.
        It will disable all the buttons and set the image mode to single mode
        '''
        if self.is_image_loaded:
            # We convert the image shape for compatibility with the segmentation module
            index = self.slide_spinbox.value()
            current_image = self.dicom_parser.get_image_data()[:,:,index:index+1,:]
            # Per construction, the masks keys and the centroids keys are the same
            initial_contours_masks = self.snake_init_widget.get_initial_contours_masks(current_image.shape)
            initial_contours_centroids = self.snake_init_widget.get_initial_contours_centroids()

            if len(list(initial_contours_masks.keys())):
                self.is_segm_running = True
                self.set_buttons_enabled(False)
                self.left_image.reset_contours()
                # We change the segmentation method based on user selection
                self.segm_module =  sh.SegmentationHub(self.bt_group.checkedId())
                final_contours = []
                for key, val in initial_contours_masks.items():
                    if not self.is_segm_running:
                        break
                    is_ok, contours = self.segm_module.start_segmentation(
                        image_set=current_image,
                        image_idx=0,
                        initial_mask=initial_contours_masks[key],
                        centroid=initial_contours_centroids[key])
                    if is_ok:
                        final_contours.append(contours)

                    # The segmentation can take a lot of time, so we refresh the Qt GUI
                    QCoreApplication.processEvents()

                if self.is_segm_running:
                    self.contours_dict[index] = final_contours
                    self.right_image.set_contours_list(final_contours)

            self.set_buttons_enabled(True)
            self.left_image.image_mode(image_mode=NORMAL_IMAGE)

    def on_click_load_file(self):
        '''
        Slot executed when the Open file button is pressed.
        It will show a file dialog where the user has to choose a file.
        If a valid string is provided, then the filepath is provided to the
        DICOM parser to try to open the file.
        '''
        fname = QFileDialog.getOpenFileName(self,
            'Open file',
            '',
            "DICOM format files ( * )")
        if fname[0]:
            self.on_click_close_image()
            self.dicom_parser.request_open_file(fname[0])
        else:
            print("No path value entered.")

    def on_click_load_folder(self):
        '''
        Slot executed when the Open folder button is pressed.
        It will show a file dialog where the user has to choose a folder.
        If a valid string is provided, then the folder path is provided to the
        DICOM parser to try to search and open files on it.

        Inspired in https://forum.qt.io/topic/62138/qfiledialog-choose-directories-only-but-show-files-as-well/8
        '''
        dialog = QFileDialog();
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        if(dialog.exec()):
            folder_name = dialog.directory().absolutePath()
            self.on_click_close_image()
            self.dicom_parser.request_open_folder(folder_name)
        else:
            print("No path value entered.")

    def on_click_close_image(self):
        '''
        Slot executed when we press the Close file button.
        It will clear the local image buffers and it will show the label of
        "No image loaded"
        '''
        self.left_image.hide()
        self.right_image.hide()
        self.is_image_loaded = False
        self.snake_init_widget.set_image_loaded(False)
        self.user_data.clear_fields()
        self.message_label.show()
        self.slide_spinbox.setEnabled(False)
        self.dicom_parser.close_file()

    def on_dicom_read_finished(self):
        '''
        Slot executed when the DICOM parser opened a file successfully.
        '''
        images = self.dicom_parser.get_image_data()
        if images.shape[0]:
            self.contours_dict = {}
            self.initial_contours_dict = {}

            self.on_new_slide_selected(0)
            self.message_label.hide()
            self.is_image_loaded = True
            self.snake_init_widget.set_image_loaded(True)
            self.slide_spinbox.setEnabled(True)
            self.slide_spinbox.setMaximum(images.shape[2] - 1)
        else:
            print("Read has no elements")
            self.on_click_close_image()

    def on_new_slide_selected(self, val):
        '''
        Set the new image and new patient data, based on which slide we select
        '''
        images = self.dicom_parser.get_image_data()
        self.change_patient_data(val)
        self.left_image.reset_contours()
        self.left_image.set_image(images[:,:,val,0])
        self.right_image.set_image(images[:,:,val,0])
        self.right_image.reset_contours()

        contour = self.contours_dict.get(val)
        if not contour is None:
            self.right_image.set_contours_list(contour)

        initial_cont = self.initial_contours_dict.get(val)
        if not initial_cont is None:
            self.snake_init_widget.restore_zones_dict(initial_cont)
            self.left_image.set_initial_contours(initial_cont)
        else:
            self.snake_init_widget.clear_contours()

        self.slide_spinbox.setValue(val)

    def on_anonymize(self):
        '''
        Slot that will be called if the user clicks the Anonymize button.
        It will pop up a dialog to chose the destination folder for the anonymized
        files
        '''
        images = self.dicom_parser.get_image_data()
        if images.shape[0]:
            dialog = QFileDialog(self, 'Select destination folder')
            dialog.setFileMode(QFileDialog.DirectoryOnly)
            dialog.setOption(QFileDialog.ShowDirsOnly, False)
            if(dialog.exec()):
                folder_path = dialog.directory().absolutePath()
                self.dicom_parser.annonymize(folder_path)
        else:
            print("Before annonymize, open a file or folder")

    def on_show_3d_model(self):
        '''
        Slot that will be called whenever the user wants to see the 3D
        representation of the segmentation.

        This will show a matplotlib plot, where each contour is repeated in
        Z axis. Only if the image is loaded and there are contours, the function
        will do something.

        If the loaded information does not contain the spatial resolution,
        or the slice thickness, a QMessageBox will be shown
        '''
        if self.is_image_loaded:
            index = self.slide_spinbox.value()
            contour = self.contours_dict.get(index)
            if not contour is None:
                thickness = self.user_data.user_data_dict['Slice thickness']
                spatial_res = self.user_data.user_data_dict['Pixel spacing']
                # we check if the DICOM file provided the data or not,
                # and if it is a float value
                if not thickness or not is_number(thickness):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("The DICOM file does not have thickness information\n" + \
                        "The 3D plot height is not representative of the actual size\n" + \
                        "of the organs")
                    msg.setWindowTitle("No thickness information")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec()
                    thickness = 10
                else:
                    thickness = float(thickness)

                # we check if the DICOM file provided the data or not,
                # and if it is a list of two float values
                if not all(spatial_res) or \
                    len(spatial_res) != 2 or \
                    not is_number(spatial_res[0]) or \
                    not is_number(spatial_res[1]):
                    msg = QMessageBox()
                    msg.setText("The DICOM file does not have spatial resolution information\n" + \
                        "The 3D plot X and Y axis is not representative of the actual size\n" + \
                        "of the organs. The measure is done in pixels")
                    msg.setWindowTitle("No spatial resolution information")
                    spatial_res = [1, 1]
                else:
                    spatial_res = [float(spatial_res[0]), float(spatial_res[1])]

                index = self.slide_spinbox.value()
                current_image = self.dicom_parser.get_image_data()[:,:,index:index+1,:]
                self.model_3d.show_3d_model(contour, spatial_res, thickness, current_image.shape)

    def request_end_of_contour(self):
        '''
        Slot executed when the user clicks the button OK to finish
        entering the initial snake. If it is in manual mode, and the user
        entered more than 2 points, then a QMessage box will be pop up saying
        that the contour has been added correctly, and the contour will be
        transfered to the snake init widget.
        '''
        input_length = len(self.left_image.get_last_contour())
        msg = QMessageBox()
        self.left_image.image_mode(NORMAL_IMAGE)

        if input_length > 2:
            my_dict = {}
            my_dict['type'] = snake_init_list[self.snake_init_widget.snk_init_group.checkedId()]
            my_dict['data'] = self.left_image.get_last_contour()
            self.snake_init_widget.set_last_requested_contour(my_dict)
            msg.setIcon(QMessageBox.Information)
            msg.setText("Initial contour added")
            msg.setWindowTitle("Success!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            self.snake_init_widget.set_buttons_enabled(True)
            # Inspired in https://www.tutorialspoint.com/pyqt/pyqt_qmessagebox.htm
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Cannot create initial contour")
            msg.setInformativeText(
                "If you are using circular snake, you must click in the image the center" +
                " of the area of interest. If you are using the manual snake, it must contain" +
                " more than 2 points. Please retry")
            msg.setWindowTitle("Error in initial snake")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()

    def on_pixel_coords_received(self, x_coord, y_coord):
        '''
        Slot executed when the user clicked in the image. It will be called
        only if there is a loaded image and if program is expecting a
        an snake initialization

        Args:
            x_coord: X coordinate that corresponds to the pixel in the image
                of the point touched
            y_coord: Y coordinate that corresponds to the pixel in the image
                of the point touched
        '''
        self.left_image.image_mode(NORMAL_IMAGE)
        my_dict = {}
        my_dict['type'] = snake_init_list[self.snake_init_widget.snk_init_group.checkedId()]
        my_dict['center_x'] = x_coord
        my_dict['center_y'] = y_coord
        self.snake_init_widget.set_last_requested_contour(my_dict)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Initial contour added")
        msg.setWindowTitle("Success!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def on_initial_contours_changed(self):
        '''
        Slot executed whenver one of the initial contours stored in the snake
        init widget had changed
        '''
        polygons = self.snake_init_widget.get_initial_contours_polygons()
        self.left_image.set_initial_contours(polygons)
        index = self.slide_spinbox.value()
        self.initial_contours_dict[index] = copy.deepcopy(polygons)