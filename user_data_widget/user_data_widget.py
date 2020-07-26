'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

from PyQt5.QtWidgets import \
    QVBoxLayout, \
    QHBoxLayout, \
    QFormLayout, \
    QWidget, \
    QLabel, \
    QLineEdit

from collections import OrderedDict

from PyQt5.QtCore import \
    QDate

class UserDataWidget(QWidget):
    '''
    Class that encloses all the user information area.
    It includes a series of labels and read-only QLineEdit objects
    to show patient information.
    '''
    def __init__(self, parent=None):
        '''
        Constructor. It initializes the widget shape, and
        the internal variables with default values.
        '''
        super(UserDataWidget, self).__init__(parent)
        self.right_layout = QVBoxLayout()
        self.left_layout = QVBoxLayout()
        self.main_layout = QHBoxLayout()

        self.study_date = QDate(2010,1,1)
        self.birthday = QDate(2010,1,1)
        self.str_format = 'dd/MM/yyyy'

        self.user_data_dict = OrderedDict()
        self.user_data_dict['Patient name'] = ''
        self.user_data_dict['Patient ID'] = ''
        self.user_data_dict['Patient birth date'] = ''
        self.user_data_dict['Study ID'] = ''
        self.user_data_dict['Study date'] = ''
        self.user_data_dict['Slice location'] = ''
        self.user_data_dict['Instance number'] = ''
        self.user_data_dict['Slice thickness'] = ''
        self.user_data_dict['Pixel spacing'] = ''

        self.user_data_boxes = OrderedDict()

        self.initialize_widget()

    def set_user_data(self, input_data_dict):
        '''
        Set new patient information in the widget, and update
        the shown values.

        Args:
            input_data_dict: Dictionary with all the data to be shown.
                The dictionary must include the following keys:
                    'Patient name'       --> String
                    'Patient ID'         --> Integer number
                    'Patient birth date' --> Date with format DD/MM/YYYY
                    'Study ID'           --> String
                    'Study date'         --> Date with format DD/MM/YYYY
                    'Slice location'     --> Float number
                    'Instance number'    --> Integer number
                    'Slice thickness'    --> Float number             (milimeters)
                    'Pixel spacing'      --> List of two float number (milimeters)

                If any of these keys is missing, a KeyError exception
                is going to be thrown.
        '''
        for key in self.user_data_dict.keys():
            self.user_data_dict[key] = input_data_dict[key]
            self.user_data_boxes[key].setText(str(self.user_data_dict[key]))

    def add_field(self, label_txt, value, layout):
        '''
        Add a new line of parameters to the layout. This line will
        include a label and a read-only QLineEdit place aside the label.

        Args:
            label_txt: Text to be shown as the field name
            value: Value for the field to be added
            layout: The generated widget will be added to this layout
        '''
        line_layout = QFormLayout()
        q_label = QLabel(label_txt)

        val_edit_box = QLineEdit()
        val_edit_box.setText(str(value))
        val_edit_box.setReadOnly(True)
        self.user_data_boxes[label_txt] = val_edit_box

        line_layout.addRow(q_label, val_edit_box)

        tmp_widget = QWidget()
        tmp_widget.setLayout(line_layout)
        layout.addWidget(tmp_widget)

    def initialize_widget(self):
        '''
        Initialize the widget shape
        '''
        keys = list(self.user_data_dict.keys())
        values = list(self.user_data_dict.values())

        for i in range(int(len(keys) / 2 + 0.5)):
            self.add_field(keys[i], values[i], self.left_layout)

        for i in range(int(len(keys) / 2 + 0.5), len(keys)):
            self.add_field(keys[i], values[i], self.right_layout)

        self.set_layout_configuration()

    def set_layout_configuration(self):
        '''
        Set the layouts in such a way that there are 2 columns of QWidgets,
        and each column has a set of lines of data placed horizontally.
        '''
        left_widget = QWidget()
        right_widget = QWidget()

        left_widget.setLayout(self.left_layout)
        right_widget.setLayout(self.right_layout)

        self.main_layout.addWidget(left_widget)
        self.main_layout.addStretch()
        self.main_layout.addWidget(right_widget)
        self.setLayout(self.main_layout)

    def clear_fields(self):
        '''
        Clear all the fields shown of with the patient data
        '''
        for key in self.user_data_dict.keys():
            self.user_data_boxes[key].setText('')