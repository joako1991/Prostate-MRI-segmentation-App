'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

import numpy as np
import copy
import pydicom
import os
import glob
from collections import OrderedDict

from PyQt5.QtCore import \
    QObject, \
    pyqtSignal, \
    QDate

class DicomFormat(QObject):
    '''
    Class to wrap the DICOM file handling.
    With this class, we will be able to anonymize files and
    save them, load a folder with DICOM files, load a single file,
    extract patient information and extract the images.

    When a folder or a file is requested to be opened, a signal read_finished
    will be emited
    '''
    read_finished = pyqtSignal()

    def __init__(self, parent=None):
        '''
        Constructor
        '''
        super(DicomFormat, self).__init__(parent)
        self.last_folder_read = np.array([])
        self.image_data = np.array([])
        self.patient_data = np.array([])
        self.str_format = 'dd/MM/yyyy'

    def request_open_folder(self, folder_path):
        '''
        Open a folder, search for DICOM files and try to open them.
        This search is not recursive. if there is a folder with more
        DICOM format files, they will not be loaded. Only the files
        that are in the specified folder

        Args:
            folder_path: Root folder where the DICOM files are.
                When the read has been finished, a signal read_finished
                will be emited.
                Files that are not in DICOM format are present in this folder,
                the function will ignore them.
        '''
        assert(folder_path and folder_path.strip())
        self.last_folder_read = []
        self.image_data = []
        self.patient_data = []
        list_with_pattern = sorted(glob.glob(os.path.abspath(os.path.join(folder_path, '*'))))
        counter = 0
        for element in list_with_pattern:
            if (self.add_new_file(element, counter)):
                counter += 1

        self.normalize_images()
        # We emit a signal to inform that the read had finished
        self.read_finished.emit()

    def request_open_file(self, filepath):
        '''
        Open a file and try to open it.

        Args:
            filepath: Path of the file to load.
                When the read has been finished, a signal read_finished
                will be emited.
                If the format of the file is not DICOM, this function
                will do nothing
        '''
        assert(filepath and filepath.strip())
        self.last_folder_read = []
        self.image_data = []
        self.patient_data = []

        self.add_new_file(filepath, 0)
        self.normalize_images()

        # We emit a signal to inform that the read had finished
        self.read_finished.emit()

    def add_new_file(self, filepath, index):
        '''
        Read a file, and if it exists, and it is DICOM format, extract
        and save locally its information. The information stored is
        patient information, image, and the DICOM object.

        Args:
            filepath: Path of the file to read
            index: Reference to which image has been loaded. Useful when a
                folder is loaded with several files

        Returns:
            True if the input path is in fact a DICOM file
        '''
        ret_val = False
        if os.path.isfile(filepath):
            try:
                dataset = pydicom.filereader.dcmread(filepath)
                print("DICOM file found {f}".format(f=os.path.basename(filepath)))
                tmp_dict = OrderedDict()
                tmp_dict['filename'] = filepath
                tmp_dict['data'] = dataset
                img = dataset.pixel_array

                self.last_folder_read.append(tmp_dict)
                self.image_data.append(img)
                self.patient_data.append(self.extract_important_data(dataset, index))
                ret_val = True
            except pydicom.errors.InvalidDicomError:
                print("File rejected {f}".format(f=os.path.basename(filepath)))
                pass
        return ret_val

    def normalize_images(self):
        '''
        Modify the loaded images to have grayscale values between 0 and 255.0.
        It will translate the values if the minimum value is negative,
        and it will divide by the maximum value and multiply by 255.0
        '''
        self.image_data = np.array(self.image_data)
        if len(self.image_data):
            # We normalize the images so the values will be between 0 and 255
            min_val = np.min(self.image_data)
            if min_val < 0:
                self.image_data = self.image_data + abs(min_val)
            max_val = np.max(self.image_data)
            if max_val:
                self.image_data = np.array(self.image_data * (255.0 / max_val), dtype=np.uint8)

            # We swap the axes so the order is [X, Y, N], where N is the image index
            self.image_data = np.swapaxes(self.image_data, 0, 1)
            self.image_data = np.swapaxes(self.image_data, 1, 2)
            # We add an additional index to keep compatibility with Nifty format
            self.image_data = self.image_data.reshape(self.image_data.shape + (1,))

    def extract_important_data(self, file, counter):
        '''
        Extract patient and study information from a DICOM formatted file

        Inspired in https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html
        Args:
            file: DICOM object already loaded from a file
            counter: When a lot of files has been loaded, this variable
                makes reference to the index of the loaded image.

        Returns:
            Dictionary with the loaded data. The loaded fields are:
                    'Patient name'
                    'Patient ID'
                    'Patient birth date'
                    'Study ID'
                    'Study date'
                    'Slice location'
                    'Instance number'

                If a field is missing, then the string (missing) is attached
        '''
        file_info_dict = {}
        # We use .get() because we are not not sure the item exists,
        # and if it is not present, we put a default value
        pat_name = file.get('PatientName', None)
        if not pat_name is None:
            file_info_dict['Patient name'] = pat_name.family_name + ", " + pat_name.given_name
        else:
            file_info_dict['Patient name'] = '(missing)'

        file_info_dict['Patient ID'] = file.get('PatientID', "(missing)")
        birthday = file.get('PatientBirthDate', None)
        if not birthday is None:
            file_info_dict['Patient birth date'] = self.convert_date(birthday)
        else:
            file_info_dict['Patient birth date'] = '(missing)'

        file_info_dict['Study ID'] = file.get('StudyID', "(missing)")
        study_date = file.get('StudyDate', "(missing)")
        if not study_date is None:
            file_info_dict['Study date'] = self.convert_date(study_date)
        else:
            file_info_dict['Study date'] = '(missing)'

        file_info_dict['Slice location'] = file.get('SliceLocation', "(missing)")
        file_info_dict['Instance number'] = str(counter)
        file_info_dict['Slice thickness'] = file.get('SliceThickness', "(missing)")
        file_info_dict['Pixel spacing'] = file.get('PixelSpacing', "(missing)")
        return file_info_dict

    def convert_date(self, date):
        '''
        Take a DICOM date and transform it into human readable date.

        Args:
            date: DICOM formatted date YYYYMMDD, where YYYY is the year,
                MM is the month and DD is the date. If the string
                contains more values after this string, they are
                ignored.

        Returns:
            String with the shape DD/MM/YYYY
        '''
        date_string = str(date)
        year = int(date_string[0:4])
        month = int(date_string[4:6])
        day = int(date_string[6:8])
        return QDate(year,month,day).toString(self.str_format)

    def get_image_data(self):
        '''
        Get the previously loaded images from DICOM files
        '''
        return self.image_data

    def get_patient_data(self):
        '''
        Get the previously loaded patient and study information
        from DICOM files
        '''
        return self.patient_data

    def annonymize(self, dest_folder):
        '''
        Anonymize DICOM file. Implementation inspired in
        https://pydicom.github.io/pydicom/stable/auto_examples/metadata_processing/plot_anonymize.html

        The fields that are going to be anonymized are:
                'PatientName'      : 'NN'
                'PatientID'        : 'A00000000000'
                'PatientBirthDate' : '19700101'
        The second column are the values that are going to replace the
        original values. It will anonymize the previously loaded file/s.

        The resulting anonymized files are going to be stored in the
        given folder

        Args:
            dest_folder: Destination folder where to store the resulting files
        '''
        if len(self.last_folder_read):
            abs_dest_folder =  os.path.abspath(dest_folder)
            if not os.path.exists(abs_dest_folder):
                raise ValueError("Destination folder does not exists")

            anonymation_fields = {
                'PatientName': 'NN',
                'PatientID': 'A00000000000',
                'PatientBirthDate': '19700101',
            }
            for dicom_data in self.last_folder_read:
                dicom = dicom_data['data']
                for key, val in anonymation_fields.items():
                    if key in dicom:
                        dicom.data_element(key).value = val

                filename = dicom_data['filename']
                if 'anonymized' in os.path.basename(filename):
                    dest_file = os.path.join(abs_dest_folder, os.path.basename(filename))
                else:
                    dest_file = os.path.join(abs_dest_folder, os.path.basename(filename) + '_anonymized')
                pydicom.filewriter.dcmwrite(dest_file, dicom, write_like_original=False)

    def close_file(self):
        '''
        When the file is closed, then the internal buffers are erased
        '''
        self.last_folder_read = np.array([])
        self.image_data = np.array([])
        self.patient_data = np.array([])

def main():
    '''
    Main testing function of different functionalities of the DICOM format module
    '''
    # Dicom Class test
    dicom = DicomFormat()
    filepath = os.path.realpath(__file__)
    dirname = os.path.dirname(filepath)
    dicom.request_open_file(os.path.join(dirname, '../Database/Image00008'))
    # dicom.request_open_folder(os.path.join(dirname, '../Database'))
    # dicom.request_open_folder(os.path.join(dirname, '../output'))
    # dicom.request_open_folder(os.path.join(dirname, '../'))
    pat_data = dicom.get_patient_data()
    for elem in pat_data:
        for key,val in elem.items():
            print("{}: {}".format(key, val))

    dicom.annonymize('./output')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.ion()
    plt.show()
    images = dicom.get_image_data()

    if images.shape[0]:
        i = 0
        while True:
            img = images[:,:,i,0]
            plt.imshow(img, cmap='gray')
            plt.draw()
            plt.pause(0.05)
            i += 1
            i %= images.shape[2]

if __name__ == '__main__':
    main()