# README
This repository corresponds to the GUI application, totally developed in Python, for Prostate MRI segmentation.
This program allows to open DICOM files (or folder with files), anonymize them, segment them using different algorithms (until now, 3 snake based algorithms have been implemented), initialize the snakes using circular initial masks, or random polygon initialization, and visualize patient data, extracted from the DICOM file.
It includes also a basic 3D visualization of the segmentation result, using MatPlotLib like visualization.
<br /><br />
This program not only works only, but it also provides a code base for the user interface for developing novel prostate algorithms in Python, and the program is ready yo use. The idea of having a program like this is also to have a place where several algorithms are implemented, and the results of a new algorithm can be directly compared with the already existing ones. For a template of how to implement a class for a new segmentation segmentation method, the files ***acs_basic***, ***acs_improved*** and ***acs_gvf*** in the folder ***image_segmentation*** can be used. Each of these files receive the image, the initial mask (can be ignored in your implementation). You class must return the segmentation mask and the level set function (or None if you don't use it). Then check the Segmentation Hub class to see how to create add a new segmentation (10 lines of code are needed).

# Main program window
![GitHub Logo](/GUI.png)


# Citation
If you find this code useful for your research, please cite the paper:
```
@Article{app10186163,
    AUTHOR = {Rodríguez, Joaquín and Ochoa-Ruiz, Gilberto and Mata, Christian},
    TITLE = {A Prostate MRI Segmentation Tool Based on Active Contour Models Using a Gradient Vector Flow},
    JOURNAL = {Applied Sciences},
    VOLUME = {10},
    YEAR = {2020},
    NUMBER = {18},
    ARTICLE-NUMBER = {6163},
    URL = {https://www.mdpi.com/2076-3417/10/18/6163},
    ISSN = {2076-3417},
    DOI = {10.3390/app10186163}
}
```


# Getting stated
In order to run this program, some dependencies are required. The commands are Linux-based commands, but the libraries are available for windows and Mac too, since they are for Python and Qt.

### 1. Python3
sudo apt-get update
sudo apt-get install python3

### 2. PyQt5 (for Python3)
sudo apt-get install python3-pyqt5 <br />
 ***NOTE***: It is strongly suggested to also install Qt from the original webpage

### 3. Python Dependencies
sudo apt-get install -y python3-tk <br />
python3 -m pip install --upgrade pip <br />
python3 -m pip install pydicom matplotlib scipy scikit-image numpy opencv-python <br />
python3 -m pip install --no-cache-dir git+https://github.com/pydicom/pydicom.git <br />

# How to run?
Just go to the root folder of the repository, and execute this command
***python3 ./main.py***