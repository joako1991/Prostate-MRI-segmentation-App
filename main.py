'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''
'''
This file is the main script that needs to be executed in order to run the application.
It make uses of all the other modules defined in this folder.
'''

from mainwindow import MainWindow
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    return 0

if __name__ == "__main__":
    main()