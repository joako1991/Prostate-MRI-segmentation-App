'''
Author: Joaquin Rodriguez
Email: joaquinrodriguez1991@gmail.com
Le Creusot - 71200 France
Year: 2020
'''

from PyQt5.QtWidgets import \
    QHBoxLayout, \
    QVBoxLayout, \
    QWidget, \
    QLabel, \
    QGroupBox

from PyQt5.QtGui import \
    QFont

from PyQt5.QtCore import \
    Qt

class Copyright(QWidget):
    def __init__(self, parent=None):
        super(Copyright, self).__init__(parent)
        group_layout = QVBoxLayout()
        main_layout = QHBoxLayout()
        group = QGroupBox()

        author = QLabel("Authors: Joaquin Rodriguez - Christian Mata")
        font = author.font()
        font.setBold(True)
        author.setFont(font)
        author.setAlignment(Qt.AlignCenter)

        email = QLabel("Joaquin_Rodriguez@etu.u-bourgogne.fr\nchristian.mata@upc.edu")
        email.setAlignment(Qt.AlignCenter)
        rest = QLabel("Le Creusot - France. May 2020")
        rest.setAlignment(Qt.AlignCenter)

        group_layout.addWidget(author)
        group_layout.addWidget(email)
        group_layout.addWidget(rest)

        group.setLayout(group_layout)
        main_layout.addWidget(group)
        self.setLayout(main_layout)