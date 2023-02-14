from argparse import FileType
from fileinput import filename
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2 as cv
import os

from UI import Ui_MainWindow
from img_controller import img_controller

class MainWindow_controller(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.file_path1 = ''
        self.img_controller = img_controller(img_path1=self.file_path1)
        self.ui.LoadButton.clicked.connect(self.open_file1) 
        self.ui.GaussianButton.clicked.connect(self.img_controller.Gaussian_Blur)
        self.ui.SobelXButton.clicked.connect(self.img_controller.Sobel_X)
        self.ui.SobelYButton.clicked.connect(self.img_controller.Sobel_Y)        
        self.ui.MagnitudeButton.clicked.connect(self.img_controller.Magnitude)
        self.ui.ResizeButton.clicked.connect(self.img_controller.Resize)
        self.ui.TranslationButton.clicked.connect(self.img_controller.Translation)
        self.ui.RotationButton.clicked.connect(self.img_controller.Rotation)        
        self.ui.ShearingButton.clicked.connect(self.img_controller.Shearing)
        
    def open_file1(self):
        filepath1,filetype1= QFileDialog.getOpenFileName(self, "Open file", " ") 
        if len(filepath1) == 0: 
            print("\n Please input correct picture")
        filename1= os.path.basename(filepath1) 
        self.ui.show_file_name1.setText(filename1)
        self.img_controller.set_path(filepath1)