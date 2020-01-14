# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2_UI.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, sip
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import math


class Ui_MainWindow(object):
    def __init__(self):
        self.file_path = ''


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1284, 839)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.M = MainWindow

        # 选择图片
        self.choose_img = QtWidgets.QPushButton(self.centralwidget)
        self.choose_img.setGeometry(QtCore.QRect(20, 10, 93, 28))
        self.choose_img.setObjectName("choose_img")

        # 图片显示区域
        self.bin_img = QtWidgets.QLabel(self.centralwidget)
        self.bin_img.setGeometry(QtCore.QRect(830, 40, 400, 350))
        self.bin_img.setObjectName("bin_img")
        self.original_img = QtWidgets.QLabel(self.centralwidget)
        self.original_img.setGeometry(QtCore.QRect(350, 40, 400, 350))
        self.original_img.setObjectName("original_img")
        self.gray_img = QtWidgets.QLabel(self.centralwidget)
        self.gray_img.setGeometry(QtCore.QRect(350, 430, 400, 350))
        self.gray_img.setObjectName("gray_img")
        self.process_img = QtWidgets.QLabel(self.centralwidget)
        self.process_img.setGeometry(QtCore.QRect(830, 430, 400, 350))
        self.process_img.setObjectName("process_img")

        # 操作
        self.edge_detection = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection.setGeometry(QtCore.QRect(40, 550, 141, 31))
        self.edge_detection.setObjectName("edge_detection")
        self.C_dilation = QtWidgets.QPushButton(self.centralwidget)
        self.C_dilation.setGeometry(QtCore.QRect(40, 610, 141, 28))
        self.C_dilation.setObjectName("C_dilation")
        self.G_reconstruction = QtWidgets.QPushButton(self.centralwidget)
        self.G_reconstruction.setGeometry(QtCore.QRect(40, 670, 141, 28))
        self.G_reconstruction.setObjectName("G_reconstruction")
        self.gradient = QtWidgets.QPushButton(self.centralwidget)
        self.gradient.setGeometry(QtCore.QRect(40, 730, 141, 28))
        self.gradient.setObjectName("gradient")

        # kernel设置
        self.kernel_input = QtWidgets.QTextEdit(self.centralwidget)
        self.kernel_input.setGeometry(QtCore.QRect(20, 100, 271, 261))
        self.kernel_input.setObjectName("kernel_input")
        self.origin_x_input = QtWidgets.QLineEdit(self.centralwidget)
        self.origin_x_input.setGeometry(QtCore.QRect(130, 410, 113, 21))
        self.origin_x_input.setObjectName("origin_x_input")
        self.origin_y_input = QtWidgets.QLineEdit(self.centralwidget)
        self.origin_y_input.setGeometry(QtCore.QRect(130, 470, 113, 21))
        self.origin_y_input.setObjectName("origin_y_input")
        self.origin_x_txt = QtWidgets.QLabel(self.centralwidget)
        self.origin_x_txt.setGeometry(QtCore.QRect(30, 410, 72, 15))
        self.origin_x_txt.setObjectName("origin_x_txt")
        self.origin_t_txt = QtWidgets.QLabel(self.centralwidget)
        self.origin_t_txt.setGeometry(QtCore.QRect(30, 470, 72, 15))
        self.origin_t_txt.setObjectName("origin_t_txt")
        self.kernel_txt = QtWidgets.QLabel(self.centralwidget)
        self.kernel_txt.setGeometry(QtCore.QRect(30, 54, 171, 31))
        self.kernel_txt.setObjectName("kernel_txt")
        self.line_1 = QtWidgets.QLabel(self.centralwidget)
        self.line_1.setGeometry(QtCore.QRect(210, 40, 72, 15))
        self.line_1.setObjectName("line_1")
        self.line_2 = QtWidgets.QLabel(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(210, 60, 72, 15))
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QLabel(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(210, 80, 72, 15))
        self.line_3.setObjectName("line_3")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1284, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # bind function

        self.choose_img.clicked.connect(self.openFile)
        self.edge_detection.clicked.connect(self.process_img)
        self.C_dilation.clicked.connect(self.process_img)
        self.G_reconstruction.clicked.connect(self.process_img)
        self.gradient.clicked.connect(self.process_img)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.choose_img.setText(_translate("MainWindow", "选择图片"))
        self.bin_img.setText(_translate("MainWindow", "二值图像"))
        self.original_img.setText(_translate("MainWindow", "原图像"))
        self.gray_img.setText(_translate("MainWindow", "灰度图像"))
        self.process_img.setText(_translate("MainWindow", "处理后图像"))
        self.edge_detection.setText(_translate("MainWindow", "edge_detection"))
        self.C_dilation.setText(_translate("MainWindow", "C_dilation"))
        self.G_reconstruction.setText(_translate("MainWindow", "G_reconstruction"))
        self.gradient.setText(_translate("MainWindow", "gradient"))
        self.kernel_input.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'SimSun\';\"><br /></p></body></html>"))
        self.origin_x_txt.setText(_translate("MainWindow", "origin_x:"))
        self.origin_t_txt.setText(_translate("MainWindow", "origin_y:"))
        self.kernel_txt.setText(_translate("MainWindow", "kernel 格式（如3*3）："))
        self.line_1.setText(_translate("MainWindow", "1，1，1"))
        self.line_2.setText(_translate("MainWindow", "1，0，1"))
        self.line_3.setText(_translate("MainWindow", "0，0，1"))

    def openFile(self):
        try:
            get_filename_path, ok = QFileDialog.getOpenFileName(self.choose_img,
                                                                "打开图片",
                                                                "C:/",
                                                                "*.jpg;;*.png;;All Files(*)")
            if ok:
                self.file_path=str(get_filename_path)
                try:
                    jpg = QtGui.QPixmap(self.file_path)
                    width = jpg.width()
                    height = jpg.height()
                    x = self.original_img.width() / width
                    y = self.original_img.height() / height
                    ratio = 1
                    if x < 1 or y < 1:
                        ratio = min(x, y)
                    print(ratio)
                    jpg = jpg.scaled(math.floor(width * ratio), math.floor(height * ratio))
                    self.original_img.setPixmap(jpg)
                    self.path_flag = 1
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                    self.file_path = ''
                    print('path err')
                    return
        except:
            QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
            print('path err')
            self.file_path = ''

    def read_img(self,path):
        try:
            img = cv2.imread(path)
        except:
            print('cannot find img')
            img = cv2.imread('./pic/pkq.jpg')
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayImage

    def gray2bin(self,path, threshold=200):
        grayImage = self.read_img(path)
        ret, binImage = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY_INV)
        print(binImage[1][2])

        return binImage

    def erode_bin(self,binImage, kernel, x, y):
        d_image = np.zeros(shape=binImage.shape)
        kernel_size = kernel.shape[0]
        for i in range(x, binImage.shape[0] - kernel_size + x + 1):
            for j in range(y, binImage.shape[1] - kernel_size + y + 1):
                flag = 1
                for k_x in range(0, kernel_size):
                    if flag == 0:
                        break
                    for k_y in range(0, kernel_size):
                        if kernel[k_x][k_y] == 1 and binImage[i - x + k_x][j - y + k_y] < 255:
                            flag = 0
                            break
                if flag == 1:
                    d_image[i, j] = 255
        return d_image

    def dilate_bin(self,binImage, kernel, x, y):
        kernel_size = kernel.shape[0]
        d_image = np.ones(shape=binImage.shape)
        for i in range(x, binImage.shape[0] - kernel_size + x + 1):
            for j in range(y, binImage.shape[1] - kernel_size + y + 1):
                if binImage[i, j] == 255:
                    for k_x in range(0, kernel_size):
                        for k_y in range(0, kernel_size):
                            if kernel[k_x][k_y] > 0:
                                d_image[i - x + k_x, j - y + k_y] = 255
        return d_image

    def edge_detection(self,binImage, kernel, x, y):
        return self.dilate_bin(binImage, kernel, x, y) - self.erode_bin(binImage, kernel, x, y)

    def erode_gray(self,grayImage, kernel, x, y):
        kernel_size = kernel.shape[0]
        d_image = np.ones(shape=grayImage.shape)
        for i in range(x, grayImage.shape[0] - kernel_size + x + 1):
            for j in range(y, grayImage.shape[1] - kernel_size + y + 1):
                tmp_array = []
                for k_x in range(0, kernel_size):
                    for k_y in range(0, kernel_size):
                        tmp_array.append(grayImage[i - x + k_x, j - y + k_y] - kernel[k_x, k_y])
                d_image[i][j] = min(tmp_array)
        return d_image

    def dilate_gray(self,grayImage, kernel, x, y):
        kernel_size = kernel.shape[0]
        d_image = np.ones(shape=grayImage.shape)
        for i in range(x, grayImage.shape[0] - kernel_size + x + 1):
            for j in range(y, grayImage.shape[1] - kernel_size + y + 1):
                tmp_array = []
                for k_x in range(0, kernel_size):
                    for k_y in range(0, kernel_size):
                        tmp_array.append(grayImage[i - x + k_x, j - y + k_y] + kernel[k_x, k_y])
                d_image[i][j] = max(tmp_array)
        return d_image

    def gradient(self,grayImage, kernel, x, y):
        return (self.dilate_gray(grayImage, kernel, x, y) - self.erode_gray(grayImage, kernel, x, y)) * 0.5

    def conditional_dilation(self,marker, mask, kernel, x, y):
        while True:
            new = np.min((self.dilate_bin(marker, kernel, x, y), mask), axis=0)
            if (new == marker).all():
                return marker
            marker = new

    # 膨胀重建
    def grayscale_reconstruction(self,marker, mask, kernel, x, y):
        c = 0
        while True:
            new = np.min((cv2.dilate(marker, kernel, iterations=1), mask), axis=0)
            if (new == marker).all():
                return marker
            marker = new
            c += 1
            print(c)

    def process_img(self):
        print('ddd')
        return