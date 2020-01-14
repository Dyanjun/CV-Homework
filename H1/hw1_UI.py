# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw1_UI.ui'
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

matplotlib.use("Qt5Agg")


class MyFigure(FigureCanvas):
    def __init__(self, width, height, dpi):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)


class Ui_MainWindow(object):
    def __init__(self):
        self.file_path = ''
        self.c = 0
        self.path_flag = 1

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("img_txt_2")
        MainWindow.resize(895, 832)
        self.M = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        palette1 = QtGui.QPalette()
        palette1.setColor(palette1.Background, QtGui.QColor(255, 255, 255))
        self.M.setPalette(palette1)

        # self.M.setWindowFlags(Qt.FramelessWindowHint)  # 无边框，置顶
        # self.M.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景色


        self.F = MyFigure(width=3, height=2, dpi=100)
        self.groupBox = QtWidgets.QGroupBox(MainWindow)
        self.groupBox.setGeometry(QtCore.QRect(460, 410, 360, 360))
        self.groupBox.setObjectName("groupBox")
        self.gridlayout = QtWidgets.QGridLayout(self.groupBox)
        # filters
        self.Median_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Median_Button.setGeometry(QtCore.QRect(30, 400, 93, 28))
        self.Median_Button.setObjectName("Median_Button")
        self.Mean_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Mean_Button.setGeometry(QtCore.QRect(170, 400, 93, 28))
        self.Mean_Button.setMouseTracking(False)
        self.Mean_Button.setObjectName("Mean_Button")
        self.Gaussian_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Gaussian_Button.setGeometry(QtCore.QRect(170, 560, 93, 28))
        self.Gaussian_Button.setMouseTracking(False)
        self.Gaussian_Button.setObjectName("Gaussian_Button")

        # set kernel_size and sigma
        self.Kernel_IN = QtWidgets.QLineEdit(self.centralwidget)
        self.Kernel_IN.setGeometry(QtCore.QRect(150, 350, 113, 31))
        self.Kernel_IN.setObjectName("Kernel_IN")
        self.Kernel_IN.setAlignment(QtCore.Qt.AlignRight)
        self.Kernel_IN.setPlaceholderText(str(3))
        self.Kernel_IN2 = QtWidgets.QLineEdit(self.centralwidget)
        self.Kernel_IN2.setGeometry(QtCore.QRect(150, 470, 113, 31))
        self.Kernel_IN2.setObjectName("Kernel_IN")
        self.Kernel_IN2.setAlignment(QtCore.Qt.AlignRight)
        self.Kernel_IN2.setPlaceholderText(str(3))
        self.Sigma_IN = QtWidgets.QLineEdit(self.centralwidget)
        self.Sigma_IN.setGeometry(QtCore.QRect(150, 520, 113, 31))
        self.Sigma_IN.setObjectName("Sigma_IN")
        self.Sigma_IN.setAlignment(QtCore.Qt.AlignRight)
        self.Sigma_IN.setPlaceholderText(str(1))

        self.Kernel_Label = QtWidgets.QLabel(self.centralwidget)
        self.Kernel_Label.setGeometry(QtCore.QRect(30, 350, 101, 16))
        self.Kernel_Label.setObjectName("Kernel_Label")
        self.Kernel_Label2 = QtWidgets.QLabel(self.centralwidget)
        self.Kernel_Label2.setGeometry(QtCore.QRect(30, 470, 101, 16))
        self.Kernel_Label2.setObjectName("Kernel_Label")
        self.Sigma__Label = QtWidgets.QLabel(self.centralwidget)
        self.Sigma__Label.setGeometry(QtCore.QRect(60, 520, 72, 15))
        self.Sigma__Label.setObjectName("Sigma__Label")

        # operators
        self.Roberts_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Roberts_Button.setGeometry(QtCore.QRect(30, 610, 93, 28))
        self.Roberts_Button.setObjectName("Roberts_Button")
        self.Prewitt_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Prewitt_Button.setGeometry(QtCore.QRect(30, 660, 93, 28))
        self.Prewitt_Button.setObjectName("Prewitt_Button")
        self.Sobel_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Sobel_Button.setGeometry(QtCore.QRect(30, 710, 93, 28))
        self.Sobel_Button.setObjectName("Sobel_Button")

        # choose files
        self.ChooseFile_Button = QtWidgets.QPushButton(self.centralwidget)
        self.ChooseFile_Button.setGeometry(QtCore.QRect(30, 230, 120, 28))
        self.ChooseFile_Button.setObjectName("ChooseFile_Button")
        self.ChooseFile_Button2 = QtWidgets.QPushButton(self.centralwidget)
        self.ChooseFile_Button2.setGeometry(QtCore.QRect(170, 230, 93, 28))
        self.ChooseFile_Button2.setObjectName("ChooseFile_Button2")
        self.FilePath_IN = QtWidgets.QLineEdit(self.centralwidget)
        self.FilePath_IN.setGeometry(QtCore.QRect(30, 190, 231, 31))
        self.FilePath_IN.setObjectName("FilePath_IN")

        # show images
        self.img_label = QtWidgets.QLabel(self.centralwidget)
        self.img_label.setGeometry(QtCore.QRect(460, 20, 360, 360))
        self.img_label.setText("")
        self.img_label.setObjectName("img_label")
        self.img2_label = QtWidgets.QLabel(self.centralwidget)
        self.img2_label.setGeometry(QtCore.QRect(460, 410, 360, 360))
        self.img2_label.setText("")
        self.img2_label.setObjectName("img2_label")

        self.img_txt = QtWidgets.QLabel(self.centralwidget)
        self.img_txt.setGeometry(QtCore.QRect(370, 200, 72, 15))
        self.img_txt.setAlignment(QtCore.Qt.AlignCenter)
        self.img_txt.setObjectName("img_txt")
        self.img2_txt = QtWidgets.QLabel(self.centralwidget)
        self.img2_txt.setGeometry(QtCore.QRect(380, 580, 72, 15))
        self.img2_txt.setAlignment(QtCore.Qt.AlignCenter)
        self.img2_txt.setObjectName("img2_txt")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 895, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # bind function
        self.ChooseFile_Button.clicked.connect(self.openFile)
        self.ChooseFile_Button2.clicked.connect(self.confirmFile)
        self.Median_Button.clicked.connect(lambda: self.process_img('med'))
        self.Mean_Button.clicked.connect(lambda: self.process_img('mean'))
        self.Gaussian_Button.clicked.connect(lambda: self.process_img('gaus'))
        self.Roberts_Button.clicked.connect(lambda: self.process_img('r'))
        self.Prewitt_Button.clicked.connect(lambda: self.process_img('p'))
        self.Sobel_Button.clicked.connect(lambda: self.process_img('s'))

    def retranslateUi(self, img_txt_2):
        _translate = QtCore.QCoreApplication.translate
        img_txt_2.setWindowTitle(_translate("img_txt_2", "MainWindow"))
        self.Median_Button.setText(_translate("img_txt_2", "中值滤波"))
        self.Mean_Button.setText(_translate("img_txt_2", "均值滤波"))
        self.Gaussian_Button.setText(_translate("img_txt_2", "高斯滤波"))
        self.Kernel_Label.setText(_translate("img_txt_2", "kernel size:"))
        self.Kernel_Label2.setText(_translate("img_txt_2", "kernel size:"))
        self.Sigma__Label.setText(_translate("img_txt_2", "sigma:"))
        self.Roberts_Button.setText(_translate("img_txt_2", "Roberts"))
        self.Prewitt_Button.setText(_translate("img_txt_2", "Prewitt"))
        self.Sobel_Button.setText(_translate("img_txt_2", "Sobel"))
        self.ChooseFile_Button.setText(_translate("img_txt_2", "从文件夹中选择"))
        self.ChooseFile_Button2.setText(_translate("img_txt_2", "确定"))
        self.img_txt.setText(_translate("img_txt_2", "原图像"))
        self.img2_txt.setText(_translate("img_txt_2", "处理后"))

    # 三种filter
    def median(self, img, x, y, size):
        sum_s = []
        size1 = int(size / 2)
        for k in range(-size1, size1 + 1):
            for m in range(-size1, size1 + 1):
                sum_s.append(img[x + k][y + m])
        sum_s.sort()
        return sum_s[(int(size * size / 2) + 1)]

    def median_filter(self, im_copy_med, img, size):
        size1 = int(size / 2)
        img = cv2.copyMakeBorder(img, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        im_copy_med = cv2.copyMakeBorder(im_copy_med, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        # for i in range(int(size / 2), img.shape[0] - int(size / 2)):
        #     for j in range(int(size / 2), img.shape[1] - int(size / 2)):
        #         im_copy_med[i][j] = self.median(img, i, j, size)
        for i in range(size1, img.shape[0] - size1):
            for j in range(size1, img.shape[1] - size1):
                im_copy_med[i][j] = self.median(img, i, j, size)
        return im_copy_med[size1:im_copy_med.shape[0] - size1, size1:im_copy_med.shape[1] - size1]

    def mean(self, img, x, y, size):
        sum_s = 0
        size1 = int(size / 2)
        for k in range(-size1, size1 + 1):
            for m in range(-size1, size1 + 1):
                sum_s += img[x + k][y + m] / (size * size)
        return sum_s

    def mean_filter(self, im_copy_mean, img, size):
        size1 = int(size / 2)
        img = cv2.copyMakeBorder(img, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        im_copy_mean = cv2.copyMakeBorder(im_copy_mean, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        # for i in range(int(size / 2), img.shape[0] - int(size / 2)):
        #     for j in range(int(size / 2), img.shape[1] - int(size / 2)):
        #         im_copy_mean[i][j] = self.mean(img, i, j, size)
        # return im_copy_mean
        for i in range(size1, img.shape[0] - size1):
            for j in range(size1, img.shape[1] - size1):
                im_copy_mean[i][j] = self.mean(img, i, j, size)
        return im_copy_mean[size1:im_copy_mean.shape[0] - size1, size1:im_copy_mean.shape[1] - size1]

    def gaussian(self, kernel_size, sigma):
        print('gaussion kernel')
        center = kernel_size // 2
        print('hi')
        kernel = np.zeros((kernel_size, kernel_size))
        print('here')
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = sigma ** 2
        sum_val = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
                sum_val += kernel[i, j]
        kernel = kernel / sum_val
        return kernel

    def gaussian_filter(self, img, kernel):
        print('gaussian_filter')
        res_h = img.shape[0] - kernel.shape[0] + 1
        res_w = img.shape[1] - kernel.shape[1] + 1
        res = np.zeros((res_h, res_w))
        dh = kernel.shape[0]
        dw = kernel.shape[1]
        for i in range(res_h):
            for j in range(res_w):
                res[i, j] = np.sum(img[i:i + dh, j:j + dw] * kernel)
        return res

    def read_img(self, path):
        try:
            img = cv2.imread(path)
        except:
            print('cannot find img')
            img = cv2.imread('./pic/pkq.jpg')
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayImage

    def Filter(self, path, type, kernel_size, sigma=1):
        if self.c:
            print('delete F')
            # plt.close()
            self.gridlayout.removeWidget(self.F)
            sip.delete(self.F)
            print(self.gridlayout.count())
            self.F = MyFigure(width=3, height=2, dpi=100)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        if type == 'med':
            grayImage = self.read_img(path)
            im_copy_med = self.read_img(path)
            medimg = self.median_filter(im_copy_med, grayImage, kernel_size)
            print(medimg.shape[0])
            print(medimg.shape[1])
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
            # plt.subplot(122),
            plt.rcParams['figure.figsize'] = (medimg.shape[0], medimg.shape[1])
            plt.imshow(medimg, cmap=plt.cm.gray), plt.title(u'中值'), plt.axis('off')
            # plt.show()
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 'mean':
            grayImage = self.read_img(path)
            im_copy_mean = self.read_img(path)
            meanimg = self.mean_filter(im_copy_mean, grayImage, kernel_size)
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
            # plt.subplot(122),
            plt.imshow(meanimg, cmap=plt.cm.gray), plt.title(u'均值'), plt.axis('off')
            # plt.show()
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 'gaus':
            grayImage = self.read_img(path)
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
            gausimg = self.gaussian_filter(grayImage, self.gaussian(kernel_size, sigma))
            # plt.subplot(122),
            plt.imshow(gausimg, cmap=plt.cm.gray), plt.title(u'高斯'), plt.axis('off')
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            print('gaus')
            return
        print('Type Erro')
        return

    # 三种算子
    def RobertsOperator(self, roi):
        operator_first = np.array([[-1, 0], [0, 1]])
        operator_second = np.array([[0, -1], [1, 0]])
        return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))

    def RobertsAlogrithm(self, image):
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        for i in range(1, image.shape[0]):
            for j in range(1, image.shape[1]):
                image[i, j] = self.RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
        return image[1:image.shape[0], 1:image.shape[1]]

    def PreWittOperator(self, roi):
        prewitt_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # result = np.abs(np.sum(roi * prewitt_x)) + np.abs(np.sum(roi * prewitt_y))
        # result = np.abs(np.sum(roi * prewitt_x))*0.5+np.abs(np.sum(roi * prewitt_y))*0.5
        result = (np.abs(np.sum(roi * prewitt_x)) ** 2 + np.abs(np.sum(roi * prewitt_y)) ** 2) ** 0.5
        return result

    def PreWittAlogrithm(self, image):
        new_image = np.zeros(image.shape)
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                new_image[i - 1, j - 1] = self.PreWittOperator(image[i - 1:i + 2, j - 1:j + 2])
        return new_image.astype(np.uint8)

    def SobelOperator(self, roi):
        sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        result = (np.abs(np.sum(roi * sobel_x)) ** 2 + np.abs(np.sum(roi * sobel_y)) ** 2) ** 0.5
        return result

    def SobelAlogrithm(self, image):
        new_image = np.zeros(image.shape)
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                new_image[i - 1, j - 1] = self.SobelOperator(image[i - 1:i + 2, j - 1:j + 2])
        return new_image.astype(np.uint8)

    def operator(self, path, type):
        if self.c:
            print('delete F')
            # plt.close()
            self.gridlayout.removeWidget(self.F)
            sip.delete(self.F)
            print(self.gridlayout.count())
            self.F = MyFigure(width=3, height=2, dpi=100)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        grayImage = self.read_img(path)
        if type == 'r':
            Roberts_saber = self.RobertsAlogrithm(grayImage)
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'gray'), plt.axis('off')
            # plt.subplot(122),
            plt.imshow(Roberts_saber, cmap=plt.cm.gray), plt.title(u'Roberts'), plt.axis('off')
            # plt.show()
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 'p':
            PreWitt_saber = self.PreWittAlogrithm(grayImage)
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'gray'), plt.axis('off')
            # plt.subplot(122),
            plt.imshow(PreWitt_saber, cmap=plt.cm.gray), plt.title(u'PreWitt'), plt.axis('off')
            # plt.show()
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 's':
            Sobel_saber = self.SobelAlogrithm(grayImage)
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'gray'), plt.axis('off')
            # plt.subplot(122),
            plt.imshow(Sobel_saber, cmap=plt.cm.gray), plt.title(u'Sobel'), plt.axis('off')
            # plt.show()
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        print('Type Erro')
        return

    # 打开图片，获取文件路径，显示原图像
    def openFile(self):
        try:
            get_filename_path, ok = QFileDialog.getOpenFileName(self.ChooseFile_Button,
                                                                "打开图片",
                                                                "C:/",
                                                                "*.jpg;;*.png;;All Files(*)")
            if ok:
                self.FilePath_IN.setText(str(get_filename_path))
        except:
            print('need choose again')
            self.file_path = ''
        # self.label.setPixmap(jpg)

    def confirmFile(self):
        # self.FilePath_IN.
        print('confirm')
        if self.FilePath_IN.text() == '':
            QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
            print('path err')
            return
        try:
            self.file_path = self.FilePath_IN.text()
            jpg = QtGui.QPixmap(self.FilePath_IN.text())
            width = jpg.width()
            height = jpg.height()
            x = self.img_label.width() / width
            y = self.img_label.height() / height
            ratio = 1
            if x < 1 or y < 1:
                ratio = min(x, y)
            print(ratio)
            jpg = jpg.scaled(math.floor(width * ratio), math.floor(height * ratio))
            self.img_label.setPixmap(jpg)
            self.path_flag = 1
        except:
            QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
            self.file_path = ''
            self.path_flag = 0
            print('path err')
            return

    # 处理图像
    def process_img(self, type):

        if self.file_path == '':
            print('Path Err')
            return
        print(self.file_path)
        if type == 'med' or type == 'mean':
            # get kernel_size
            kernel_size = self.Kernel_IN.text()
            print('kernel_size')
            print(kernel_size)
            if kernel_size == '':
                kernel_size = 3
            else:
                print('int ker')
                try:
                    kernel_size = int(kernel_size)
                    if kernel_size <= 0 :
                        QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                        return
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                    return
            self.Filter(self.file_path, type, kernel_size)
            return
        if type == 'gaus':
            # get kernel_size
            kernel_size = self.Kernel_IN2.text()
            print(kernel_size)
            if kernel_size == '':
                kernel_size = 3
            else:
                try:
                    kernel_size = int(kernel_size)
                    if kernel_size <=0 :
                        QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                        return
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                    return
            sigma = self.Sigma_IN.text()
            if sigma == '':
                sigma = 1.0
            else:
                try:
                    sigma = float(sigma)
                    if sigma <= 0.0:
                        QMessageBox.critical(self.centralwidget, "错误", "请重新设置sigma")
                        return
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "请重新设置sigma")
                    return
            print('sigma')
            print(sigma)
            if sigma == '':
                sigma = 1
            else:
                sigma = int(sigma)
            self.Filter(self.file_path, type, kernel_size, sigma)
            return
        if type == 'r':
            self.operator(self.file_path, 'r')
            return
        if type == 'p':
            self.operator(self.file_path, 'p')
            return
        if type == 's':
            self.operator(self.file_path, 's')
            return
