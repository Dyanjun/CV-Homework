# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2_UI.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, sip
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import math


class MyFigure(FigureCanvas):
    def __init__(self, width, height, dpi):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)


class Ui_MainWindow(object):
    def __init__(self):
        self.file_path = ''
        self.file_path_mask = ''
        self.c = 0
        array = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        tmp_kernel = np.array(array)
        self.kernel = tmp_kernel
        self.x = 1
        self.y = 1
        self.n = 250

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1284, 839)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.F = MyFigure(width=3, height=2, dpi=100)
        self.groupBox = QtWidgets.QGroupBox(MainWindow)
        self.groupBox.setGeometry(QtCore.QRect(830, 430, 400, 350))
        self.groupBox.setObjectName("groupBox")
        self.gridlayout = QtWidgets.QGridLayout(self.groupBox)

        self.choose_img = QtWidgets.QPushButton(self.centralwidget)
        self.choose_img.setGeometry(QtCore.QRect(20, 10, 93, 28))
        self.choose_img.setObjectName("choose_img")
        self.choose_img_mask = QtWidgets.QPushButton(self.centralwidget)
        self.choose_img_mask.setGeometry(QtCore.QRect(130, 10, 111, 28))
        self.choose_img_mask.setObjectName("choose_img_mask")

        self.mask_img = QtWidgets.QLabel(self.centralwidget)
        self.mask_img.setGeometry(QtCore.QRect(830, 40, 400, 350))
        self.mask_img.setObjectName("mask_img")
        self.original_img = QtWidgets.QLabel(self.centralwidget)
        self.original_img.setGeometry(QtCore.QRect(350, 40, 400, 350))
        self.original_img.setObjectName("original_img")
        self.process_img = QtWidgets.QLabel(self.centralwidget)
        self.process_img.setGeometry(QtCore.QRect(830, 430, 400, 350))
        self.process_img.setObjectName("process_img")

        self.edge_detection = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection.setGeometry(QtCore.QRect(20, 550, 90, 31))
        self.edge_detection.setObjectName("edge_detection")
        self.edge_detection_external = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection_external.setGeometry(QtCore.QRect(120, 550, 90, 31))
        self.edge_detection_external.setObjectName("edge_detection_external")
        self.edge_detection_internal = QtWidgets.QPushButton(self.centralwidget)
        self.edge_detection_internal.setGeometry(QtCore.QRect(220, 550, 90, 31))
        self.edge_detection_internal.setObjectName("edge_detection_internal")
        self.C_dilation = QtWidgets.QPushButton(self.centralwidget)
        self.C_dilation.setGeometry(QtCore.QRect(20, 670, 141, 28))
        self.C_dilation.setObjectName("C_dilation")
        self.G_reconstruction = QtWidgets.QPushButton(self.centralwidget)
        self.G_reconstruction.setGeometry(QtCore.QRect(20, 720, 100, 28))
        self.G_reconstruction.setObjectName("G_reconstruction")
        self.G_reconstruction_C = QtWidgets.QPushButton(self.centralwidget)
        self.G_reconstruction_C.setGeometry(QtCore.QRect(140, 720, 100, 28))
        self.G_reconstruction_C.setObjectName("C_reconstruction")
        self.gradient_button = QtWidgets.QPushButton(self.centralwidget)
        self.gradient_button.setGeometry(QtCore.QRect(20, 760, 90, 28))
        self.gradient_button.setObjectName("gradient")
        self.gradient_external_button = QtWidgets.QPushButton(self.centralwidget)
        self.gradient_external_button.setGeometry(QtCore.QRect(120, 760, 90, 28))
        self.gradient_external_button.setObjectName("gradient")
        self.gradient_internal_button = QtWidgets.QPushButton(self.centralwidget)
        self.gradient_internal_button.setGeometry(QtCore.QRect(220, 760, 90, 28))
        self.gradient_internal_button.setObjectName("gradient")

        origin_Validator = QIntValidator(0, 65536)
        self.n_input = QtWidgets.QLineEdit(self.centralwidget)
        self.n_input.setGeometry(QtCore.QRect(130, 600, 113, 21))
        self.n_input.setObjectName("origin_y_input")
        self.n_input.setValidator(origin_Validator)
        self.n_txt = QtWidgets.QLabel(self.centralwidget)
        self.n_txt.setGeometry(QtCore.QRect(30, 600, 72, 15))
        self.n_txt.setObjectName("origin_t_txt")
        self.n_button = QtWidgets.QPushButton(self.centralwidget)
        self.n_button.setGeometry(QtCore.QRect(200, 630, 93, 28))
        self.n_button.setObjectName("kernel_button")

        self.kernel_input = QtWidgets.QTextEdit(self.centralwidget)
        self.kernel_input.setGeometry(QtCore.QRect(20, 100, 271, 261))
        self.kernel_input.setObjectName("kernel_input")

        self.origin_x_input = QtWidgets.QLineEdit(self.centralwidget)
        self.origin_x_input.setGeometry(QtCore.QRect(130, 410, 113, 21))
        self.origin_x_input.setObjectName("origin_x_input")
        self.origin_x_input.setValidator(origin_Validator)
        self.origin_y_input = QtWidgets.QLineEdit(self.centralwidget)
        self.origin_y_input.setGeometry(QtCore.QRect(130, 470, 113, 21))
        self.origin_y_input.setObjectName("origin_y_input")
        self.origin_y_input.setValidator(origin_Validator)

        self.kernel_button = QtWidgets.QPushButton(self.centralwidget)
        self.kernel_button.setGeometry(QtCore.QRect(200, 505, 93, 28))
        self.kernel_button.setObjectName("kernel_button")

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
        self.choose_img.clicked.connect(lambda: self.openFile(1))
        self.choose_img_mask.clicked.connect(lambda: self.openFile(2))
        self.edge_detection.clicked.connect(lambda: self.process('e_d_s'))
        self.edge_detection_external.clicked.connect(lambda: self.process('e_d_e'))
        self.edge_detection_internal.clicked.connect(lambda: self.process('e_d_i'))
        self.C_dilation.clicked.connect(lambda: self.process('c_d'))
        self.G_reconstruction.clicked.connect(lambda: self.process('g_r'))
        self.G_reconstruction_C.clicked.connect(lambda: self.process('g_r_c'))
        self.gradient_button.clicked.connect(lambda: self.process('g_s'))
        self.gradient_external_button.clicked.connect(lambda: self.process('g_e'))
        self.gradient_internal_button.clicked.connect(lambda: self.process('g_i'))
        self.kernel_button.clicked.connect(self.get_kernel)
        self.n_button.clicked.connect(self.get_n)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.choose_img.setText(_translate("MainWindow", "选择图片"))
        self.kernel_button.setText(_translate("MainWindow", "确认kernel"))
        self.mask_img.setText(_translate("MainWindow", "mask图像"))
        self.original_img.setText(_translate("MainWindow", "原图像"))
        self.process_img.setText(_translate("MainWindow", "处理后图像"))
        self.edge_detection.setText(_translate("MainWindow", "边缘检测_st"))
        self.edge_detection_external.setText(_translate("MainWindow", "边缘检测_ex"))
        self.edge_detection_internal.setText(_translate("MainWindow", "边缘检测_in"))
        self.C_dilation.setText(_translate("MainWindow", "C_dilation"))
        self.G_reconstruction.setText(_translate("MainWindow", "OBR"))
        self.G_reconstruction_C.setText(_translate("MainWindow", "CBR"))
        self.gradient_button.setText(_translate("MainWindow", "梯度_st"))
        self.gradient_external_button.setText(_translate("MainWindow", "梯度_ex"))
        self.gradient_internal_button.setText(_translate("MainWindow", "梯度_in"))
        self.kernel_input.setHtml(_translate("MainWindow",
                                             "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                             "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                             "p, li { white-space: pre-wrap; }\n"
                                             "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                             "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'SimSun\';\"><br /></p></body></html>"))
        self.origin_x_txt.setText(_translate("MainWindow", "origin_x:"))
        self.origin_t_txt.setText(_translate("MainWindow", "origin_y:"))
        self.n_txt.setText(_translate("MainWindow", "迭代次数n:"))
        self.n_button.setText(_translate("MainWindow", "确认n"))
        self.kernel_txt.setText(_translate("MainWindow", "kernel 格式（如3*3）："))
        self.line_1.setText(_translate("MainWindow", "1，1，1"))
        self.line_2.setText(_translate("MainWindow", "1，0，1"))
        self.line_3.setText(_translate("MainWindow", "0，0，1"))
        self.choose_img_mask.setText(_translate("MainWindow", "选择mask图片"))

    def openFile(self, i):
        try:
            get_filename_path, ok = QFileDialog.getOpenFileName(self.choose_img,
                                                                "打开图片",
                                                                "C:/",
                                                                "*jpeg;;*.jpg;;*.png;;All Files(*)")
            if ok:
                if i == 1:
                    self.file_path = str(get_filename_path)
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
                else:
                    self.file_path_mask = str(get_filename_path)
                    try:
                        jpg = QtGui.QPixmap(self.file_path_mask)
                        width = jpg.width()
                        height = jpg.height()
                        x = self.original_img.width() / width
                        y = self.original_img.height() / height
                        ratio = 1
                        if x < 1 or y < 1:
                            ratio = min(x, y)
                        print(ratio)
                        jpg = jpg.scaled(math.floor(width * ratio), math.floor(height * ratio))
                        self.mask_img.setPixmap(jpg)
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

    def read_img(self, path):
        try:
            img = cv2.imread(path)
        except:
            print('cannot find img')
            img = cv2.imread('./pic/pkq.jpg')
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayImage

    def gray2bin(self, path, threshold=200):
        grayImage = self.read_img(path)
        ret, binImage = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY_INV)
        return binImage

    def erode_bin(self, binImage, kernel, x, y):
        # d_image = np.zeros(shape=binImage.shape)
        d_image = binImage.copy()
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
                if flag == 0:
                    d_image[i, j] = 0
        return d_image

    def dilate_bin(self, binImage, kernel, x, y):
        kernel_size = kernel.shape[0]
        d_image = binImage.copy()
        for i in range(x, binImage.shape[0] - kernel_size + x + 1):
            for j in range(y, binImage.shape[1] - kernel_size + y + 1):
                if binImage[i, j] == 255:
                    for k_x in range(0, kernel_size):
                        for k_y in range(0, kernel_size):
                            if kernel[k_x][k_y] > 0:
                                d_image[i - x + k_x, j - y + k_y] = 255
                                break
        return d_image

    # 二值边缘检测
    def e_detection_standard(self, binImage, kernel, x, y):
        return self.dilate_bin(binImage, kernel, x, y) - self.erode_bin(binImage, kernel, x, y)

    def e_detection_external(self, binImage, kernel, x, y):
        return self.dilate_bin(binImage, kernel, x, y) - binImage

    def e_detection_internal(self, binImage, kernel, x, y):
        return binImage - self.erode_bin(binImage, kernel, x, y)

    # 二值条件膨胀
    def conditional_dilation(self, marker, mask_path, kernel, x, y, n):
        mask = self.gray2bin(mask_path)
        mask1 = cv2.resize(mask, (marker.shape[0], marker.shape[1]))
        while n >= 0:
            new = np.min((self.dilate_bin(marker, kernel, x, y), mask1), axis=0)
            if (new == marker).all():
                return marker
            marker = new
            n -= 1
            print(n)
        return marker

    def erode_gray(self, grayImage, kernel, x, y):
        kernel_size = kernel.shape[0]
        d_image = grayImage.copy()
        for i in range(x, grayImage.shape[0] - kernel_size + x + 1):
            for j in range(y, grayImage.shape[1] - kernel_size + y + 1):
                tmp_array = []
                for k_x in range(-x, kernel_size - x):
                    for k_y in range(-y, kernel_size - x):
                        tmp_array.append(grayImage[i + k_x, j + k_y] - kernel[x + k_x, y + k_y])
                d_image[i][j] = min(tmp_array)
        return d_image

    def dilate_gray(self, grayImage, kernel, x, y):
        kernel_size = kernel.shape[0]
        d_image = grayImage.copy()
        '''for i in range(kernel_size - x - 1, grayImage.shape[0] - x):
            for j in range(kernel_size - y - 1, grayImage.shape[1] - y):
                tmp_array = []
                for k_x in range(-x, kernel_size - x):
                    for k_y in range(-y, kernel_size - y):
                        tmp_array.append(grayImage[i - k_x, j - k_y] + kernel[x + k_x, y + k_y])
                d_image[i][j] = max(tmp_array)'''
        for i in range(x, grayImage.shape[0] - kernel_size + x + 1):
            for j in range(y, grayImage.shape[1] - kernel_size + y + 1):
                tmp_array = []
                for k_x in range(-x, kernel_size - x):
                    for k_y in range(-y, kernel_size - x):
                        tmp_array.append(grayImage[i + k_x, j + k_y] + kernel[x + k_x, y + k_y])
                d_image[i][j] = max(tmp_array)
        return d_image

    def gradient_standard(self, grayImage, kernel, x, y):
        return (self.dilate_gray(grayImage, kernel, x, y) - self.erode_gray(grayImage, kernel, x, y)) * 0.5

    def gradient_external(self, grayImage, kernel, x, y):
        return (self.dilate_gray(grayImage, kernel, x, y) - grayImage) * 0.5
        # return (cv2.dilate(grayImage, kernel) - grayImage) * 0.5

    def gradient_internal(self, grayImage, kernel, x, y):
        return (grayImage - self.erode_gray(grayImage, kernel, x, y)) * 0.5

    # 膨胀重建
    def grayscale_reconstruction(self, marker, mask, kernel, x, y, n):
        print("重建开操作")
        while n >= 0:
            new = np.min((self.dilate_gray(marker, kernel, x, y), mask), axis=0)
            if (new == marker).all():
                return marker
            marker = new
            n -= 1
            print(n)
        return marker

    def grayscale_reconstruction_C(self, marker, mask, kernel, x, y, n):
        print("重建闭操作")
        while n >= 0:
            new = np.max((self.erode_gray(marker, kernel, x, y), mask), axis=0)
            if (new == marker).all():
                return marker
            n -= 1
            print(n)
        return marker

    def OBR(self, grayImage, kernel, x, y, n):
        opening = self.dilate_gray(self.erode_gray(grayImage, kernel, x, y), kernel, x, y)
        return self.grayscale_reconstruction(opening, grayImage, kernel, x, y, n)

    def CBR(self, grayImage, kernel, x, y, n):
        closing = self.erode_gray(self.dilate_gray(grayImage, kernel, x, y), kernel, x, y)
        return self.grayscale_reconstruction_C(closing, grayImage, kernel, x, y, n)

    # 得到kernel和origin
    def get_kernel(self):
        print('get_kernel:')
        tmp_kernel = []
        if self.kernel_input.toPlainText() != '':
            tmp_list = self.kernel_input.toPlainText().split('\n')
            try:
                for line in tmp_list:
                    if line == '':
                        continue
                    line = line.replace('，', ',')
                    item = line.split(',')
                    item = list(map(int, item))
                    max_one = max(item)
                    if max_one > 255:
                        print('kernel 过大')
                        QMessageBox.critical(self.centralwidget, "错误", "kernel中元素值不能超过255，请重新设置")
                        return
                    tmp_kernel.append(item)
            except:
                QMessageBox.critical(self.centralwidget, "错误", "kernel请重新设置")
                return
            tmp_kernel = np.array(tmp_kernel)
            print(tmp_kernel)
            self.kernel = tmp_kernel
            self.x = tmp_kernel.shape[0] // 2
            self.y = tmp_kernel.shape[1] // 2
        else:
            array = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            tmp_kernel = np.array(array)
            self.kernel = tmp_kernel
            self.x = 1
            self.y = 1
            return
        # kernel 完成
        # 接下来 搞 origin
        center = int(self.kernel.shape[0] / 2)
        print(center)
        o_x = self.origin_x_input.text()
        o_y = self.origin_y_input.text()
        if o_x == '':
            self.x = center
        else:
            o_x = int(o_x)
            if o_x < self.kernel.shape[0]:
                self.x = o_x
            else:
                QMessageBox.critical(self.centralwidget, "错误", "origin_x错误")
                return
        if o_y == '':
            self.y = center
        else:
            o_y = int(o_y)
            if o_y < self.kernel.shape[1]:
                self.y = o_y
            else:
                QMessageBox.critical(self.centralwidget, "错误", "origin_y错误")
                return

    # 得到迭代次数n
    def get_n(self):
        if self.n_input.text() != '':
            n = self.n_input.text()
            self.n = int(n)
        else:
            self.n = 250
        print(self.n)

    # 综合分类处理
    def process(self, type):
        if self.c:
            print('delete F')
            # plt.close()
            self.gridlayout.removeWidget(self.F)
            sip.delete(self.F)
            print(self.gridlayout.count())
            self.F = MyFigure(width=3, height=2, dpi=100)
        kernel = self.kernel
        o_x = self.x
        o_y = self.y
        n = self.n
        if type == 'e_d_s':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            else:
                image = self.e_detection_standard(self.gray2bin(self.file_path), kernel, o_x, o_y)
                plt.rcParams['figure.figsize'] = (image.shape[0], image.shape[1])
                plt.imshow(image, cmap=plt.cm.gray), plt.title(u'edge_detection_standard'), plt.axis('off')
                print(self.gridlayout.count())
                self.gridlayout.addWidget(self.F, 0, 0)
                print(self.gridlayout.count())
                self.c = 1
                return
        if type == 'e_d_e':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            else:
                image = self.e_detection_external(self.gray2bin(self.file_path), kernel, o_x, o_y)
                plt.rcParams['figure.figsize'] = (image.shape[0], image.shape[1])
                plt.imshow(image, cmap=plt.cm.gray), plt.title(u'edge_detection_external'), plt.axis('off')
                print(self.gridlayout.count())
                self.gridlayout.addWidget(self.F, 0, 0)
                print(self.gridlayout.count())
                self.c = 1
                return
        if type == 'e_d_i':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            else:
                image = self.e_detection_internal(self.gray2bin(self.file_path), kernel, o_x, o_y)
                plt.rcParams['figure.figsize'] = (image.shape[0], image.shape[1])
                plt.imshow(image, cmap=plt.cm.gray), plt.title(u'edge_detection_internal'), plt.axis('off')
                print(self.gridlayout.count())
                self.gridlayout.addWidget(self.F, 0, 0)
                print(self.gridlayout.count())
                self.c = 1
                return
        if type == 'c_d':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            if self.file_path_mask == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择mask")
                return
            image_c_d = self.conditional_dilation(self.gray2bin(self.file_path), self.file_path_mask,
                                                  kernel, o_x, o_y,n)
            plt.rcParams['figure.figsize'] = (image_c_d.shape[0], image_c_d.shape[1])
            plt.imshow(image_c_d, cmap=plt.cm.gray), plt.title(u'conditional_dilation'), plt.axis('off')
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 'g_r':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            image_g_r = self.OBR(self.read_img(self.file_path), kernel, o_x, o_y, n)
            plt.rcParams['figure.figsize'] = (image_g_r.shape[0], image_g_r.shape[1])
            plt.imshow(image_g_r, cmap=plt.cm.gray), plt.title(u'OBR'), plt.axis('off')
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 'g_r_c':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            image_g_r = self.CBR(self.read_img(self.file_path), kernel, o_x, o_y, n)
            plt.rcParams['figure.figsize'] = (image_g_r.shape[0], image_g_r.shape[1])
            plt.imshow(image_g_r, cmap=plt.cm.gray), plt.title(u'CBR'), plt.axis('off')
            print(self.gridlayout.count())
            self.gridlayout.addWidget(self.F, 0, 0)
            print(self.gridlayout.count())
            self.c = 1
            return
        if type == 'g_s':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            else:
                image_g = self.gradient_standard(self.read_img(self.file_path), kernel, o_x, o_y)
                plt.rcParams['figure.figsize'] = (image_g.shape[0], image_g.shape[1])
                plt.imshow(image_g, cmap=plt.cm.gray), plt.title(u'gradiant_standard'), plt.axis('off')
                print(self.gridlayout.count())
                self.gridlayout.addWidget(self.F, 0, 0)
                print(self.gridlayout.count())
                self.c = 1
                return
        if type == 'g_e':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            else:
                image_g = self.gradient_external(self.read_img(self.file_path), kernel, o_x, o_y)
                plt.rcParams['figure.figsize'] = (image_g.shape[0], image_g.shape[1])
                plt.imshow(image_g, cmap=plt.cm.gray), plt.title(u'gradiant_external'), plt.axis('off')
                print(self.gridlayout.count())
                self.gridlayout.addWidget(self.F, 0, 0)
                print(self.gridlayout.count())
                self.c = 1
                return
        if type == 'g_i':
            if self.file_path == '':
                QMessageBox.critical(self.centralwidget, "错误", "图片路径错误，请重新选择")
                return
            else:
                image_g = self.gradient_internal(self.read_img(self.file_path), kernel, o_x, o_y)
                plt.rcParams['figure.figsize'] = (image_g.shape[0], image_g.shape[1])
                plt.imshow(image_g, cmap=plt.cm.gray), plt.title(u'gradiant__internal'), plt.axis('off')
                print(self.gridlayout.count())
                self.gridlayout.addWidget(self.F, 0, 0)
                print(self.gridlayout.count())
                self.c = 1
                return
