import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters


def RobertsOperator(roi):
    operator_first = np.array([[-1, 0], [0, 1]])
    operator_second = np.array([[0, -1], [1, 0]])
    return np.abs(np.sum(roi[1:, 1:] * operator_first)) * 0.5 + np.abs(np.sum(roi[1:, 1:] * operator_second)) * 0.5


def RobertsAlogrithm(image):
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            image[i, j] = RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
    return image[1:image.shape[0], 1:image.shape[1]]


def PreWittOperator(roi):
    prewitt_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # result = np.abs(np.sum(roi * prewitt_x)) + np.abs(np.sum(roi * prewitt_y))
    # result = np.abs(np.sum(roi * prewitt_x))*0.5+np.abs(np.sum(roi * prewitt_y))*0.5
    result = (np.abs(np.sum(roi * prewitt_x)) ** 2 + np.abs(np.sum(roi * prewitt_y)) ** 2) ** 0.5
    return result


def PreWittAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = PreWittOperator(image[i - 1:i + 2, j - 1:j + 2])
    return new_image.astype(np.uint8)


def SobelOperator(roi):
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    result = (np.abs(np.sum(roi * sobel_x)) ** 2 + np.abs(np.sum(roi * sobel_y)) ** 2) ** 0.5
    return result


def SobelAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = SobelOperator(image[i - 1:i + 2, j - 1:j + 2])
    return new_image.astype(np.uint8)


def read_img(path):
    img = cv2.imread(path)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage


def operator(path, type):
    grayImage = read_img(path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    if type == 'R' or type == 'r':
        Roberts_saber = RobertsAlogrithm(grayImage)
        plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'gray'), plt.axis('off')
        plt.subplot(122), plt.imshow(Roberts_saber, cmap=plt.cm.gray), plt.title(u'Roberts'), plt.axis('off')
        plt.show()
        return
    if type == 'P' or type == 'p':
        PreWitt_saber = PreWittAlogrithm(grayImage)
        plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'gray'), plt.axis('off')
        plt.subplot(122), plt.imshow(PreWitt_saber, cmap=plt.cm.gray), plt.title(u'PreWitt'), plt.axis('off')
        plt.show()
        return
    if type == 'S' or type == 's':
        Sobel_saber = SobelAlogrithm(grayImage)
        plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'gray'), plt.axis('off')
        plt.subplot(122), plt.imshow(Sobel_saber, cmap=plt.cm.gray), plt.title(u'Sobel'), plt.axis('off')
        plt.show()
        return
    print('Type Erro')
    return


# img = cv2.imread('./pic/pkq.jpg')
# img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Robert_saber = RobertsAlogrithm(grayImage)
# PreWitt_saber = PreWittAlogrithm(grayImage)
# Sobel_saber = SobelAlogrithm(grayImage)
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# # edges = filters.prewitt(grayImage)
# # edges = filters.sobel(grayImage)
# # plt.subplot(121), plt.imshow(img_RGB), plt.title(u'原始图像'), plt.axis('off')  # 坐标轴关闭
# plt.subplot(121), plt.imshow(edges, cmap=plt.cm.gray), plt.title(u'Roberts算子'), plt.axis('off')
# plt.subplot(122), plt.imshow(PreWitt_saber, cmap=plt.cm.gray), plt.title(u'Prewitt算子'), plt.axis('off')
# plt.show()

# operator('./pic/pkq.jpg', 'r')
# operator('./pic/pkq.jpg', 'p')
# operator('./pic/pkq.jpg', 's')