import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, io


def median(img, x, y, size):
    sum_s = []
    size1 = int(size / 2)
    for k in range(-size1, size1 + 1):
        for m in range(-size1, size1 + 1):
            sum_s.append(img[x + k][y + m])
    sum_s.sort()
    return sum_s[(int(size * size / 2) + 1)]


def median_filter(im_copy_med, img, size):
    for i in range(int(size / 2), img.shape[0] - int(size / 2)):
        for j in range(int(size / 2), img.shape[1] - int(size / 2)):
            im_copy_med[i][j] = median(img, i, j, size)
    return im_copy_med


def mean(img, x, y, size):
    sum_s = 0
    size1 = int(size / 2)
    for k in range(-size1, size1 + 1):
        for m in range(-size1, size1 + 1):
            sum_s += img[x + k][y + m] / (size * size)
    return sum_s


def mean_filter(im_copy_mean, img, size):
    for i in range(int(size / 2), img.shape[0] - int(size / 2)):
        for j in range(int(size / 2), img.shape[1] - int(size / 2)):
            im_copy_mean[i][j] = mean(img, i, j, size)
    return im_copy_mean


def imgConvolve(image, kernel):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve


def imgGaussian(sigma):
    img_h = img_w = 2 * sigma + 1
    gaussian_mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return gaussian_mat


def imgAverageFilter(image, kernel):
    return imgConvolve(image, kernel) * (1.0 / kernel.size)


def read_img(path):
    try:
     img = cv2.imread(path)
    except:
        print('cannot find img')
        img = cv2.imread('./pic/pkq.jpg')
    grayImage1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage1, grayImage2


def Filter(path, type, kernel_size, sigma=1):
    print('hello')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    if type == 'med':
        grayImage, im_copy_med = read_img(path)
        medimg = median_filter(im_copy_med, grayImage, kernel_size)
        plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
        plt.subplot(122), plt.imshow(medimg, cmap=plt.cm.gray), plt.title(u'中值'), plt.axis('off')
        plt.show()

        return medimg
    if type == 'mean':
        grayImage, im_copy_mean = read_img(path)
        meanimg = mean_filter(im_copy_mean, grayImage, kernel_size)
        plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
        plt.subplot(122), plt.imshow(meanimg, cmap=plt.cm.gray), plt.title(u'均值'), plt.axis('off')
        plt.show()
        return meanimg
    if type == 'gaus':
        grayImage, im_copy_guas = read_img(path)
        plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
        gausimg = imgAverageFilter(grayImage, imgGaussian(sigma))
        plt.subplot(122), plt.imshow(gausimg, cmap=plt.cm.gray), plt.title(u'高斯'), plt.axis('off')
        plt.show()
        return gausimg
    print('Type Erro')
    return


#     # 读取图像
# img = cv2.imread('./pic/pkq.jpg')
# im_copy_med = cv2.imread('./pic/pkq.jpg')
# im_copy_mean = cv2.imread('./pic/pkq.jpg')
#
# img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 灰度化处理图像
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# grayImage1 = cv2.cvtColor(im_copy_med, cv2.COLOR_BGR2GRAY)
#
# # 设置plt可显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# # medimg = median_filter(im_copy_med, grayImage, 8)
# # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
# # plt.subplot(122), plt.imshow(medimg, cmap=plt.cm.gray), plt.title(u'中值'), plt.axis('off')
# # plt.show()
#
# # meanimg=mean_filter(im_copy_mean, grayImage, 3)
# plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
#
# img_gaus1 = imgAverageFilter(grayImage, imgGaussian(3))
# plt.subplot(122), plt.imshow(img_gaus1, cmap=plt.cm.gray), plt.title(u'均值'), plt.axis('off')
# plt.show()

# Filter('./pic/pkq.jpg', 'med', 3)
# Filter('./pic/pkq.jpg', 'mean', 3)
# Filter('./pic/pkq.jpg', 'gaus', 3, 3)
