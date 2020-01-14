import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img(path):
    try:
        img = cv2.imread(path)
    except:
        print('cannot find img')
        img = cv2.imread('./pic/pkq.jpg')
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage


def gray2bin(path, threshold=200):
    grayImage = read_img(path)
    ret, binImage = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY_INV)
    print(binImage[1][2])

    return binImage


def erode_bin(binImage, kernel, x, y):
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


def dilate_bin(binImage, kernel, x, y):
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


def edge_detection(binImage, kernel, x, y):
    return dilate_bin(binImage, kernel, x, y) - erode_bin(binImage, kernel, x, y)


def erode_gray(grayImage, kernel, x, y):
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


def dilate_gray(grayImage, kernel, x, y):
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


def gradient(grayImage, kernel, x, y):
    return (dilate_gray(grayImage, kernel, x, y) - erode_gray(grayImage, kernel, x, y)) * 0.5


def conditional_dilation(marker, mask, kernel, x, y):
    while True:
        new = np.min((dilate_bin(marker, kernel, x, y), mask), axis=0)
        if (new == marker).all():
            return marker
        marker = new


# 膨胀重建
def grayscale_reconstruction(marker, mask, kernel, x, y):
    c=0
    while True:
        new = np.min((cv2.dilate(marker,kernel,iterations=1), mask), axis=0)
        if (new == marker).all():
            return marker
        marker = new
        c+=1
        print(c)


array = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
kernel = np.array(array)
print(kernel)
'''
# 二值腐蚀
image_e = erode_bin(gray2bin('./pic/pkq.jpg'), kernel, 1, 2)
plt.title('erode_bin')
plt.imshow(image_e, 'gray')
plt.show()
# 二值膨胀
image_d = dilate_bin(gray2bin('./pic/pkq.jpg'), kernel, 1, 2)
plt.title('dilate_bin')
plt.imshow(image_d, 'gray')
plt.show()
# 边缘检测
image = edge_detection(gray2bin('./pic/pkq.jpg'), kernel, 1, 2)
plt.title('edge_detection')
plt.imshow(image, 'gray')
plt.show()
#灰度腐蚀
image_e_g = erode_gray(read_img('./pic/people2.png'), kernel, 1, 2)
plt.title('erode_gray')
plt.imshow(image_e_g, 'gray')
plt.show()
# 灰度膨胀
image_d_g = dilate_gray(read_img('./pic/people2.png'), kernel, 1, 2)
plt.title('dilate_gray')
plt.imshow(image_d_g, 'gray')
plt.show()
# 形态学梯度
image_g = gradient(read_img('./pic/people2.png'), kernel, 1, 2)
plt.title('image_g')
plt.imshow(image_g, 'gray')
plt.show()'''

'''
# 测地膨胀
image_e = erode_bin(gray2bin('./pic/pkq.jpg'), kernel, 1, 2)
plt.title('erode_bin')
plt.imshow(image_e, 'gray')
plt.show()
image_d = dilate_bin(gray2bin('./pic/pkq.jpg'), kernel, 1, 2)
plt.title('dilate_bin')
plt.imshow(image_d, 'gray')
plt.show()
image_c_d = conditional_dilation(image_e, image_d, kernel, 1, 2)
plt.title('image_g')
plt.imshow(image_c_d, 'gray')
plt.show()'''

# 灰度腐蚀
image_e_g = erode_gray(read_img('./pic/people2.png'), kernel, 1, 2)
plt.title('erode_gray')
plt.imshow(image_e_g, 'gray')
plt.show()
# 灰度膨胀
image_d_g = dilate_gray(read_img('./pic/people2.png'), kernel, 1, 2)
plt.title('dilate_gray')
plt.imshow(image_d_g, 'gray')
plt.show()
image_g_r = grayscale_reconstruction(image_e_g, read_img('./pic/people2.png'), kernel, 1, 2)
plt.title('image_g_r')
plt.imshow(image_g_r, 'gray')
plt.show()
