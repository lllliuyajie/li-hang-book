'''import numpy as np

trian_size = np.zeros((1, 2, 2))
# print(trian_size)'''
'''import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def binary_img(img):
    cv_img = img.astype(np.uint8)

    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)   # threshold(src, thresh, maxval, type, dst=None)
    return cv_img

if __name__ == '__main__':
    print("开始读取数据\n")

    data = pd.read_csv("../data/train.csv", header=0)
    feature = data.values

    img = feature[0:, 1:]
    lable = feature[:, 0]

    train_set, test_set, trian_lable, test_lable = train_test_split(img, lable, test_size=0.7, random_state=23325)

    print("读取数据完毕\n")

    print("进行图像二值化处理")
    for i in range(len(trian_lable)):
        b_img = binary_img(train_set[i])
        print("图像二值化后：", b_img, "\n")
        b_imgs =np.shape(b_img)
        print("矩阵形状", b_imgs)'''

