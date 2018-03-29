import pandas as pd
import numpy as np
import time
import cv2
from sklearn.model_selection import train_test_split


def binary_img(img):
    cv_img = img.astype(np.uint8)     # threshold(src, thresh, maxval, type, dst=None)   np.uint8(0-255)就是cv2图片格式
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img

def trian_data(train_set, train_lable):
    prior_pro = np.zeros(class_num)  # 先验概率矩阵
    conditional_pro = np.zeros((class_num, feature, 2)) # 条件概率


    # 计算条件概率和先验概率
    for i in range(len(train_lable)):
        img = binary_img(train_set[i])
        print(img)
        lable = train_lable[i]

        prior_pro[lable] += 1

        for j in range(feature):
            conditional_pro[lable][j][img[j]] += 1

           # conditional_pro = np.shape(conditional_pro)
           # print(conditional_pro)


    for i in range(class_num):
        for j in range(feature):

            # 二值化后图像只有0,1取值
            pix_0 = conditional_pro[i][j][0]
            print(pix_0)
            pix_1 = conditional_pro[i][j][1]

            # 计算0,1像素点对应条件概率点
            probaility_0 = (float(pix_0)/float(pix_0+pix_1)) * 1000000 +1
            probaility_1 = (float(pix_1)/float(pix_0+pix_1)) * 1000000 +1

            conditional_pro[i][j][0] = probaility_0
            conditional_pro[i][j][1] = probaility_1
    print(prior_pro)
    return prior_pro, conditional_pro


# 计算概率
def calculate_pro(img, lable):
    probability = int(prior_pro[lable])

    for i in range(len(img)):
        probability *= int(conditional_probability[lable][i][img[i]])

    return probability


def predict(test_set, prior_pro, conditional_pro):
    predict_array = []

    for img in test_set:
        img = binary_img(img)

        max_lable = 0
        max_probability = calculate_pro(img,0)

        for j in range(1, 10):
            probability = calculate_pro(img, j)

            if max_probability < probability:
                max_lable = j
                max_probability = probability

        predict_array.append(max_lable)

    return np.array(predict_array)


class_num = 10     # 类标个数
feature = 784      # 784 维


if __name__ == '__main__':
    print("开始读取数据\n")
    time1 = time.time()

    data = pd.read_csv("../data/train.csv",header=0)
    features = data.values

    img = features[0:, 1:]
    lable = features[:, 0]

    trian_set, test_set, train_lable, test_lable = train_test_split(img, lable, test_size=0.75, random_state=23325)

    time2 = time.time()
    print("处理数据完毕，用时%s\n " % str(time2-time1))

    print("开始数据训练和图像二值化\n")
    prior_pro, conditional_probability = trian_data(trian_set, train_lable)

    time3 = time.time()
    print("数据训练完毕，用时%s\n" % str(time3-time2))

    test_prdict = predict(test_set, prior_pro, conditional_probability)
    print(test_prdict)

    print("预测结束")







