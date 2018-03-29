import numpy as np
import cv2
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score     # 模型评估
'''感知机模型： f(x) = yi * (wx + b),使用梯度下降法训练模型，更新w，b，找出一个超平面（wx*b在移动）能够正确分类
    正例点yi>0， wx + b >0;负例点yi<0,wx + b<0;若找出的超平面能够准确分类，f(x) >0
    w <- w +yi*xi
    b <- b +yi
    
    hog特征：
    
    
    
'''

# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features =[]
    hog = cv2.HOGDescriptor("../hog.xml")
    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        # hog_feature = hog.compute(cv_img)     # hog: 方向梯度直方图
        features.append(cv_img)

    features = np.array(features)
    # features = np.reshape(features, (-1, 324))   # 不知feature有多少行，但总共有324列
    return features


def train(trainset, train_lable):
    train_size = len(train_lable)

    # 初始化 w,b
    w = np.zeros((features_length, 1))  # 列向量
    # print(w)
    b = 0

    study_count = 0             # 学习的次数
    nochange_count = 0          # 连续分类正确的次数
    nochange_up_limit = 10000   # 连续分类正确的上界，达到上界后，程序退出

    while True:
        nochange_count += 1
        if nochange_count > nochange_up_limit:
            break
        index = random.randint(0, train_size-1)  # randint(a, b) 随机生成a和b之间的数
        img = trainset[index]
        # print(img)
        lable = train_lable[index]

        # 计算 yi = wx+b
        yi = int(lable != object_num) *2 -1   # int(true) = 0  若标签和识别的数字不一样 int（true） = 0
        result = yi * (np.dot(img, w)+b)

        # 如果 result <= 0 ,则进行更新迭代w， b

        if result <= 0 :

            img = np.reshape(trainset[index], (features_length, 1))   # 更新维度
            print(img)
            w += img*yi*study_step
            b += yi*study_step
            study_count += 1

            if study_count > study_total:
                break
            nochange_count = 0    # 连续分类正确次数重新置0

    return w, b


def predict(test_set, w, b):
    predicts = [ ]
    for img in test_set:
        result = np.dot(img, w)+b
        result = result >0
        predicts.append(result)

    return predicts


features_length = 784
study_count = 0
study_step = 0.0001  # 学习步长
study_total = 1000   # 学习的次数
object_num = 0       # 识别的数字


if __name__ == '__main__':
    print("开始读取数据\n")

    time1 = time.time()
    raw_data = pd.read_csv("../data/train_binary.csv", header=0)
    data = raw_data.values

    img = data[0:, 1:]
    lables = data[:, 0]

    # features1 = get_hog_features(img)

    train_features, test_features, train_lable, test_lable = train_test_split(img, lables, test_size=0.33, random_state=23323)
    #print(train_features)

    time2 = time.time()
    print("读取数据完毕，用时："+str(time2-time1)+"\n")


    print("开始训练数据\n")
    w, b = train(train_features, train_lable)
    time3 = time.time()
    print("训练数据完毕，用时:"+str(time3 - time2)+"\n")
    print("使用测试集测试\n")

    predicts = predict(test_features, w, b)
    time4 = time.time()
    print("数据测试完毕，用时："+str(time4-time3)+"\n")

    # 模型分类准确率
    socre = accuracy_score(test_lable, predicts)
    print("模型分类准确率:"+str(socre)+"\n")


