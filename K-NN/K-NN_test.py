import numpy as np
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''K-NN算法，K值：表明的是与K个点之间的距离（最近），本代码设置K值为10，则表明是与10个点之间的距离（最近）
   先放进去10个点，剩下的点计算的距离，会与之前10个点中最远的点进行比较，若此点距离小于最远点距离，则进行替换
   并将重置最远点的标记和最远距离。剩下的点依次进行
   


'''


def train(test_set, train_set, train_lable):
    predict = []
    count = 0

    for test_vec in test_set:
        print(count)
        count += 1

        knn_list =[]  # 当前最近的k个邻居
        max_index = -1  # 当前k个最近邻居中最远点的坐标
        max_dist = 0    # 当前k个最近邻居中最远点的距离

        # 放进前k个点
        for i in range(k):
            lable = train_lable[i]
            train_vec = train_set[i]

            dist = np.linalg.norm(train_vec - test_vec)
            knn_list.append((dist, lable))  # 列表存入字典  [(dist, lable), (dist, lable)]

        # 剩下的元素
        for i in range(10, len(train_lable)):
            lable = train_lable[i]
            train_vec = train_set[i]

            dist = np.linalg.norm(train_vec - test_vec)

            # 寻找最先进去的10个点最远的距离
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            # 如果当前的存在比最近邻居中最远点还要近的点，则进行替换
            if dist < max_dist:
                knn_list[max_index] = (dist, lable)
                max_index = -1
                max_dist = 0

    # 统计选票
    class_total = 10
    class_count = [0 for i in range(class_total)]
    # print(class_count+"\n")
    # class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for dis, lable in knn_list:
        class_count[lable] += 1
    # 找出最大选票
    mmax = max(class_count)

    # 找出最大选票标签
    for i in range(class_total):
        if mmax == class_count[i]:
            predict.append(i)
            break

    return np.array(predict)


k = 10


if __name__ == '__main__':
    print('开始读取数据\n')
    time1 =time.time()

    raw_data = pd.read_csv("../data/train.csv", header=0)
    features = raw_data.values

    img = features[0:, 1:]
    lables = features[:, 0]
    imgs = np.shape(img)
    print(imgs, "\n")

    train_set, test_set, train_lable, test_lable = train_test_split(img, lables, test_size=0.33, random_state=23323)

    time2 = time.time()
    print("读取数据完毕,用时：%s\n" % str(time2 - time1))
    print("knn开始工作\n")
    test_predict = train(test_set, train_set, train_lable)
    print("train：", test_predict, "\n")

    time3 = time.time()
    print("使用完毕，用时%s\n" % str(time3 - time2) )

    socre = accuracy_score(test_lable, test_predict)
    print("准确度是：%d\n" % socre)

