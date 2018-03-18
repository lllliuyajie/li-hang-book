'''import pandas as pd
import numpy  as np
raw_data = pd.read_csv("../data/train_binary.csv", header=0)
data = raw_data.values

img = data[0:, 1:]
lables = data [:, 0]

print(lables)'''
'''import time

print(time.time())
# print(time.localtime(time.time()))
print(time.asctime(time.localtime()))

 # strftime（）接收元组，表示成字符串
print(time.strftime("%Y-%m-%d %h:%M:%S", time.localtime(time.time())))'''
