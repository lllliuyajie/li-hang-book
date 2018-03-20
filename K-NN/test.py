'''class_count = [0 for i in range(10)]
print(class_count)

class_count[0] += 1
print(class_count)'''
list1 = []
labl = -1
dist = 1000
class_count = [0 for i in  range(10)]

for i in range(10):
    labl += 1
    dist += 1000
    list1.append((dist, labl))
print(list1)
for dist, labl in list1:
    class_count[labl] += 1
    print(labl)
    print(dist)
print(class_count)


