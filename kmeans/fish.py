import numpy as np
import random
import matplotlib.pyplot as plt

points=np.loadtxt('points.txt')
herring_r = np.loadtxt('distribution.txt')
herring=np.zeros((802,350))
for i in range(350):
    for j in range(802):
        herring[j,349-i]=herring_r[i,j]

# s=np.zeros(10)
#
# for i in range(10):
#     x=int(round(points[i,0]))-1
#     y=int(round(points[i,1]))
#
#     for xx in range(x-11,x+12):
#         for yy in range(y-11,y+12):
#             if herring[xx,yy]>0:
#                 s[i]+=herring[xx,yy]
#
# f = open('fish_count.txt', 'w')
# for i in range(10):
#     f.write(str(s[i])+'\n')
# f.close()
s=0
for i in range(802):
    for j in range(350):
        if herring[i,j]>0:
            s+=herring[i,j]

print(s)


