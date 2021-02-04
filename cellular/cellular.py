# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random

def ram_mig_1(move_pro,n): # 按概率寻找迁移方向
    pro = move_pro.reshape((1,9))[0]
    mig_num = np.zeros((1,9))
    for i in range(n):
        x = random.uniform(0,1)
        p = 0.0
        for i in range(9):
            p += pro[i]
            if x < p:
                break
        mig_num[0][i] += 1
    mig_num = mig_num.reshape(3,3)
    return mig_num

# def ram_mig_2(move_pro,n): # 往概率最大方向迁移
#     pro = list(move_pro.reshape((1,9))[0])
#     mig_num = np.zeros((1,9))
#     mig_num[0][pro.index(max(pro))] = n
#     mig_num = mig_num.reshape(3,3)
#     return mig_num

# def ram_mig_3(move_pro,n): # 往温度变化最小方向迁移
#     pro = list(move_pro.reshape((1,25))[0])
#     mig_num = np.zeros((1,25))
#     mig_num[0][pro.index(min(pro))] = n
#     mig_num = mig_num.reshape(5,5)
#     return mig_num

# initial
# temperature
temp_data_r = np.loadtxt('year202008.txt')
temp_data = np.zeros((350,784))
for i in range(349):
    for j in range(783):
        temp_data[i][j] = temp_data_r[349-i][j]

# plt.title('temperature')
# plt.imshow(temp_data,cmap=plt.cm.get_cmap('ocean'))
# plt.colorbar()
# plt.show()

# herring shoals: 45580 1.3mt
herring_r = np.loadtxt('herring_all.txt')
herring = np.zeros((350,784))
for i in range(350):
    for j in range(784):
        if herring_r[i][j] != -9999:
            herring[i][j] = np.rint(herring_r[i][j] * 20)
        else:
            herring[i][j] = np.nan

# plt.title('herring')
# plt.imshow(herring,cmap=plt.cm.get_cmap('Pastel1'))
# plt.colorbar()
# plt.show()


# update
np.random.seed(56784)

for year in range(50):
    for p in range(350):
        for q in range(784):
            if not np.isnan(temp_data[p][q]):
                temp_data[p][q] += 1.2

    for i_ter in range(3):
        buf = np.zeros(herring.shape)
        for i in range(1,349):
            for j in range(1,783):
                if not np.isnan(herring[i][j]):
                    # ram_mig_1, ram_mig_2
                    neighbor = herring[i-1:i+2,j-1:j+2]
                    temp_neighbor = temp_data[i-1:i+2,j-1:j+2]
                    temp_per = np.array([[18.5,18.5,18.5],[18.5,18.5,18.5],[18.5,18.5,18.5]])# + np.random.randn(3,3)
                    temp_neighbor[np.isnan(temp_neighbor)] = -20
                    temp_diff = np.abs(temp_neighbor-temp_per)
                    move_pro = np.zeros((3,3))
                    for p in range(3):
                        for q in range(3):
                            move_pro[p][q] = np.exp(-np.square(temp_diff[p][q]/3.0))
                    move_pro = move_pro/np.sum(move_pro)
                    move_num = ram_mig_1(move_pro,int(herring[i][j]))
                    for p in range(3):
                        for q in range(3):
                            if not np.isnan(herring[i-1+p][j-1+q]):
                                buf[i-1+p][j-1+q] = buf[i-1+p][j-1+q] + move_num[p][q]


                    # ram_mig_3
                    # temp_neighbor = temp_data[i-2:i+3,j-2:j+3]
                    # temp_neighbor[np.isnan(temp_neighbor)] = -20
                    # temp_diff = np.zeros((5,5))
                    # for p in range(5):
                    #     for q in range(5):
                    #         temp_diff[p][q] = temp_neighbor[p][q] + 0.08 - temp_data[i][j]
                    # move_num = ram_mig_3(np.abs(temp_diff),int(herring[i][j]))
                    # for p in range(5):
                    #     for q in range(5):
                    #         if not np.isnan(herring[i-2+p][j-2+q]):
                    #             buf[i-2+p][j-2+q] = buf[i-2+p][j-2+q] + move_num[p][q]
                else:
                    buf[i][j] = np.nan
        herring = buf
        plt.title('herring_{}_{}'.format(year+2021,i_ter))
        plt.imshow(herring,cmap=plt.cm.get_cmap('Pastel1'))
        plt.savefig('pictures/herring_{}_{}.png'.format(year+2021,i_ter))
        plt.pause(1)
        herring_1 = herring.copy()
        herring_1[np.isnan(herring_1)] = 0
        print(np.sum(herring_1))





