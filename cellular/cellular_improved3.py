# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random

def process_nan(x):
    if np.isnan(x):
        return 0
    else:
        return x

# def ram_mig_1(move_pro,n): # 按概率寻找迁移方向
#     pro = move_pro.reshape((1,9))[0]
#     mig_num = np.zeros((1,9))
#     for i in range(n):
#         x = random.uniform(0,1)
#         p = 0.0
#         for i in range(9):
#             p += process_nan(pro[i])
#             if x < p:
#                 break
#         mig_num[0][i] += 1
#     mig_num = mig_num.reshape(3,3)
#     return mig_num


def ram_mig_1(move_pro,n): # 按概率寻找迁移方向 5*5
    pro = move_pro.reshape((1,25))[0]
    mig_num = np.zeros((1,25))
    for i in range(n):
        x = random.uniform(0,1)
        p = 0.0
        for i in range(25):
            p += process_nan(pro[i])
            if x < p:
                break
        mig_num[0][i] += 1
    mig_num = mig_num.reshape(5,5)
    return mig_num



# def ram_mig_2(move_pro,n): # 往概率最大方向迁移
#     pro = list(move_pro.reshape((1,9))[0])
#     mig_num = np.zeros((1,9))
#     mig_num[0][pro.index(np.nanmax(pro))] = n
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
temp_data = np.loadtxt('continuous_sst.txt')
for i in range(350):
    for j in range(784):
        if temp_data[i][j] < 0:
            temp_data[i][j] = np.nan


# plt.title('temperature')
# plt.imshow(temp_data,cmap=plt.cm.get_cmap('tab20c'))
# plt.colorbar()
# plt.show()

# herring shoals: 45580 1.3mt
herring_r = np.loadtxt('filtered_distribution.txt')
herring = np.zeros((350,784))
for i in range(350):
    for j in range(784):
        if herring_r[i][j] != -9999:
            herring[i][j] = np.ceil(herring_r[i][j]/1000)
        else:
            if np.isnan(temp_data[i][j]):
                herring[i][j] = np.nan
            else:
                herring[i][j] = 0.0


# plt.title('herring')
# plt.imshow(herring,cmap=plt.cm.get_cmap('tab20c'))
# plt.colorbar()
# plt.show()


# update
np.random.seed(56772)

FIT_TMP = 16
DLT_TMP = 1.5

def tmp_membership(x,a):
    fit_tmp = FIT_TMP + a
    if fit_tmp < x and x < fit_tmp + DLT_TMP:
        return 1.1 - (x+fit_tmp)/DLT_TMP
    elif  fit_tmp - DLT_TMP< x and x <= fit_tmp:
        return 1.1 - (fit_tmp - x)/DLT_TMP
    else:
        return np.exp(-np.square(x - fit_tmp))

def tmp_membership_2(x,a):
    fit_tmp = FIT_TMP + a
    return np.exp(-np.abs(x - fit_tmp))

# def fish_membership(fish):
#     if fish < 15:
#         return 1 - fish/30
#     else:
#         return 0.5*np.exp(np.square((fish - 15)/5))


def membership(x,fish,a):
    return tmp_membership(x,a) # +0.5*fish_membership(fish)


# print(np.nansum(herring),np.nanmax(herring),np.nanmedian(herring))

for year in range(50):
#     for p in range(350):
#         for q in range(784):
#             if not np.isnan(temp_data[p][q]):
#                 temp_data[p][q] += 0.12
    for i_ter in range(10):
        buf = np.zeros(herring.shape)
        for i in range(2,348):
            for j in range(2,782):
                if not np.isnan(herring[i][j]):
                    # ram_mig_1, ram_mig_2
                    neighbor = herring[i-2:i+3,j-2:j+3]
                    temp_neighbor = temp_data[i-2:i+3,j-2:j+3]
                    # temp_per = np.array([[18.5,18.5,18.5],[18.5,18.5,18.5],[18.5,18.5,18.5]])# + np.random.randn(3,3)
                    temp_neighbor[np.isnan(temp_neighbor)] = -20
                    # temp_diff = np.abs(temp_neighbor-temp_per)
                    move_pro = np.zeros((5,5))
                    a = np.random.randn(1)[0]/10
                    for p in range(5):
                        for q in range(5):
                            move_pro[p][q] = membership(temp_neighbor[p][q]+0.1*year,neighbor[p][q],a)
                    move_pro = move_pro/np.nansum(move_pro)
                    move_num = ram_mig_1(move_pro,int(herring[i][j]))
                    for p in range(5):
                        for q in range(5):
                            if not np.isnan(herring[i-2+p][j-2+q]):
                                buf[i-2+p][j-2+q] += move_num[p][q]

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
        plt.imshow(herring,cmap=plt.cm.get_cmap('tab20c'))
        plt.savefig('pictures_0.1/herring_{}_{}.png'.format(year+2021,i_ter))
        plt.pause(1)
        np.savetxt('data_0.1/herring_{}_{}.txt'.format(year+2021,i_ter),herring)
        herring_1 = herring.copy()
        herring_1[np.isnan(herring_1)] = 0
        print(np.sum(herring_1),np.nanmax(herring_1),np.nanmedian(herring_1))
