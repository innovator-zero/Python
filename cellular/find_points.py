import numpy as np

fish_data = np.loadtxt('herring_2070.txt')

center=np.zeros((15,34))
for i in range(15):
    for j in range(34):
        s=0
        for x in range(i*23,(i+1)*23):
            for y in range(j*23,(j+1)*23):
                if fish_data[x][y]>0:
                    s+=fish_data[x][y]

        center[i][j]=s

f = open('center2_3.txt', 'w')
for i in range(15):
    for j in range(34):
        f.write(str(center[i][j])+' ')
    f.write('\n')
f.close()


