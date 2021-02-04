import numpy as np
import matplotlib.pyplot as plt

N = 1e7 + 10 + 5
T = 170

s = np.zeros([T])
e = np.zeros([T])
i = np.zeros([T])
r = np.zeros([T])
lamda = 0.5
gamma = 0.0821
sigma = 0.25

i[0] = 10.0 / N
s[0] = 1e7 / N
e[0] = 40.0 / N

for t in range(T - 1):
    s[t + 1] = s[t] - lamda * s[t] * i[t]
    e[t + 1] = e[t] + lamda * s[t] * i[t] - sigma * e[t]
    i[t + 1] = i[t] + sigma * e[t] - gamma * i[t]
    r[t + 1] = r[t] + gamma * i[t]

fix, ax = plt.subplots(figsize=(8, 4))
ax.plot(s, c='b', lw=2, label='S')
ax.plot(i, c='r', lw=2, label='I')
ax.plot(r, c='g', lw=2, label='R')
ax.plot(e, c='y', lw=2, label='E')
ax.set_xlabel('Day', fontsize=20)
ax.set_ylabel('Infective Ratio', fontsize=20)
ax.grid(1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.show()
