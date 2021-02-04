import numpy as np
import matplotlib.pyplot as plt

N = 1e7
T = 70

s = np.zeros([T])
i = np.zeros([T])
r = np.zeros([T])
lamda = 1.0
gamma = 0.5

i[0] = 45.0 / N
s[0] = 1 - i[0]

for t in range(T - 1):
    i[t + 1] = i[t] + i[t] * lamda * s[t] - gamma * i[t]
    s[t + 1] = s[t] - lamda * s[t] * i[t]
    r[t + 1] = r[t] + gamma * i[t]

fix, ax = plt.subplots(figsize=(8, 4))
ax.plot(s, c='b', lw=2, label='S')
ax.plot(i, c='r', lw=2, label='I')
ax.plot(r, c='g', lw=2, label='R')
ax.set_xlabel('Day', fontsize=20)
ax.set_ylabel('Infective Ratio', fontsize=20)
ax.grid(1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.show()
