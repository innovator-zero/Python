import numpy as np
import matplotlib.pyplot as plt

N = 1e7
T = 70

s = np.zeros([T])
i = np.zeros([T])
lamda = 1.0
gamma = 0.5

i[0] = 45.0 / N

for t in range(T - 1):
    i[t + 1] = i[t] + i[t] * lamda * (1.0 - i[t]) - gamma * i[t]

fix, ax = plt.subplots(figsize=(8, 4))
ax.plot(i, c='r', lw=2)
ax.set_xlabel('Day', fontsize=20)
ax.set_ylabel('Infective Ratio', fontsize=20)
ax.grid(1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
