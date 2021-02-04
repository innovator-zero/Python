import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# read dataset from csv
dataset = pd.read_csv("assignment 2-supp.csv", header=0)
X = dataset[
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
X = np.array(X)

# normalize X data to [-1,1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X = min_max_scaler.fit_transform(X)
Y = dataset[['Outcome']]
Y = np.array(Y)

# split train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


def Sigmoid(z):
    S = float(1 / float(1 + np.exp(-z)))
    return S


def Hypothesis(theta, b, x):
    z = b
    for i in range(len(theta)):
        z += x[i] * theta[i]
    return z


def Logistic_Regression(X, Y, alpha, theta, b, num_iters):
    m = len(X)
    z = np.zeros(m)
    a = np.zeros(m)

    for n in range(num_iters):
        L = 0
        dz = np.zeros(m)  # derivative of z
        dtheta = np.zeros(len(theta))  # gradient

        for i in range(m):  # m = numbers of samples
            # forward propagation
            z[i] = Hypothesis(theta, b, X[i])
            a[i] = Sigmoid(z[i])
            L += (- Y[i] * np.log(a[i]) - (1 - Y[i]) * np.log(1 - a[i]))  # loss function
            # back propagation
            dz[i] = a[i] - Y[i]
            for j in range(len(theta)):
                dtheta[j] += X[i][j] * dz[i]
            db = float(np.sum(dz) / m)
        L /= m

        # gradient descent
        b = b - alpha * db
        for j in range(len(theta)):
            dtheta[j] /= m
            theta[j] = theta[j] - alpha * dtheta[j]
        '''
        if n % 10 == 0:
            print ('Loss is ', L)
            Li[int(n / 10)] = L
        '''
    return theta, b

#initialize parameters
init_theta = np.zeros(8)
init_b = 0
iterations = 200

'''
plot_num = int(iterations / 10)
it = list(range(1, plot_num + 1))
Li = np.zeros(plot_num)
x_major_locator = MultipleLocator(5)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
'''

# test accuracy
t = len(X_test)
test_z = np.zeros(t)
test_a = np.zeros(t)
at = []
act = []
alpha=0.005

for a in range(1, 100):
    alpha += 0.1   # learning rates
    trained_theta, trained_b = Logistic_Regression(X_train, Y_train, alpha, init_theta, init_b, iterations)

    right = 0
    for i in range(t):
        test_z[i] = Hypothesis(trained_theta, trained_b, X_test[i])
        test_a[i] = Sigmoid(test_z[i])
        if test_a[i] >= 0.5:  # thershold=0.5
            output = 1
        else:
            output = 0
        if output == Y_test[i]:
            right += 1

    accu = float(right / t)
    print("learning rate:{:.3f}".format(alpha) + " Accuracy:{:.3f}".format(accu))

    # add to graph
    at.append(alpha)
    act.append(accu)

plt.plot(at, act)
plt.show()
