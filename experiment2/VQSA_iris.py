from numpy.random import rand
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets

#normalization
def normalization(X):
    X_normalize = normalize(X, norm='l2', axis=1)

    return X_normalize

#mean
def mean(X):      
    X_mean = np.mean(X,axis=0)     
    X_zero_mean = X - X_mean
    
    return X_zero_mean

#PCA
def pca(X, d):
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    U_reduced = U[:,:d]

    return np.dot(X, U_reduced), U_reduced

#rotation operator
def Ry(theta):
    Ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    return Ry

#parameterized unitary
def Unitary(x):
    U_H = np.kron(Hadamard, Hadamard)
    U1 = np.kron(Ry(x[0]), Ry(x[1]))
    U_CNOT = np.dot(CNOT2, CNOT1)
    U2 = np.kron(Ry(x[2]), Ry(x[3]))
    U3 = np.kron(Ry(x[4]), Ry(x[5]))
    U4 = np.kron(Ry(x[6]), Ry(x[7]))
    U = np.dot(U4, np.dot(U_CNOT, np.dot(U3, np.dot(U_CNOT, np.dot(U2, np.dot(U_CNOT, np.dot(U1, U_H)))))))

    return U

#cost function
def cost_function(x):
    Unitary(x)
    cost = np.linalg.norm(np.dot(Ps, Unitary(x)) - Pt)

    return cost

#compute gradient
def numerical_gradient(x):
    t = 1e-4
    grad = np.zeros_like(x)
    cost_function(x)

    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + t 
        fxh1 = cost_function(x)

        x[i] = tmp_val - t
        fxh2 = cost_function(x)

        grad[i] = (fxh1 - fxh2) / (2*t)
        x[i] = tmp_val
    return grad

#optimization with AdaGrad
def fit(x):
    cost_history = []

    for i in range(step_num):
        grad = numerical_gradient(x)
        cost_function(x)
        cost_history.append(cost_function(x))

        h = np.zeros_like(grad)

        for index in range(len(x)):
            x[index] -= lr * grad[index]
            h[index] += grad[index] * grad[index]
            x[index] -= lr * grad[index] / (np.sqrt(h[index]) + 1e-7)

        #print(cost_function(x))

    return cost_history, x

#classification
def fit_predict(x):
    Xs_new = np.dot(hat_Xs, Unitary(x))
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_new, Ys.ravel())
    y_pred = clf.predict(hat_Xt)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred)

    return acc, y_pred

if __name__ == '__main__':
    zero_state = np.array([[1], [0]])
    one_state = np.array([[0], [1]])
    
    #Pauli operators
    PauliX = np.array([[0, 1], [1, 0]])
    PauliY = np.array([[0, -1j], [1j, 0]])
    PauliZ = np.array([[1, 0], [0, -1]])
    PauliI = np.eye(2)

    #Hadamard operator
    Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    #CNOT operators
    CNOT1 = np.kron(np.dot(zero_state, zero_state.T), PauliI) + np.kron(np.dot(one_state, one_state.T), PauliX)
    CNOT2 = np.kron(PauliI, np.dot(zero_state, zero_state.T)) + np.kron(PauliX, np.dot(one_state, one_state.T))

    #theta_initialization
    np.random.seed(6)
    init_theta = np.random.uniform(low=0, high=2*np.pi, size=8)

    #learning rate
    lr = 0.001

    #step numbers
    step_num=1000

    #original dimension
    D = 4

    #principal components number
    d = 4
    
    data = ['X3.txt', 'Iris.txt']
    labels = ['Y3.txt', 'Y_Iris.txt']
    for i in range(2):
        for j in range(2):
            if i != j:
                Xs = np.loadtxt(data[i])
                Xt = np.loadtxt(data[j])
                Ys = np.loadtxt(labels[i])
                Yt = np.loadtxt(labels[j])
    
                Xs = normalization(Xs)
                Xt = normalization(Xt)

                Xs = mean(Xs)
                Xt = mean(Xt)

                hat_Xs, Ps = pca(Xs, d)
                hat_Xt, Pt = pca(Xt, d)

                x = init_theta
                cost_history, final_theta = fit(x)
                acc, yrep = fit_predict(x)
                print(final_theta)
                print(acc)

    

    






   