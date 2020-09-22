import numpy as np
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt

#zero_normalization
def zero_normalize(X):
    for i in range(np.size(X, axis=0)):
        X_mean = np.mean(X, axis=1)
        X[i, :] -= X_mean[i]
        X_norm = np.linalg.norm(X, axis=1)
        X[i, :] /= X_norm[i]
    return X

#classification
def fit_predict(Xs, Ys, Xt, Yt):
    Xs_new = Xs
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_new, Ys.ravel())
    y_pred = clf.predict(Xt)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred)

    return acc, y_pred


if __name__ == '__main__':
    data = ['X3.txt', 'Iris.txt']
    labels = ['Y3.txt', 'Y_Iris.txt']
    for i in range(2):
        for j in range(2):
            if i != j:
                Xs = np.loadtxt(data[i])
                Xt = np.loadtxt(data[j])
                Ys = np.loadtxt(labels[i])
                Yt = np.loadtxt(labels[j])
                acc, ypre = fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
