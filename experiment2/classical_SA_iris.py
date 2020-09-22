import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn.decomposition import PCA

#zero mean
def zero_mean(X):      
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

#subspace alignment
def sa(hat_Xs, Ps, Pt):
    M = np.dot(Ps.T, Pt)
    hat_Xa = np.dot(hat_Xs, M)

    return hat_Xa

#target predictioin
def fit_predict(hat_Xs, Ys, hat_Xt, Yt, Ps, Pt):
    hat_Xa = sa(hat_Xs, Ps, Pt)
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(hat_Xa, Ys.ravel())
    y_pred = clf.predict(hat_Xt)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred)

    return acc, y_pred

if __name__ == '__main__':
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
                Xs = zero_mean(Xs)
                Xt = zero_mean(Xt)
                hat_Xs, Ps = pca(Xs, d)
                hat_Xt, Pt = pca(Xt, d)
                hat_Xa = sa(hat_Xs, Ps, Pt)
                acc, y_pred = fit_predict(hat_Xs, Ys, hat_Xt, Yt, Ps, Pt)
                print(acc)

    