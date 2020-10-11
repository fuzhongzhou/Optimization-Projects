import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def sampleCov(X):
    """
    :param X: an N*p matrix,
    :return: a p*p matrix, covariance matrix
    """
    a = pd.DataFrame(X)
    a = a - a.mean()
    return a.cov().to_numpy()


def constCorrCov(X, rho=None):
    X_corr = pd.DataFrame(X).corr().to_numpy()

    if rho is None:
        # compute the correlation, and solve for an average
        rho_avg = np.triu(X_corr, k=1).reshape(1, -1)[0]
        rho_avg = rho_avg[rho_avg != 0].mean()
        rho = rho_avg

    X_std = X.std(axis=0)
    cov = rho * (X_std.reshape((-1, 1)) @ X_std.reshape((1, -1))) + \
          (1 - rho) * np.diag(X_std**2)

    return cov


def singleFactorCov(X, f):
    """
    :param X: N*p matrix
    :param f: an array of length N
    :return: a p*p matrix
    """
    reg = LinearRegression()
    reg.fit(f.reshape((-1, 1)), X)
    beta = reg.coef_
    sig = f.std()
    u = X - f.reshape((-1, 1)) @ beta.T     # residual
    w = u.std(axis=0)**2
    cov = sig**2 * beta @ beta.T + np.diag(w)
    return cov


def multiFactorCov(X, f):
    """
    :param X: N*p matrix
    :param f: an N*k matrix, k is the number of factors
    :return: a p*p matrix
    """
    p = X.shape[1]
    beta = []
    for i in range(p):
        mod = sm.OLS(f, X[:, i])
        res = mod.fit()
        beta.append(res.params[0])
    beta = np.array(beta)
    f_cov = np.cov(f.T)

    u = X - f @ beta.T     # residual
    w = u.std(axis=0)**2

    cov = np.matmul(beta, f_cov)
    cov = np.matmul(cov, beta.T)
    cov += np.diag(w)
    return cov


if __name__ == '__main__':
    x = np.random.multivariate_normal(np.ones(10), np.eye(10), 2000)
    f = np.random.randn(2000)
    ff = np.random.randn(2000, 5)

    sampleCov(x)
    constCorrCov(x)
    singleFactorCov(x, f)
    multiFactorCov(x, ff)

    print(0)
