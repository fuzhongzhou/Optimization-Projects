from cov_est import *
import cvxpy as cp

def min_var_port(covmat):
    '''
    Compute minimum-variance portfolio
    :param covmat: covariance matrix
    :return: weights of each asset
    '''

    N = covmat.shape[0]
    X = cp.Variable(N)
    problem = cp.Problem(cp.Minimize(cp.quad_form(X, covmat)),[np.ones(N)@X==1, X>=0])
    problem.solve()
    return X.value


def in_sample_eval(r, f=None):
    '''
    Evaluate different covariance estimators based on corresponding minimum-variance portfolios
    :param r: N*p matrix, in-sample return data
    :param f: N*k matrix, optional multiple factors
    :return: Sharpe ratios of the portfolios
    '''

    ret_mean = r.mean(axis=0)
    N = r.shape[0]
    sr = []

    # sample covariance
    cov_sample = sampleCov(r)
    x_sample = min_var_port(cov_sample)
    ret_sample = x_sample@ret_mean
    sig_sample = np.sqrt(x_sample.T@cov_sample@x_sample)
    sr.append(ret_sample/sig_sample)

    # single factor
    bench_f = r@x_sample # use minimum-variance portfolio based on sample covariance as benchmark
    cov_f = singleFactorCov(r, bench_f)
    x_f = min_var_port(cov_f)
    ret_f = x_f@ret_mean
    sig_f = np.sqrt(x_f.T@cov_f@x_f)
    sr.append(ret_f/sig_f)

    # const correlation
    cov_const = constCorrCov(r)
    x_const = min_var_port(cov_const)
    ret_const = x_const@ret_mean
    sig_const = np.sqrt(x_const.T@cov_const@x_const)
    sr.append(ret_const/sig_const)

    # multi-factor
    if f is None:
        k = 20
        f_cov = np.random.randn(k)
        f_cov = f_cov.reshape((-1, 1))@f_cov.reshape((1, -1))
        f_mean = np.random.randn(k) + 10
        f = np.random.multivariate_normal(f_mean, f_cov, N)

    cov_F = multiFactorCov(r, f)
    x_F = min_var_port(cov_F)
    ret_F = x_F@ret_mean
    sig_F = np.sqrt(x_F.T@cov_F@x_F)
    sr.append(ret_F/sig_F)

    return sr

