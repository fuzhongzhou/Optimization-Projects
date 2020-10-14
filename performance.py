from cov_est import *
import cvxpy as cp
import cvxopt as cpt
from scipy.optimize import minimize

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

def max_div_port(covmat):
    '''
    Compute maximum-diversification portfolio
    :param covmat: covariance matrix
    :return: weights of each asset
    '''

    N = covmat.shape[0]
    sig = np.sqrt(covmat.diagonal())
    X = cp.Variable(N)
    k = cp.Variable(1)
    problem = cp.Problem(cp.Minimize(cp.quad_form(X, covmat)), [np.ones(N)@X==k, X >= 0, sig.T@X==1])
    problem.solve()
    return X.value/k.value

def risk_par_port(covmat):
    '''
    Compute risk-parity portfolio
    :param covmat: covariance matrix
    :return: weights of each asset
    '''

    # N = covmat.shape[0]
    # X = cp.Variable(N)
    # problem = cp.Problem(cp.Minimize(cp.quad_form(X,covmat)/2-np.ones(N)@cp.log(X)/N), [X >= 0])
    # problem.solve()
    # return X.value/X.value.sum()

    N = covmat.shape[0]
    w0 = np.ones(N)/N
    x0 = w0/(w0.T@covmat@w0)**0.5
    def optfunc(x):
        return (x.T@covmat@x/2-np.ones(N)@np.log(x)/N)**2
    cons = {'type':'ineq','fun':lambda x: x-1e-5}
    ops = {'maxiter':1e8}
    res = minimize(optfunc, x0, constraints=cons,options=ops)

    def res_check(x):
        w = x/x.sum()
        sig = np.sqrt(w.T@covmat@w)
        rhs = sig/N
        lhs = w*(covmat@w)/sig
        diff = np.sum((lhs-rhs)**2)
        return diff

    assert res_check(res.x)<1e-6

    return res.x/res.x.sum()

# def risk_par_port(covmat):
#     '''
#         Compute risk-parity portfolio
#         :param covmat: covariance matrix
#         :return: weights of each asset
#         '''
#
#     N = covmat.shape[0]
#     covmat = cpt.matrix(covmat)
#     def F(x=None, z=None):
#         if x is None: return (0, cpt.matrix(np.exp(np.random.rand(N))))
#         if cpt.min(x)<=0: return 0
#         f = 0.5*x.T*covmat*x - np.sum(cpt.log(x)/N)
#         df = covmat*x - 1/N*x**-1
#         if z is None: return f, df
#         H = z[0]*(covmat+1/N*cpt.matrix(np.array(x**-1)@np.array(x**-1).T))
#         return f, df, H
#     res = cpt.solvers.cp(F)['x']
#     return res/res.sum()

def port_weight(covmat, port_type='minvar'):
    '''
    Provide an interface to calculate portfolio weights
    :param covmat: covariance matrix
    :param port_type: type of portfolio to calculate, default to be 'minvar'
                    possible values include ['minvar','maxdiv','riskpar']
    :return: weights of each asset
    '''

    if port_type=='minvar':
        return min_var_port(covmat)
    elif port_type=='maxdiv':
        return max_div_port(covmat)
    elif port_type=='riskpar':
        return risk_par_port(covmat)
    else:
        raise ValueError("port_type has to be one of ['minvar','maxdiv','riskpar']")

def in_sample_eval(r, f, port_type, bench_f=None):
    '''
    Evaluate different covariance estimators based on corresponding type of portfolios
    :param r: N*p matrix, in-sample return data
    :param f: N*k matrix, multiple factors
    :param port_type: type of portfolio to calculate, default to be 'minvar'
                    possible values include ['minvar','maxdiv','riskpar']
    :param bench_f: optional single factor, default to be None
    :return: Weights and Sharpe ratios of each portfolio
    '''

    ret_mean = r.mean(axis=0)
    N = r.shape[0]
    weights = []
    sr = []

    # sample covariance
    cov_sample = sampleCov(r)
    x_sample = port_weight(cov_sample, port_type)
    ret_sample = x_sample@ret_mean
    sig_sample = np.sqrt(x_sample.T@cov_sample@x_sample)
    weights.append(x_sample)
    sr.append(ret_sample/sig_sample)

    # const correlation
    cov_const = constCorrCov(r)
    x_const = port_weight(cov_const, port_type)
    ret_const = x_const@ret_mean
    sig_const = np.sqrt(x_const.T@cov_const@x_const)
    weights.append(x_const)
    sr.append(ret_const/sig_const)

    # single factor
    if bench_f is None:
        bench_f = r@x_const  # use benchmark portfolio based on constant correlation covariance
    cov_f = singleFactorCov(r, bench_f)
    x_f = port_weight(cov_f, port_type)
    ret_f = x_f@ret_mean
    sig_f = np.sqrt(x_f.T@cov_f@x_f)
    weights.append(x_f)
    sr.append(ret_f/sig_f)

    # multi-factor
    cov_F = multiFactorCov(r, f)
    x_F = port_weight(cov_F, port_type)
    ret_F = x_F@ret_mean
    sig_F = np.sqrt(x_F.T@cov_F@x_F)
    weights.append(x_F)
    sr.append(ret_F/sig_F)

    return weights, sr

def out_sample_eval(r, weights, step_t):
    sr = []
    N = len(r)
    for w in weights:
        actual_r = r@w
        total_r = np.prod(1+actual_r)**(1/(N*step_t)) - 1
        total_std = np.std(actual_r/step_t)
        sr.append(total_r/total_std)
    return sr


def eval(r,step_t,f=None,bench_f=None):

    port_type = ['minvar','maxdiv','riskpar']
    # port_type = ['minvar', 'maxdiv']
    eval_type = ['_in','_out']
    result = pd.DataFrame(np.zeros((4,6)),index=['sample','const','single_fac','multi_fac'],columns=[i+j for i in port_type for j in eval_type])

    r_train, r_test = r
    N_train = r_train.shape[0]
    N_test = r_test.shape[0]
    N = N_train + N_test

    if f is None:
        k = 20
        f_cov = np.random.randn(k)
        f_cov = f_cov.reshape((-1, 1))@f_cov.reshape((1, -1))
        f_mean = np.random.randn(k)
        f = np.random.multivariate_normal(f_mean, f_cov, N_train)


    for i, p_type in enumerate(port_type):
        in_weights, in_sr = in_sample_eval(r_train/step_t, f, p_type, bench_f)
        out_sr = out_sample_eval(r_test, in_weights, step_t)
        result.loc[:, p_type + '_in'] = in_sr
        result.loc[:, p_type + '_out'] = out_sr

    return result