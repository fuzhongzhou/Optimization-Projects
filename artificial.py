from cov_est import *
from peformance import *
import matplotlib.pyplot as plt

N = 252*2   # period
T = 2       # Year
k = 10      # num of factors
p = 30      # num of securities
S0 = np.random.randn(p) + 10

train_ratio = 0.8
train_index = int(N*train_ratio)


''' without factor, using geometric Brownian Motion here '''
rf = 0.03
np.random.seed(0)

BM_corr = pd.DataFrame(np.random.randn(20, p)).corr().to_numpy()
sig = np.diag(np.random.uniform(0.15, 0.3, p))
BM_cov = sig @ BM_corr @ sig
delta = T / N
S = np.zeros((N+1, p))
S[0] = S0
z = np.random.multivariate_normal(np.zeros(p), BM_cov, N)
for i in range(N):
    S[i+1] = S[i] * np.exp((rf-(BM_cov**2).sum(axis=1)/2)*delta + np.sqrt(delta)*z[i])
r = (S[1:] - S[:-1]) / S[:-1]


r_train = r[:train_index]
r_test = r[train_index:]
res_nonlinear = eval((r_train,r_test),delta)
print(res_nonlinear)


''' with linear factor structure '''
f_corr = pd.DataFrame(np.random.randn(20, k)).corr().to_numpy()
f_sig = np.diag(np.random.uniform(0.01, 0.02, k))
f_cov = f_sig @ f_corr @ f_sig
f_start = np.random.uniform(-0.4, 0.4, k)

f = np.zeros((N+1, k))
f[0] = f_start
z = np.random.multivariate_normal(np.zeros(k), f_cov, N)
for i in range(N):
    f[i+1] = f[i] + z[i]

f = f[1:]
# pl.plot(f)
beta = np.random.uniform(-0.02, 0.02, (k, p))
u = np.random.randn(N, p) / 100
r = f @ beta + u

r_train = r[:train_index]
r_test = r[train_index:]
res_linear = eval((r_train,r_test),delta,f[:train_index])
print(res_linear)

print(0)