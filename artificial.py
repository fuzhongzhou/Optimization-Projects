from cov_est import *
from peformance import *
import matplotlib.pyplot as plt

N = 2520    # period
T = 10      # Year
k = 20      # num of factors
p = 30      # num of securities
S0 = np.random.randn(p) + 10

# without factor, using geometric Brownian Motion here
rf = 0.03
BM_cov = np.random.randn(p)
BM_cov = BM_cov.reshape((-1, 1)) @ BM_cov.reshape((1, -1)) / 10
delta = T / N
S = np.zeros((N+1, p))
S[0] = S0
z = np.random.multivariate_normal(np.zeros(p), BM_cov, N)
for i in range(N):
    S[i+1] = S[i] * np.exp((rf-(BM_cov**2).sum(axis=1)/2)*delta + np.sqrt(delta)*z[i])

r = (S[1:] - S[:-1]) / S[:-1]

train_ratio = 0.7
train_index = int(N*train_ratio)
r_train = r[:train_index]
r_test = r[train_index:]
sr_in = in_sample_eval(r_train/delta)


# with linear factor structure
f_cov = np.random.randn(k)
f_cov = f_cov.reshape((-1, 1)) @ f_cov.reshape((1, -1))
f_mean = np.random.randn(k)
f = np.random.multivariate_normal(f_mean, f_cov, N)

beta = np.random.randn(k, p) / 100
u = np.random.randn(N, p) / 1000
r = f @ beta + u

print(0)