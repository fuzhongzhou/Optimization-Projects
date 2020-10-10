from cov_est import *
import matplotlib.pyplot as plt

N = 2000    # period
k = 20      # num of factors
p = 30      # num of securities
S0 = np.random.randn(p) + 10

# without factor, using geometric Brownian Motion here
r = 0.03
BM_cov = np.random.randn(p)
BM_cov = BM_cov.reshape((-1, 1)) @ BM_cov.reshape((1, -1)) / 10
delta = 1 / N
S = np.zeros((N+1, p))
S[0] = S0
z = np.random.multivariate_normal(np.zeros(p), BM_cov, N)
for i in range(N):
    S[i+1] = S[i] * np.exp((r-(BM_cov**2).sum(axis=1)/2)*delta + np.sqrt(delta)*z[i])

# plot the trajectory
plt.plot(S)
plt.show()
r = (S[1:] - S[:-1]) / S[:-1]


# with linear factor structure
f_cov = np.random.randn(k)
f_cov = f_cov.reshape((-1, 1)) @ f_cov.reshape((1, -1))
f_mean = np.random.randn(k)
f = np.random.multivariate_normal(f_mean, f_cov, N)

beta = np.random.randn(k, p) / 100
u = np.random.randn(N, p) / 1000
r = f @ beta + u

print(0)