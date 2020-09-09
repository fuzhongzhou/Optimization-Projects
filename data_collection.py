import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.style.use('ggplot')


def compoundSemiAnnual(rate, time):
    """return discount factor"""
    return 1 / (1 + rate / 2)**(2 * time)


# source: https://fixedincome.fidelity.com/ftgw/fi/FILanding#tbcurrent-yields|highest-yield
term_struct_trea = pd.Series(np.array([0.11,	0.12, 0.12,	0.13, 0.14,	0.17, 0.28, 0.69, 1.24, 1.45]) / 100,
                             index=[0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 20, 30])
'''
fig = plt.figure(figsize=(10, 6))
ax = fig.subplots(1, 1)
term_struct_trea.plot(ax=ax)
ax.set_title('Treasury term structure')
plt.show()
 '''

liab_stream = pd.Series([11, 9, 7, 9, 9, 12, 8, 10, 6, 5, 7, 7, 8, 7, 9, 9],
                        index=3.5 / 12 + np.arange(0, 8, 0.5))  # there are 3.5 months from mid-Sep to end-Dec 2020

term_struct_interp = interp1d(term_struct_trea.index, term_struct_trea.values,
                              kind='linear')  # linear interp function
liab_discount_factors = pd.Series(map(compoundSemiAnnual, liab_stream.index, term_struct_interp(liab_stream.index)),
                                  index=liab_stream.index)
liab_discounted = liab_stream * liab_discount_factors


print(0)


