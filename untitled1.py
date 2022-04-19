import parsers
parsers.clean_all_data_and_make_positions_combs(1819, 
     "data/pl_csv/players_incnew_lin_", dest = "data_cleaned/pl/incnew/")

#parsers.clean_all_data_and_make_positions_combs(1819, 
#    "data/pl_csv/players_incnew_", dest = "data_cleaned/pl/incnew/")

#%%

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

plt.axvline(x=mu-sigma, color='r', ls='--')
plt.axvline(x=mu+sigma, color='r', ls='--')

plt.axvline(x=mu+2*sigma, color='b', ls='--') # *1.645 for 90%
plt.axvline(x=mu-2*sigma, color='b', ls='--') # *1.960 for 95 %

plt.yticks(color='w')
plt.title('Normal distribution')
plt.show()   
