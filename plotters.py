# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:52:56 2022

@author: jgull
"""

import plot_helpers
import pandas as pd
season = 21
players = pd.read_csv("data/allsvenskan/players_raw.csv")
cost_list = players.now_cost.to_list()
points_list= players.total_points.to_list()

plot_helpers.plot_hist_of_costs(cost_list, "Cost of AF players season " +str(season), season,nbins =20)
plot_helpers.plot_hist_of_points(points_list, "Total points of AF players season " +str(season), season,nbins =20)


#%%

import matplotlib.pyplot as plt
import numpy as np
inds = np.linspace(0,10,11, dtype = int)
cost = np.linspace(45,100,11, dtype=int)
fig,ax = plt.subplots()
ax.plot([inds[0],inds[-1]],[cost[0],cost[-1]])
ax.scatter(inds,cost)
ax.set_xlabel("Individual player")
ax.set_ylabel("Cost")
ax.set_title("Example of ideal team")
plt.savefig("results/pl/data_viz/example_lin.png", bbox_inches = "tight")
plt.show()
