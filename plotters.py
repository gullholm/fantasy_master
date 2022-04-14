# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:52:56 2022

@author: jgull
"""

import plot_helpers
import pandas as pd
import helpers_calc_div as hcd
import ast
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}
import matplotlib.pyplot as plt
import numpy as np
import os
import getters as get
def load_():
    one = pd.read_csv(os.path.join("data_cleaned","pl","1819", "[4, 4, 2].csv"), converters =conv)
    playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", 1819)
    return(one,playerspldata)
#%%
season = 21
players = pd.read_csv("data/allsvenskan/players_raw.csv")
cost_list = players.now_cost.to_list()
points_list= players.total_points.to_list()

plot_helpers.plot_hist_of_costs(cost_list, "Cost of AF players season " +str(season), season,nbins =20)
plot_helpers.plot_hist_of_points(points_list, "Total points of AF players season " +str(season), season,nbins =20)


#%%


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

#%%

one, playerspldata = load_()
#%%
ones = one.sample(100)
all_teams = ones["indexes"].to_list()
all_points = ones['points_total'].to_list()
all_costs = ones['cost'].to_list()
for t,p,c in zip(all_teams, all_points, all_costs):
    each_team = hcd.team(t,playerspldata)
    each_team.lin_fit()
    if each_team.r2 >= 0.75:
        fig, ax = plt.subplots()
        ax.scatter(each_team.x, each_team.ind_cost, edgecolor = "k")    
        ax.plot(each_team.x, each_team.y_pr)
        ax.set_title(r'$R^2:$' + str(round(each_team.r2,2)))
        ax.set_xlabel("Player")
        ax.set_ylabel("Cost")
    
    