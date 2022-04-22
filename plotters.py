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

def load_cost_points(season, typ = "raw"):
    players = get.get_players_feature_pl("data/pl_csv/players_" + typ + "_", str(season))
    cost_list = [get.get_cost_player(players, i) for i in range(1,len(players)+1)]
    points_list = [get.get_points_player(players, i) for i in range(1, len(players)+1)]
    cost_list, points_list = zip(*[(x,y) for x,y in zip(cost_list, points_list) if x >0 and y >0])

    return(cost_list, points_list)

#%%
seasons = [1617,1718,1819,1920,2021]
for season in seasons:
    cost_list, points_list = load_cost_points(season)
    fig, ax = plt.subplots()
    ax.scatter(cost_list, points_list, s = 2/3, color = "k" , marker = 'o')
    ax.set_xlim(38,140)
    ax.set_ylim(0,300)
    fan = "FPL "
    ax.set_title("Cost vs. Total points for "+ fan  + str(season))
    ax.set_xlabel("Cost")
    ax.set_ylabel("Total points")
    plt.savefig("results/pl/data_viz/costvspoints/pc_" + str(season))
    
#%% AS
players = get.get_players_feature_pl("data/allsvenskan/players_raw_", str(21))
cost_list = [get.get_cost_player(players, i) for i in range(1,len(players)+1)]
points_list = [get.get_points_player(players, i) for i in range(1, len(players)+1)]
cost_list, points_list = zip(*[(x,y) for x,y in zip(cost_list, points_list) if x >0 and y >0])
fig, ax = plt.subplots()
ax.scatter(cost_list, points_list, s = 2/3, color = "k" , marker = 'o')
ax.set_xlim(38,140)
ax.set_ylim(0,300)
fan = "AF "
ax.set_title("Cost vs. Total points for "+ fan  + str(season))
ax.set_xlabel("Cost")
ax.set_ylabel("Total points")
plt.savefig("results/as/data_viz/pc_as")

#%% 
c_l_1, p_l_1 = load_cost_points(1617, typ = "incnew_lin")
c_l_2, p_l_2 = load_cost_points(1819, typ = "incnew_lin")

plot_helpers.plot_hist_of_costs(c_l_1, "Cost of players FPL season 1617 including new players", 1617, typ ="incnew")
plot_helpers.plot_hist_of_costs(c_l_2, "Cost of players FPL season 1617 including new players", 1819, typ ="incnew")

plot_helpers.plot_hist_of_points(p_l_1, "Total points of players FPL season 1617 including new players", 1617, typ ="incnew")
plot_helpers.plot_hist_of_points(p_l_2, "Cost of players FPL season 1617 including new players", 1819, typ ="incnew")
