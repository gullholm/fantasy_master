import helpers_calc_div as hcd
import pandas as pd
import numpy as np
import ast
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}
import getters as get
import os
import cleaners


df = pd.read_csv("data_cleaned/pl/1617/[3, 4, 3].csv", converters = conv)
one = df.sample(frac = 1)
all_teams = one["indexes"].to_list()
all_points = one['points_total'].to_list()
all_costs = one['cost'].to_list()
playerspldata = get.get_players_feature_pl("data/pl_csv/players_raw_", 1617)

#%%
import matplotlib.pyplot as plt
i = 0
for t,p,c in zip(all_teams, all_points, all_costs):
    each_team = hcd.team(t,playerspldata)
    each_team.create_int()
    each_team.check_int(3)
    if(each_team.zero_count < 6):
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,10,11).reshape(-1), each_team.ind_cost, edgecolor = "k")
        ax.set_title("Number of empty bins: " + str(each_team.zero_count))
        ax.set_xlabel("Player")
        ax.set_ylabel("Cost")
        i += 1
        if (i == 50):
            break
    
    