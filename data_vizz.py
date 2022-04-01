# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:03:50 2022

@author: jgull
"""
import pandas as pd
import os
import plot_helpers
seasons = [1617, 1718,1819,1920,2021]

for season in seasons:
    df = pd.read_csv("data/pl_csv/players_raw_" +  str(season) + ".csv" )
    cost_list = df.now_cost.to_list()
    title = "Costs of FPL players season " + str(season)
    plot_helpers.plot_hist_of_costs(cost_list, title , season)    