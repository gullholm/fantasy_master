# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:03:50 2022

@author: jgull
"""
import pandas as pd
import plot_helpers
seasons = [1617, 1718,1819,1920,2021]

for season in seasons:
    df = pd.read_csv("data/pl_csv/players_raw_" +  str(season) + ".csv" )
    cost_list = df.now_cost.to_list()
    cost_change_list = df.cost_change_start
    cost_change_list_no0 = [x for x in cost_change_list if x != 0]
    
    points_list = df.total_points.to_list()
    points_list_no0 = [x for x in points_list if x >0]
    
    title_c = "Costs of FPL players season " + str(season)
    plot_helpers.plot_hist_of_costs(cost_list, title_c , season)
    title_p = "Total points of FPL players season " + str(season)
    plot_helpers.plot_hist_of_points(points_list_no0, title_p,season)
    print("points # of zeros " + str(season) +":", len(points_list)-len(points_list_no0))
    print("cost change # of zeros " + str(season) +":", len(cost_change_list)-len(cost_change_list_no0))
    plot_helpers.plot_hist(cost_change_list, season)