# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:39:05 2022

@author: jgull
"""

import helpers_calc_div
import parsers

seasons = [1718, 1819, 1920, 2021]

for season in seasons:
    parsers.write_full_teams("data_cleaned/pl/" + str(season) + "/")
    

seasons = [1617,1718, 1819, 1920, 2021]

for season in seasons:
    helpers_calc_div.use_linreg_pl_full_seasons(season)
