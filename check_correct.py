# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:39:05 2022

@author: jgull
"""

import helpers_calc_div
import parsers

import pandas as pd
import os
import ast
generic = lambda x: ast.literal_eval
conv = {"n linear": generic}
seasonss = [1617, 1718, 1819, 1920, 2021]
for season in seasonss:
    df = pd.read_csv(os.path.join("results","pl",str(season),"[4, 5, 1]" + "_budgets_means_raw.csv"))
                 
    nlin = df['n linear'].apply(lambda s: list(ast.literal_eval(s))).to_list()
    nNonlin = df['Non linear n'].apply(lambda s: list(ast.literal_eval(s))).to_list()
    for i in range(len(nlin)):    
        print(str(season), "[4, 5, 1]", round(100*(nNonlin[i][1]/(nlin[i][1]+nNonlin[i][1]))),
              round(100*(nlin[i][1]/(nlin[i][1]+nNonlin[i][1])))
              )

#%%
helpers_calc_div.use_linreg_pl_full_seasons(seasonss)
parsers.write_full_teams("data_cleaned/pl/incnew/1617/")
parsers.write_full_teams("data_cleaned/pl/incnew/1819/")