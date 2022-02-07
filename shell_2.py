# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:21:55 2022

@author: jgull
"""


import getters
import numpy as np
import time
import matplotlib.pyplot as plt
import calculations as calc
import cleaners
import parsers


formations = [[3,4,5],[3,4,5],[1,2,3]]
form_name = ["df", "mf", "fw"]
all_parts_but_goalie = cleaners.all_forms_as_df_cleaned()[1:]


for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
    for p in part:
        all_cleaned = cleaners.run_all_cleans(df, p)
        combs = parsers.create_all_combs_from_cleaned_df(all_cleaned, p)[0]
        combs.to_csv("data_cleaned/as/" + pos + "/" + str(p) + ".csv")
        
        
all_cleaned = cleaners.run_all_cleans(all_parts_but_goalie[1], 4)
combs = parsers.create_all_combs_from_cleaned_df(all_cleaned, 4)[0]
combs.to_csv("data_cleaned/as/df/4.csv",index=False)
