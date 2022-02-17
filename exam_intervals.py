#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:38:11 2022

@author: josef
"""

import pandas as pd
import numpy as np
import getters as get
import ast
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}

data = get.get_data()

one = pd.read_csv("data_cleaned/as/[3, 4, 3].csv", converters = conv)
#%%
def filter_df(df, lwr, upper):
    df = df[df['cost'] < upper]
    df_new = df[df['cost'] > lwr]
    return(df_new)

def lower_bound(df_list):
    return 0

def flatten_all_ids(all_teams_id):
    return(set([item for sublist in all_teams_id for item in sublist]))

def det_a(data, all_ids):
    return(min(get.get_cost_team(data,all_ids)))

def thry_interval(a,upper):
    inter = np.linspace(a+2, (2*upper/11)-a, 11)
    return(np.round(inter))

def is_diverse(team_ids,theory_int, full_data):
    team_cost = get.get_cost_team(full_data, team_ids).sort()
    for (th, re) in zip(theory_int, team_cost):
        if (re > th + 2 OR re < th - 2):
            return(False)
    
    return(True)
all_ids = list(flatten_all_ids(one["indexes"].to_list()))

a = det_a(data, all_ids)

inter = thry_interval(a, 700)
