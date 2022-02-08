# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:20:12 2022

@author: jonat

"""


#%%
import pandas as pd
import getters
import calculations as calc
import cleaners
import parsers

# 2021

playerspl = pd.read_csv("data/pl_csv/players_raw_2021.csv") 
playerspl = playerspl.to_dict('index')
playerspldata = getters.get_players_feature_pl(playerspl)

formations = [[3,4,5],[3,4,5],[1,2,3]]
form_name = ["df", "mf", "fw"]
all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl()[1:]


for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
    print(part)
    print(df)
    print(pos)
    for p in part:
        all_cleaned = cleaners.run_all_cleans(df, p)
        combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)[0]
        combs.to_csv("data_cleaned/pl/2021" + pos + "/" + str(p) + ".csv")
        combs.to_csv("data_cleaned/pl/2021" + pos + "/" + str(p) + ".csv",index = False)

# In[]

# goalkeepers

gk, df,mf,fw = getters.get_diff_pos(playerspldata)

df_gk = pd.DataFrame.from_dict(gk, orient='index')

sorted_df_gk = df_gk.sort_values(by= ['now_cost'])

cleaned_gk = cleaners.clean_gk(sorted_df_gk)
cleaned_gk.reset_index(inplace=True)
cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
cleaned_gk.drop('element_type', inplace=True, axis=1)
cleaned_gk.to_csv("data_cleaned/pl/2021/gk.csv")