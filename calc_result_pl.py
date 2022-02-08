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

data2 = getters.get_data()
players = getters.get_players_feature(data2)
# gk, df, mf, fw = getters.get_diff_pos(players)
# gk1, def4, mid4, forw2 = calc.createFormation(gk,df,mf,fw, 4, 4, 2, 100)

playerspl = pd.read_csv("data/pl_csv/players_raw_2021.csv") 
#playerspl = playerspl.set_index('id').T.to_dict()
playerspl = playerspl.to_dict('index')
#print(playerspl[1])
 

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
        combs.to_csv("data_cleaned/pl/" + pos + "/" + str(p) + ".csv")
        combs.to_csv("data_cleaned/pl/" + pos + "/" + str(p) + ".csv",index = False)


