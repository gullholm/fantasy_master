# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:15:57 2022

@author: jgull
"""

import numpy as np
import getters as get
import matplotlib.pyplot as plt
from collections import Counter
import random

data_as = get.get_data()
players = get.get_players_feature(data_as)

def parse_players_feature_to_list(players):
    ids, pos, cost, point = [], [], [], []

    for (k,v) in players.items():
#        ids.append(k)
        pos.append(v.get("element_type"))
        cost.append(v.get("now_cost"))
        point.append(v.get("total_points"))

    return(pos,cost,point)

def create_new_players(lists, inds):
    new_players = []
    for i in range(n):
        randind = random.randint(0, len(inds)-1)
        print(randind)
        temp = [x[randind] for x in lists]
        new_players.append(temp)
    return(new_players)

pos, cost, point = parse_players_feature_to_list(players)

rang = np.linspace(min(cost), max(cost), dtype = int,num = 10)

bins = np.digitize(cost, rang).tolist()

len(Counter(bins)) == len(rang)

cou = Counter(bins).most_common()
most = cou.pop(0)
new_players = []
all_lists = [pos,cost,point]
for co in cou:
    n = most[1]-co[1]
    inds = [i for (i,x) in enumerate(bins) if x == co[0]]
    all_lists_inds = [[y[x] for x in inds] for y in all_lists]
    temp = create_new_players(all_lists_inds, inds)
    new_players.extend(temp)



new_all_lists = [[x[j] for x in new_players] for j in range(3)]

snew_all_lists = [lis + lis2 for (lis,lis2) in zip(all_lists,new_all_lists)]

plt.hist(snew_all_lists[1], bins = 10)

cou = Counter(bins).most_common()

rang = np.linspace(min(cost), max(cost), dtype = int,num = 10)

bins2 = np.digitize(cost, rang).tolist()

print(Counter(cost).most_common())

import pandas as pd

inds = [i+1 for (i,x) in enumerate(snew_all_lists[0])]
snew_all_lists.append(inds)
new_data = pd.DataFrame(snew_all_lists).transpose()
new_data.columns = ["element_type", "now_cost", "total_points", "id"]
new_data.to_csv("data/inc_copy_players/as21.csv")


#%%

import pandas as pd
import getters
import calculations as calc
import cleaners
import parsers

def clean_all_data_and_make_positions_combs(season, bas = "data/pl_csv/players_raw_", dest = "data_cleaned/pl/",  clean_all = True):
    
    csv_file = str(bas) + str(season) + ".csv"
    playerspl = pd.read_csv(csv_file) 
    playerspl = playerspl.to_dict('index')
    playerspldata = getters.get_players_feature_pl(playerspl)
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(csv_file)[1:]
    
    
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        #print(part)
        #print(df)
        print(pos)
        for p in part:
            print(p)
            all_cleaned = cleaners.run_all_cleans(df, p)
            
            if clean_all: 
                combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv")
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)
            else: 
                combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
                combs.to_csv("individual_data_cleaned/pl/" + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)

    
    # Goalkeepers
    
    gk, df,mf,fw = getters.get_diff_pos(playerspldata)
    
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    
    cleaned_gk = cleaners.clean_gk(sorted_df_gk)
    cleaned_gk.reset_index(inplace=True)
    cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    if clean_all: 
        cleaned_gk.to_csv(dest + str(season) + "/gk.csv")
    else : 
        cleaned_gk.to_csv(dest + str(season) + "/gk.csv")
        
    print("Done with " + str(season))
    
#%%
clean_all_data_pl(21, "data/inc_copy_players/as","data_cleaned/inc_copy_players/")
clean_all_data_pl(1819)

