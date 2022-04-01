# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:30:18 2022

@author: jgull
"""

import numpy as np
import calculations as calc
import getters
import pandas as pd
import cleaners
import os
import ast


def parse_formations_points_or_cost(all_combs): # Arguments is cost/points for each formation part
    """
    Adds together the cost and points of different formation parts into 
    all possible whole teams

    """
    
    gk = np.array(all_combs[0])
    df = np.array(all_combs[1])
    mf = np.array(all_combs[2])
    fw = np.array(all_combs[3])

    gk_df = np.add.outer(df, gk)
    gk_df_mf = np.add.outer(mf, gk_df)
    gk_df_mf_fw = np.add.outer(fw, gk_df_mf)

    return gk_df_mf_fw

# (fw,mf,df,gk) index


def find_best_team(under_cost, points):
    """   
    Finds best team (wrt points) index amongst team under cost limit


    """   
    point_f = np.zeros(under_cost.shape[0])
  
    for i in range(under_cost.shape[0]):
        point_f[i] = points[under_cost[i][0],
                           under_cost[i][1], under_cost[i][2], under_cost[i][3]]
        
    return(np.argmax(point_f))


#def convert_from_matrix_to_list(all_combs, all_points, all_costs)


def create_all_combs_from_cleaned_df(df_full, df_part, form_n):
    combs = np.transpose(calc.nump2(len(df_part), form_n))
#    print(df_part.index[:10])
    combs_indexes = calc.calcIndexOld(combs, df_part.index) 
    pointsList = calc.createPointsList(df_full)
    costList = calc.createCostList(df_full)
    combsPoints, combsCost = [], []
    for i in range(len(combs)): 
        combsPoints.append(calc.pointsPerTeam4(combs_indexes[i],pointsList))
        combsCost.append(calc.costPerTeam4(combs_indexes[i], costList)) 
    combs_parts = pd.DataFrame(list(zip(combsPoints, combsCost, combs_indexes)),
                           columns =['total_points', 'now_cost', 'indexes'])
    sortedCombs_parts = combs_parts.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])
    print("before return")
    return(cleaners.delete_worse_points_when_increasing_cost(sortedCombs_parts, 1))

def worst_create_all_combs_from_cleaned_df(df_full, df_part, form_n):
    combs = np.transpose(calc.nump2(len(df_part), form_n))
#    print(df_part.index[:10])
    combs_indexes = calc.calcIndexOld(combs, df_part.index) 
    pointsList = calc.createPointsList(df_full)
    costList = calc.createCostList(df_full)
    combsPoints, combsCost = [], []
    for i in range(len(combs)): 
        combsPoints.append(calc.pointsPerTeam4(combs_indexes[i],pointsList))
        combsCost.append(calc.costPerTeam4(combs_indexes[i], costList)) 
    combs_parts = pd.DataFrame(list(zip(combsPoints, combsCost, combs_indexes)),
                           columns =['total_points', 'now_cost', 'indexes'])
    sortedCombs_parts = combs_parts.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])
    print("before return")
    return(cleaners.delete_better_points_when_decreasing_cost(sortedCombs_parts, 1))
    
def calc_full_teams(all_combs):
    all_points = calc.calc_from_combs(all_combs, "total_points")
    all_costs = calc.calc_from_combs(all_combs, "now_cost" )

    points_full = parse_formations_points_or_cost(all_points)
    costs_full = parse_formations_points_or_cost(all_costs)
    
    it = np.nditer(points_full, flags=['multi_index'])
    bit = np.nditer(costs_full, flags = ['multi_index'])
    sep_ids  = [combs['indexes'].values.tolist() for combs in reversed(all_combs)]
    costs_total, points_total, indexes = [], [], []

    for x, y in zip(it,bit):
        points_total.append(x)
        costs_total.append(y)
        reg_list = [x[it.multi_index[i]] for (i,x) in enumerate(sep_ids)]
        indexes.append([item for sublist in reg_list for item in sublist])
        
    full_teams_df = pd.DataFrame({"cost": costs_total, "points_total": points_total, "indexes": indexes})
    return(full_teams_df)

def write_full_teams(loc):
    generic = lambda x: ast.literal_eval(x)
    conv = {'indexes': generic}
    all_pass_combs = [[3,5,2],[3,4,3],[4,3,3], [4,4,2], [4,5,1], [5,3,2], [5,4,1]]
    form_name = ["/df", "/mf", "/fw"]
    all_combs = []
    
    for comb in all_pass_combs: 

        all_combs = [pd.read_csv(loc + form + "/" + str(c) + ".csv", converters = conv) for (c,form) in zip(comb,form_name)]
        #all_combs.insert(0, pd.read_csv(loc + "gk.csv", engine = "pyarrow"))
        all_combs.insert(0, pd.read_csv(loc + "gk.csv"))

        all_combs[0]['indexes'] = all_combs[0]['indexes'].apply(lambda x: [x])
        done_df = calc_full_teams(all_combs)
        done_df.to_csv(loc + str(comb) + ".csv", index = False)


def clean_all_data_and_make_positions_combs(season, bas = "data/pl_csv/players_raw_", dest = "data_cleaned/pl/", k = "0.2", clean_all = True):
    
    playerspldata = getters.get_players_feature_pl(bas,season)
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(bas, season)[1:]
    new_dest = os.path.join(dest, str(season))
    os.makedirs(new_dest, exist_ok = True)
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        #print(part)
        #print(df)
        print(len(df))
        os.makedirs(os.path.join(new_dest, pos) , exist_ok = True)

        for p in part:
            print(p)
            
            all_cleaned = cleaners.run_all_cleans(df, p)
            if clean_all: 
            
#                print(len(all_cleaned))
                combs = create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
#                print(combs.head(5))
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv")
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)
            else: 
                combs = create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
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


def clean_all_data_and_make_positions_combs_worst(season, bas = "data/pl_csv/players_raw_", dest = "data_cleaned/pl/worst/", k = "0.2", clean_all = True):
    
    playerspldata = getters.get_players_feature_pl(bas,season)
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(bas, season)[1:]
    new_dest = os.path.join(dest, str(season))
    os.makedirs(new_dest, exist_ok = True)
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        #print(part)
        #print(df)
        print(len(df))
        os.makedirs(os.path.join(new_dest, pos) , exist_ok = True)

        for p in part:
            print(p)
            
            all_cleaned = cleaners.worst_clean_all_data_pl(df, p)
            if clean_all: 
            
#                print(len(all_cleaned))
                combs = worst_create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
#                print(combs.head(5))
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv")
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)
            else: 
                combs = worst_create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
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


"""
def calc_p_c_per_part(gk_comb, def_comb,  mf_comb, fw_comb): 


   
  #   Calculate points and cost for the different combinations of the 
  #   formation parts
 #   
    
    
    forPoints, midPoints, defPoints, gkPoints, forCosts, midCosts, defCosts, gkCosts = [],[],[],[],[],[], [], []
    
    costList = calc.createCostList()
    pointsList = calc.createPointsList()
    
    for i in range(len(fw_comb)):
        forPoints.append(calc.pointsPerTeam4(fw_comb[i],pointsList))
        midPoints.append(calc.pointsPerTeam4(mf_comb[i],pointsList))        
        defPoints.append(calc.pointsPerTeam4(def_comb[i],pointsList))

        forCosts.append(calc.costPerTeam4(fw_comb[i], costList))
        midCosts.append(calc.costPerTeam4(mf_comb[i], costList))
        defCosts.append(calc.costPerTeam4(def_comb[i], costList))
    
    for i in range(len(gk_comb)):
        gkCosts.append(calc.costPerTeam4(gk_comb[i], costList))    
        gkPoints.append(calc.pointsPerTeam4(gk_comb[i], pointsList))
    
    points = [gkPoints, defPoints, midPoints, forPoints]
    costs = [gkCosts, defCosts, midCosts, forCosts]
    return points, costs


    
def get_best_team_from_random(n, formation = [4, 4, 2], cost_limit = 750, seed = 123):
    data2 = getters.get_data()
    players = getters.get_players_feature(data2)
    gk, df, mf, fw = getters.get_diff_pos(players)


#  Create n combinations of different formation  (max at goalies)
#    gk_combs, df_combs, mf_combs, fw_combs = calc.createFormation(gk, df, mf, fw, formation[0], 
 #                                                                 formation[1], 
 #                                                                 formation[2], 
#                                                                  n = n)
#   print(gk_combs)

#    points_comb, costs_comb = calc_p_c_per_part(gk_combs,
#                                                df_combs, mf_combs, fw_combs)

    gk =?
    
    points_full = parse_formations_points_or_cost(points_comb[0],
                                                     points_comb[1], 
                                                     points_comb[2], points_comb[3])
    costs_full = parse_formations_points_or_cost(costs_comb[0],
                                                     costs_comb[1], 
                                                     costs_comb[2], costs_comb[3])
    


    
    defe = np.transpose(calc.nump2(len(df),4))
    midf = np.transpose(calc.nump2(len(mf),4))
    forw = np.transpose(calc.nump2(len(fw),2))    
    glk = np.transpose(calc.nump2(len(gk),1))


    forwards = calc.calcindex(forw, fw, formation[2], n, seed) 
    defenders = calc.calcindex(defe, df, formation[0], n, seed )
    midfielders = calc.calcindex(midf, mf, formation[1], n, seed)
    goalkeepers = calc.calcindex(glk, gk, 1, len(gk), seed)

    under_cost =  np.argwhere(costs_full < cost_limit) 
    best = find_best_team(under_cost, points_full)

    sep_ids = [forwards, midfielders, defenders, goalkeepers]
#   print(sep_ids)
#   print(best)
#   print(under_cost)
#   print(under_cost.shape)
#   print(len(sep_ids[2]))
#    print(sep_ids)
#    print(under_cost)
    best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]
    
    return best_team_ids # FW-MF-DF-GK

def change_dict_to_2darray(dictt, value):
    dfPoints = {k:v.get(value) for (k,v) in dictt.items()}
    dfPointz = np.array(list(dfPoints.items()))
    
    return dfPointz
    """