# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:53:29 2022

@author: jonat
"""


import pandas as pd
import parsers
import ast
import numpy as np
import getters
generic = lambda x: ast.literal_eval(x)


def calc_best_team_under_budget(players, max_cost, full_costs, full_points, combs):
    gk_combs=combs[0]
    df_combs=combs[1]
    mf_combs=combs[2]
    fw_combs=combs[3]
    
    #print(len(costs_full))
    under_cost =  np.argwhere(full_costs <= max_cost) 
    #print(len(under_cost))
    best = parsers.find_best_team(under_cost, full_points)
    sep_ids = [fw_combs['indexes'].values.tolist(),mf_combs['indexes'].values.tolist() 
               , df_combs['indexes'].values.tolist(), gk_combs['indexes'].values.tolist()]
    best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]
    best_team_i = [item for sublist in best_team_ids for item in sublist]    
    best_team_ids_values = [players[ids] for ids in best_team_i]
    #best_team_names = getters.get_full_name_team(data, best_team_i)

    best_sum_cost = [player['now_cost'] for player in best_team_ids_values] 
    best_sum_points = [player['total_points'] for player in best_team_ids_values]
    
    return best_sum_cost, best_sum_points, best_team_i

#%%

#For as 

conv = {'indexes': generic}

data = getters.get_data()
players = getters.get_players_feature(data)

formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

for i in range(len(formations)):
    print("Running formation " + str(formations[i]))
    df = formations[i][0]
    mf = formations[i][1]
    fw = formations[i][2]

    df_csv = "data_cleaned/as/df/" + str(df) + ".csv"
    mf_csv = "data_cleaned/as/mf/" + str(mf) + ".csv"
    fw_csv = "data_cleaned/as/fw/" + str(fw) + ".csv"

    gk_combs = pd.read_csv("data_cleaned/gk.csv", converters = conv)
    df_combs = pd.read_csv(df_csv, converters = conv)
    mf_combs = pd.read_csv(mf_csv, converters = conv)
    fw_combs = pd.read_csv(fw_csv, converters = conv)
    gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])
    all_combs=[gk_combs, df_combs, mf_combs, fw_combs]
    
    gk_points = gk_combs['total_points'].values
    df_points = df_combs['total_points'].values
    mf_points = mf_combs['total_points'].values
    fw_points = fw_combs['total_points'].values
    
    all_points=[gk_points, df_points, mf_points, fw_points]
    points_full = parsers.parse_formations_points_or_cost(all_points)
    
    gk_costs = gk_combs['now_cost'].values
    df_costs = df_combs['now_cost'].values
    mf_costs = mf_combs['now_cost'].values
    fw_costs = fw_combs['now_cost'].values
    
    all_costs = [gk_costs, df_costs, mf_costs, fw_costs]
    costs_full = parsers.parse_formations_points_or_cost(all_costs)

    best_costs, sorted_costs, best_ids = [],[], []
    best_total_costs, best_total_points =[],[]
    
    for j in range(500, 900, 50):
        costs, points, ids = calc_best_team_under_budget(players, j, costs_full,points_full, all_combs)
        best_costs.append(costs)
        sorted_costs.append(sorted(costs))
        best_total_costs.append(sum(costs))
        best_total_points.append(sum(points))
        best_ids.append(ids)
     
    budget = range(500, 900, 50)    
    dataframe = pd.DataFrame()
    dataframe['Budget'] = budget
    dataframe['Best total cost'] = best_total_costs 
    dataframe['Best total points'] = best_total_points
    dataframe['Individual costs'] = best_costs
    dataframe['Sorted individual costs'] = sorted_costs
    #print (dataframe)
    if i == 0:
        best_dataframe = dataframe
        df_form= []
        for l in range(len(best_dataframe)):
            df_form.append(formations[0])
        best_dataframe['Formation'] = df_form 
    else:
        for j in range(len(dataframe)):
            if dataframe.loc[j]['Best total points'] > best_dataframe.loc[j]['Best total points']:
                best_dataframe.loc[j]=dataframe.loc[j]
                df_form[j]=formations[i]
                best_dataframe['Formation'] = df_form
                ''
    print(dataframe['Best total cost'])            

    # Uncomment to save as csv
    
    formation= str(df) + '-' + str(mf) + '-' + str(fw)
    csv_output ='results/as/' + formation + '.csv'
    dataframe.to_csv(csv_output, index=False)
best_dataframe.to_csv('results/as/best.csv', index=False)

# In[]

def calc_best_team_under_budget2(max_cost):
    
    under_cost =  np.argwhere(all_costs <= max_cost) 
            
    point_f = []
    for idx in under_cost:
        point_f.append(all_points[idx])
    
    best = np.argmax(point_f)
    best_id = under_cost[best][0]
    
    best_indexes = all_combs.loc[best_id]['indexes']
    
    best_costs=[]
    for idx in best_indexes:
        best_costs.append(getters.get_cost_player(data, idx))
    
    return best_id, best_costs

conv = {'indexes': generic}

data = getters.get_data()

formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

for i in range(len(formations)):
    print("Running formation " + str(formations[i]))
    
    all_csv = "data_cleaned/as/" + str(formations[i]) + ".csv"
    all_combs = pd.read_csv(all_csv, converters = conv)    
    all_points = all_combs['points_total'].values
    all_costs = all_combs['cost'].values
    
    best_costs, best_points = [],[]
    best_total_costs, best_total_points =[],[]
    
    for j in range(500, 900, 50):
        best_id, id_costs = calc_best_team_under_budget2(j)
    
        best_costs.append(sorted(id_costs))
        best_total_costs.append(all_costs[best_id])
        best_total_points.append(all_points[best_id])
     
    budget = range(500, 900, 50)    
    dataframe = pd.DataFrame()
    dataframe['Budget'] = budget
    dataframe['Best total cost'] = best_total_costs 
    dataframe['Best total points'] = best_total_points
    dataframe['Individual costs'] = best_costs
   
# In[]
def calc_best_per_season_pl(season): 
    conv = {'indexes': generic}
    csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
    
    playerspl = pd.read_csv(csv_file).to_dict('index')
    playerspldata = getters.get_players_feature_pl(playerspl)
    formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]
    
    for i in range(len(formations)):
        print("Running formation " + str(formations[i]))
        df = formations[i][0]
        mf = formations[i][1]
        fw = formations[i][2]
    
        df_csv = "data_cleaned/pl/" + str(season) + "/df/" + str(df) + ".csv"
        mf_csv = "data_cleaned/pl/" + str(season) + "/mf/" + str(mf) + ".csv"
        fw_csv = "data_cleaned/pl/" + str(season) + "/fw/" + str(fw) + ".csv"
    
        gk_combs = pd.read_csv("data_cleaned/pl/" + str(season) + "/gk.csv", converters = conv)
        df_combs = pd.read_csv(df_csv, converters = conv)
        mf_combs = pd.read_csv(mf_csv, converters = conv)
        fw_combs = pd.read_csv(fw_csv, converters = conv)
        gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])
        combs_all = [gk_combs, df_combs, mf_combs, fw_combs]
        

        gk_points = gk_combs['total_points'].values
        df_points = df_combs['total_points'].values
        mf_points = mf_combs['total_points'].values
        fw_points = fw_combs['total_points'].values
        
        
        all_points=[gk_points, df_points, mf_points, fw_points]
        points_full = parsers.parse_formations_points_or_cost(all_points)
        gk_costs = gk_combs['now_cost'].values
        df_costs = df_combs['now_cost'].values
        mf_costs = mf_combs['now_cost'].values
        fw_costs = fw_combs['now_cost'].values
        
        all_costs = [gk_costs, df_costs, mf_costs, fw_costs]
        costs_full = parsers.parse_formations_points_or_cost(all_costs)
        best_costs= []
        best_total_costs, best_total_points =[],[]
        best_ids=[]
        
        for j in range(500, 1050, 50):
            costs, points, ids = calc_best_team_under_budget(playerspldata, j, costs_full, points_full, combs_all)
            best_costs.append(sorted(costs))
            best_total_costs.append(sum(costs))
            best_total_points.append(sum(points))
            best_ids.append(ids)
         
        budget = range(500, 1050, 50)    
        dataframe = pd.DataFrame()
        dataframe['Budget'] = budget
        dataframe['Best total cost'] = best_total_costs 
        dataframe['Best total points'] = best_total_points
        dataframe['Individual costs'] = best_costs
        dataframe['Id'] = best_ids
        #print (dataframe)
        if i == 0:
            best_dataframe = dataframe
            df_form= []
            for l in range(len(best_dataframe)):
                df_form.append(formations[0])
            best_dataframe['Formation'] = df_form 
        else:
            for j in range(len(dataframe)):
                if dataframe.loc[j]['Best total points'] > best_dataframe.loc[j]['Best total points']:
                    best_dataframe.loc[j]=dataframe.loc[j]
                    df_form[j]=formations[i]
                    best_dataframe['Formation'] = df_form
                    
        #print(dataframe['Best total cost'])              
        # Uncomment to save as csv
        #formation= str(df) + '-' + str(mf) + '-' + str(fw)
        #csv_output ='results/pl/' + str(season) + '/' + formation + '.csv'
        #dataframe.to_csv(csv_output, index=False)
    #best_dataframe.to_csv('results/pl/' + str(season) + '/best.csv', index=False)
    return all_points, points_full
    
# In[]

 
#PL
# Change to get different season
seasons = [1617, 1718, 1819, 1920, 2021]
#season = seasons[3]

for season in seasons:
    print("Calculating season: " + str(season))
    calc_best_per_season_pl(season)
#season=seasons[0]
#all_points , points_full = calc_best_per_season_pl(season)