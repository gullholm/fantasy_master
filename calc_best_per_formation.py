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
import matplotlib.pyplot as plt 
import calculations as calc 
import getters


def calc_best_team_under_budget(max_cost):
    
    under_cost =  np.argwhere(costs_full <= max_cost) 
    print(under_cost.shape)
    best = parsers.find_best_team(under_cost, points_full)
    sep_ids = [fw_combs['indexes'].values.tolist(),mf_combs['indexes'].values.tolist() 
               , df_combs['indexes'].values.tolist(), gk_combs['indexes'].values.tolist()]
    
    best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]
    best_team_i = [item for sublist in best_team_ids for item in sublist]    
    best_team_ids_values = [players[ids] for ids in best_team_i]
    #best_team_names = getters.get_full_name_team(data, best_team_i)

    best_sum_cost = [player['now_cost'] for player in best_team_ids_values] 
    best_sum_points = [player['total_points'] for player in best_team_ids_values]
    
    return best_sum_cost, best_sum_points

conv = {'indexes': generic}

data = getters.get_data()
players = getters.get_players_feature(data)

formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

for i in range(len(formations)):
    print("Running formation " + str(formations[i]))
    df = formations[i][0]
    mf =formations[i][1]
    fw = formations[i][2]
    

    df_csv = "data_cleaned/as/df/" + str(df) + ".csv"
    mf_csv = "data_cleaned/as/mf/" + str(mf) + ".csv"
    fw_csv = "data_cleaned/as/fw/" + str(fw) + ".csv"
    
    #if df == 4:
    #    df_csv= "data_cleaned/df.csv"

    gk_combs = pd.read_csv("data_cleaned/gk.csv", converters = conv)
    df_combs = pd.read_csv(df_csv, converters = conv)
    mf_combs = pd.read_csv(mf_csv, converters = conv)
    fw_combs = pd.read_csv(fw_csv, converters = conv)
    gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])
    
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

    best_costs, best_points = [],[]
    best_total_costs, best_total_points =[],[]
    
    for j in range(500, 900, 50):
        costs, points = calc_best_team_under_budget(j)
        best_costs.append(sorted(costs))
        best_points.append(sorted(points))
        best_total_costs.append(sum(costs))
        best_total_points.append(sum(points))
     
    budget = range(500, 900, 50)    
    dataframe = pd.DataFrame()
    dataframe['Budget'] = budget
    dataframe['Best total cost'] = best_total_costs 
    dataframe['Best total points'] = best_total_points
    dataframe['Individual costs'] = best_costs
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

    # Uncomment to save as csv
    
    #formation= str(df) + '-' + str(mf) + '-' + str(fw)
    #csv_output ='results/as/' + formation + '.csv'
    #dataframe.to_csv(csv_output, index=False)
         
# In[]

best_dataframe.to_csv('results/as/best.csv', index=False)

# In[]
sum_best_costs = list(map(sum, best_costs))
sum_best_points = list(map(sum, best_points))


x = range(1,12,1)
y = best_costs
plt.xlabel("Player")
plt.ylabel("Cost")
#print(y[0])
for i in range(len(y)):
    plt.plot(x,[pt for pt in y[i]], 'o', label = '< %s'%(500+i*50))
plt.legend()
plt.show()
    

# Plots
#plt.plot(sum_best_costs,'o')


# In[]

#Theory 
# minimum salary, in our case 37?
# total budget, in our case 700
# C = 700/11 ~63.64, a = 37

#spanning the range from a to 2C-a

# maybe total budget should be the total cost of the best team 
# maybe minimum salary, "a", should be the cheapest in the best team

for idx in range(len(y)):

    k,m= np.polyfit(x,y[idx],1)
    plt.plot(x, k*x+m)
    
    theory_m = y[idx][0]
    theory_expensive = round(sum(y[idx])*2/11-theory_m)  
    theory_k = np.float64(theory_expensive-theory_m)/11
    plt.plot(x, theory_k*x+theory_m)
    
    total_diff = 0
    for i in range(1,12):
        first = k*i+m
        second =  theory_k*i + theory_m
        diff = np.abs(first-second)
        total_diff += diff
    
    mean_diff =  total_diff/11
    print(mean_diff)
    
 # In[]
costList= calc.createCostList()
plt.hist(costList)
plt.xlabel("Cost")
plt.ylabel("Amount")
plt.show()

   
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

print(all_combs.loc[1]['cost'])    