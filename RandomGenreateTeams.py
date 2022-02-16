# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:22:56 2022

@author: jonade
"""

# generate random integer values
from random import *
import pandas as pd
import getters
import calculations
import cleaners
import ast
generic = lambda x: ast.literal_eval(x)


#Random generate teams from ditribution or randomly.

def generateRandomTeam(allpositions, budget, formation):
        
    gkcost, gkpoints, gkindCost = addPositions(allpositions[0], 1)
    dfcost, dfpoints, dfindCost = addPositions(allpositions[0], formation[0])
    mfcost, mfpoints, mfindCost = addPositions(allpositions[0], formation[1])
    fwcost, fwpoints, fwindCost = addPositions(allpositions[0], formation[2])
    
    teamCost = gkcost + dfcost + mfcost + fwcost 
    
    lowerBudget= 0
    if teamCost <= budget and teamCost >= lowerBudget:
        teamPoints = gkpoints + dfpoints + mfpoints + fwpoints
        teamDynamics =  gkindCost + dfindCost + mfindCost + fwindCost
        return teamCost, teamPoints, teamDynamics
    else:
        return None, None, None
    



def addPositions(df, n):
    indexes = sample(list(df.keys()), n)
    cost = 0
    points = 0
    indCost = [] 
    for idx in indexes:
        cost += df[idx]['now_cost']
        points += df[idx]['total_points'] 
        indCost.append(df[idx]['now_cost'])
    
    return cost, points, indCost    
        
def generateRandomTeamFromCleaned(budget,formation, combs_all):

    tPoints, tCost = 0, 0 
    tIndexes = []

    for comb in combs_all:
        row = comb.loc[choice(comb.index)]
        tPoints += + row['total_points']
        tCost += row['now_cost']
        tIndexes.extend(row['indexes'])        
    lower = 700
    if tCost <= budget and tCost >= lower:    
        return tCost, tPoints, tIndexes     
    else: 
        return None, None, None
    
def saveAllCombs(season, formation):
    
    conv = {'indexes': generic}
    df = formation[0]
    mf = formation[1]
    fw = formation[2]
    
    df_csv = "data_cleaned/pl/" + str(season) + "/df/" + str(df) + ".csv"
    mf_csv = "data_cleaned/pl/" + str(season) + "/mf/" + str(mf) + ".csv"
    fw_csv = "data_cleaned/pl/" + str(season) + "/fw/" + str(fw) + ".csv"
    
    gk_combs = pd.read_csv("data_cleaned/pl/" + str(season) + "/gk.csv", converters = conv)
    df_combs = pd.read_csv(df_csv, converters = conv)
    mf_combs = pd.read_csv(mf_csv, converters = conv)
    fw_combs = pd.read_csv(fw_csv, converters = conv)
    gk_combs['indexes'] = gk_combs['indexes'].apply(lambda x: [x])
    gk_combs.drop('Unnamed: 0', axis=1, inplace=True)
    all_combs = [gk_combs, df_combs, mf_combs, fw_combs]
    
    return all_combs
        
# In[]
season = 1617
csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
playerspl = pd.read_csv(csv_file) 
playerspl = playerspl.to_dict('index')
playerspldata = getters.get_players_feature_pl(playerspl)

allpositions = getters.get_diff_pos(playerspldata)

allCosts, allPoints, allDynamics =[], [], []

budget = 500
formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

while len(allCosts) < 100:
    formation = choice(formations)
    teamCost, teamPoints, teamDynamics = generateRandomTeam(allpositions, budget, formation)
    
    if teamCost is not None:
        allCosts.append(teamCost)
        allPoints.append(teamPoints)
        allDynamics.append(teamDynamics)

print(max(allCosts))




# In[]


budget = 1000
formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

allCosts, allPoints, allDynamics =[], [], []

season = 1617

allCombsPerFormation = []
for formation in formations:
    allCombsPerFormation.append(saveAllCombs(season, formation))

while len(allCosts) < 1000:
    idx = randint(0, len(formations)-1)
    formation = formations[idx]
    allCombs = allCombsPerFormation[idx] 
    teamCost, teamPoints, teamDynamics = generateRandomTeamFromCleaned(budget, formation, allCombs)
    
    if teamCost is not None:
        allCosts.append(teamCost)
        allPoints.append(teamPoints)
        allDynamics.append(teamDynamics)

# In[]
# for understanding
max_value = max(allPoints)
max_index = allPoints.index(max_value)
#print('Best index: ' +str(max_index))
print('Best total points: ' +str(max_value))
print('Total cost for best team: ' + str(allCosts[max_index]))
#print(allDynamics[max_index])
print("Individual costs: " +str(getters.get_cost_team_pl(playerspl, allDynamics[max_index])))
print("Players in best team: " + str(getters.get_full_name_team_pl(playerspl, allDynamics[max_index])))
print("Mean total points: " + str(sum(allPoints)/len(allPoints)))
print("Mean total cost: " + str(sum(allCosts)/len(allCosts)))





# In[]

import matplotlib.pyplot as plt 

plt.hist(allCosts)
plt.show()
plt.hist(allPoints)



# In[]

def cleanAllPositions(season):
    csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
    playerspl = pd.read_csv(csv_file) 
    playerspl = playerspl.to_dict('index')
    playerspldata = getters.get_players_feature_pl(playerspl)
    
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    all_parts_but_goalie = cleaners.all_forms_as_df_cleaned_pl(csv_file)[1:]
    
    individualCleansPerPosition =[]
    
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        print(pos)
        for p in part:
            print(p)
            all_cleaned = cleaners.run_all_cleans(df, p)  
            individualCleansPerPosition.append(all_cleaned)
            
    
    # Goalkeepers
    
    gk, _,_,_ = getters.get_diff_pos(playerspldata)
    
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    
    cleaned_gk = cleaners.clean_gk(sorted_df_gk)
    cleaned_gk.reset_index(inplace=True)
    cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    
    individualCleansPerPosition.append(cleaned_gk)
    
    print("Done with " + str(season))
    return individualCleansPerPosition

# In[]
import collections
#cleaned = cleanALlPositions(1819)

season=1819
csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
playerspl = pd.read_csv(csv_file) 
playerspl = playerspl.to_dict('index')
playerspldata = getters.get_players_feature_pl(playerspl)
gk, df, mf,fw = getters.get_diff_pos(playerspldata)

allmfCost=[]
for m in mf.items():
    allmfCost.append(m[1]['now_cost'])

occurrences = collections.Counter(allmfCost)
print(sorted(occurrences.items()))    
plt.hist(allmfCost)   

 
#gk,df,mf,fw = getters.get_diff_pos(playerspldata)

#for g in gk.items():
 #   print(g[1]['now_cost'])
    
#def splitDfByCost():
 #   return
 
 
 # In[]
 
sortIdxByCost = sorted(playerspldata, key=lambda k: (playerspldata[k]['now_cost']))

test_dictionary = { idx : playerspldata[idx] for idx in sortIdxByCost }
#print(test_dictionary[testsort[0]]) 

for idx in sortIdxByCost:
    print(playerspldata[idx])
    
print(playerspldata[sortIdxByCost[2]])

# In[]

# get the mean of all PL-seasons








