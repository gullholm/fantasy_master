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
import matplotlib.pyplot as plt
generic = lambda x: ast.literal_eval(x)


#Random generate teams from ditribution or randomly.

def generateRandomTeam(allpositions, budget, formation):
        
    gkcost, gkpoints, gkindCost = addPositions(allpositions[0], 1)
    dfcost, dfpoints, dfindCost = addPositions(allpositions[0], formation[0])
    mfcost, mfpoints, mfindCost = addPositions(allpositions[0], formation[1])
    fwcost, fwpoints, fwindCost = addPositions(allpositions[0], formation[2])
    
    teamCost = gkcost + dfcost + mfcost + fwcost 
    
    lowerBudget= 500
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

budget = 550
formations = [[3,5,2],[3,4,3],[4,4,2],[4,3,3],[4,5,1],[5,3,2],[5,4,1]]

while len(allCosts) < 10000:
    formation = choice(formations)
    teamCost, teamPoints, teamDynamics = generateRandomTeam(allpositions, budget, formation)
    
    if teamCost is not None:
        allCosts.append(teamCost)
        allPoints.append(teamPoints)
        allDynamics.append(teamDynamics)

print(max(allCosts))

#%%
plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "random")

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
def printAndPlotSummary(allCosts, allPoints, allDynamics, budget): 
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
plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "Random")

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
  
 # In[]
 
sortIdxByCost = sorted(playerspldata, key=lambda k: (playerspldata[k]['now_cost']))

test_dictionary = { i : playerspldata[idx] for idx, i in zip(sortIdxByCost, range(len(sortIdxByCost))) }
#print(test_dictionary[testsort[0]]) 

for idx in sortIdxByCost:
    print(playerspldata[idx])
    
print(playerspldata[sortIdxByCost[2]])

# In[]
value = -1
theList = [None]*128 #127 highest value for cost
templist=[]
for key in test_dictionary.values(): 
    if key['now_cost'] <= value:
        templist.append(key['total_points'])
        #print(key)
    else: 
        theList[value] = templist  
        templist = []
        value = key['now_cost']
        templist.append(key['total_points'])
        if key['now_cost'] == 127:
            theList[value] = templist

print(choice(theList[38]))

#%%

# minimum salary, in our case 38 # maybe cheapest in best team
# total budget, in our case 700 # can change
# C = 700/11 ~63.64, a = 37
#spanning the range from a to 2C-a

#theory_m is first not None value
def calculateTheoryDistribution(theList, budget):
    n = 11
    theory_m = next(idx for idx,item in zip(range(len(theList)),theList) if item is not None)
    #TEST
    #theory_m += 30
    
    C = budget/n
    theory_expensive = round(2*C-theory_m)  
    theory_k = (theory_expensive-theory_m)/(n-1)
    for x in range(11):
        low = round(theory_k*x+theory_m-theory_k/2)
        high = round(theory_k*x+theory_m+theory_k/2)
        print(low)
        print(high)
    return theory_k, theory_m

def generateTeamFromDistribution(theList, budget, theory_k, theory_m):
    
    n = 11
    totalPoints, totalCost = 0, 0
    teamDistr = []
    for x in range(n):
        low = round(theory_k*x+theory_m-theory_k/2)
        high = round(theory_k*x+theory_m+theory_k/2)
        
        if x == 0: 
            low = theory_m
        cost = randint(low, high)
        templist = theList[cost]
        while (templist is None): 
            cost = randint(low, high)    
            templist = theList[cost]
        
        teamDistr.append(cost)    
        totalPoints += choice(templist) # add points
        totalCost += cost
    
    return totalPoints, totalCost, teamDistr

#%%
budget = 800
allCosts, allPoints, allDynamics =[], [], []

theory_k, theory_m = calculateTheoryDistribution(theList, budget)

while (len(allCosts)<10000):
    points, costs, dynamics = generateTeamFromDistribution(theList, budget, theory_k, theory_m)
    
    if costs < budget:
        allCosts.append(costs)
        allPoints.append(points)
        allDynamics.append(dynamics)

# Plot results distribution
plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "distribution")       


#%%

def generateRandomTeamFromAllPlayers(theList, budget): 
    n=11
    
    res = [i for i in range(len(theList)) if theList[i] is not None]
    totalPoints, totalCost = 0, 0
    teamDistr = []
    
    for i in range(n):
        cost = choice(res)    
        templist = theList[cost]
        while not templist: 
            cost = choice(res)    
            templist = theList[cost]
        teamDistr.append(cost)
        totalPoints += choice(templist) # add points
        totalCost += cost
    
    return totalPoints, totalCost, teamDistr

#%%
budget = 800
lowerbudget = 770
allCosts, allPoints, allDynamics =[], [], []

while (len(allCosts)<10000):
    points, costs, dynamics = generateRandomTeamFromAllPlayers(theList, budget)
    if costs < budget and costs > lowerbudget:
        allCosts.append(costs)
        allPoints.append(points)
        allDynamics.append(dynamics)

plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, "random")        
#%%
# Plot results

def plotHistOfAllCostsAndPoints(allCosts, allPoints, budget, title):        
    plt.hist(allCosts)
    plt.title("All costs from" + title + " under budget: " + str(budget))
    plt.xlabel("Costs")
    plt.ylabel("Amount")
    plt.show()
    
    plt.hist(allPoints)    
    plt.title("All points from " + title + " under budget: " + str(budget)) 
    plt.xlabel("Points")
    plt.ylabel("Amount")       
    plt.show()
    
#%%   
 
cleaned = cleanAllPositions(1819)
#%%
#Cleaned players for 5 def, 5 mid, 3 forw, 1 keeper

values=[2,5,8,9]
cleanedPlayers = []
for i in range(len(cleaned)):
    if i in values:
        cleanedPlayers.append(cleaned[i])

#%%
#gk,df,mf,fw = getters.get_diff_pos(playerspldata)

#for g in gk.items():
 #   print(g[1]['now_cost'])
    
#def splitDfByCost():
 #   return-    
#%%

printAndPlotSummary(allCosts, allPoints, allDynamics, budget)

#%%

# Create a combinations of all seasons in PL 



def combineAllSeasonsPl():
    
    seasons=[1617, 1718,1819,1920,2021]
    for season in seasons: 
        csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
        playerspl = pd.read_csv(csv_file) 
        playerspl = playerspl.to_dict('index')
        playerspldata = getters.get_players_feature_pl(playerspl)

        sortIdxByCost = sorted(playerspldata, key=lambda k: (playerspldata[k]['now_cost']))
        
        test_dictionary = { i : playerspldata[idx] for idx, i in zip(sortIdxByCost, range(len(sortIdxByCost))) }
        value = -1
        if season == 1617:
            theList = [None]*133 #132 highest value for cost
            templist=[]
            for key in test_dictionary.values(): 
                if key['now_cost'] <= value:
                    templist.append(key['total_points'])
                #print(key)
                else: 
                    theList[value]= templist  
                    templist = []
                    value = key['now_cost']
                    templist.append(key['total_points'])
                    if key['now_cost'] == 127:
                        theList[value] = templist
        
        else:
            value= -1
            for key in test_dictionary.values(): 
                if key['now_cost'] <= value:
                    templist.append(key['total_points'])
                #print(key)
                else: 
                    if theList[value] is None:
                        theList[value]= templist 
                    else:    
                        beforeList = theList[value]
                        newList = beforeList + templist
                        theList[value]= newList
                    templist = []
                    value = key['now_cost']
                    templist.append(key['total_points'])
                    if key['now_cost'] == 132:
                        theList[value] = templist
            

    return theList
        
    
#%%
#De bästa kommer inte med för tre mittensäsonger, annars klart
PLCombined = combineAllSeasonsPl()  

     