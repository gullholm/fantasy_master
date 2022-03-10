# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:55:46 2022

@author: jonat
"""

import numpy as np
import getters
import parsers

#data2 = getters.get_data()
#players2 = getters.get_players_feature(data2)


def nump2(n, k):
    a = np.ones((k, n-k+1), dtype=int)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j]) 
    return a

def calcindex(indexlist, dat, nr, length, seed): # Returns indexes of (length) amount of diff random players combinations
    returnlist=[]
    np.random.seed(seed)
    rand_x = np.random.randint(indexlist.shape[0], size = length)
    
    for i in range(length):

        if(len(indexlist.shape) == 1):
            temp = list(dat)[indexlist[rand_x[i]]]
        else:
            temp = [list(dat)[indexlist[rand_x[i],j]] for j in range(nr)]
        returnlist.append(temp)
        
    return returnlist

def createFormation(gk, df, mf, fw, d = 4, m = 4, f = 2, n = 100, seed = 123): # standard 4-4-2
    
    defe = np.transpose(nump2(len(df), d))
    midf = np.transpose(nump2(len(mf), m))
    forw = np.transpose(nump2(len(fw), f))    
    glk = np.transpose(nump2(len(gk), 1))

    forwards = calcindex(forw, fw, f, n, seed) 
    defenders = calcindex(defe, df, d, n, seed )
    midfielders = calcindex(midf, mf, m, n, seed)
    goalkeepers = calcindex(glk, gk, 1, len(gk), seed)
    
    return goalkeepers, defenders, midfielders, forwards

def pointsPerTeam4(team, pointsList):
    return sum([pointsList[x-1] for x in team]) 

def costPerTeam4(team, costList):
    return sum([costList[x-1] for x in team])  

def createCostList(players, fill_out = True):
    #Pick a larger number than largest key, some spots are missing / 0
    n = len(players)
    if(fill_out):
        n = max(10000,len(players))

    costList = n*[-1]
    #print(players.keys())
    for (k,v) in players.items():
        #costList.append(players[i]["now_cost"])
      #  print(i)
        costList[k-1] = v.get("now_cost")
    #for player in players:
    #    costList.append(player["now_cost"])  
    costList = [i for i in costList if i != -1]

    return tuple(costList)

def createPointsList(players):
    #Pick a larger number than largest key, some spots are missing / 0
    n = max(10000,len(players))
    pointsList = [-1]*n
    for k,v in players.items():
        
        #costList.append(players[i]["total_points"])
        pointsList[k-1] = v.get("total_points")
    #for player in players:
    #    costList.append(player["now_cost"])   
    pointsList = [i for i in pointsList if i != -1]
    return tuple(pointsList)




def printSummary(teamPoints, teamCosts):
    
    index_max = np.argmax(teamPoints)
    meanCost = round(sum(teamCosts)/len(teamCosts)) 
    meanPoints = round(sum(teamPoints)/len(teamPoints))

    print("Nr of teams: " + str(len(teamPoints)))
    print("Best index: " + str(index_max))
    #print("Indexes for the best team: " + str(teams[index_max]))
    print("Best score: "+ str(teamPoints[index_max]))
    print("Total cost for the best team: " + str(teamCosts[index_max]))
    print("Mean cost: " + str(meanCost))
    print("Mean points: " + str(meanPoints))
    
    #return None
    
# calculate n max numbers of a list and append tehm to a list and print  
def Nmaxelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        max1 = 0
          
        for j in range(len(list1)):     
            if list1[j] > max1:
                max1 = list1[j];
                  
        list1.remove(max1);
        final_list.append(max1)
          
    print(final_list)  
    
#save this, the old way to calculate all indexes without randomization
# so that all combinations occur    
    
def calcIndexOld(indexlist, dat):
    returnlist=[]
    dat = list(dat)
    for i in range(indexlist.shape[0]):
        temp = []
        for j in range(indexlist.shape[1]):                
            temp.append(dat[indexlist[i][j]])
        returnlist.append(temp)
    return returnlist    

def calc_from_combs(all_combs, column):
    return [comb[column].values for comb in all_combs]

def calc_best_team(all_combs, cost_limit):

    all_combs[0]['indexes'] = all_combs[0]['indexes'].apply(lambda x: [x])

    all_points = calc_from_combs(all_combs, "total_points")
    all_costs = calc_from_combs(all_combs, "now_cost" )

    points_full = parsers.parse_formations_points_or_cost(all_points)

    costs_full = parsers.parse_formations_points_or_cost(all_costs)


    under_cost =  np.argwhere(costs_full < cost_limit)
    
    best = parsers.find_best_team(under_cost, points_full)
    sep_ids  = [combs['indexes'].values.tolist() for combs in all_combs]
    

    
    best_team_ids = [x[under_cost[best][i]] for (i,x) in enumerate(sep_ids)]
    
    return under_cost, best_team_ids
    