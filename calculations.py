# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:55:46 2022

@author: jonat
"""

import numpy as np
import getters

data2 = getters.get_data()
players = getters.get_players_feature(data2)

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

def calcindex(indexlist, dat, nr, length):
    returnlist=[]
    for i in range(length):
        temp = []
        for j in range(nr):    
            temp.append(list(dat)[indexlist[i][j]])
        returnlist.append(temp)
    return returnlist


def pointsPerTeam4(team, pointsList):
    teampoints = 0
    #teampoints = np.sum(pointsList[team])
    for key in team: 
        teampoints = teampoints + pointsList[key-1]
     
    return teampoints


def costPerTeam4(team, costList):
    teamcost = 0
    for key in team:
        teamcost = teamcost + costList[key-1]
    return teamcost  

def createCostList():
    costList = []
    for i in range(len(players)):
        costList.append(players[i+1]["now_cost"])
    #for player in players:
    #    costList.append(player["now_cost"])    
    return tuple(costList)

def createPointsList():
    pointsList=[]
    for i in range(len(players)):
        pointsList.append(players[i+1]["total_points"])
    #for player in players:
     #   pointsList.append(player["total_points"])
    return tuple(pointsList)

def createFormation(d = 4, m = 4, f = 2, n = 100):
    
    gk, df, mf, fw = getters.get_diff_pos(players)
    
    defe = np.transpose(nump2(len(df),d))
    midf = np.transpose(nump2(len(mf),m))
    forw = np.transpose(nump2(len(fw),f))    
        
    forwards = calcindex(forw, fw, f, n) 
    defenders = calcindex(defe, df, d, n )
    midfielders = calcindex(midf, mf, m, n)
    
    return defenders, midfielders, forwards

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