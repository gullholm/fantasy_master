# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:12:50 2022

@author: jonat
"""
# In[ ]:

import getters
import numpy as np
import time
import matplotlib.pyplot as plt
import calculations as calc

# Import the data 

data2 = getters.get_data()
players = getters.get_players_feature(data2)
gk, df, mf, fw = getters.get_diff_pos(players)

# In[]

# Functions
def calcindex(indexlist, dat, nr, length):
    returnlist=[]
    for i in range(length):
        temp = []
        for j in range(nr):    
            temp.append(list(dat)[indexlist[i][j]])
        returnlist.append(temp)
    return returnlist

def pointsPerTeam3(team):
    teampoints = 0
    for key in team: 
        teampoints = teampoints + players[key]["total_points"]
     
    return teampoints

def pointsPerTeam4(team):
    teampoints = 0
    for key in team: 
        teampoints = teampoints + pointsList[key-1]
     
    return teampoints


def costPerTeam(team):
    teamcost = 0
    for key in team:
        teamcost = teamcost + players[key]["now_cost"]
    return teamcost  

def costPerTeam4(team):
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
    return costList

def createPointsList():
    pointsList=[]
    for i in range(len(players)):
        pointsList.append(players[i+1]["total_points"])
    #for player in players:
     #   pointsList.append(player["total_points"])
    return pointsList

def createFormation(d = 4, m = 4, f = 2, n = 100):
    
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

# In[]
def4, mid4, forw2 = calc.createFormation(4, 4, 2, 100)


# In[ ]:

# Calculate points and cost for all different teams (Time inefficient)    

# n = 50 takes ~130s, n = 30 ~40s

n = 10 # number of players used on each position
teamPointsList=[]
teamCostList=[]
teams=[]

# Uncomment to test time 
#timeList=[]
#for n in range(30):

start_time = time.time()
for i in gk:
    for j in range(n):
        team0 = np.append(i, forw2[j])
        for k in range(n):
            team1 = np.append(team0, def4[k]) 
            #team1 = team0.append(def4[k])
            for l in range(n):
                team2 = np.append(team1, mid4[l])
                teams.append(team2)
                teamPointsList.append(pointsPerTeam3(team2))
                teamCostList.append(costPerTeam(team2))

print("--- %s seconds ---" % (time.time() - start_time))
#runtime = (time.time() - start_time)
#timeList.append(runtime)

printSummary(teamPointsList, teamCostList)
index_max = np.argmax(teamPointsList)
print("Indexes for the best team: " + str(teams[index_max]))



# In[]


# Test for faster runtime, doesn't save each team and the indexes 
# for the team but get the rest of the results and one can 
# probably calculate the indexes...

# n=100 takes ~140s

n = 10 # number of players used on each position
teamPointsList2=[]
teamCostList2=[]
teams2=[]

start_time = time.time()
for i in gk:
    totalPoints0 = players[i]["total_points"]
    totalCost0 = players[i]["now_cost"]
    for j in range(n):
        totalPoints1 = totalPoints0 + pointsPerTeam3(forw2[j])
        totalCost1 = totalCost0 + costPerTeam(forw2[j])
        for k in range(n):
            totalPoints2 = totalPoints1 + pointsPerTeam3(def4[k])
            totalCost2 = totalCost1 + costPerTeam(def4[k])
            for l in range(n):
                totalPoints3 = totalPoints2 + pointsPerTeam3(mid4[l])
                totalCost3 = totalCost2 + costPerTeam(mid4[l])
                teamPointsList2.append(totalPoints3)
                teamCostList2.append(totalCost3)

print("--- %s seconds ---" % (time.time() - start_time))


printSummary(teamPointsList2, teamCostList2)
# In[]
# Trying with list instead of dictionary

# n= 50 ~12s

n = 10 # number of players used on each position
teamPointsList3=[]
teamCostList3=[]
teams3=[]

costList = createCostList()
pointsList = createPointsList()

start_time = time.time()
for i in gk:
    totalPoints0 = pointsList[i-1]
    totalCost0 = costList[i-1]
    for j in range(n):
        totalPoints1 = totalPoints0 + pointsPerTeam4(forw2[j])
        totalCost1 = totalCost0 + costPerTeam4(forw2[j])
        for k in range(n):
            totalPoints2 = totalPoints1 + pointsPerTeam4(def4[k])
            totalCost2 = totalCost1 + costPerTeam4(def4[k])
            for l in range(n):
                totalPoints3 = totalPoints2 + pointsPerTeam4(mid4[l])
                totalCost3 = totalCost2 + costPerTeam4(mid4[l])
                teamPointsList3.append(totalPoints3)
                teamCostList3.append(totalCost3)

print("--- %s seconds ---" % (time.time() - start_time))

printSummary(teamPointsList3, teamCostList3)


# In[]

print(players[12]["now_cost"])
print(costList[11])


# In[]

# Plot some results

plot1 = plt.figure(1)
plt.hist(teamCostList)

plot2 = plt.figure(2)
plt.hist(teamPointsList)

plt.show()

# In[]


# Can be used for testing how much time it will take to run 

xs = list(range(30))
ys = timeList 

print(xs)
print(ys)

p = np.poly1d(np.polyfit(xs ,ys ,deg=2))
print(p)

plt.plot(timeList)
plt.plot(xs, p(xs))

print("Time for 50 is ca: " +str(round(p(50))) + " sec")

print("Time for 88201170 is ca: " +str(round(p(88201170))) + " sec")
# for the moment 15.6 million years...