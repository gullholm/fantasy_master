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

# Import the data 

data2 = getters.get_data()

players = getters.get_players_feature(data2)

gk, df, mf, fw = getters.get_diff_pos(players)

# In[]

# Functions

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

def pointsPerTeam3(team):
    teampoints = 0
    for key in team: 
        teampoints = teampoints + players[key]["total_points"]
     
    return teampoints

def costPerTeam(team):
    teamcost = 0
    for key in team:
        teamcost = teamcost + players[key]["now_cost"]
    return teamcost  

# In[]
# Create combinations of positions for teams (Time inefficient)

forwards2 = np.transpose(nump2(len(fw),2))
defenders4 = np.transpose(nump2(len(df),4))
midfielders4 = np.transpose(nump2(len(mf),4))

n = 100    
forw2 = calcindex(forwards2, fw, 2, n) 
def4 = calcindex(defenders4, df, 4, n )
mid4 = calcindex(midfielders4, mf, 4, n)


# In[ ]:

# Calculate points and cost for all different teams (Time inefficient)    

# n=50 takes ~130s

# Initiate variables


n = 50 # number of players used on each position
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
            for l in range(n):
                team2 = np.append(team1, mid4[l])
                teams.append(team2)
                teamPointsList.append(pointsPerTeam3(team2))
                teamCostList.append(costPerTeam(team2))

print("--- %s seconds ---" % (time.time() - start_time))
#runtime = (time.time() - start_time)
#timeList.append(runtime)

# In[]


# Test for faster runtime, doesn't save each team and the indexes 
# for the team but get the rest of the results and one can 
# probably calculate the indexes...

# n=100 takes ~140s

n = 100 # number of players used on each position
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

index_max2 = np.argmax(teamPointsList2)
meanCost2 = round(sum(teamCostList2)/len(teamCostList2)) 
meanPoints2 = round(sum(teamPointsList2)/len(teamPointsList2))

print("Nr of teams: " + str(len(teamPointsList2)))
print("Best index: " + str(index_max2))
#print("Indexes for the best team: " + str(teams[index_max]))
print("Best score: "+ str(teamPointsList2[index_max2]))
print("Total cost for the best team: " + str(teamCostList2[index_max2]))
print("Mean cost: " + str(meanCost2))
print("Mean points: " + str(meanPoints2))


# In[]

#Printing out results from points and cost of all teams
 
index_max = np.argmax(teamPointsList)
meanCost = round(sum(teamCostList)/len(teamCostList)) 
meanPoints = round(sum(teamPointsList)/len(teamPointsList))

print("Nr of teams: " + str(len(teams)))
print("Best index: " + str(index_max))
print("Indexes for the best team: " + str(teams[index_max]))
print("Best score: "+ str(teamPointsList[index_max]))
print("Total cost for the best team: " + str(teamCostList[index_max]))
print("Mean cost: " + str(meanCost))
print("Mean points: " + str(meanPoints))


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