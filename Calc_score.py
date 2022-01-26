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
def4, mid4, forw2 = calc.createFormation(4, 4, 2, 100)



# In[ ]:

# Calculate points and cost for all different teams (Time inefficient)    

# n = 50 takes ~130s, n = 30 ~40s

n = 30 # number of players used on each position
teamPointsList=[]
teamCostList=[]
teams=[]

costList = calc.createCostList()
pointsList = calc.createPointsList()


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
                #teamPointsList.append(calc.pointsPerTeam3(team2))
                #teamCostList.append(calc.costPerTeam(team2))

print("--- %s seconds ---" % (time.time() - start_time))
#runtime = (time.time() - start_time)
#timeList.append(runtime)

start_time = time.time()

for team in teams:
    teamPointsList.append(calc.pointsPerTeam4(team,pointsList))
    teamCostList.append(calc.costPerTeam4(team,costList))

print("--- %s seconds ---" % (time.time() - start_time))
    

calc.printSummary(teamPointsList, teamCostList)
index_max = np.argmax(teamPointsList)
print("Indexes for the best team: " + str(teams[index_max]))



# In[]
# Trying with list instead of dictionary

# n= 50 ~12s

#timeList=[] 
#for i in range(10):
n = 30 # number of players used on each position
teamPointsList3=[]
teamCostList3=[]
teams3=[]

costList = calc.createCostList()
pointsList = calc.createPointsList()

start_time = time.time()
for i in gk:
    totalPoints0 = pointsList[i-1]
    totalCost0 = costList[i-1]
    for j in range(n):
        totalPoints1 = totalPoints0 + calc.pointsPerTeam4(forw2[j], (pointsList))
        totalCost1 = totalCost0 + calc.costPerTeam4(forw2[j], costList)
        for k in range(n):
            totalPoints2 = totalPoints1 + calc.pointsPerTeam4(def4[k], (pointsList))
            totalCost2 = totalCost1 + calc.costPerTeam4(def4[k],costList)
            for l in range(n):
                totalPoints3 = totalPoints2 + calc.pointsPerTeam4(mid4[l], (pointsList))
                totalCost3 = totalCost2 + calc.costPerTeam4(mid4[l],costList)
                teamPointsList3.append(totalPoints3)
                teamCostList3.append(totalCost3)

print("--- %s seconds ---" % (time.time() - start_time))
  #  runtime = (time.time() - start_time)
   # timeList.append(runtime)
    
#print(np.mean(timeList)) #2.78
calc.printSummary(teamPointsList3, teamCostList3)


# In[]
# Test with np.sum of np.arrays

n = 30 # number of players used on each position
teamPointsList3=[]
teamCostList3=[]
teams3=[]

costList = calc.createCostList()
pointsList = calc.createPointsList()
pointsList= np.array(pointsList)
#forw2= np.array(forw2)
#mid4 = np.array(mid4)
#def4= np.array(def4)


start_time = time.time()
for i in gk:
    totalPoints0 = pointsList[i-1]
    totalCost0 = costList[i-1]
    for j in range(n):
        totalPoints1 = totalPoints0 + np.sum(pointsList[forw2[j]-1])
        totalCost1 = totalCost0 + calc.costPerTeam4(forw2[j], costList)
        for k in range(n):
            totalPoints2 = totalPoints1 + np.sum(pointsList[def4[k]-1])
            totalCost2 = totalCost1 + calc.costPerTeam4(def4[k],costList)
            for l in range(n):
                totalPoints3 = totalPoints2 + np.sum(pointsList[mid4[l]-1])
                totalCost3 = totalCost2 + calc.costPerTeam4(mid4[l],costList)
                teamPointsList3.append(totalPoints3)
                teamCostList3.append(totalCost3)

print("--- %s seconds ---" % (time.time() - start_time))

calc.printSummary(teamPointsList3, teamCostList3)


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

print(ys)

p = np.poly1d(np.polyfit(xs ,ys ,deg=2))
print(p)

plt.plot(timeList)
plt.plot(xs, p(xs))

print("Time for 50 is ca: " +str(round(p(50))) + " sec")

print("Time for 88201170 is ca: " +str(round(p(88201170))) + " sec")
# for the moment 15.6 million years...