# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:50:35 2022

@author: jonat
"""

#Plot data for understanding and discussion

import matplotlib.pyplot as plt 
import pandas as pd
import getters
import calculations as calc
import numpy as np
import ast
"""
Get the data
"""
#PL
seasons=[0, 1617,1718,1819,1920,2021]

for season in seasons:
#As
    if season == 0:
        title = "Allsvenskan" 
        
        data = getters.get_data()
        playersdata = getters.get_players_feature(data)
        gk, df, mf, fw = getters.get_diff_pos(playersdata)
        
        csv_results = "results/as/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 
    
    else:
        title = "Premier league season " + str(season)
        
        csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
        playerspl = pd.read_csv(csv_file).to_dict('index') 
        playersdata = getters.get_players_feature_pl(playerspl)
        gk, df, mf, fw = getters.get_diff_pos(playersdata)  
        
        csv_results = "results/pl/" + str(season) + "/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 

        
    
    positions= [len(gk), len(df), len(mf), len(fw)]
        
    plot_per_position(positions, title)
    plot_hist_of_costs(playersdata, title)
    plot_hist_of_points(playersdata, title)
    plotIndividualCosts(results, title, degree=1, line=False)


# In[]
"""
Create different plots 
"""

# plot histogram of how many of each position
def plot_per_position(positions, title):
    
    x=["gk", "df", "mf", "fw" ]
    plt.plot(x, positions, 'o')
    plt.xlabel("Position")
    plt.ylabel("Amount")
    plt.title(title)
    ymin, ymax = [0, 310]
    plt.ylim(ymin,ymax)    
    plt.show()


# plot hist of points
def plot_hist_of_points(feature_data, title):
    pointsList= calc.createPointsList(feature_data)
    plt.hist(pointsList)
    plt.xlabel("Points")
    plt.ylabel("Amount")
    xmin, xmax, ymin, ymax = [0, 320, 0, 350]
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()

# plot hist of costs
def plot_hist_of_costs(feature_data, title):
    costList= calc.createCostList(feature_data)
    #npscostlist = [np.log(x) for x in costList if x > 0]
    plt.hist(costList)
    #plt.hist(npscostlist)
    plt.xlabel("Costs")
    plt.ylabel("Amount")
    xmin, xmax, ymin, ymax = [35, 140, 0, 400]
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()
    
def plotIndividualCosts(feature_data, title, degree, line=False):

    x = list(range(1,12,1))
    plt.xlabel("Player")
    plt.ylabel("Cost")
    for i in range(len(feature_data)):
        y = ast.literal_eval(feature_data[i]['Individual costs'])
        plt.plot(x,y, 'o', label = '< %s'%(500+i*50))
        if line:
            poly= np.polyfit(x,y,degree)
            plt.plot(x, np.polyval(poly,x))
    plt.legend()
    plt.title(title)
    plt.show()              

# In[]
# NOT DONE
# plot results: 
    # plot best score per budget
    # plot cost of all players in best team per budget

best_costs = results[7]['Individual costs']
ind_costs = [50,53,55,55,56,57,66,96,113,119,129]
x = list(range(1,12,1))
y = ind_costs
plt.xlabel("Player")
plt.ylabel("Cost")
#print(y[0])
#for i in range(len(y)):
#    plt.plot(x,[pt for pt in y[i]], 'o', label = '< %s'%(500+i*50))
plt.legend()
plt.show()  
# In[9]
#Theory , not done
# minimum salary, in our case 37?
# total budget, in our case 700
# C = 700/11 ~63.64, a = 37

#spanning the range from a to 2C-a

# maybe total budget should be the total cost of the best team 
# maybe minimum salary, "a", should be the cheapest in the best team

for idx in range(len(y)):

    poly= np.polyfit(x,y[idx],1)
    plt.plot(x, np.polyval(poly,x))
    
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
    
sum_best_costs = list(map(sum, best_costs))
sum_best_points = list(map(sum, best_points))


  

# plot "score", score = points/cost


 # In[] 
import RandomGenreateTeams
eason=1819
cleaned = RandomGenreateTeams.cleanAllPositions(season)  
# In[]
titlenames = ["df3", 'df4','df5', 'mf3', 'mf4', 'mf5', 'fw1', 'fw2', 'fw3', 'gk']
for j in range(len(cleaned)):
    totcost=[]
    for i in range(len(cleaned[j])):
        totcost.append(cleaned[j].iloc[i]['now_cost'])
    plt.hist(totcost)
    plt.xlabel("Cost")
    plt.ylabel
    plt.title("Individuella kostnader efter cleaned, "
              "innan cleaned som formation: " + titlenames[j] + 
              " season: " + str(season))
    plt.show()
    
    
    
# In[]
    

def getResultsPerSeason(season):
    if season == 0: 
        csv_results = "results/as/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 
    else: 
        csv_results = "results/pl/" + str(season) + "/best.csv"
        results = pd.read_csv(csv_results).to_dict('index') 
        
    return results 

    
# In[]        

# Mean of all PL data

seasons=[0, 1617, 1718, 1819, 1920, 2021]

allResults = []
for season in seasons: 
    allResults.append(getResultsPerSeason(season))        

meanIndCost = []
for j in range(len(allResults[1])): 
    for i in range(1,6):
        if i == 1:
            meanIndCost.append(ast.literal_eval(allResults[i][j]['Individual costs']))
        else: 
            lista = meanIndCost[j]
            listb = ast.literal_eval(allResults[i][j]['Individual costs'])
            meanIndCost[j] = [lista[i] + listb[i] for i in range(11)]
            
for k in range(11):    
    meanIndCost[k] = [x/5 for x in meanIndCost[k]]   
    
degree=3
#x= range(11)    
#plt.plot(x, meanIndCost, 'o')
#poly= np.polyfit(x,meanIndCost,degree)
#plt.plot(x, np.polyval(poly,x))
x = list(range(1,12,1))
plt.xlabel("Player")
plt.ylabel("Cost")
line=True
for i in range(len(meanIndCost)):
    y = (meanIndCost[i])
    plt.plot(x,y, 'o', label = '< %s'%(500+i*50))
    if line:
        poly= np.polyfit(x,y,degree)
        plt.plot(x, np.polyval(poly,x))
plt.legend()
plt.title("test")
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    