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
    plt.hist(costList)
    plt.xlabel("Costs")
    plt.ylabel("Amount")
    xmin, xmax, ymin, ymax = [35, 140, 0, 400]
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()

# In[]
# NOT DONE
# plot results: 
    # plot best score per budget
    # plot cost of all players in best team per budget


x = range(1,12,1)
y = best_costs
plt.xlabel("Player")
plt.ylabel("Cost")
#print(y[0])
for i in range(len(y)):
    plt.plot(x,[pt for pt in y[i]], 'o', label = '< %s'%(500+i*50))
plt.legend()
plt.show()  
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
    
sum_best_costs = list(map(sum, best_costs))
sum_best_points = list(map(sum, best_points))


  

# plot "score", score = points/cost


    