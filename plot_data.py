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


"""
Get the data
"""
#PL
seasons=[1617,1718,1819,1920,2021]
season=seasons[0]

csv_file = "data/pl_csv/players_raw_" + str(season) + ".csv"
playerspl = pd.read_csv(csv_file) 
playerspl = playerspl.to_dict('index')
playerspldata = getters.get_players_feature_pl(playerspl)
plgk, pldf, plmf, plfw = getters.get_diff_pos(playerspldata)

#As
data2 = getters.get_data()
players2 = getters.get_players_feature(data2)
asgk, asdf, asmf, asfw = getters.get_diff_pos(players2)

# In[]
"""
Create different plots 
"""

# plot histogram of how many of each position
x=["gk", "df", "mf", "fw" ]
aspositions= [len(asgk), len(asdf), len(asmf), len(asfw) ]
plpositions= [len(plgk), len(pldf), len(plmf), len(plfw) ]

plt.plot(x,aspositions, 'o')
plt.xlabel("Position")
plt.ylabel("Amount")
plt.show()

plt.plot(x,plpositions,'o')
plt.xlabel("Position")
plt.ylabel("Amount")
plt.show()

# plot hist of points

pointsList= calc.createPointsList()
plt.hist(pointsList)
plt.xlabel("Points")
plt.ylabel("Amount")
plt.show()

# plot hist of costs

costList= calc.createCostList()
plt.hist(costList)
plt.xlabel("Cost")
plt.ylabel("Amount")
plt.show()

# plot "score", score = points/cost


# plot results: 
    # plot best score per budget
    # plot cost of all players in best team per budget
    