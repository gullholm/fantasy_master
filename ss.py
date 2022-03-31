# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:28:21 2022

@author: jgull
"""

from sklearn.linear_model import LinearRegression as lin
import getters as get

class team:
    def __init__(self, team_ids, season, playersdata, typ = "raw"):
        self.ids = team_ids
        self.season = season 
        self.typ = typ
        self.cost = get.get_cost_team(playersdata, self.ids)
    def det_lin_reg(self):
        X = np.array(range(10)).reshape(-1,1)
        Y = np.array(self.cost).reshape(-1,1)
        linmod = lin()
        linmod.fit(X,Y)
        r2 = linmod.score(X,Y)
        res = (linmod.predict(X) - Y).flatten()
        return r2, res