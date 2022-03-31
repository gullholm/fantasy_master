# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:28:21 2022

@author: jgull
"""

from sklearn.linear_model import LinearRegression as lin
import getters as get
import numpy as np

class team:
    def __init__(self, team_ids, points,season, playersdata,  typ = "raw"):
        self.ids = team_ids
        self.season = season 
        self.typ = typ
        self.points = points
        self.ind_cost = get.get_cost_team(playersdata, self.ids)
        self.cost = sum(self.ind_cost)
        
    def det_lin_reg(self):
        X = np.linspace(0,11,11).reshape(-1,1)
        Y = np.array(self.ind_cost).reshape(-1,1)
        linmod = lin()
        linmod.fit(X,Y)
        self.r2 = linmod.score(X, y_pred)
        self.res = (linmod.predict(X) - Y).flatten()
        
    def get_is_linear(self,r2_lim, ret = True):
        if (self.r2 > r2_lim):
            self.linear = True
        else: 
            self.linear = False
        if(ret): return self.linear
        
class budget:
    def __init__(self, low_lim, high_lim, season, typ):
        self.low_lim = low_lim
        self.high_lim = high_lim
        self.season = season
        self.typ = typ

    def set_teams(self, team_list, point_list, playersdata):
        self.team_list = [team(x, y, self.season, playersdata) for (x,y) in zip(team_list,point_list)]

    def lin_reg(self, r2_lim = 0.75):
        [x.det_lin_reg() for x in self.team_list]

    def get_all_res(self):
        temp = [x.res for x in self.team_list if(x.linear)]
        self.res = np.array(temp).flatten()
        
    def get_is_linear(self, r2_lim):
        [x.get_is_linear(r2_lim) for x in self.team_list]
        if len(self.team_list)>0: self.prop_lin = sum([x.linear for x in self.team_list])/len(self.team_list)
        else: self.prop_lin = -1