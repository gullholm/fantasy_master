# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:06:40 2022

@author: jgull
"""

import helpers_calc_div as hcd

def plot_all_residuals(seasons= [1617,1819], league = "pl"):
    for season in seasons:
        if season == 1617 or season ==1819:
            for typ in ["incnew", "noexp"]:
                hcd.get_residuals(season, typ)
        else: hcd.get_residuals(season)
        
plot_all_residuals()
hcd.get_residuals(21, league = "as")


import numpy as np

hej = np.array([1,2,3])
np.savetxt("test.out", hej)


hej2 = np.loadtxt("test.out")
