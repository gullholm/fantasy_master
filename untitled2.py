# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:06:52 2022

@author: jgull
"""

import parsers
import helpers_calc_div

seasons = [1718,1819,1920,2021]
helpers_calc_div.get_residuals([1617]) 
helpers_calc_div.get_residuals([1617,1819], typ = "incnew")
helpers_calc_div.get_residuals([1617,1819], typ = "noexp")