# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:34:35 2022

@author: jgull
"""
import pandas as pd
import numpy as np
import ast
generic = lambda x: ast.literal_eval(x)
conv = {'indexes': generic}
import getters as get
import os

class team:
    def __init__(self, team_ids, playersdata,  typ = "raw"):
            self.ids = team_ids
            self.ind_cost = get.get_cost_team(playersdata, self.ids)
            self.tot_cost = sum(self.ind_cost)

    def lin_fit(self):
        self.x = np.linspace(0,10,11).reshape(-1)
        mean_x = np.mean(self.x)
        mean_y = np.mean(self.ind_cost)

        sub_mean_x = np.subtract(self.x, mean_x)
        sub_mean_y = np.subtract(self.ind_cost,mean_y)

        prod = np.multiply(sub_mean_x,sub_mean_y)

        numer = np.sum(prod)
        denom = np.sum(np.power(sub_mean_x,2))

        m = np.divide(numer,denom)
        c = np.subtract(mean_y,np.multiply(m,mean_x))

        self.y_pr = np.add(c,np.multiply(m,self.x))
        ss_t = np.sum(np.power((self.ind_cost - mean_y),2))
        ss_r = np.sum(np.power((self.ind_cost - self.y_pr), 2))         

        self.r2 = np.subtract(1,np.divide(ss_r,ss_t)) 
    def check_lin(self, r2lim):
        if self.r2 >= r2lim: self.linear = True
        else: self.linear = False
        
    def create_int(self,  s = 2, z_count = 4, rang = 2): # s determines interval length

            self.min_cost = min(self.ind_cost)
            self.max_cost= max(self.ind_cost)
            self.theory_int = np.round(np.linspace(self.min_cost, self.max_cost, 11, dtype=float),1).tolist()
            self.gap = np.round((self.theory_int[1]-self.theory_int[0])/2)

    def check_int(self, z_count = 3):
            theory_int_l = [round(x - self.gap) for x in self.theory_int]
            theory_int_h = [round(x + self.gap) for x in self.theory_int]
            self.counts = [0]*11

            for re in self.ind_cost:
                for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
                    if(re >= low and re <= up):
                        self.counts[i] += 1
                        break

            if(self.counts > z_count): 
                self.diverse = True
            else:
                self.diverse = False

def create_nested_lists(lis):
    return([lis for _ in range(3)])

def mean_of_lists(lis):
    return([np.mean(x) for x in lis])

def use_linreg_pl_full_seasons(seasons, typ = "raw", r2_vals = [0.8,0.85,0.9], empty_all = [2,3,4]):
    formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]']
    
    for season in seasons:
        print(str(season))
        lin_cost, lin_points, lin_n, nonlin_cost, nonlin_points, nonlin_n = [],[],[],[],[],[]
        for formation in formations:
            print('Preparing data', str(formation))
            if typ == "raw":
                loc =  'data_cleaned/pl/'+str(season)+'/'
                one = pd.read_csv(loc+str(formation)+ '.csv', converters =conv)
            elif typ == "incnew": 
                loc = 'data_cleaned/pl/'+ typ + "/"
                one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)
            elif typ == "noexp":
                loc = 'data_cleaned/pl/'+ "noexp_01" + "/"
                one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)
    
            playerspldata = get.get_players_feature_pl("data/pl_csv/players_" + typ + "_", season)
            one.sort_values(by="points_total", inplace=True, ascending = False)
        
            print('Done')
            all_teams = one["indexes"].to_list()
            all_points = one['points_total'].to_list()
            all_costs = one['cost'].to_list()
           
            cost_lin, points_lin = create_nested_lists([]), create_nested_lists([])
            non_cost_lin,non_points_lin = create_nested_lists([]), create_nested_lists([])
            
            cost_div, points_div = create_nested_lists([]), create_nested_lists([])
            non_cost_div,non_points_div= create_nested_lists([]), create_nested_lists([])
            
            cost_both, points_both= create_nested_lists([]), create_nested_lists([])
            non_cost_both,non_points_both= create_nested_lists([]), create_nested_lists([])

    
            #linear_cost_085, linear_points_085, non_cost_085, non_points_085 = [],[],[],[]
            #linear_cost_09, linear_points_09, non_cost_09, non_points_09 = [],[],[],[]
            for t,p,c in zip(all_teams, all_points, all_costs):
                each_team = team(t,playerspldata)
                each_team.create_int()
                
                each_team.lin_fit()
                
                for i, (r2,z) in enumerate(zip(r2_vals, empty_all)):
                    each_team.check_lin(r2)
                    each_team.check_int(z)
                    if(each_team.linear):
                        cost_lin[i].append(c)
                        points_lin[i].append(p)
                    else:
                        non_cost_lin[i].append(c)
                        non_points_lin[i].append(p)
                    if(each_team.diverse):
                        cost_div[i].append(c)
                        points_div[i].append(p)
                    else:
                        non_cost_div[i].append(c)
                        non_points_div[i].append(p)
                    if(each_team.diverse and each_team.linear):
                        cost_both[i].append(c)
                        points_both[i].append(p)
                    else: 
                        non_cost_both[i].append(c)
                        non_points_both[i].append(p)

        lin_cost, lin_points = mean_of_lists(linear_cost), mean_of_lists(linear_points)
        lin_n = [len(x) for x in linear_cost]
            
        nonlin_cost.append(np.mean(non_cost))
        nonlin_points.append(np.mean(non_points))
        nonlin_n.append(len(non_cost))
        df = pd.DataFrame({'formation': formation, 'Linear mean cost': lin_cost, 'Linear mean points': lin_points, 'n linear': lin_n,
                      'Non linear cost': nonlin_cost, 'Non linear points': nonlin_points, 'Non linear n': nonlin_n
            })
        
        dest = os.path.join("results","pl",str(season), "linvsnonlin_" + typ + ".csv")
        df.to_csv(dest)
