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
        
    def create_int(self, playersdata, team_ids, s = 2, z_count = 4, rang = 2): # s determines interval length
            team_cost = get.get_cost_team(playersdata, team_ids)

            team_cost.sort()
            self.min_cost = min(self.ind_cost)
            self.max_cost= max(self.ind_cost)
            self.theory_int = np.round(np.linspace(self.min_cost, self.max_cost, 11, dtype=float),1).tolist()
            self.gap = np.round((theory_int[1]-theory_int[0])/2,1)

    def check_int(self):
            theory_int_l = [round(x - self.gap,1) for x in self.theory_int]
            theory_int_h = [round(x + self.gap,2) for x in self.theory_int]
            counts = [0]*11

            for re in team_cost:
                
                for i,(low,up) in enumerate(zip(theory_int_l, theory_int_h)):
                    if(re >= low and re <= up):
                        counts[i] += 1
                        break
        #    print(counts)
            mas = max(counts)
            max_ind = [i for (i,x) in enumerate(counts) if x == mas]
            normals = []
            for i in max_ind:
        #        print(i)
                if i == 0: i+=rang
        #        if i == 1: i +=1
                if i == 10: i-=rang
        #        if i == 9: i-=1
                normals.append(sum([counts[j] for j in range(i-rang,i+rang+1)]))
        #    print(normals)
            if(sum(counts)> 11): print("hi")    
            normal = max(normals)
        #    print(counts)
        #    print(normal)
            if(normal > z_count): 
        #        print("normal")        
                return(0)
            else:
        #        print(counts)
        #        print("linear")
                return(1)

        
def use_linreg_pl_full_seasons(seasons, typ = "raw"):
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
           
            linear_cost = [[] for _ in range(3)]
            linear_points = [[] for _ in range(3)]
            non_cost = [[] for _ in range(3)]
            non_points = [[] for _ in range(3)]
            
            #linear_cost_085, linear_points_085, non_cost_085, non_points_085 = [],[],[],[]
            #linear_cost_09, linear_points_09, non_cost_09, non_points_09 = [],[],[],[]
            for t,p,c in zip(all_teams, all_points, all_costs):
                y = get.get_cost_team(playerspldata, t)
                linmod = team(t,playerspldata)
                linmod.lin_fit()
                for i,r2 in enumerate([0.8,0.85,0.9]):
                    linmod.check_lin(r2)
                    if(linmod.linear):
                        linear_cost[i].append(c)
                        linear_points[i].append(p)
                    else:
                        non_cost[i].append(c)
                        non_points[i].append(p)
            break
        lin_cost.append(np.mean(linear_cost))
        lin_points.append(np.mean(linear_points))
        lin_n.append(len(linear_cost))
            
        nonlin_cost.append(np.mean(non_cost))
        nonlin_points.append(np.mean(non_points))
        nonlin_n.append(len(non_cost))
        df = pd.DataFrame({'formation': formation, 'Linear mean cost': lin_cost, 'Linear mean points': lin_points, 'n linear': lin_n,
                      'Non linear cost': nonlin_cost, 'Non linear points': nonlin_points, 'Non linear n': nonlin_n
            })
        
        dest = os.path.join("results","pl",str(season), "linvsnonlin_" + typ + ".csv")
        df.to_csv(dest)
