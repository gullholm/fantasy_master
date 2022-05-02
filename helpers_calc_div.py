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
import cleaners

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
        self.res = np.subtract(self.ind_cost, self.y_pr)
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
            self.zero_count = self.counts.count(0)
            if(self.zero_count <= z_count): 
                self.diverse = True
            else:
                self.diverse = False

def create_nested_lists(lis):
    return([lis for _ in range(3)])

def mean_of_lists(lis, r2= [0.8,0.85,0.9]):
    new  = {}
    for x,r in zip(lis, r2):
        #print(np.mean(x),r)
        new[r]  = np.round(np.mean(x))
        
    return(new)

def use_linreg_pl_full_seasons(seasons, typ = "raw", r2_vals = [0.8,0.85,0.9], empty_all = [4,3,2]):
    formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]']
    

    for season in seasons:
        print(str(season))
        cost_lin_t,points_lin_t, non_cost_lin_t,non_points_lin_t = [],[], [],[]
        cost_div_t,points_div_t, non_cost_div_t,non_points_div_t = [],[], [],[]
        cost_both_t,points_both_t, non_cost_both_t,non_points_both_t = [],[], [],[]
        lin_n_t, non_lin_n_t, div_n_t, non_div_n_t, both_n_t, non_both_n_t = [],[], [],[], [],[]
        for formation in formations:
            cost_lin,points_lin, non_cost_lin,non_points_lin= [],[], [],[]
            cost_div,points_div, non_cost_div,non_points_div = [],[], [],[]
            cost_both,points_both, non_cost_both,non_points_both= [],[], [],[]
            
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
           
            cost_lin, points_lin = [[] for _ in range(3)], [[] for _ in range(3)]
            non_cost_lin,non_points_lin = [[] for _ in range(3)], [[] for _ in range(3)]
            
            cost_div, points_div = [[] for _ in range(3)], [[] for _ in range(3)]
            non_cost_div,non_points_div= [[] for _ in range(3)], [[] for _ in range(3)]
            
            cost_both, points_both= [[] for _ in range(3)], [[] for _ in range(3)]
            non_cost_both,non_points_both= [[] for _ in range(3)], [[] for _ in range(3)]

    
            #linear_cost_085, linear_points_085, non_cost_085, non_points_085 = [],[],[],[]
            #linear_cost_09, linear_points_09, non_cost_09, non_points_09 = [],[],[],[]
            for t,p,c in zip(all_teams, all_points, all_costs):
                each_team = team(t,playerspldata)
                each_team.create_int()
                
                each_team.lin_fit()
                
                for i, (r,z) in enumerate(zip(r2_vals, empty_all)):
                    each_team.check_int(z)
                    if(each_team.r2 >= r):
                        cost_lin[i].append(c)
                        points_lin[i].append(p)
                    else:
                        non_cost_lin[i].append(c)
                        non_points_lin[i].append(p)
                    if(each_team.zero_count <= z):
                        cost_div[i].append(c)
                        points_div[i].append(p)
                    else:
                        non_cost_div[i].append(c)
                        non_points_div[i].append(p)
                    if(each_team.zero_count <= z and each_team.r2 >= r):
                        cost_both[i].append(c)
                        points_both[i].append(p)
                    else: 
                        non_cost_both[i].append(c)
                        non_points_both[i].append(p)
            cost_lin_t.append(mean_of_lists(cost_lin))
            points_lin_t.append(mean_of_lists(points_lin))
            non_cost_lin_t.append(mean_of_lists(non_cost_lin))
            non_points_lin_t.append(mean_of_lists(non_points_lin))
            
            cost_div_t.append(mean_of_lists(cost_div, empty_all))
            points_div_t.append(mean_of_lists(points_div,empty_all))
            non_points_div_t.append(mean_of_lists(non_points_div, empty_all))
            non_cost_div_t.append(mean_of_lists(non_cost_div,empty_all))
            
            cost_both_t.append(mean_of_lists(cost_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
            points_both_t.append(mean_of_lists(points_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
            non_cost_both_t.append(mean_of_lists(non_cost_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
            non_points_both_t.append(mean_of_lists(non_points_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
    
    
            lin_n_t.append([len(x) for x in cost_lin])
            non_lin_n_t.append([len(x) for x in non_cost_lin])
            
            div_n_t.append([len(x) for x in cost_div])
            non_div_n_t.append([len(x) for x in non_cost_div]) 
            
            both_n_t.append([len(x) for x in cost_both])
            non_both_n_t.append([len(x) for x in non_cost_both])
        

        df = pd.DataFrame({'formation': formations, 'Linear mean cost': cost_lin_t, 'Linear mean points': points_lin_t, 'n linear': lin_n_t, 
                           'Non linear cost': non_cost_lin_t, 'Non linear points': non_points_lin_t, 'Non linear n': non_lin_n_t,
                          'Div mean cost': cost_div_t, 'Div mean points': points_div_t, 'n div': div_n_t,
                                        'Non div cost': non_cost_div_t, 'Non div points': non_points_div_t, 'Non div n': non_div_n_t,
                                        'Both mean cost': cost_both_t, 'Both mean points': points_both_t, 'n both': both_n_t,
                                                      'Non both cost': non_cost_both_t, 'Non both points': non_points_both_t, 'Non both n': non_both_n_t
                          
                              
                })
            
        dest = os.path.join("results","pl",str(season), "all_on_all_" + typ + ".csv")
        df.to_csv(dest)


#%%
import matplotlib.pyplot as plt
def create_lin_perteam(all_teams, all_points, all_costs, res, playerspldata):
    i = 0

    for t,p,c in zip(all_teams, all_points, all_costs):
        each_team = team(t,playerspldata)
        each_team.create_int()    
        each_team.lin_fit()
        if(each_team.r2 > 0.85):
            res = np.append(res, each_team.res)
            i += 1
            if i == 5:
                return(res)
    
def get_residuals(season, typ = "raw", league = "pl"):
    formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]']

    res = np.array([])

    if typ=="raw":
        if league == "pl": playerspldata = get.get_players_feature_pl("data/pl_csv/players_" + typ + "_", season)
        elif league == "as": playerspldata = get.get_players_feature_pl(os.path.join("data","allsvenskan", "players_raw_"),season)
    elif typ =="noexp":
        playerspldata = get.get_players_feature_pl(os.path.join("data","pl_csv", "players_noexp_0.1_"),season)
    elif typ == "incnew":
        playerspldata = get.get_players_feature_pl(os.path.join("data","pl_csv", "players_incnew_lin_"),season)
    print(str(season))


    for formation in formations:
        if typ == "raw":
            if league=="pl":
                loc =  'data_cleaned/'+ league + '/'+str(season)+'/'
                one = pd.read_csv(loc+str(formation)+ '.csv', converters =conv)
            if league == "as":
                one = pd.read_csv(os.path.join("data_cleaned",league,str(formation)+ '.csv'), converters =conv)
                
        elif typ == "incnew": 
            loc = 'data_cleaned/pl/'+ typ + "/"
            one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)
        elif typ == "noexp":
            loc = 'data_cleaned/pl/'+ "noexp" + "/"
            one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)

        one = one.sample(frac=1)
        all_teams = one["indexes"].to_list()
        all_points = one['points_total'].to_list()
        all_costs = one['cost'].to_list()
        res = create_lin_perteam(all_teams, all_points, all_costs, res, playerspldata)
    np.savetxt(typ+str(season)+league + ".out", res)
    cmap = plt.get_cmap('gnuplot')
    print(len(res)/11)
    colors = [cmap(i) for i in np.linspace(0, 1, int(len(res)/11))]
    print(len(colors))
    colors = np.repeat(np.array(colors),11, axis = 0)
    fig, ax = plt.subplots()
    ax.scatter(range(len(res)), res, facecolors = "none", s=0.5,c = colors, marker = 'o')
    ax.plot(0,len(res))
    ax.set_ylabel("Cost")
    ax.set_xlabel("Individual")
    ax.set_ylim([-20,20])
    dirs = os.path.join("results", league, "linreg")
    os.makedirs(dirs, exist_ok = True)

    if league == "pl":
        ax.set_title("Residuals of FPL season " + str(season))
    else: ax.set_title("Residuals of AF season " + str(season))
    if typ == "noexp": ax.set_title("Residuals of FPL season  " + str(season) + " without exp. players")
    elif typ == "incnew": ax.set_title("Residuals of FPL season " + str(season) + " with 'new' players")

    plt.savefig(dirs +  "/residuals_" + str(season) + "_" + typ + ".png", bbox_inches = "tight")
    
    plt.show()
    #return(res)
                    

def use_linreg_pl_full_seasons_on_budgets(seasons, typ = "raw", r2_vals = [0.8,0.85,0.9], empty_all = [4,3,2], league = "pl"):
    formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]','[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]']


    for season in seasons:
        if typ=="raw":
            if league == "pl": playerspldata = get.get_players_feature_pl("data/pl_csv/players_" + typ + "_", season)
            elif league == "as": playerspldata = get.get_players_feature_pl(os.path.join("data","allsvenskan", "players_raw_"),season)
        elif typ =="noexp":
            playerspldata = get.get_players_feature_pl(os.path.join("data","pl_csv", "players_noexp_0.1_"),season)
        elif typ == "incnew":
            playerspldata = get.get_players_feature_pl(os.path.join("data","pl_csv", "players_incnew_lin_"),season)
        print(str(season))

        

        
        for formation in formations:
            cost_lin_t,points_lin_t, non_cost_lin_t,non_points_lin_t = [],[], [],[]
            cost_div_t,points_div_t, non_cost_div_t,non_points_div_t = [],[], [],[]
            cost_both_t,points_both_t, non_cost_both_t,non_points_both_t = [],[], [],[]
            lin_n_t, non_lin_n_t, div_n_t, non_div_n_t, both_n_t, non_both_n_t = [],[], [],[], [],[]
            budget_int_l = np.arange(450,1001,50)
            budget_int_h = np.add(budget_int_l, 50)

            print('Preparing data', str(formation))
            if typ == "raw":
                if league=="pl":
                    loc =  'data_cleaned/'+ league + '/'+str(season)+'/'
                    one = pd.read_csv(loc+str(formation)+ '.csv', converters =conv)
                if league == "as":
                    one = pd.read_csv(os.path.join("data_cleaned",league,str(formation)+ '.csv'), converters =conv)
                    
            elif typ == "incnew": 
                loc = 'data_cleaned/pl/'+ typ + "/"
                one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)
            elif typ == "noexp":
                loc = 'data_cleaned/pl/'+ "noexp" + "/"
                one = pd.read_csv(loc +str(season)+'/'+str(formation)+ '.csv', converters =conv)
            del_ind = np.array([],dtype = int)
            for (i,low) in enumerate(budget_int_l):

                budget = low+50
                print('-------------------------------------')
                print(budget)
                
                ones = cleaners.filter_df(one, low, budget)
                
                if(len(ones)<50):
                    del_ind = np.append(del_ind, i)
                    continue
                cost_lin,points_lin, non_cost_lin,non_points_lin= [],[], [],[]
                cost_div,points_div, non_cost_div,non_points_div = [],[], [],[]
                cost_both,points_both, non_cost_both,non_points_both= [],[], [],[]
                
#                ones = ones.sample(50)
                all_teams = ones["indexes"].to_list()
                all_points = ones['points_total'].to_list()
                all_costs = ones['cost'].to_list()
               
                cost_lin, points_lin = [[] for _ in range(3)], [[] for _ in range(3)]
                non_cost_lin,non_points_lin = [[] for _ in range(3)], [[] for _ in range(3)]
                
                cost_div, points_div = [[] for _ in range(3)], [[] for _ in range(3)]
                non_cost_div,non_points_div= [[] for _ in range(3)], [[] for _ in range(3)]
                
                cost_both, points_both= [[] for _ in range(3)], [[] for _ in range(3)]
                non_cost_both,non_points_both= [[] for _ in range(3)], [[] for _ in range(3)]
        
                #linear_cost_085, linear_points_085, non_cost_085, non_points_085 = [],[],[],[]
                #linear_cost_09, linear_points_09, non_cost_09, non_points_09 = [],[],[],[]
                for t,p,c in zip(all_teams, all_points, all_costs):
                    each_team = team(t,playerspldata)
                    each_team.create_int()
                    
                    each_team.lin_fit()
                    
                    for i, (r,z) in enumerate(zip(r2_vals, empty_all)):
                        each_team.check_int(z)
                        if(each_team.r2 >= r):
                            cost_lin[i].append(c)
                            points_lin[i].append(p)
                        else:
                            non_cost_lin[i].append(c)
                            non_points_lin[i].append(p)
                        if(each_team.zero_count <= z):
                            cost_div[i].append(c)
                            points_div[i].append(p)
                        else:
                            non_cost_div[i].append(c)
                            non_points_div[i].append(p)
                        if(each_team.zero_count <= z and each_team.r2 >= r):
                            cost_both[i].append(c)
                            points_both[i].append(p)
                        else: 
                            non_cost_both[i].append(c)
                            non_points_both[i].append(p)
                cost_lin_t.append(mean_of_lists(cost_lin))
                points_lin_t.append(mean_of_lists(points_lin))
                non_cost_lin_t.append(mean_of_lists(non_cost_lin))
                non_points_lin_t.append(mean_of_lists(non_points_lin))
                
                cost_div_t.append(mean_of_lists(cost_div, empty_all))
                points_div_t.append(mean_of_lists(points_div,empty_all))
                non_points_div_t.append(mean_of_lists(non_points_div, empty_all))
                non_cost_div_t.append(mean_of_lists(non_cost_div,empty_all))
                
                cost_both_t.append(mean_of_lists(cost_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
                points_both_t.append(mean_of_lists(points_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
                non_cost_both_t.append(mean_of_lists(non_cost_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
                non_points_both_t.append(mean_of_lists(non_points_both, ["0.8, 2", "0.85, 3", "0.9, 4"]))
        
        
                lin_n_t.append([len(x) for x in cost_lin])
                non_lin_n_t.append([len(x) for x in non_cost_lin])
                
                div_n_t.append([len(x) for x in cost_div])
                non_div_n_t.append([len(x) for x in non_cost_div]) 
                
                both_n_t.append([len(x) for x in cost_both])
                non_both_n_t.append([len(x) for x in non_cost_both])
            if(del_ind.size !=0):
                budget_int_h = np.delete(budget_int_h, del_ind)
            df = pd.DataFrame({'Budget': budget_int_h, 'Linear mean cost': cost_lin_t, 'Linear mean points': points_lin_t, 'n linear': lin_n_t, 
                               'Non linear cost': non_cost_lin_t, 'Non linear points': non_points_lin_t, 'Non linear n': non_lin_n_t,
                              'Div mean cost': cost_div_t, 'Div mean points': points_div_t, 'n div': div_n_t,
                                            'Non div cost': non_cost_div_t, 'Non div points': non_points_div_t, 'Non div n': non_div_n_t,
                                            'Both mean cost': cost_both_t, 'Both mean points': points_both_t, 'n both': both_n_t,
                                                          'Non both cost': non_cost_both_t, 'Non both points': non_points_both_t, 'Non both n': non_both_n_t
                              
                                  
                    })
                
            dest = os.path.join("results",league,str(season), formation +"_budgets_means_" + typ + ".csv")
            df.to_csv(dest)
