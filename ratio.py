# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:38:33 2022

@author: jgull
"""
import pandas as pd
import ast
generic = lambda x: ast.literal_eval(x)
import os

def calc_perc_n(loc = "results/pl/1617/[3, 4, 3]_budgets_means_incnew.csv",
               dest= "results/pl/1617/ratio/perc/[3, 4, 3]_budgets_means_incnew_div.csv",
               kind = "div", k = 1):
    df = pd.read_csv(loc)
    df = df.applymap(lambda x: x.replace("nan", "-1") if isinstance(x, str) else x)
    df = df.applymap(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
    perc_div, perc_non_div= [],[]
    for ind in df.index:
        total = df["n " +kind][ind][k] + df["Non " + kind.lower() + " n"][ind][k]
        perc_div.append(round(100*(df["n " +kind][ind][k]/total)))
        perc_non_div.append(round(100*(df["Non " + kind.lower() + " n"][ind][k]/total)))
    new_df = pd.DataFrame({"Budget": df["Budget"], "Percent div ": perc_div, "Percent non div ": perc_non_div})
    
    new_df.to_csv(dest)
    return(0)

def visualize_formation_results(season = 1617, formation = "[4, 4, 2]", typ = "raw", league = "pl"):
     return(0)    

def calc_season_perc(season, league = "pl", types = ["raw"]
                      , formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]',
                                      '[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]'], kind = "div"):
    dirs = os.path.join("results",league, str(season))
    dirs_rat = os.path.join(dirs,"ratio","perc", kind + "_%")
    os.makedirs(dirs_rat, exist_ok = True)
    for typ in types:
        print(typ)
        for formation in formations:
            print(formation)
            loc = dirs + "/" + str(formation) + "_budgets_means_" + str(typ) + ".csv"
            
            dest = os.path.join(dirs_rat, str(formation) +"_" +  str(typ) + ".csv")
            calc_perc_n(loc, dest, kind)
def calc_all_perc(seasons, league = "pl", kind = "div"):
    
    for season in seasons:
        print(str(season))
        if season == 1617 or season ==1819:
            calc_season_perc(season, league, types = ["incnew","raw", "noexp"],kind = kind)
        else: calc_season_perc(season, league, types = ["raw"])
    return(0)               


def calc_ratio(loc = "results/pl/1617/[3, 4, 3]_budgets_means_incnew.csv",
               dest= "results/pl/1617/ratio/[3, 4, 3]_budgets_means_incnew_div.csv",
               kind = "Div", k = 3):
    df = pd.read_csv(loc)
    df = df.applymap(lambda x: x.replace("nan", "-1") if isinstance(x, str) else x)
    df = df.applymap(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
    ratio_cost, ratio_points= [],[]
    for ind in df.index:
        ratio_cost.append(df[kind +" mean cost"][ind][k]/df["Non " + kind.lower() + " cost"][ind][k])
        ratio_points.append(df[kind +" mean points"][ind][k]/df["Non " + kind.lower() + " points"][ind][k])
    new_df = pd.DataFrame({"Budget": df["Budget"], "Cost ratio " + str(k): ratio_cost, "Points ratio " + str(k): ratio_points})
    
    new_df.to_csv(dest)
    return(0)

def visualize_formation_results(season = 1617, formation = "[4, 4, 2]", typ = "raw", league = "pl"):
     return(0)    

def calc_season_ratio(season, league = "pl", types = ["raw"]
                      , formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]',
                                      '[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]'], kind = "Div"):
    dirs = os.path.join("results",league, str(season))
    dirs_rat = os.path.join(dirs,"ratio", kind)
    os.makedirs(dirs_rat, exist_ok = True)
    for typ in types:
        print(typ)
        for formation in formations:
            print(formation)
            loc = dirs + "/" + str(formation) + "_budgets_means_" + str(typ) + ".csv"
            
            dest = os.path.join(dirs_rat, str(formation) +"_" +  str(typ) + ".csv")
            calc_ratio(loc, dest, kind)

def calc_all_ratios(seasons, league = "pl", kind = "Div"):
    
    for season in seasons:
        print(str(season))
        if season == 1617 or season ==1819:
            calc_season_ratio(season, league, types = ["incnew","raw", "noexp"],kind = kind)
        else: calc_season_ratio(season, league, types = ["raw"])
    return(0)               

def calc_mean_ratio_all_form(season, typ = "raw", league = "pl",formations = ['[3, 4, 3]','[3, 5, 2]','[4, 3, 3]',
                '[4, 4, 2]','[4, 5, 1]','[5, 3, 2]','[5, 4, 1]'], kind = "Div", k = 3):
    loc = os.path.join("results", league, str(season),"ratio", kind + "/")
    cost_matr = np.full([7,13], np.nan)
    points_matr = np.full([7,13], np.nan)
    print(season)
    print(typ)
    for (i,formation) in enumerate(formations):
        df = pd.read_csv(loc +  formation +"_" +  typ + ".csv")
        print(formation)
        for ind in df.index:
            if(df["Cost ratio " + str(k)][ind] > 0 and df["Points ratio "+ str(k)][ind] > 0):
                cost_matr[i,ind] = df["Cost ratio "+ str(k)][ind]
                points_matr[i,ind] = df["Points ratio " + str(k)][ind]
            else: print("HEJEJ")            
    cost_mean = np.nanmean(cost_matr, axis = 0)
    print(cost_mean)
    points_mean = np.nanmean(points_matr, axis = 0)
    budgets = np.arange(500,1101, step = 50,dtype = int)
    df_new = pd.DataFrame({"Budgets": budgets, "Mean cost ratio" : cost_mean, "Mean points ratio": points_mean})
    df_new.to_csv(loc + "_" + typ + "_meanRatio.csv")
    return(0)

import numpy as np

def calc_all_mean_ratio(seasons= [1617,1718,1819,1920,2021], league = "pl", kind = "Div"):
    for season in seasons:
        if season == 1617 or season ==1819:
            for typ in ["raw", "incnew", "noexp"]:
                calc_mean_ratio_all_form(season,typ, kind = kind)
        else: calc_mean_ratio_all_form(season, kind =kind)
        
def plot_mean_ratios(season, typ = "raw", league = "pl", kind = "Div"):
    budgets = np.arange(500,1101, step = 50,dtype = int)
    if kind == "Div": loc = os.path.join("results", league, str(season),"ratio", kind + "/_")

    else: loc = os.path.join("results", league, str(season),"ratio/_")
    
    dirs = loc +  typ + "_meanRatio.csv"
    df = pd.read_csv(dirs)
    cost_ratio =df["Mean cost ratio"].to_numpy()
    points_ratio = df["Mean points ratio"].to_numpy()
    inds_del = ~np.isnan(cost_ratio)
    cost_ratio = cost_ratio[inds_del]
    points_ratio = points_ratio[inds_del]

    fig, ax = plt.subplots()
    ax.plot(budgets[0:cost_ratio.shape[0]], cost_ratio, label = "Cost", marker = 'o')
    ax.plot(budgets[0:cost_ratio.shape[0]], points_ratio, label = "Points", marker = 'o')
    ax.legend()
    ax.set_xlabel("Budget constraint")
    ax.set_ylabel("Ratio (div/non div)")
    if league == "pl":
        if typ == "raw": ax.set_title("Ratio of div. and non div. teams FPL season " + str(season) )
        elif typ == "incnew":ax.set_title("Ratio of div. and non div. teams FPL season " + str(season) + "inc. new players" )
        elif typ == "noexp":ax.set_title("Ratio of div. and non div. teams FPL season " + str(season) + " no exp. players")
    else: ax.set_title("Ratio of div. and non div. teams AF season " + str(season) )
    ax.set_ylim(0.85,1.15)
    plt.savefig("results/" + league +"/plots/ratio/" + kind +  "/_lines_mean_"+ str(season) + typ + kind + ".png", bbox_inches = "tight")
def plot_all_mean_ratio(seasons= [1617,1718,1819,1920,2021], league = "pl", kind = "Div"):
    for season in seasons:
        if season == 1617 or season ==1819:
            for typ in ["raw", "incnew", "noexp"]:
                plot_mean_ratios(season,typ, kind = "Div")
        else: plot_mean_ratios(season, kind = "div")
#%%
import pandas as pd
import matplotlib.pyplot as plt
import math



