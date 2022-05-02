# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:05:41 2022

@author: jgull
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import check_div_methods as cdm
import getters as get
# plot histogram of how many of each position


def load_cost_points(season, typ = "raw"):
    players = get.get_players_feature_pl("data/pl_csv/players_" + typ + "_", str(season))
    cost_list = [get.get_cost_player(players, i) for i in range(1,len(players)+1)]
    points_list = [get.get_points_player(players, i) for i in range(1, len(players)+1)]
    cost_list, points_list = zip(*[(x,y) for x,y in zip(cost_list, points_list) if x >0 and y >0])

    return(cost_list, points_list)

def plot_cost_vs_points_as():
    players = get.get_players_feature_pl("data/allsvenskan/players_raw_", str(21))
    cost_list = [get.get_cost_player(players, i) for i in range(1,len(players)+1)]
    points_list = [get.get_points_player(players, i) for i in range(1, len(players)+1)]
    cost_list, points_list = zip(*[(x,y) for x,y in zip(cost_list, points_list) if x >0 and y >0])
    fig, ax = plt.subplots()
    ax.scatter(cost_list, points_list, s = 2/3, color = "k" , marker = 'o')
    ax.set_xlim(38,140)
    ax.set_ylim(0,300)
    fan = "AF "
    ax.set_title("Cost vs. Total points for "+ fan  + str(21))
    ax.set_xlabel("Cost")
    ax.set_ylabel("Total points")
    plt.savefig("results/as/data_viz/pc_as")

def plot_cost_vs_points(seasons):
    for season in seasons:
        cost_list, points_list = load_cost_points(season)
        fig, ax = plt.subplots()
        ax.scatter(cost_list, points_list, s = 2/3, color = "k" , marker = 'o')
        ax.set_xlim(38,140)
        ax.set_ylim(0,300)
        fan = "FPL "
        ax.set_title("Cost vs. Total points for "+ fan  + str(season))
        ax.set_xlabel("Cost")
        ax.set_ylabel("Total points")
        plt.savefig("results/pl/data_viz/costvspoints/pc_" + str(season))
def plot_ideal_linreg():
    x = np.arange(0,11)
    y = 45 + 5*x
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.scatter(x,y, marker = 'o', ec = "k")
    ax.set_title("Example of ideal diverse team")
    ax.set_xlabel("Player")
    ax.set_ylabel("Cost")
    plt.savefig("plots/examples/linreg_ideal.png", bboxinches ="tight")
    plt.show()


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
def plot_hist_of_points(pointsList, title,season, nbins = 20, dest = "results/pl/data_viz",  typ = "raw"):
    fig, ax = plt.subplots()
    ax.hist(pointsList, bins = nbins)
    #plt.hist(npscostlist)
    ax.set_xlabel("Points")
    ax.set_ylabel("Amount")
    xmin, xmax, ymin, ymax = [0, max(pointsList)+20, 0, 180]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    dest = os.path.join(dest, "points_hist_"+ str(season) + typ +".png")
    fig.savefig(dest, bbox_inches = "tight")

# plot hist of costs
def plot_hist(list_to_count, season, xlabel = "Change of cost", 
                         title = "Cost change for FPL season ", 
                         nbins =20, lims = [-16,16,0,200], dest = "results/pl/data_viz", typ = "cost_change"):
    fig, ax = plt.subplots()
    ax.hist(list_to_count, bins = nbins)
    #plt.hist(npscostlist)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amount")
    xmin, xmax, ymin, ymax = lims
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title + str(season))
    dest = os.path.join(dest, typ +  "_" + str(season) + ".png")
    fig.savefig(dest, bbox_inches = "tight")
def plot_hist_of_costs(costList, title, season, nbins = 20, 
                       dest = "results/pl/data_viz", ylim= 300, typ = "raw"):
    #npscostlist = [np.log(x) for x in costList if x > 0]
    fig, ax = plt.subplots()
    ax.hist(costList, bins = nbins)
    #plt.hist(npscostlist)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Amount")
    xmin, xmax, ymin, ymax = [35, 140, 0, ylim]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    
    dest = os.path.join(dest, "cost_hist_"+ str(season) +typ +".png")
    fig.savefig(dest, bbox_inches = "tight")

def plot_example_bins(all_teams, all_points, all_costs):
    i = 0
    for t,p,c in zip(all_teams, all_points, all_costs):
        each_team = cdm.team(t,playerspldata)
        
        each_team.create_int()
        each_team.check_int(3)
        if(each_team.zero_count < 6):
            fig, ax = plt.subplots()
            ax.scatter(np.linspace(0,10,11).reshape(-1), each_team.ind_cost, edgecolor = "k")
            ax.set_title("Number of empty bins: " + str(each_team.zero_count))
            ax.set_xlabel("Player")
            ax.set_ylabel("Cost")
            i += 1
            if (i == 50):
                break

def exam_r2_plots(budg):
    empt = 5*[True]
    i=0
    for team in budg.team_list:                
        for r1, r2 in zip(range(5,10), range(6,11)):
            print(r1)
            if(team.r2 > r1/10 and team.r2 <= r2/10 and empt[r1-5]):
                empt[r1-5] = False
                i+=1
                        
                fig, ax = plt.subplots()
                ax.scatter(range(11), team.cost, marker = 'o', facecolors= "none", edgecolors='r')
                plt.title(team.r2)
                plt.show()
                break
    
    

