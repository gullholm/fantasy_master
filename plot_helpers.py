# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:05:41 2022

@author: jgull
"""

import matplotlib.pyplot as plt
import os

# plot histogram of how many of each position
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
def plot_hist_of_points(pointsList, title,season, nbins = 20, dest = "results/pl/data_viz"):
    fig, ax = plt.subplots()
    ax.hist(pointsList, bins = nbins)
    #plt.hist(npscostlist)
    ax.set_xlabel("Points")
    ax.set_ylabel("Amount")
    xmin, xmax, ymin, ymax = [0, max(pointsList)+20, 0, 180]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    dest = os.path.join(dest, "points_hist_"+ str(season) +".png")
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
    
def plotIndividualCosts(feature_data, title, degree, line=False):

    x = list(range(1,12,1))
    plt.xlabel("Player")
    plt.ylabel("Cost")
    for i in range(len(feature_data)):
        y = ast.literal_eval(feature_data[i]['Individual costs'])
        plt.plot(x,y, 'o', label = '< %s'%(500+i*50))
        if line:
            poly= np.polyfit(x,y,degree)
            plt.plot(x, np.polyval(poly,x))
    plt.legend()
    plt.title(title)
    plt.show()              

