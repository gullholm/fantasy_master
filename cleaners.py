# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:47:51 2022

@author: jgull
"""
import pandas as pd
import getters


def all_forms_as_df_cleaned(league = "allsvenskan"):
    data2 = getters.get_data()
    players = getters.get_players_feature(data2)

    all_form = getters.get_diff_pos(players)

    all_form_df = [pd.DataFrame.from_dict(part, 
                                          orient = 'index').drop("element_type", axis=1) for part in all_form]

    sorted_dfs = [part.sort_values(by = ['total_points','now_cost']) for part in all_form_df]

    sorted_dfs_del_0 = del_zeros(sorted_dfs,[1,4,4,2])
    
    return sorted_dfs_del_0

def saveBetterPointsWhenIncreasingCost(df):
    pointsmax=0  
    saveIndexes=[]
    for i in range(df.shape[0]):
        if (df.iloc[i]['total_points']) > pointsmax:
            pointsmax =  (df.iloc[i]['total_points']) 
            saveIndexes.append(df.iloc[i].name)
    df = df.loc[saveIndexes]
    return df


def clean_gk(sorted_df_gk):
    
    idx = sorted_df_gk.groupby(['now_cost'])['total_points'].transform(max) == sorted_df_gk['total_points']
    gkBestPerSalary = sorted_df_gk[idx] # Remove the ones that costs the same but less points

    cost_best_gk = sorted_df_gk.loc[sorted_df_gk['total_points'].idxmax()]['now_cost']
    # Remove all that are more expansive than the best gk 
    gkFinal = gkBestPerSalary[gkBestPerSalary['now_cost'] <= cost_best_gk]      

    gkFinalSorted = gkFinal.sort_values(by=['now_cost', 'total_points'], ascending=[True, False])
    bestGK = saveBetterPointsWhenIncreasingCost(gkFinalSorted)
    
    return bestGK 
    
def dropRows(df, indexes):
    df = df.drop(indexes, axis=0)
    return df

def del_zeros(sorted_dfs, formation): # Delete #n_part zeros from formation df
    del_sorted_dfs = []
    for (i,df) in enumerate(sorted_dfs):
        dele = list(df.index[(df['total_points'] == 0)][formation[i]:])
        del_sorted_dfs.append(dropRows(df, dele))
    return(del_sorted_dfs)