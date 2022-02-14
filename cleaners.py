# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:47:51 2022

@author: jgull
"""
import pandas as pd
import getters
import parsers

def all_forms_as_df_cleaned(league = "allsvenskan"):
    data2 = getters.get_data()
    players = getters.get_players_feature(data2)

    all_form = getters.get_diff_pos(players)

    all_form_df = [pd.DataFrame.from_dict(part, 
                                          orient = 'index').drop("element_type", axis=1) for part in all_form]

    sorted_dfs = [part.sort_values(by = ['total_points','now_cost']) for part in all_form_df]
    
    return sorted_dfs

def all_forms_as_df_cleaned_pl(csv_file):
    playerspl = pd.read_csv(csv_file).to_dict('index')
    playerspldata = getters.get_players_feature_pl(playerspl)

    all_form = getters.get_diff_pos(playerspldata)

    all_form_df = [pd.DataFrame.from_dict(part, 
                                          orient = 'index').drop("element_type", axis=1) for part in all_form]

    sorted_dfs = [part.sort_values(by = ['total_points','now_cost']) for part in all_form_df]
    
    return sorted_dfs

def run_all_cleans(df_part, n_part):
    
    df_clean =  del_n_zeros(df_part, n_part)
    df_clean_1 = del_multiple_point_per_cost(df_clean, n_part)
    df_clean_2 = del_multiple_cost_per_point(df_clean_1,n_part)        
    df_clean_3 = delete_worse_points_when_increasing_cost(df_clean_2, n_part)
    return(df_clean_3)

def del_n_zeros(df_part, n_part):
    dfDelet = list(df_part.index[(df_part['total_points'] == 0)][n_part:] )
    return(dropRows(df_part, dfDelet))
    

#def clean_for_all_combs(func, df_part):
#    return([func(df) for df in dfs])

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
    bestGK = delete_worse_points_when_increasing_cost(gkFinalSorted, 1)
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

def del_multiple_cost_per_point(sorted_df_part, n):
    """
    delete if there are more than 
    n players that have the same total points
    """
    sorted_df_part = sorted_df_part.sort_values(by=["total_points", "now_cost"])    
    deleteIndexes=[]    

    for i in range(max(sorted_df_part['total_points'])+1):
        if((sorted_df_part['total_points'] == i).sum() > n):
            
            delete = list(sorted_df_part.index[(sorted_df_part['total_points'] == i) ][n:])
            deleteIndexes.extend(delete)
    return(dropRows(sorted_df_part,deleteIndexes))



def del_multiple_point_per_cost(sorted_df_part, n):
    """
    delete if there are more than 
    n players that have the same cost
    """

    deleteIndexes=[]    
    sorted_df_part = sorted_df_part.sort_values(by=['now_cost',"total_points"])    
    
    for i in range(max(sorted_df_part['now_cost'])):
        if((sorted_df_part['now_cost'] == i).sum() > n):
            
            delete = list(sorted_df_part.index[(sorted_df_part['now_cost'] == i) ][:-n])
            deleteIndexes.extend(delete)
            
    return(dropRows(sorted_df_part,deleteIndexes))

def delete_worse_points_when_increasing_cost(df_part, n_form):
    
    df_part.sort_values(by=['now_cost','total_points'], 
                        ascending=[True, False], inplace = True)
    
    best = df_part.head(n_form)['total_points'].to_list()
    ind_to_del = []

    for i in range(n_form,len(df_part)):
        if df_part.iloc[i]['total_points'] > min(best):
            best.remove(min(best))
            best.append(df_part.iloc[i]['total_points'])
        else: 
            ind_to_del.append(df_part.iloc[i].name)
    return(dropRows(df_part, ind_to_del))


             