# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:47:51 2022

@author: jgull
"""
import pandas as pd
import getters
#import parsers

def filter_df(df, lwr, upper):
    df = df[df['cost'] <= upper]
    df_new = df[df['cost'] >= lwr]
    return(df_new)

def all_forms_as_df_cleaned(league = "allsvenskan"):
    data2 = getters.get_data()
    players = getters.get_players_feature(data2)

    all_form = getters.get_diff_pos(players)

    all_form_df = [pd.DataFrame.from_dict(part, 
                                          orient = 'index').drop("element_type", axis=1) for part in all_form]

    sorted_dfs = [part.sort_values(by = ['total_points','now_cost']) for part in all_form_df]
    
    return sorted_dfs

def all_forms_as_df_cleaned_pl(bas, season):

    playerspldata = getters.get_players_feature_pl(bas, season)
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
    
#    print(df_part.head(10))
    tot_points = df_part["total_points"].to_list()
    indexes = list(df_part.index)
    best = tot_points[:n_form]
#    print(tot_points[:10])
#    print(indexes[:10])
#    print(best)
    
    ind_to_del = []

    for point, ind in zip(tot_points[n_form:], indexes[n_form:]):
        if point > min(best):
            best.remove(min(best))
            best.append(point)
        else: 
            ind_to_del.append(ind)
            
    return(dropRows(df_part, ind_to_del))


"""
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
#            print(df_part.iloc[i].name)
            ind_to_del.append(df_part.iloc[i].name)
#    print(ind_to_del)
    return(dropRows(df_part, ind_to_del))
"""

def cleanToWorstTeams(df_part, n_part): 
    # budget är i detta fall vad laget minst måste kosta
       
    df_clean=worst_del_multiple_point_per_cost(df_part, n_part)
    df_clean_2 = worst_del_multiple_cost_per_point(df_clean,n_part)  
    df_clean_3 = delete_better_points_when_decreasing_cost(df_clean_2, n_part)  

    return(df_clean_3)

def worst_del_multiple_point_per_cost(sorted_df_part, n):
    """
    delete if there are more than 
    n players that have the same cost
    """

    deleteIndexes=[]    
    sorted_df_part = sorted_df_part.sort_values(by=['now_cost',"total_points"])    
    for i in range(max(sorted_df_part['now_cost'])):
        if((sorted_df_part['now_cost'] == i).sum() > n):
            
            delete = list(sorted_df_part.index[(sorted_df_part['now_cost'] == i) ][n:])
            deleteIndexes.extend(delete)
            
    return(dropRows(sorted_df_part,deleteIndexes))

def worst_del_multiple_cost_per_point(sorted_df_part, n):
    """
    delete if there are more than 
    n players that have the same total points
    """
    sorted_df_part = sorted_df_part.sort_values(by=["total_points", "now_cost"])    
    deleteIndexes=[]    

    for i in range(max(sorted_df_part['total_points'])+1):
        if((sorted_df_part['total_points'] == i).sum() > n):
            
            delete = list(sorted_df_part.index[(sorted_df_part['total_points'] == i) ][:-n])
            deleteIndexes.extend(delete)
    return(dropRows(sorted_df_part,deleteIndexes))

def delete_better_points_when_decreasing_cost(df_part, n_form):
    
    df_part.sort_values(by=['now_cost','total_points'], 
                        ascending=[False, True], inplace = True)
    
    #print(df_part)
    tot_points = df_part["total_points"].to_list()
    indexes = list(df_part.index)
    worst = tot_points[:n_form]
    #print(worst)
    
    ind_to_del = []

    for point, ind in zip(tot_points[n_form:], indexes[n_form:]):
        if point < max(worst):
     #       print('byt',point)
            worst.remove(max(worst))
            worst.append(point)
        else: 
      #      print('bort',point)
            ind_to_del.append(ind)
            
    return(dropRows(df_part, ind_to_del))

#%%
season=1617
budget = 950 # i detta fall vad laget minst måste kosta 
playerspldata = getters.get_players_feature_pl("data/pl_csv/players_raw_", season)
 
formations = [[3,4,5],[3,4,5],[1,2,3]]
form_name = ["df", "mf", "fw"]
all_parts_but_goalie = all_forms_as_df_cleaned_pl("data/pl_csv/players_raw_",season)[1:]
individualCleansPerPosition =[]
 
for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
    print(pos)
    for p in part:
        print(p)
        all_cleaned = cleanToWorstTeams(df, p)  
        individualCleansPerPosition.append(all_cleaned)

worst3=individualCleansPerPosition

#%%
def cleanWorst(season):

    playerspldata = getters.get_players_feature_pl("data/pl_csv/players_raw_", season)
    
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
    all_parts_but_goalie = all_forms_as_df_cleaned_pl("data/pl_csv/players_raw_",season)[1:]
    individualCleansPerPosition =[]
    
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        print(pos)
        for p in part:
            print(p)
            all_cleaned = cleanToWorstTeams(df, p)  
            individualCleansPerPosition.append(all_cleaned)
    # Goalkeepers
    gk, _,_,_ = getters.get_diff_pos(playerspldata)
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    cleaned_gk = cleanToWorstTeams(sorted_df_gk, 1)
    #cleaned_gk.reset_index(inplace=True)
    cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    
    individualCleansPerPosition.append(cleaned_gk)
    
    print("Done with " + str(season))
    return individualCleansPerPosition
#%%

def worst_clean_all_data_pl(season, bas = "data/pl_csv/players_raw_", dest = "data_cleaned/pl/worst/",  clean_all = True, ns = 3):
    
    playerspldata = getters.get_players_feature_pl(bas, season)
    formations = [[3,4,5],[3,4,5],[1,2,3]]
    form_name = ["df", "mf", "fw"]
#    csv_file = str(bas) + str(season) + ".csv"
    all_parts_but_goalie = all_forms_as_df_cleaned_pl(bas,season)[1:]
    
    
    for part, df, pos in zip(formations, all_parts_but_goalie, form_name):
        print(pos)
        for p in part:
            print(p)
            all_cleaned =cleanToWorstTeams(df, p)
            
            if clean_all: 
                print(len(all_cleaned))
                combs=[] # kommentera bort raden nedan för att köra
 #               combs = parsers.worst_create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv")
                combs.to_csv(dest + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)
            else: 
  #              combs = parsers.create_all_combs_from_cleaned_df(playerspldata, all_cleaned, p)
                combs.to_csv("individual_data_cleaned/pl/" + str(season) + "/" + pos + "/" + str(p) + ".csv",index = False)

    
    # Goalkeepers
    
    gk, _,_,_ = getters.get_diff_pos(playerspldata)
    
    df_gk = pd.DataFrame.from_dict(gk, orient='index')
    
    sorted_df_gk = df_gk.sort_values(by= ['now_cost'])
    
    cleaned_gk = cleanToWorstTeams(sorted_df_gk, 1)
    # cleaned_gk.reset_index(inplace=True)
    cleaned_gk.rename(columns={'index':'indexes'}, inplace=True)
    cleaned_gk.drop('element_type', inplace=True, axis=1)
    if clean_all: 
        cleaned_gk.to_csv(dest + str(season) + "/gk.csv")
    else : 
        cleaned_gk.to_csv("individual_data_cleaned/pl/" + str(season) + "/gk.csv")
        
    print("Done with " + str(season))

# #%%
# seasons=[1617,1718,1819,1920,2021]
# for season in seasons:
    
#    worst = worst_clean_all_data_pl(season)

# #%%
# for season in seasons: 
    
#     parsers.clean_all_data_and_make_positions_combs_worst(season)
    
# #%%
#För att göra kombinationer av worst men krånglar, det är klart iaf
# seasons=[1617,1718,1819,1920,2021]
# import parsers
# for season in seasons:
#     location =  "data_cleaned/pl/worst/" + str(season)+"/"
#     parsers.write_full_teams(location) 


   