# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sns

data_set = pd.read_csv("players.csv")
#print(data_set.head())
#print(data_set.tail())
#pd.set_option('display.max_columns', 5)
#print(data_set[data_set.isna().any(axis=1)])
#print("mean",data_set["Weight"].mean())
#print("max",data_set["Weight"].max())
#print("min",data_set["Weight"].min())
pd.set_option('display.max_rows', 25)
#print(data_set["Nationality"].value_counts())
ds1 = data_set[(data_set["Growth"] > 4) & (data_set["Potential"]> 84) ]
sns.swarmplot(x="ClubPosition", y="Growth", data=ds1, s=2)
#print(ds1["Club"].value_counts())
#print(len(data_set[(data_set["ContractUntil"] == 2021) & (data_set["NationalTeam"] == "Not in team") ]))
#mt = data_set[data_set["FullName"] == "Mehdi Taremi"]
#print(mt[["Club", "WageEUR", "ClubPosition"]])