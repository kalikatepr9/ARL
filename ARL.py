#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:04:54 2019

@author: pranavkalikate
"""

# Association Rule Learning -  Apriori 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)   
transactions = []       
for i in range(0, 7501):              #there are 7500 observations
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) #20 attributes

    
# Training Apriori on the dataset
from apyori import apriori            
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

#check result..it will show the relation, support,confidence & lift

#type results in console
