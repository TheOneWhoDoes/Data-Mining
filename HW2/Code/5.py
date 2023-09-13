# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:57:59 2021

@author: Amirhosein Khoshbin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from sklearn import datasets
 

#Load Data
worms = pd.read_csv('worms.csv')
print(worms)
worms = worms.iloc[: , 1:]
#worms = worms.to_numpy()
 

worms.plot(x ='X', y='Y', s=0.001 ,kind = 'scatter')
plt.legend()
plt.show()


