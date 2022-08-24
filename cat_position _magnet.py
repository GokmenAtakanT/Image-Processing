# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:26:25 2021

@author: gat06
"""

import matplotlib.pyplot as plt

x=[50,40,30,20,15,10,5,0,-5,-8]
y=[0,8,14,20,25,25,26,26,29,29]
plt.plot(x,y)
plt.xlabel("Magnet Position[mm]")
plt.ylabel("Catheter Pulling Distance[mm]")
