#!/usr/bin/env python
# coding: utf-8

"""
Assela Pathirana, 2022 (C)

Input data: (Ensure year is a continious sequence)
year,maxvalue
1995,137
...
2013,161
2014,120.8

Year will be ignored, other than to get the order. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import time
sns.set()

"""
Do a linear regression
"""
def linfit(df,xcol,ycol):
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    X_ = df[xcol].values.reshape(-1,1)
    Y_ = df[ycol].values.reshape(-1,1)
    regr.fit(X_, Y_)
    return regr.coef_[0][0], regr.intercept_[0]

ser_=pd.read_csv('data.SHIA.csv', sep=',') # you may want to change the seperatore. 
ser=ser_[:10]

ser=ser.sort_values('maxvalue', ascending=False).reset_index(drop=True)


#https://serc.carleton.edu/hydromodules/steps/166250.html
#Gringorten's plotting position https://glossary.ametsoc.org/wiki/Gringorten_plotting_position
# qi=(i=a)/(N+1-2a) a= constant for estimation=0.44 using Gringorten's method
# p=(1-qi)
# Tr=1/qi
NY=len(ser)
ser['qi'] = (ser.index.values+1-.44)/(NY+1-2*.44)
ser['p'] = 1-ser['qi']
ser['llp']=-np.log(-np.log(ser['p']))
xticklabels=[2,3,5,10,20,50]
xtl=[x for x in xticklabels]
xticks=[-np.log(-np.log(1-1/x)) for x in xtl]

fig, ax = plt.subplots(figsize=(10, 5))
#xlim = [0,8] 
#ax.set_xlim(xlim)    
g=sns.regplot(x='llp',y='maxvalue',data=ser,fit_reg=True, ax=ax) 
sns.regplot(x='llp', y='maxvalue', data=ser, fit_reg=False, ax=ax)
g.set(xticklabels=xticklabels)
g.set(xticks=xticks)
g.set_title("Extreme Value Fit")
g.set_xlabel("Return period (years)")
g.set_ylabel("Maximum rainfall (mm/day)")

# rolling average
#ser_rol=ser.rolling(ROLL, win_type='triang').mean()
#sns.regplot(x='llp', y='maxvalue', data=ser_rol, fit_reg=False, ax=ax)

m,c=linfit(ser,'llp','maxvalue')
props = dict(boxstyle='round', alpha=0.5,color=sns.color_palette()[0])
textstr = r'$y={:4f}(-ln(-ln(1-1/(Tr))+{:4f} $'.format(m, c)
g.text(0.1, 0.9, textstr, transform=ax.transAxes, fontsize=14, bbox=props)

print(ser)

plt.show()





