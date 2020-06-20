#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  20 April 2020                                 
# REVISED DATE :  19 jun   2020
# PURPOSE: Create a program to track the daily covid raw data from the Johns Hopkins University
#          and generate two charts containning the top 5 countries and the central america an Mx data 
#          
# 
# Command Line Arguments:
# 
#  1. Top countries number                      --top 
#  2. Scale                                     --scale
##

# Imports python modules


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports functions created for this program
from get_args import get_args

#URL to get the raw data 

URL ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

def cleandata(dataset):
    columna = dataset.columns
    dataset.set_index(columna[0], inplace=True)  # Para regenerar el indice por pais
    dataset.drop(['Lat', 'Long'], axis=1, inplace=True)  # Para eliminar las colunnas de Lat y Long


def graph(dataset):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    #print(axes)
    subdata = dataset.groupby('Country/Region', axis=0).sum()   #Agrupa los dataos por país y los suma
    columnas = list(subdata.columns)
    subdata.sort_values(columnas[-1], ascending=False, inplace=True) #Ordena el DF por la ultima columna

    tograph = subdata.head(6)
    #tograph.drop('China',axis = 0,inplace=True) #Get the top 6 countries with covid positive cases
    #graphcr = tograph.append(subdata.loc[['Costa Rica','Panama','Brazil','Russia','Mexico']])      #Add Costa Rica data to graph
    #print(subdata.index)
    #print(graphcr)
    #plt.figure(figsize=[12, 5])


    #plt.subplot(1, 2, 1)
    #plt.plot(tograph.T['US'])
    tograph.T.plot(ax=axes[0],grid=True, title='Datos por país',logy=True)  # Graph the transpose data
    scale = [1, 10, 100, 1000, 10000, 100000]
    logscale = ['1', '10', '100', '1K', '10K', '100K']
    plt.yticks(scale, logscale)
    plt.xticks(fontsize=8, rotation=75)
    plt.grid(True, which='both')
    plt.ylabel('#CASOS')
    #plt.show()

    #plt.subplot(1, 2, 2)
    #print(graphcr.index)
    graphcr = subdata.loc[['Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Mexico','El Salvador']]  # Add Costa Rica data to graph
    #print(subdata.index)
    #print(graphcr)

    graphcr.T.plot(ax=axes[1],grid=True, title='Datos por país', logy=True)  # Graph the transpose data
    scale = [1, 10, 100, 1000, 10000, 100000]
    logscale = ['1', '10', '100', '1K', '10K', '100K']
    plt.yticks(scale, logscale)
    plt.xticks(fontsize=8, rotation=75)
    plt.grid(True, which='both')
    plt.ylabel('#CASOS')

    plt.show()

    #daytotal = dataset.sum()  # Total cases per day
    #daytotal.plot(grid=True,title='Total cases per day',logy=True)
    #plt.show()


if __name__ == '__main__':
    
    #Get variables from command line
    
    in_arg = get_args()
    dataset = pd.read_csv(URL,index_col=0)  #Se lee los datos de github en formato .csv
    cleandata(dataset)
    graph(dataset)
