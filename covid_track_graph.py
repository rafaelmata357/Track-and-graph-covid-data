#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  2 April 2020                                 
# REVISED DATE :  21 jun   2020
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
import matplotlib.dates as mdates

# Imports functions created for this program
from get_args import get_args

#URL to get the raw data from JHU update

URL ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

def get_and_cleandata(URL):

    '''
    Download the data from the JHU github repository and prepare the data to graph
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
   
    
    Returns:
     dataset  : a pandas DF with the comple covid dataset   
    '''

    dataset = pd.read_csv(URL,index_col=0)  #Se lee los datos de github en formato .csv
    columna = dataset.columns
    dataset.set_index(columna[0], inplace=True)  # Para regenerar el indice por pais
    dataset.drop(['Lat', 'Long'], axis=1, inplace=True)  # Para eliminar las colunnas de Lat y Long
    return dataset


def graph(dataset, scale, top_n):
    '''
    From the Dataset this function graph the data for the top countries and central america countries 
    upto date.
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
    Returns:
         None  
    '''

    
    months = mdates.MonthLocator()  # every month
    mdays = mdates.DayLocator(interval=7)
    months_fmt = mdates.DateFormatter('%b')
    
    subdata = dataset.groupby('Country/Region', axis=0).sum()        #Sum the daily data by country
    columnas = list(subdata.columns)
    subdata.columns = pd.to_datetime(columnas)  #Change the format date
    
    initial_day = subdata.columns [0]
    last_day = subdata.columns [-1]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,7))  #Generate subplots
    fig.suptitle('2020 Accumulated Covid Cases until {}'.format(last_day.strftime('%d/%m')), fontsize=17, c='b')
          
    subdata.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the last column

    tograph = subdata.iloc[:top_n]   #Get top_n coutnries based on acumulated cases
 
    
    if scale == 'log':
        tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n),logy=True)  # Transpose and graph
        scale_log = [1, 10, 100, 1000, 10000, 100000,1000000,10000000]
        logscale = ['1', '10', '100', '1K', '10K', '100K','1M','10M']
        plt.sca(axes[0])
        plt.yticks(scale_log, logscale)
        #axes[0].set_yticks(scale_log)
        y_label = '#Cases Log Scale'
    else:
        tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n),logy=False)  # Transpose and graph
        y_label = '#Cases Linear Scale'
    
    axes[0].grid(True, which='major')
    axes[0].grid(which='minor', color='k', linestyle=':', alpha=0.5)
    #axes[0].set_xlabel('Date', fontsize= 5)
    axes[0].set_ylabel(y_label)
    
    r = axes[0].get_xticklabels()
    for i in r:
        #i.set_rotation(75)
        i.set_fontsize(10)
    
    # format the ticks
    axes[0].xaxis.set_major_locator(months)
    axes[0].xaxis.set_major_formatter(months_fmt)
    axes[0].xaxis.set_minor_locator(mdays)
    #axes[0].set_xticks(minor=False)

    
    # Set date min and date max for the x axis
  
    datemin = np.datetime64(initial_day, 'M')
    datemax = np.datetime64(last_day, 'M') + np.timedelta64(1, 'M')
    axes[0].set_xlim(datemin, datemax)

    graphca = subdata.loc[['Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Mexico','El Salvador']]  # Get  CA data to graph
    graphca.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the total cases   
     
    if scale == 'log':
       
        graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mexico', logy=True)  # Plot the transpose data
        scale_log = [1, 10, 100, 1000, 10000, 100000,1000000]
        logscale = ['1', '10', '100', '1K', '10K', '100K','1M']
        plt.sca(axes[1])
        plt.yticks(scale_log, logscale)
    else:
        graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mexico', logy=False)  # Plot the transpose data
    
    
    plt.xticks(fontsize=10)
    plt.grid(True, which='major')
    plt.grid(which='minor', color='k', linestyle=':', alpha=0.5)

    #axes[1].set_xlabel('Date')
    axes[1].set_ylabel(y_label)

    axes[1].xaxis.set_major_locator(months)
    axes[1].xaxis.set_major_formatter(months_fmt)
    axes[1].xaxis.set_minor_locator(mdays)
    axes[1].set_xlim(datemin, datemax)
    

    plt.show()



if __name__ == '__main__':
    
    in_arg = get_args()               #Get variables from command line
    scale = in_arg.scale
    top_n = in_arg.top_n
    dataset = get_and_cleandata(URL)
    graph(dataset, scale, top_n)
