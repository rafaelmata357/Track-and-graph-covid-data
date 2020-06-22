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

#URL to get the raw data 

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
    mdays = mdates.DayLocator(interval=10)
    months_fmt = mdates.DateFormatter('%b')
    
    subdata = dataset.groupby('Country/Region', axis=0).sum()        #Sum the daily data by country
    columnas = list(subdata.columns)
    subdata.columns = pd.to_datetime(columnas)  #Change the format date
    
    initial_day = subdata.columns [0]
    last_day = subdata.columns [-1]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,7))  #Generate subplots
    fig.suptitle('Accumulated Covid Cases until {}'.format(last_day.strftime('%d/%m/%Y')), fontsize=17, c='b')
          
    subdata.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the last column

    tograph = subdata.iloc[:top_n]   #Get top_n coutnries based on acumulated cases
 
    tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n),logy=True)  # Transpose and graph
    scale = [1, 10, 100, 1000, 10000, 100000]
    logscale = ['1', '10', '100', '1K', '10K', '100K']

  
    axes[0].set_yticks(scale)
    axes[0].grid(True, which='both')
    axes[0].set_xlabel('Date', fontsize= 5)
    axes[0].set_ylabel('#Cases')
    
    r = axes[0].get_xticklabels()
    for i in r:
        i.set_rotation(75)
        i.set_fontsize(8)
    
    # format the ticks
    axes[0].xaxis.set_major_locator(months)
    axes[0].xaxis.set_major_formatter(months_fmt)
    axes[0].xaxis.set_minor_locator(mdays)

    
    # Set date min and date max for the x axis
  
    datemin = np.datetime64(initial_day, 'M')
    datemax = np.datetime64(last_day, 'M') + np.timedelta64(1, 'M')
    axes[0].set_xlim(datemin, datemax)
    
    # format the ticks
    #axes[0].xaxis.set_major_locator(months)
    #axes[0].xaxis.set_major_formatter(months_fmt)
    #axes[0].xaxis.set_minor_locator(mdays)
    

    graphca = subdata.loc[['Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Mexico','El Salvador']]  # Get  CA data to graph
    graphca.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the total cases   
    print(columnas) 
    graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mx', logy=True)  # Plot the transpose data
    scale = [1, 10, 100, 1000, 10000, 100000]
    logscale = ['1', '10', '100', '1K', '10K', '100K']
    plt.yticks(scale, logscale)
    plt.xticks(fontsize=8, rotation=75)
    plt.grid(True, which='both')

    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('#Cases')
    

    plt.show()



if __name__ == '__main__':
    
    in_arg = get_args()               #Get variables from command line
    scale = in_arg.scale
    top_n = in_arg.top_n
    dataset = get_and_cleandata(URL)
    graph(dataset, scale, top_n)
