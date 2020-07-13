#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  2 April 2020                                 
# REVISED DATE :  9 july   2020
# PURPOSE: Create a program to track the daily covid raw data from the Johns Hopkins University
#          and generate two charts containning the top 5 countries and the central america an Mx data 
#          
# 
# Command Line Arguments:
# 
#  1. Top countries number                      --top 
#  2. Scale                                     --scale
#  3. Countries  list to plot                   --country
##

# Imports python modules


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

# Imports functions created for this program
from get_args import get_args

#URL to get the raw data JHU CSSE COVID-19 Dataset

URL ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
URL_RECOVERED = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
URL_DEATHS = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'


def get_and_cleandata(URL):

    '''
    Download the data from the JHU github repository and prepare the data to graph
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
   
    
    Returns:
     dataset  : a pandas DF with the comple covid dataset  
     population: dataset with the population per country 
    '''

    dataset = pd.read_csv(URL,index_col=0)  #Se lee los datos de github en formato .csv
    columna = dataset.columns
    dataset.set_index(columna[0], inplace=True)  # Para regenerar el indice por pais
    dataset.drop(['Lat', 'Long'], axis=1, inplace=True)  # Para eliminar las colunnas de Lat y Long


    population = pd.read_excel('population.xlsx', 'data', index_col=0, na_values=['NA'])
    subadata = dataset.groupby('Country/Region', axis=0).sum()        #Sum the daily data by country
    
    return subadata, population



def cases_population_ratio(population, dataset):
    '''
    From the Dataset this function calculate the accumulated cases/population ratio
      
    Args:
        population : population per country
        dataset : dataset with the accumulated cases
       
    Returns:
         dataset: dataset with the accumulated cases/ population 
    '''

    for country in dataset.index:
        try:
            pop = population[population['Country']==country]['Population'].values[0] 
            dataset.loc[country] = dataset.loc[country] / pop*1000000
            #print('{} Population:{}'.format(country,pop))
        except:
            pass #print('Pais no encontrado {}'.format((country)))
    
    return dataset

def get_log_scale(dataset):

    '''
    From the Dataset this function calculate the log scale to plot the data
      
    Args:
       
        dataset : dataset with the accumulated cases
       
    Returns:
         logscale: list with the logscale values
    '''

    max_value = dataset.max().max()     #Get maximum value from the dataset 
    scale_log_labels = {1:'1',10:'10',100:'100',1000:'1K',10000:'10K',100000:'100K',1000000:'1M',10000000:'10M',100000000:'100M'}
    positions = math.floor(math.log(max_value,10)) + 2          #Number of 10s power to generate the scale 
    scale_log = [10**i for i in range(positions)]
    logscale = [scale_log_labels[i] for i in scale_log]

    return scale_log, logscale, max_value



def graph(dataset, scale, top_n, countries, pop, population):
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
    
    
    columnas = list(dataset.columns)
    dataset.columns = pd.to_datetime(columnas)  #Change the format date
    
    initial_day = dataset.columns [0]
    last_day = dataset.columns [-1]
    dataset.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the last column
    
    if pop == 'y':
        title = '2020 Accumulated Covid  Cases until {} per 1M Population'.format(last_day.strftime('%d/%m'))
        dataset = cases_population_ratio(population, dataset)  # Calculate the cases/population ratio
    else:
        title = '2020 Accumulated Covid Cases until {}'.format(last_day.strftime('%d/%m'))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,7))  #Generate subplots
    fig.suptitle(title, fontsize=17, c='b')
          
    tograph = dataset.iloc[:top_n]   #Get top_n coutnries based on acumulated cases
    
    if scale == 'log':
        tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n),logy=True)  # Transpose and graph
        scale_log, logscale, max_value = get_log_scale(tograph)
        plt.sca(axes[0])
        plt.yticks(scale_log, logscale)
        #axes[0].set_yticks(scale_log)
        y_label = '#Cases Log Scale'
    else:
        tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n),logy=False)  # Transpose and graph
        y_label = '#Cases Linear Scale'
    
    axes[0].grid(True, which='major')
    axes[0].grid(which='minor', color='k', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Source Data: JHU CSSE COVID-19 Dataset', fontsize= 5)
    axes[0].set_ylabel(y_label)
    
    r = axes[0].get_xticklabels()
    for i in r:
        #i.set_rotation(75)
        i.set_fontsize(10)
    
    # format the ticks
    axes[0].xaxis.set_major_locator(months)
    axes[0].xaxis.set_major_formatter(months_fmt)
    axes[0].xaxis.set_minor_locator(mdays)
      
    # Set date min and date max for the x axis
  
    datemin = np.datetime64(initial_day, 'M')
    datemax = np.datetime64(last_day, 'M') + np.timedelta64(1, 'M')
    axes[0].set_xlim(datemin, datemax)
    
    graphca = dataset.loc[countries]  # Get  CA data to graph
    graphca.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the total cases  
    
    scale_log, logscale, max_value = get_log_scale(graphca)
    if scale == 'log':
        graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mexico', logy=True)  # Plot the transpose data
        
        plt.sca(axes[1])
        plt.yticks(scale_log, logscale)
    else:
        graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mexico', logy=False)  # Plot the transpose data
    
    
    plt.xticks(fontsize=10)
    plt.grid(True, which='major')
    plt.grid(which='minor', color='k', linestyle=':', alpha=0.5)

    axes[1].set_xlabel('Source Data: JHU CSSE COVID-19 Dataset',fontsize=5)
    axes[1].set_ylabel(y_label)

    axes[1].xaxis.set_major_locator(months)
    axes[1].xaxis.set_major_formatter(months_fmt)
    axes[1].xaxis.set_minor_locator(mdays)
    axes[1].set_xlim(datemin, datemax)
    if pop == 'y':
        maxvalue_str = '{:.2f}'.format(max_value)
    else:
        maxvalue_str = str(max_value)

    plt.text(datemax,max_value, maxvalue_str)
    

    plt.show()



if __name__ == '__main__':
    
    in_arg = get_args()               #Get variables from command line
    scale = in_arg.scale
    top_n = in_arg.top_n
    countries = in_arg.country
    pop = in_arg.pop

    if countries == '': #If no countries specified assume all centroamerica countries and Mexico
        countries = ['Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Mexico','El Salvador','Nicaragua']
    

    
    dataset, population = get_and_cleandata(URL_RECOVERED)
    graph(dataset, scale, top_n, countries, pop, population)
