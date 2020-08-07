#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  2 April 2020                                 
# REVISED DATE :  31 july 2020
# PURPOSE: Create a program to track the daily covid raw data from the Johns Hopkins University
#          and generate two charts containning the top 5 countries and the central america an Mx data 
#          
# 
# Command Line Arguments:
# 
#  1. Top countries number                      --top {number}
#  2. Scale                                     --scale {log/lin}
#  3. Countries  list to plot                   --country {Country Name}
#  4. Cases/Population                          --pop {y/n}
## 5. Dataset accumulated, recoverd, deaths     --ds {acc, rec, death}

# Imports python modules


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import datetime

# Imports functions created for this program
from get_args import get_args

#URL to get the raw data JHU CSSE COVID-19 Dataset

URL_ACCUMULATED_CASES ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
URL_RECOVERED = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
URL_DEATHS = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
URL_TESTING ='https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv'

def get_and_cleandata(URL, start_date):

    '''
    Download the data from the JHU github repository and prepare the data to graph
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
   
    
    Returns:
     dataset  : a pandas DF with the comple covid dataset  
     population: dataset with the population per country 
    '''
    url_split = URL.split('/')
    print('Reading dataset {}'.format(url_split[-1]))
    dataset = pd.read_csv(URL,index_col=0)  #Se lee los datos de github en formato .csv
 
    columna = dataset.columns
    dataset.set_index(columna[0], inplace=True)  # Para regenerar el indice por pais
    dataset.drop(['Lat', 'Long'], axis=1, inplace=True)  # Para eliminar las colunnas de Lat y Long


    population = pd.read_excel('population.xlsx', 'data', index_col=0, na_values=['NA'])
    subdata = dataset.groupby('Country/Region', axis=0).sum()        #Sum the daily data by country
    subdata =  columns_dataset_to_timestamp(subdata) #Change the columns format date to timestamp

    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    #Filter the dataset using the start date selected

    subdata = subdata.loc[:,start_date_obj:]
    
    #Sort datasets using last day data as as key
    subdata = sort_dataset(subdata)
    
    return subdata, population



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
            tmp_dataset = dataset.loc[country] / pop*1000000
            #print('{} Population:{}'.format(country,pop))
        except:
            pass #print('Pais no encontrado {}'.format((country)))
 
    return tmp_dataset

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

def get_daily_values(dataset):
    '''
    From the accumulated Dataset this function calculate the daily values
      
    Args:
       
        dataset : dataset with the accumulated cases
       
    Returns:
         daily_dataset: data frame with the daily values
    '''
    columns = dataset.columns
    daily_value = np.empty(len(dataset)) # create a temporary numpy array to store the daily values 
    daily_dataset = pd.DataFrame(index=dataset.index)
    for country in columns: 
        country_data = dataset[country] 
        for i in range(len(country_data)-1): 
            daily_value[-i-1] = country_data[-i-1] - country_data[-i-2] 
            daily_value[0] = country_data[0] 
        daily_dataset[country] = daily_value
    
    return daily_dataset

def rolling_window(dataset, window, countries):

    rolling = dataset[countries].rolling(window)
    
    return rolling

def benford(dataset):
    '''
    From the daily Dataset values this function calculate the benford laws to analyze the data reported
      
    Args:
       
        dataset : dataset with the accumulated cases
       
    Returns:
         benford_dataset: data with the accumulated values for the first digit
    '''
    
    digits_map = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0}  #Digits hash table
    daily_values = dataset.values

    if len(dataset.shape) > 1:   #Check if is a vector single country data or matrix more than one country data
        for vector in daily_values:
            for number in vector:
                first_digit = str(number)[0]
                if first_digit in ['1','2','3','4','5','6','7','8','9']:
                    digits_map[first_digit] += 1                         #Count the number of firts digits for the daily values
    else:
        for Number in daily_values:
            first_digit = str(number)[0]
            if first_digit in ['1','2','3','4','5','6','7','8','9']:
                digits_map[first_digit] += 1          


    return digits_map  

def calculate_active_cases(accumulated_dataset, recovered_dataset, death_dataset):
    '''
    From the recoverd Dataset values and accumulated dataset this function calculate the active cases
      
    Args:
       
        recovered_dataset : dataset with the accumulated recovered cases
        accumulated_dataset : accumulated cases
       
    Returns:
         active_dataset: dataset with the active cases
    '''
    active_cases_dataset = accumulated_dataset - recovered_dataset - death_dataset

   
    pct_recovered_dataset =  recovered_dataset/accumulated_dataset*100 #Calculate the percentage of recovery cases
    

    return active_cases_dataset, pct_recovered_dataset
    
 
     
def daily_test(URL, countries, daily_dataset, time_frame):
    '''
    From the tests Dataset URL from the our world in data (https://ourworldindata.org/coronavirus)
      
    Args:
       
        URL : Git hub raw URL data
        countries : countries to read the tests covid data
        dataset : dataset with the accumulated cases
       
    Returns:
         test_dataset: dataset with the daily tests performed and merge with the accumulated cases
    '''   

    COUNTRY_MAP_TO_JSU = {'Argentina', 'Australia', 'Austria', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bolivia', 'Brazil', 'Bulgaria', 'Canada',
                          'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba', 'Czech Republic', 'Democratic Republic of Congo', 'Denmark', 'Ecuador',
                          'El Salvador', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Hong Kong', 'Hungary', 'Iceland',
                          'India', 'Indonesia', 'Iran', 'Ireland', 'Israel' , 'Italy', 'Japan', 'Kazakhstan', 'Kenya', 'Kuwait', 'Latvia', 'Lithuania',
                          'Luxembourg', 'Malaysia', 'Maldives', 'Malta', 'Mexico', 'Morocco', 'Myanmar', 'Nepal', 'Netherlands', 'New Zealand', 'Nigeria',
                          'Norway', 'Oman', 'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia',
                          'Rwanda', 'Saudi Arabia', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain',
                          'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Togo', 'Tunisia', 'Turkey' , 'Uganda', 'Ukraine', 'United Arab Emirates',
                          'United Kingdom', 'United States', 'Uruguay', 'Vietnam', 'Zimbabwe'}
    
    MAP_JSU_to_OWD = {'Korea, South':'South Korea', 'Taiwan*':'Taiwan', 'US':'United States', 'Czechia':'Czech Republic'}
    
    if countries[0] in MAP_JSU_to_OWD:
        country = MAP_JSU_to_OWD[countries[0]]
    else:
        country = countries[0]
        
    print('Reading dataset {}'.format(URL.split('/')[-1]))
    dataset = pd.read_csv(URL,index_col=0) 

    #print(dataset.head())
    daily_test_dataset = dataset[dataset.index.str.contains(country)][['Date','Daily change in cumulative total']]
    daily_test_dataset.dropna(axis=0, inplace=True)

    daily_test_dataset_tests = daily_test_dataset[daily_test_dataset.index.str.contains('tests')]
    daily_test_dataset_people = daily_test_dataset[daily_test_dataset.index.str.contains('people')]
   
    size_tests_dataset = daily_test_dataset_tests['Daily change in cumulative total'].count()
    size_people_dataset = daily_test_dataset_people['Daily change in cumulative total'].count()

    if size_tests_dataset >= size_people_dataset:
        daily_test_dataset =  daily_test_dataset_tests
    else:
        daily_test_dataset = daily_test_dataset_people


    dates = pd.to_datetime(daily_test_dataset.Date.values)
    daily_test_dataset.set_index(dates,inplace=True)
    daily_test_dataset.drop(['Date'], axis=1, inplace=True)

    df = pd.concat([daily_dataset,daily_test_dataset],axis=1)
 
    df.dropna(axis=0, inplace=True)
    
    df['week'] = df.index.week
    df['month'] = df.index.month

    if time_frame == 'weekly':
        df2 = df.groupby('week').sum()[[countries[0],'Daily change in cumulative total']] 
        df2['Positive Cases'] = df2[countries[0]]/df2['Daily change in cumulative total']*100
    else:
        df2 = df.groupby('month').sum()[[countries[0],'Daily change in cumulative total']]
        df2['Positive Cases'] = df2[countries[0]]/df2['Daily change in cumulative total']*100
    df2['WHO Recommend value'] = 10  #Set the WHO reccomended test/cases ratio 10% 
    
    return df2



def dashboard_1(dataset, scale, top_n, countries,  title_option):
    '''
    From the Dataset this function graph the data for the top countries and central america countries 
    upto date.
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
    Returns:
         None  
    '''

    #months = mdates.MonthLocator()  # every month
    #mdays = mdates.DayLocator(interval=7)
    #months_fmt = mdates.DateFormatter('%b')
    
    
    columnas = list(dataset.columns)

    initial_day = dataset.columns[0]
    last_day = dataset.columns[-1]
    dataset.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the last column
    
    title = '2020 {} Covid Accumulated confirmed cases until {}'.format(title_option, last_day.strftime('%d/%m'))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,7))  #Generate subplots
    fig.suptitle(title, fontsize=17, c='b')

    graphca = dataset.loc[countries]  # Get  CA data to graph
    graphca.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the total cases 
    tograph = dataset.iloc[:top_n]   #Get top_n coutnries based on acumulated cases

    
    
    if scale == 'log':
        tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n),logy=True)  # Transpose and graph
        scale_log, logscale, max_value = get_log_scale(tograph)
        plt.sca(axes[0])
        plt.yticks(scale_log, logscale)
        #axes[0].set_yticks(scale_log)
        y_label = '#Cases Log Scale'
    else:
        tograph.T.plot(ax=axes[0],grid=True, title='Top {} countries'.format(top_n), logy=False)  # Transpose and graph
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
    #axes[0].xaxis.set_major_locator(months)
    #axes[0].xaxis.set_major_formatter(months_fmt)
    #axes[0].xaxis.set_minor_locator(mdays)
      
    # Set date min and date max for the x axis
  
    datemin = np.datetime64(initial_day, 'M')
    datemax = np.datetime64(last_day, 'M') + np.timedelta64(1, 'M')
    axes[0].set_xlim(datemin, datemax)
    
    scale_log, logscale, max_value = get_log_scale(graphca)
  
    if scale == 'log':
        graphca.T.plot(ax=axes[1],grid=True, title='Other countries', logy=True)  # Plot the transpose data
        plt.sca(axes[1])
        plt.yticks(scale_log, logscale)
    else:
        graphca.T.plot(ax=axes[1],grid=True, title='Other countries', logy=False, fontsize=8)  # Plot the transpose data
    
    plt.xticks(fontsize=10)
    plt.grid(True, which='major')
    plt.grid(which='minor', color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlim(datemin, datemax) 
            
  
    maxvalue_str = str(max_value)
            
    plt.text(datemax,max_value, maxvalue_str) 
    axes[1].set_xlabel('Source Data: JHU CSSE COVID-19 Dataset',fontsize=5) 
   

    
    axes[1].set_ylabel(y_label)

    #axes[1].xaxis.set_major_locator(months)
    #axes[1].xaxis.set_major_formatter(months_fmt)
    #axes[1].xaxis.set_minor_locator(mdays)
    
    plt.show()


def columns_dataset_to_timestamp(dataset):
    '''
    From the Dataset this function take the columns in string format and conver to timestamp 
    
      
    Args:
        dataset : pandas datarframe dataset with string date format columns
       
    Returns:
         datset: with columns in timestamp format
    '''

    columns = list(dataset.columns)
    dataset.columns = pd.to_datetime(columns)

    return dataset


def graph_subplot(dataset, log, title, ylabel, xlabel, ax, bar, tf):

    '''
    Fuction to graph a subplot 
    Args:
        dataset : pandas dataframe to plot
        log:  boolena to plot logarithmic y scale
        ytitle : String with y title
        xtitle : String with x title
        ax : axes subplot 
        bar: boolean to plot a bar chart
    
    Returns:
         None
    '''

    if bar:
        dataset.plot.bar(ax=ax, grid=True, logy=log )
    else:
        dataset.plot(ax=ax, grid=True, logy=log )
    
    if log:
        scale_log, logscale, max_value = get_log_scale(dataset)
        plt.sca(ax)
        plt.yticks(scale_log, logscale)
      
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel)
    #if tf != 'daily':
    ax.set_xlabel(xlabel)

    ax.grid(True, which='major')
    #ax.grid(which='minor', color='k', linestyle=':', alpha=0.5)
    r = ax.get_xticklabels()
    for i in r:
        #i.set_rotation(75)
        i.set_fontsize(6)

def graph_subplot2(dataset, log, title, ylabel, xlabel, ax, bar, tf):

    '''
    Fuction to graph a subplot from a dataframe and combine with bar and line
    Args:
        dataset : pandas dataframe to plot
        log:  boolena to plot logarithmic y scale
        ytitle : String with y title
        xtitle : String with x title
        ax : axes subplot 
        bar: boolean to plot a bar chart
    
    Returns:
         None
    '''
    if log:
        scale = 'log'
    else:
        scale='linear'
   
    daily_aggregate = dataset.groupby(tf).sum()
    daily_average = dataset.groupby(tf).mean()
    daily_max = dataset.groupby(tf).max()
    
    ax2 = daily_aggregate.iloc[:,0].plot(ax=ax,grid=True, legend=True, label='Accumulated', logy=log, kind='bar')
    #daily_average.iloc[:,0].plot(ax=ax,grid=True, color='blue', label='Average',legend=True, lw=2, logy=log)
    ax.plot(ax2.get_xticks(),daily_max.iloc[:,0].values, color='green', label='Maximun', lw=2)
    ax.plot(ax2.get_xticks(),daily_average.iloc[:,0].values, color='blue', label='Average', lw=2)
    ax.legend()
    ax.set_yscale(scale)
    #daily_max.iloc[:,0].plot(ax=ax,grid=True, color='green', label='Maximun',legend=True, lw=2, logy=log,secondary_y=False)
   
    
    if log:
        scale_log, logscale, max_value = get_log_scale(dataset)
        plt.sca(ax)
        plt.yticks(scale_log, logscale)
      
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel)
    #if tf != 'daily':
    ax.set_xlabel(xlabel)

    ax.grid(True, which='major')
    #ax.grid(which='minor', color='k', linestyle=':', alpha=0.5)
    r = ax.get_xticklabels()
    for i in r:
        #i.set_rotation(75)
        i.set_fontsize(6)

def plot_benford(ax, dataset):
    '''
    From the Dataset this function graph the benford analysis
      
    Args:
        ax: axes subplot
        dataset: dataframe with the daily values
       
    Returns:
         None  
    '''

    digits_map = benford(dataset)
    ylabel = '%Probability'
    digits_values = np.array(list(digits_map.values()))
    digits_values = digits_values / digits_values.sum()*100 # Calculate the percentage
    df = pd.DataFrame({'P(D)':digits_values,'Benford´s Law':[30.1,17.6,12.5,9.7,7.9,6.7,5.8,5.1,4.6]},index=digits_map.keys())
    df.plot.bar(ax=ax, grid=True, logy=False, fontsize=8)
    ax.set_xlabel('First Digits of the dataset',fontsize=6)
    title='Benford Law Analysis'
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel)
    r = ax.get_xticklabels()
    for i in r:
        #i.set_rotation(75)
        i.set_fontsize(6)

def unify_datasets(datasetA, datasetB, nameA, nameB):
    '''
    From two datasets, unify both and add an id if the columns are the same
      
    Args:
        datsetA: data Frame
        datasetB: data Frame
       
    Returns:
         unified_dataset: data Frame 
    '''
    
    datasetA = datasetA.T
    datasetB = datasetB.T
    columnsA = list(datasetA.columns)
    columnsB = list(datasetB.columns)
   

    for i in range(len(columnsA)):
        columnsA[i] = '{} {}'.format(columnsA[i], nameA)
    
    for i in range(len(columnsA)):
        columnsB[i] = '{} {}'.format(columnsB[i], nameB)
    
   

    datasetA.columns = columnsA
    datasetB.columns = columnsB

  
    unified_dataset = pd.concat([datasetA, datasetB],axis=1)

    return unified_dataset

def sort_dataset(dataset):
    '''
    From the datasets, sort it using the last reported value
      
    Args:
        datsetA: data Frame
        
       
    Returns:
        sort_dataset: data Frame 
    '''
    last_day = dataset.columns[-1]
    dataset.sort_values(last_day, ascending=False, inplace=True)
    return dataset

def dashboard_2(accumulated_dataset, recovered_dataset, death_dataset, scale, countries, population, time_frame, URL, dash):
    '''
    From the Dataset this function graph the data for the top countries and central america countries 
    upto date.
      
    Args:
        URL : url to the github raw data from JHU updated daily
        accumulate_dataset: data frame with confirmed acculated cases
        recovered_dataset: data frame with recovered cases
        death_dataset: data frame with deats cases
        scale: string y or n
        countries: list with country
        time_frame: string daily, weekly, monthly
        aggregate: string aggregate sum, mean, max
        
       
    Returns:
         None  
    '''
    
    active_dataset, pct_recovered = calculate_active_cases(accumulated_dataset, recovered_dataset, death_dataset) #Calculate active cases and %recovered cases            
    daily_dataset = get_daily_values(accumulated_dataset.T)                 # Calculate the daily values 
     
    # Calculate the positive/accumulate test ratio if there is data
    try:
        test_ratio_df = daily_test(URL, countries, daily_dataset, time_frame) 
        test_data = True
    except:
        test_data = False
    
    acc_rec_dataset = unify_datasets(accumulated_dataset, recovered_dataset, 'Acc', 'Rec')
  
    acc_dataset_pop = cases_population_ratio(population, accumulated_dataset) 
     
    top_n_countries = acc_rec_dataset.iloc[:top_n]   #Get top_n coutnries based on acumulated cases

    countries_str = ', '.join(countries)
    last_day = accumulated_dataset.columns[-1]
    title = '2020 Covid  Cases until {} for {}'.format(last_day.strftime('%d/%m'), countries_str)
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,10))  #Generate subplots 3 x 2 
    fig.suptitle(title, fontsize=20, c='b')
    #fig.suptitle('Source Data: JHU CSSE COVID-19 Dataset',x=1, y=15, ha='left', fontsize=10, c='b')

    active_daily_dataset = get_daily_values(active_dataset.T)
    daily_dataset['week'] = daily_dataset.index.week
    daily_dataset['month'] = daily_dataset.index.month

    active_daily_dataset['week'] = active_daily_dataset.index.week
    active_daily_dataset['month'] = active_daily_dataset.index.month
    
    if scale == 'log':
        log = True
        ylabel = '#Cases Log Scale'
    else:
        log = False
        ylabel = '#Cases Linear Scale'
    
    
    if time_frame == 'daily' or time_frame == 'monthly':
        tf = 'month'
    else:
        tf = 'week'
    
    if dash==2:
        graph_subplot(dataset=acc_rec_dataset, log=log, title='Accumulated and Recovered cases', ylabel=ylabel, xlabel='', ax=axes[0,0], bar=False, tf='daily')
        graph_subplot(dataset=active_dataset.T, log=log, title='Accumulate Active cases', ylabel=ylabel, xlabel='', ax=axes[1,0], bar=False, tf='daily')
        graph_subplot(dataset=death_dataset.T, log=log, title='Accumulated Death cases', ylabel=ylabel, xlabel='*Source Data: JHU CSSE COVID-19 Dataset', ax=axes[2,0], bar=False, tf='daily')
        graph_subplot2(dataset=daily_dataset, log=log, title='Accumulated {}tly cases'.format(tf), ylabel='', xlabel='', ax=axes[0,1], bar=True, tf=tf)
        graph_subplot2(dataset=active_daily_dataset, log=log, title='Active {}tly cases'.format(tf), ylabel='', xlabel='', ax=axes[1,1], bar=True, tf=tf)
        graph_subplot(dataset=acc_dataset_pop.T, log=log, title='Accumulated cases by 1M population', ylabel='', xlabel='', ax=axes[2,1], bar=False, tf='daily')
    
        if test_data and not test_ratio_df.empty:
            graph_subplot(dataset=test_ratio_df[['Positive Cases','WHO Recommend value']], log=False, title='%Test to positive cases ratio {}tly'.format(tf), ylabel='%', xlabel='', ax=axes[0,2], bar=True, tf=tf)
       
        graph_subplot(dataset=pct_recovered.T, log=False, title='%Recovered cases', ylabel='%', xlabel='', ax=axes[1,2], bar=False, tf='daily')
        plot_benford(ax=axes[2,2], dataset=daily_dataset)
    else:
        graph_subplot(dataset=daily_dataset[countries], log=log, title='Daily confirmed cases', ylabel=ylabel, xlabel='', ax=axes[0,0], bar=False, tf='daily')
        rolling = daily_dataset[countries].rolling('14D')  #Rolling window for 14 days

        death_daily_dataset = get_daily_values(death_dataset.T)
    
        fatality_rate_dataset = death_dataset/accumulated_dataset*100
        graph_subplot(dataset=rolling.sum(), log=log, title='Cumulative 14 days rolling window', ylabel=ylabel, xlabel='', ax=axes[1,0], bar=False, tf='daily')
        graph_subplot(dataset=death_daily_dataset, log=log, title='Daily Death cases', ylabel=ylabel, xlabel='*Source Data: JHU CSSE COVID-19 Dataset', ax=axes[2,0], bar=False, tf='daily')
        graph_subplot(dataset=fatality_rate_dataset.T, log=log, title='Fatality rate', ylabel='', xlabel='', ax=axes[0,1], bar=False, tf=tf)
        graph_subplot2(dataset=active_daily_dataset, log=log, title='Active {}tly cases'.format(tf), ylabel='', xlabel='', ax=axes[1,1], bar=True, tf=tf)
        death_dataset_pop = cases_population_ratio(population, death_dataset) 

        graph_subplot(dataset=death_dataset_pop.T, log=log, title='Death Accumulated cases by 1M population', ylabel='', xlabel='', ax=axes[2,1], bar=False, tf='daily')
    
        if test_data and not test_ratio_df.empty:
            graph_subplot(dataset=test_ratio_df[['Positive Cases','WHO Recommend value']], log=False, title='%Test to positive cases ratio {}tly'.format(tf), ylabel='%', xlabel='', ax=axes[0,2], bar=True, tf=tf)
       
        graph_subplot(dataset=pct_recovered.T, log=False, title='%Recovered cases', ylabel='%', xlabel='', ax=axes[1,2], bar=False, tf='daily')
        plot_benford(ax=axes[2,2], dataset=daily_dataset)

    plt.show()

#MAIN PROGRAM

if __name__ == '__main__':
    
    in_arg = get_args()               #Get variables from command line
    scale = in_arg.scale
    top_n = in_arg.top_n
    countries = in_arg.country
    time_frame = in_arg.tf

    dash = in_arg.dash
    start_date = in_arg.start
 

    if countries == '': #If no countries specified assume all centroamerica countries 
        countries = ['Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'El Salvador','Nicaragua']
    
    
    
    if dash==1:
        accumulated_dataset, population = get_and_cleandata(URL_ACCUMULATED_CASES, start_date)
        dashboard_1(accumulated_dataset, scale, top_n, countries,  'Accumulated')
    else:
        #Read and clean data from datasets github repositories
        accumulated_dataset, population = get_and_cleandata(URL_ACCUMULATED_CASES, start_date)
        recovered_dataset, population = get_and_cleandata(URL_RECOVERED, start_date)
        death_dataset, population = get_and_cleandata(URL_DEATHS, start_date)
    
         #Filter the countries to explore and analyze
        accumulated_dataset = accumulated_dataset.loc[countries]
        recovered_dataset = recovered_dataset.loc[countries]
        death_dataset = death_dataset.loc[countries]
        dashboard_2(accumulated_dataset, recovered_dataset, death_dataset, scale, countries, population, time_frame, URL_TESTING, dash)

        