#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  2 April 2020                                 
# REVISED DATE :  15 july 2020
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

# Imports functions created for this program
from get_args import get_args

#URL to get the raw data JHU CSSE COVID-19 Dataset

URL_ACCUMULATED_CASES ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
URL_RECOVERED = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
URL_DEATHS = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
URL_TESTING ='https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv'

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

def calculate_active_cases(recovered_dataset, accumulated_dataset, death_dataset):
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
    
   
    
    dataset = pd.read_csv(URL,index_col=0) 
    daily_test_dataset = dataset[dataset.index.str.contains(country)][['Date','Daily change in cumulative total']]
    dates = pd.to_datetime(daily_test_dataset.Date.values)
    daily_test_dataset.set_index(dates,inplace=True)
    daily_test_dataset.drop(['Date'], axis=1, inplace=True)
    daily_test_dataset.dropna(axis=0, inplace=True)

  

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
    df2['WHO Recommend tests ratio'] = 10  #Set the WHO reccomended test/cases ratio 10% 
    
    return df2



def graph(dataset, pct_recovered, scale, top_n, countries, pop, population, title_option, time_frame, benf, ratio, URL):
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
    dataset.columns = pd.to_datetime(columnas)  #Change the format date to timestamp
    pct_recovered.columns = pd.to_datetime(columnas)  #Change the format date to timestamp
    
    initial_day = dataset.columns[0]
    last_day = dataset.columns[-1]
    dataset.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the last column
    
    if pop == 'y':
        title = '2020 {} Covid  Cases until {} per 1M Population'.format(title_option, last_day.strftime('%d/%m'))
        dataset = cases_population_ratio(population, dataset)  # Calculate the cases/population ratio
    else:
        title = '2020 {} Covid Cases until {}'.format(title_option, last_day.strftime('%d/%m'))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,7))  #Generate subplots
    fig.suptitle(title, fontsize=17, c='b')

    graphca = dataset.loc[countries]  # Get  CA data to graph
    graphca.sort_values(last_day, ascending=False, inplace=True) #Sort the data by the total cases 
          
    
    
    if ratio == 'rec' and time_frame == 'daily':
        tograph = graphca
        gtitle = 'Active Cases {}'.format(countries)
    elif benf == 'n' and time_frame == 'daily':
        tograph = dataset.iloc[:top_n]   #Get top_n coutnries based on acumulated cases
        gtitle = 'Top {} countries'.format(top_n)
    else:
        tograph = graphca
        gtitle = '{} countries'.format(countries)
    
    
    if scale == 'log':
        tograph.T.plot(ax=axes[0],grid=True, title=gtitle,logy=True)  # Transpose and graph
        scale_log, logscale, max_value = get_log_scale(tograph)
        plt.sca(axes[0])
        plt.yticks(scale_log, logscale)
        #axes[0].set_yticks(scale_log)
        y_label = '#Cases Log Scale'
    else:
        tograph.T.plot(ax=axes[0],grid=True, title=gtitle, logy=False)  # Transpose and graph
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
    
    
    
    if time_frame != 'daily': #Plot accumulated values weekly or monthly
        daily_dataset = get_daily_values(graphca.T)   #Calculate the daily values
        
        if ratio == 'test':
            y_label = '%Positive Cases'
            test_ratio_df = daily_test(URL, countries, daily_dataset, time_frame)
            
            if time_frame == 'weekly':
                test_ratio_df[['Positive Cases','WHO Recommend tests ratio']].plot.bar(ax=axes[1],grid=True, title='%Positive cases vs tests  {}'.format(countries), logy=False)
                #test_ratio_df[['WHO Recommend tests ratio']].plot.bar(ax=axes[1])
                axes[1].set_xlabel('Week',fontsize=8)
            else:
                test_ratio_df[['Positive Cases','WHO Recommend tests ratio']].plot.bar(ax=axes[1],grid=True, title='%Positive cases vs tests {}'.format(countries), logy=False)
                axes[1].set_xlabel('Month',fontsize=8)
                
        
        else:
            daily_dataset['week'] = daily_dataset.index.week
            daily_dataset['month'] = daily_dataset.index.month
            y_label = '#Cases Linear Scale'
        
            if time_frame == 'weekly':
                daily_dataset.groupby('week').sum()[countries].plot.bar(ax=axes[1],grid=True, title='Weekly accumuled values', logy=False)  # Plot the transpose data
                axes[1].set_xlabel('Week',fontsize=8)
            elif time_frame == 'monthly':
                daily_dataset.groupby('month').sum()[countries].plot.bar(ax=axes[1],grid=True, title='Monthly accumulated values', logy=False)  # Plot the transpose data
                axes[1].set_xlabel('Month',fontsize=8)
    else:
        scale_log, logscale, max_value = get_log_scale(graphca)
        if benf == 'y':    #Plot Benford analysis
            daily_dataset = get_daily_values(graphca.T) 
            digits_map = benford(daily_dataset)
            y_label = '%Probability'
            
            digits_values = np.array(list(digits_map.values()))
            digits_values = digits_values / digits_values.sum()*100 # Calculate the percentage
            df = pd.DataFrame({'P(D)':digits_values,'Benford´s Law':[30.1,17.6,12.5,9.7,7.9,6.7,5.8,5.1,4.6]},index=digits_map.keys())
            df.plot.bar(ax=axes[1],grid=True, title='Benford Law Analysis {}'.format(countries), logy=False)
            axes[1].set_xlabel('First Digits of the dataset',fontsize=8)
        else:
            if ratio == 'rec':
                graphca = pct_recovered.loc[countries]
            if scale == 'log':
            
                graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mexico', logy=True)  # Plot the transpose data
                plt.sca(axes[1])
                plt.yticks(scale_log, logscale)
            else:
                graphca.T.plot(ax=axes[1],grid=True, title='Central America and Mexico', logy=False)  # Plot the transpose data
    
            plt.xticks(fontsize=10)
            plt.grid(True, which='major')
            plt.grid(which='minor', color='k', linestyle=':', alpha=0.5)
            axes[1].set_xlim(datemin, datemax) 
            if pop == 'y':
                maxvalue_str = '{:.2f}'.format(max_value)
            else:
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


def graph_subplot(dataset, log, title, ytitle, xtitle, ax, bar):

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
        dataset.plot.bar(ax=ax, grid=True, log=log )
    else:
        dataset.plot.bar(ax=ax, grid=True, log=log )
    
    if log:
        scale_log, logscale, max_value = get_log_scale(dataset)
        plt.sca(ax)
        plt.yticks(scale_log, logscale)
        y_label = '#Cases Log Scale'

    initial_day = dataset.columns[0]
    last_day = dataset.columns[-1]
    datemin = np.datetime64(initial_day, 'M')
    datemax = np.datetime64(last_day, 'M') + np.timedelta64(1, 'M')
    axes[0].set_xlim(datemin, datemax)
    ax.set_title(title)
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    ax.grid(True, which='major')
    ax.grid(which='minor', color='k', linestyle=':', alpha=0.5)
    r = ax.get_xticklabels()
    for i in r:
        #i.set_rotation(75)
        i.set_fontsize(10)

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
    y_label = '%Probability'
    digits_values = np.array(list(digits_map.values()))
    digits_values = digits_values / digits_values.sum()*100 # Calculate the percentage
    df = pd.DataFrame({'P(D)':digits_values,'Benford´s Law':[30.1,17.6,12.5,9.7,7.9,6.7,5.8,5.1,4.6]},index=digits_map.keys())
    df.plot.bar(ax=ax, grid=True,  title='Benford Law Analysis {}'.format(countries), logy=False)
    ax.set_xlabel('First Digits of the dataset',fontsize=8)

def unify_datasets(datasetA, datasetB):
    '''
    From two datasets, unify both and add an id if the columns are the same
      
    Args:
        datsetA: data Frame
        datasetB: data Frame
       
    Returns:
         unified_dataset: data Frame 
    '''

    
    
    datasetA = sort_dataset(datasetA)
    datasetB = sort_dataset(datasetB)    
    columnsA = list(datasetA.T.columns)
    columnsB = list(datasetB.T.columns)

    for i in range(len(columnsA)):
        columnsA[i] = '{} {}'.format(columnsA[i],datasetA.name)
    
    for i in range(len(columnsA)):
        columnsB[i] = '{} {}'.format(columnsB[i],datasetB.name)

    datasetA.T.columns = columnsA
    datasetB.T.columns = columnsB
    unified_dataset = pd.concat([datasetA.T, datasetB.T],axis=1)

    return unified_dataset

def sort_dataset(dataset):
    '''
    From the datasets, sort it using the last reported value
      
    Args:
        datsetA: data Frame
        
       
    Returns:
        sort_dataset: data Frame 
    '''
    last_day = datasetB.columns[-1]
    dataset.sort_values(last_day, ascending=False, inplace=True)
    return sort_dataset

def graph2(accumulated_dataset, recovered_dataset, death_dataset, scale, top_n, countries, pop, population, title_option, time_frame, benf, ratio, URL):
    '''
    From the Dataset this function graph the data for the top countries and central america countries 
    upto date.
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
    Returns:
         None  
    '''

    
    
    #Change the columns format date to timestamp
    accumulated_dataset =  columns_dataset_to_timestamp(accumulated_dataset)   
    recovered_dataset = columns_dataset_to_timestamp(recovered_dataset)        
    death_dataset = columns_dataset_to_timestamp(recovered_dataset)

    active_dataset, pct_recovered = calculate_active_cases(recovered_dataset, accumulated_dataset, death_dataset) #Calculate active cases and %recovered cases            
    daily_dataset = get_daily_values(accumulated_dataset.T)                 # Calculate the daily values 
    
    # Calculate the positive/accumulate test ratio
    test_ratio_df = daily_test(URL, countries, daily_dataset, time_frame) 

    accumulated_dataset.name = 'Acc'
    recovered_dataset.name = 'Rec'

    acc_rec_dataset = unify_datasets(accumulated_dataset, recovered_dataset)

    acc_rec_country_pop = cases_population_ratio(population, acc_rec_dataset) 

    tograph = dataset.iloc[:top_n]   #Get top_n coutnries based on acumulated cases


    test_ratio_df[['Positive Cases','WHO Recommend tests ratio']].plot.bar(ax=axes[1],grid=True, title='%Positive cases vs tests  {}'.format(countries), logy=False)
   
    daily_dataset['week'] = daily_dataset.index.week
    daily_dataset['month'] = daily_dataset.index.month
    y_label = '#Cases Linear Scale'

    #if pop == 'y':
    #    pass #maxvalue_str = '{:.2f}'.format(max_value)
    #else:
    #    maxvalue_str = str(max_value)
    #plt.text(datemax,max_value, maxvalue_str) 
    #axes[1].set_xlabel('Source Data: JHU CSSE COVID-19 Dataset',fontsize=5) 
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,9))  #Generate subplots 3 x 2 
    fig.suptitle(title, fontsize=17, c='b')

    title = '2020 {} Covid  Cases until {} per 1M Population'.format(title_option, last_day.strftime('%d/%m'))
   
    plt.show()

#MAIN PROGRAM

if __name__ == '__main__':
    
    in_arg = get_args()               #Get variables from command line
    scale = in_arg.scale
    top_n = in_arg.top_n
    countries = in_arg.country
    pop = in_arg.pop
    dataset_option = in_arg.ds
    time_frame = in_arg.tf
    benf = in_arg.benf
    ratio = in_arg.ratio

    if countries == '': #If no countries specified assume all centroamerica countries and Mexico
        countries = ['Costa Rica', 'Panama', 'Guatemala', 'Honduras', 'Mexico','El Salvador','Nicaragua']
    
    if dataset_option == 'acc':
        URL = URL_ACCUMULATED_CASES
        title_option = 'ACCUMULATED'
    elif dataset_option == 'rec':
        URL = URL_RECOVERED
        title_option = 'RECOVERED'
    else:
        URL = URL_DEATHS
        title_option = 'DEATHS'
    
   

    
                                       
    accumulated_dataset, population = get_and_cleandata(URL_ACCUMULATED_CASES)
    recovered_dataset, population = get_and_cleandata(URL_RECOVERED)
    death_dataset, population = get_and_cleandata(URL_DEATHS)
    dataset, pct_recovered = calculate_active_cases(recovered_dataset, accumulated_dataset, death_dataset) # Calculate active cases
    if ratio == 'rec':
        title_option = '%RECOVERED'
    else:
        title_option = 'ACTIVE'
    if dataset_option == 'acc':
        dataset =  accumulated_dataset
 

    graph(dataset, pct_recovered, scale, top_n, countries, pop, population, title_option, time_frame, benf, ratio, URL_TESTING)
