#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  2 April 2020                                 
# REVISED DATE :  19 jun   2020
# PURPOSE: Create a function that retrieves the args for the covid track program
#          from the user using the Argparse Python module. If the user does not 
#          input the params default value is used. 
# 
# Command Line Arguments:
# 
#  1. Top n countries                           --top  default value 5
#  2. Scale                                     --scale default value log
##

# Imports python modules

import os
import argparse


# 
def get_args():
    '''
    Retrieves and parses two command line arguments provided by the user from 
    command line. argparse module is used. 
    If some arguments is missing default value is used. 
    Command Line Arguments:
    
    Args:
        1. Top n countries                           --top_n
        2. Log or linear scale for Y axis            --scale
   
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    '''
    
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser('covid_track_graph.py',description='Graph covid data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
       
    # Argument 1: Number of countries to display 
    parser.add_argument('--top_n', type = int, default= 5,
                    help = 'Top n countries to display')  
    
    # Argument 2: Y axis scale to use
    parser.add_argument('--scale', type = str, default = 'log', choices=['log', 'lin'],
                    help = 'Y scale log or linear')    

    # Argument 3: Country list
    parser.add_argument('--country',type = str, default = '',  nargs='+',
                    help = 'Country list') 
    
   #wrapper_descriptor

    return parser.parse_args()


