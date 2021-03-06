# Track-and-graph-covid-data
Python app to track and graph the daily covid data using the JHU CSSE COVID-19 Dataset (https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data)
The script uses the Pandas and Numpy libraries to process the data and matplotlib to create the charts.

Two charts with the accumulated cases are generated with the app:
- Top countries 
- Central America region  and México 

# Files in the repository

- covid_track_graph.py  : Main script
- get_args.py : Script to read the command line arguments
- example.png : Output example image

# Install
Clone the repository to the local machine

`$ git clone https://github.com/rafaelmata357/Track-and-graph-covid-data.git`

# Running

The app has two options:
- top_n: Option to specify the number (n) of countries to show in the graph
- scale: Option to select between Log or Linear scale for the y axis of the graphs
- country : specify the countries to plot the data
- tf  : Time frame option [daily, weekly, monthly]
- dash: Dashboard number to display [1,2,3,4] 
- start: start date format yyyy-mm-dd 
- end: end date format yyyy-mm-dd 

From terminal command line execute:

`$ python covid_track_graph.py --top_n n  --scale [log | lin] -tf [daily | weekly | monthly] --dash [1|2|4]  --start yyyy-mm-dd --end yyyy-mm-dd`

Example:

`$ python covid_track_graph.py --top_n 6 --scale log --dash 1`
`$ python covid_track_graph.py --country 'Costa Rica' --scale log --dash 2 --start 2020-03-01`

In addition to get help execute:

`$ python covid_track_graph.py -h `

# Output Example:

Dashboard #1
![Example](https://github.com/rafaelmata357/Track-and-graph-covid-data/blob/master/example1.png)
Dashboard #2
![Example](https://github.com/rafaelmata357/Track-and-graph-covid-data/blob/master/example2.png)
Dashboard #3
![Example](https://github.com/rafaelmata357/Track-and-graph-covid-data/blob/master/example3.png)

# Terms of use:

This script is made only to show how to read a dataset repository, process and graph using python, follow https://github.com/CSSEGISandData/COVID-19 regarding the dataset uses. 

# License:

The code follows this license: https://creativecommons.org/licenses/by/3.0/us/
