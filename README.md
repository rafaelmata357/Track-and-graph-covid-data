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

From terminal command line execute:

`$ python covid_track_graph.py --top_n n  --scale [log | lin]`

Example:

`$ python covid_track_graph.py --top_n 6 --scale log`

In addition to get help execute:

`$ python covid_track_graph.py -h `

# Output Example:

![Example](https://github.com/rafaelmata357/Track-and-graph-covid-data/blob/master/example.png)

# Terms of use:

This script is made only to show how to read a dataset repository, process and graph using python, follow https://github.com/CSSEGISandData/COVID-19 regarding the dataset uses. 

# License:

The code follows this license: https://creativecommons.org/licenses/by/3.0/us/
