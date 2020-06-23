# Track-and-graph-covid-data
Python app to track and graph the daily covid data using the JHU CSSE COVID-19 Dataset (https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data)
The script uses the Pandas and Numpy libraries to process the data and matplotlib to create the charts.

Two charts with the accumulated cases are generated with the app:
- Top countries 
- Central America region  and México 

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
