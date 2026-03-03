from Sub_Functions.Evaluate import open_popup
from Sub_Functions.Analysis import Analysis

from Sub_Functions.Read_data import Read_data
from Sub_Functions.plot import ALL_GRAPH_PLOT
DB = ["DB"]

for i in range(len(DB)):
    # Read_data(DB[i])

    T = Analysis(DB[i])

    # T.COMP_Analysis()

    T.PERF_Analysis()

    Plot = ALL_GRAPH_PLOT()

    Plot.GRAPH_RESULT(DB[i])

