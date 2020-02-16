from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotTrainTest(index_name, headers, data, axs, color, marker, label):
    """Generate a side by side plot of expected and predicted

    Args:
        headers: list of train/test metric names
        data: data frame where each frow is an iteration and each column is a train/test metric
        axs
        color: the color to use for the plots
        maker: the marker to use for the plots
        label: the label to use for the plots
    """
    n_metric_pairs = len(headers)
    data_downsampled = data.iloc[::2] # down sample by 1/2
    for n in range(0,n_metric_pairs):
        if n==0:
            axs[n, 0].scatter(data_downsampled.loc[:, index_name],  data_downsampled.loc[:, headers[n][0]],
                       alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=label)
            axs[n, 0].title.set_text("Train")
            axs[n, 1].scatter(data_downsampled.loc[:, index_name],  data_downsampled.loc[:, headers[n][1]],
                       alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=label)
            axs[n, 1].title.set_text("Test")
        else:
            axs[n, 0].scatter(data_downsampled.loc[:, index_name],  data_downsampled.loc[:, headers[n][0]],
                       alpha=0.5, c=color, marker=marker, edgecolors='none', s=20)
            axs[n, 1].scatter(data_downsampled.loc[:, index_name],  data_downsampled.loc[:, headers[n][1]],
                       alpha=0.5, c=color, marker=marker, edgecolors='none', s=20)

def aggregateTrainTestStats(headers, agg_funcs, index_name, data, label, n_agg_values = 10):
    """Creates a pandas data frame with headers according to headers
    and row labels according to labels where the column data from the
    specified header is aggregated according to the specified aggregation function.
    In addition, the number of iterations required to reach the aggregation function
    100 times is also reported.

    Args:
        headers: list of train/test metric names
        agg_func: name of an aggregation function
        index_name: name of the index header
        data: the actual data frame of values to aggregate according to the specified statistics

    Returns:
        pandas data frame
    """
    assert(len(headers)==len(agg_funcs))
    row_data = {"label":label}    
    for header_tup, agg_func in zip(headers, agg_funcs):
        for header in header_tup:
            # calculate the aggregation statistic
            agg_func_value = None
            if agg_func == "max":
                agg_func_value = data.loc[:,header].max()
            elif agg_func == "min":
                agg_func_value = data.loc[:,header].min()
            row_data.update({header:agg_func_value})

            # calculate # of iterations using the agg stat
            indices = []
            if agg_func == "max":
                indices = data[data[header]>=agg_func_value*0.99].index.tolist()
            elif agg_func == "min":
                indices = data[data[header]<=agg_func_value*1.01].index.tolist()
            # TODO: added only for loss error
            n_agg_values = 10
            if "_Error" in header:
                n_agg_values = 1
            index = indices[n_agg_values-1] if len(indices) >= n_agg_values else np.NaN
            row_data.update({header + "_itersToValue":index})
    return row_data

def readDataFilenamesCsv(filename):
    """Creates a pandas data frame with headers for 'filenames','labels','color','marker'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'filenames','labels','colors','markers'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    filenames = list(data_used.loc[:, "filenames"]) 
    labels = list(data_used.loc[:, "labels"])
    #filenames, labels, colors, markers = data_used.loc[:, "filenames"], data_used.loc[:, "labels"], data_used.loc[:, "colors"], data_used.loc[:, "markers"]
    return filenames, labels

def readDataHeaders(filename):
    """Creates a pandas data frame with headers for 'train_headers','test_headers','agg_funcs'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'filenames','labels','colors','markers'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    headers = list(zip(list(data_used.loc[:, "train_headers"]),list(data_used.loc[:, "test_headers"])))
    agg_funcs = list(data_used.loc[:, "agg_funcs"])
    #filenames, labels, colors, markers = data_used.loc[:, "filenames"], data_used.loc[:, "labels"], data_used.loc[:, "colors"], data_used.loc[:, "markers"]
    return headers, agg_funcs

def main(data_dir, data_filename, headers_filename, index_name, n_rows):
    """Run main script"""

    # read in the input files
    filenames, labels = readDataFilenamesCsv(data_filename)
    headers, agg_funcs = readDataHeaders(headers_filename)

    # make the empty data frame for the aggregate statistics
    flat_headers = [item for sublist in headers for item in sublist]
    custom_headers = [item + "_itersToValue" for item in flat_headers]
    flat_headers.extend(custom_headers)
    flat_headers.insert(0, "label")
    agg_stats = pd.DataFrame(columns=flat_headers)

    # make the initial figure
    n_metric_pairs = len(headers)
    n_data = len(filenames)
    fig, axs = plt.subplots(n_metric_pairs, 2, sharex=True, sharey=False)

    # read in the data and anlayze each train/test metric
    all_colors = ["b","r","g","m","c","y","k"]
    all_markers = [".","+",",","o","x","v"]
    marker_iter = 0
    color_iter = 0
    for n in range(0, n_data):
        # trim the data
        data = pd.read_csv(filenames[n]).iloc[:n_rows,:]

        # plot each train/test metric
        plotTrainTest(index_name, headers, data, 
                      axs, all_colors[color_iter], all_markers[marker_iter], labels[n])
        color_iter += 1
        if color_iter >= len(all_colors):
            color_iter = 0;
            marker_iter += 1
            if marker_iter >= len(all_markers):
                marker_iter = 0

        # calculate the aggregate statistics
        row_data = aggregateTrainTestStats(headers, agg_funcs, index_name, data, labels[n])
        agg_stats = agg_stats.append(row_data, ignore_index=True)

    # store the aggregate statistics
    agg_stats.to_csv(data_dir + "TrainTestMetrics.csv")

    # make the legend
    axs[0,0].legend(loc="upper left", markerscale=2)

    # show the image
    plt.show()

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/"
    data_filename = data_dir + "TrainTestMetricsInput.csv"
    headers_filename = data_dir + "TrainTestMetricsHeaders.csv"

    # Name of the index
    index_name = "Epoch"; n_rows = 90000
    main(data_dir, data_filename, headers_filename, index_name, n_rows)