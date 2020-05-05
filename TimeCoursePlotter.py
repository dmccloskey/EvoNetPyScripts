from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotTimeCourse(index_name, headers, data, axs, color, marker, label):
    """Generate a side by side plot of the input node values, output node values, and expected output node values.
    Each figure is of a single train/text batch.
    Each panel is of a single node.

    Args:
        headers: list of train/test data type names
        data:
        axs
        color: the color to use for the plots
        maker: the marker to use for the plots
        label: the label to use for the plots
    """
    n_cols = len(headers)
    n_rows = 20; # FIXME
    data_downsampled = data.iloc[::2] # down sample by 1/2
    for n_col in range(0,n_cols):
        for n_row in range(0,n_rows):
            if n_row==0:
                axs[n_row, n_col].scatter(data_downsampled.loc[:, index_name],  data_downsampled.loc[:, headers[n_col]],
                           alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=label)
                #axs[n_row, n_row].title.set_text("Train")
            else:
                axs[n_row, n_col].scatter(data_downsampled.loc[:, index_name],  data_downsampled.loc[:, headers[n_col]],
                           alpha=0.5, c=color, marker=marker, edgecolors='none', s=20)

def makeColumnHeaders(n_batch, n_node, memory_size): 
    """Make the expected headers for the input, output, and expected data files.

    Args:
        headers: list of train/test metric names
        agg_func: name of an aggregation function
        index_name: name of the index header
        data: the actual data frame of values to aggregate according to the specified statistics

    Returns:
        pandas data frame
    """   
    input_headers = ["Input_{:012d}_Input_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(memory_size)]
    output_headers = ["Output_{:012d}_Output_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(memory_size)]
    expected_headers = ["Output_{:012d}_Expected_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(memory_size)]

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
                agg_func_value = data.loc[:,header].abs().max() # just use the magnitude and not the direction
            elif agg_func == "min":
                agg_func_value = data.loc[:,header].abs().min() # just use the magnitude and not the direction
            row_data.update({header:agg_func_value})

            # calculate # of iterations using the agg stat
            indices = []
            if agg_func == "max":
                indices = data[data[header].abs()>=agg_func_value*0.98].index.tolist()
            elif agg_func == "min":
                indices = data[data[header].abs()<=agg_func_value*1.02].index.tolist()
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
    return data_used

def readDataNodes(filename):
    """Creates a pandas data frame with headers for 'nodes','labels'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'nodes','labels'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]

    return data_used

def main(data_dir, data_filename, nodes_filename, index_name, n_epoch, n_batch, n_nodes, memory_size):
    """Run main script"""

    # read in the input files
    filenames = pd.read_csv(data_filename)
    filenames = filenames[filenames["used_"]==True] # filter on used
    nodes_to_labels = pd.read_csv(nodes_filename)
    nodes_to_labels = nodes_to_labels[nodes_to_labels["used_"]==True] # filter on used

    # make the initial figure
    if display_plot:
        fig, axs = plt.subplots(n_nodes, 3, sharex=True, sharey=False)

    # read in the data and anlayze each train/test metric
    all_colors = ["b","r","g","m","c","y","k"]
    all_markers = [".","+",",","o","x","v"]
    marker_iter = 0
    color_iter = 0
    for n in range(0, n_nodes):
        # make the expected column headers
        input_headers, output_headers, expected_headers = makeColumnHeaders(n_batch, n, memory_size) 

        # read in and trim the data
        input_data = pd.read_csv(filenames['input_filenames'], usecols = input_headers, dtype=np.float32)
        input_data = input_data[input_data[index_name]==n_epoch] # filter on epoch
        output_data = pd.read_csv(filenames['output_filenames'], usecols = output_headers, dtype=np.float32)
        output_data = output_data[output_data[index_name]==n_epoch] # filter on epoch
        expected_data = pd.read_csv(filenames['expected_filenames'], usecols = expected_headers, dtype=np.float32)
        expected_data = expected_data[expected_data[index_name]==n_epoch] # filter on epoch

        # plot each node time-course
        plotTrainTest(index_name, headers, data, 
                        axs, all_colors[color_iter], all_markers[marker_iter], labels[n])
        color_iter += 1
        if color_iter >= len(all_colors):
            color_iter = 0;
            marker_iter += 1
            if marker_iter >= len(all_markers):
                marker_iter = 0
        
    # make the legend
    axs[0,0].legend(loc="upper left", markerscale=2)

    # show the image
    plt.show()

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = ""
    data_filename = data_dir + "TimeCourseFilenames.csv"
    nodes_filename = data_dir + "TimeCourseNodes.csv"

    # Input parameters
    index_name = "Epoch"
    n_epoch = 0
    n_batch = 0
    n_nodes = 23
    memory_size = 128
    main(data_dir, data_filename, nodes_filename, index_name, n_epoch, n_batch, n_nodes, memory_size)