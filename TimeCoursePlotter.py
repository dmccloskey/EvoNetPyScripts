from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotTimeCourse(n_node, memory_size, subplot_titles, input_data, output_data, expected_data, nodes_to_labels, axs, color, marker):
    """Generate a side by side plot of the input node values, output node values, and expected output node values.
    Each figure is of a single train/text batch.
    Each panel is of a single node.

    Args:
        n_node
        subplot_titles
        input_data
        output_data
        expected_data
        nodes_to_labels
        axs
        color: the color to use for the plots
        maker: the marker to use for the plots
    """
    n_cols = len(subplot_titles)
    x_data = np.arrange(memory_size)
    label = nodes_to_labels[nodes_to_labels["nodes"]==n_node]["labels"].to_series()[0]

    # Make the subplots
    axs[n_node, 0].scatter(x_data, input_data.iloc[0],
                alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=label)
    axs[n_node, 1].scatter(x_data, output_data.iloc[0],
                alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=label)
    axs[n_node, 2].scatter(x_data, expected_data.iloc[0],
                alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=label)
    if n_node==0:
        axs[n_node, 0].title.set_text(subplot_titles[0])
        axs[n_node, 1].title.set_text(subplot_titles[1])
        axs[n_node, 2].title.set_text(subplot_titles[2])

def makeColumnHeaders(n_batch, n_node, memory_size, index_name): 
    """Make the expected headers for the input, output, and expected data files.

    Args:
        headers: list of train/test metric names
        agg_func: name of an aggregation function
        index_name: name of the index header
        data: the actual data frame of values to aggregate according to the specified statistics

    Returns:
        pandas data frame
    """   
    input_headers = ["Input_{:012d}_Input_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(0, memory_size)]
    output_headers = ["Output_{:012d}_Output_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(0, memory_size)]
    expected_headers = ["Output_{:012d}_Expected_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(0, memory_size)]
    input_headers.append(index_name)
    output_headers.append(index_name)
    expected_headers.append(index_name)
    return input_headers, output_headers, expected_headers

def main(data_dir, data_filename, nodes_filename, index_name, n_epoch, n_batch, n_nodes, memory_size):
    """Run main script"""

    # read in the input files
    filenames = pd.read_csv(data_filename)
    filenames = filenames[filenames["used_"]==True] # filter on used
    nodes_to_labels = pd.read_csv(nodes_filename)
    nodes_to_labels = nodes_to_labels[nodes_to_labels["used_"]==True] # filter on used

    # make the initial figure
    fig, axs = plt.subplots(n_nodes, 3, sharex=True, sharey=False)

    # read in the data and anlayze each train/test metric
    all_colors = ["b","r","g","m","c","y","k"]
    all_markers = [".","+",",","o","x","v"]
    subplot_titles = ["Input", "Output", "Expected"]
    marker_iter = 0
    color_iter = 0
    for n in range(0, n_nodes):
        # make the expected column headers
        input_headers, output_headers, expected_headers = makeColumnHeaders(index_name, n_batch, n, memory_size) 

        for series in range(0,len(filenames)):
            # read in and trim the data
            input_data = pd.read_csv(filenames['input_filenames'][series], usecols = input_headers, dtype=np.float32)
            print(len(input_data))
            input_data = input_data[input_data[index_name]==n_epoch] # filter on epoch
            output_data = pd.read_csv(filenames["output_filenames"][series], usecols = output_headers, dtype=np.float32)
            output_data = output_data[output_data[index_name]==n_epoch] # filter on epoch
            expected_data = pd.read_csv(filenames["expected_filenames"][series], usecols = expected_headers, dtype=np.float32)
            expected_data = expected_data[expected_data[index_name]==n_epoch] # filter on epoch

            # plot each node time-course
            plotTimeCourse(n, memory_size, subplot_titles, input_data, output_data, expected_data, nodes_to_labels,
                            axs, all_colors[color_iter], all_markers[marker_iter])
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