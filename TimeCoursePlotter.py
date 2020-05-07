from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotTimeCourse(n_node, n_row, memory_size, subplot_titles, input_data, output_data, expected_data, nodes_to_labels, axs, color, marker, series):
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
    x_data = np.arange(memory_size)
    label = nodes_to_labels[nodes_to_labels["nodes"]==n_node]["labels"][n_node]

    # Make the subplots
    try:
        axs[n_row, 0].scatter(x_data, input_data.iloc[0][1:],
                    alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=series)
        axs[n_row, 1].scatter(x_data, output_data.iloc[0][1:],
                    alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=series)
        axs[n_row, 2].scatter(x_data, expected_data.iloc[0][1:],
                    alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=series)

        # Make the titles 
        axs[n_row, 0].title.set_text("{} {}".format(label, subplot_titles[0]))
        axs[n_row, 1].title.set_text("{} {}".format(label, subplot_titles[1]))
        axs[n_row, 2].title.set_text("{} {}".format(label, subplot_titles[2]))

        # make the legend
        axs[n_row,0].legend(loc="upper left", markerscale=2)

    except:
        axs[0].scatter(x_data, input_data.iloc[0][1:],
                    alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=series)
        axs[1].scatter(x_data, output_data.iloc[0][1:],
                    alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=series)
        axs[2].scatter(x_data, expected_data.iloc[0][1:],
                    alpha=0.5, c=color, marker=marker, edgecolors='none', s=20, label=series)

        # Make the titles 
        axs[0].title.set_text("{} {}".format(label, subplot_titles[0]))
        axs[1].title.set_text("{} {}".format(label, subplot_titles[1]))
        axs[2].title.set_text("{} {}".format(label, subplot_titles[2]))

        # make the legend
        axs[0].legend(loc="upper left", markerscale=2)

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
    input_headers = ["Input_{:012d}_Input_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(memory_size)]
    output_headers = ["Output_{:012d}_Output_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(memory_size)]
    expected_headers = ["Output_{:012d}_Expected_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(memory_size)]
    input_headers.append(index_name)
    output_headers.append(index_name)
    expected_headers.append(index_name)
    return input_headers, output_headers, expected_headers

def main(data_dir, data_filename, nodes_filename, index_name, n_epoch, n_batch, nodes, memory_size):
    """Run main script"""

    # read in the input files
    filenames = pd.read_csv(data_filename)
    filenames = filenames[filenames["used_"]==True] # filter on used
    nodes_to_labels = pd.read_csv(nodes_filename)
    nodes_to_labels = nodes_to_labels[nodes_to_labels["used_"]==True] # filter on used

    # make the initial figure
    n_nodes = len(nodes)
    fig, axs = plt.subplots(n_nodes, 3, sharex=True, sharey=True)
    print("Preparing the plot...")

    # read in the data and anlayze each train/test metric
    all_colors = ["b","r","g","m","c","y","k"]
    all_markers = [".","+",",","o","x","v"]
    subplot_titles = ["Input", "Output", "Expected"]
    marker_iter = 0
    color_iter = 0
    for n_row, n in enumerate(nodes):
        print("adding node {}...".format(n))
        # make the expected column headers
        input_headers, output_headers, expected_headers = makeColumnHeaders(n_batch, n, memory_size, index_name)

        for series in range(0,len(filenames)):
            # read in and trim the data
            input_data = pd.read_csv(filenames['input_filenames'][series], usecols = input_headers, dtype=np.float32)
            input_data = input_data[input_data[index_name]==n_epoch] # filter on epoch
            output_data = pd.read_csv(filenames["output_filenames"][series], usecols = output_headers, dtype=np.float32)
            output_data = output_data[output_data[index_name]==n_epoch] # filter on epoch
            expected_data = pd.read_csv(filenames["expected_filenames"][series], usecols = expected_headers, dtype=np.float32)
            expected_data = expected_data[expected_data[index_name]==n_epoch] # filter on epoch

            # plot each node time-course
            plotTimeCourse(n, n_row, memory_size, subplot_titles, input_data, output_data, expected_data, nodes_to_labels,
                            axs, all_colors[color_iter], all_markers[marker_iter], filenames['series'][series])
            color_iter += 1
            if color_iter >= len(all_colors):
                color_iter = 0;
                marker_iter += 1
                if marker_iter >= len(all_markers):
                    marker_iter = 0

    # show the image
    print("Showing the plot...")
    plt.show()

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/HarmonicOscillator/Gpu0-0a/"
    data_filename = data_dir + "TimeCourseFilenames.csv"
    nodes_filename = data_dir + "TimeCourseNodes.csv"

    # Input parameters
    index_name = "Epoch"
    n_epoch = 2000 # 6000 (kinetic)
    n_batch = 0
    #nodes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    nodes = [0] # [11, 19, 20, 22] (kinetic)
    memory_size = 64 # 128 (kinetic)
    main(data_dir, data_filename, nodes_filename, index_name, n_epoch, n_batch, nodes, memory_size)