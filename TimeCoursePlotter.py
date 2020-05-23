from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotTimeCourse(n_node, n_row, memory_size, sequence_length, subplot_titles, input_data, output_data, expected_data, label, axs, color, marker, series):
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
    x_data = np.arange(sequence_length-1, sequence_length - memory_size, -1)

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

def makeColumnHeaders(n_batch, n_node, memory_size, sequence_length, index_name): 
    """Make the expected headers for the input, output, and expected data files.
    The sequence is generated downwards to match the memory steps.

    Args:
        headers: list of train/test metric names
        agg_func: name of an aggregation function
        index_name: name of the index header
        data: the actual data frame of values to aggregate according to the specified statistics

    Returns:
        pandas data frame
    """   
    input_headers = ["Input_{:012d}_Input_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(sequence_length-1, sequence_length - memory_size, -1)]
    output_headers = ["Output_{:012d}_Output_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(sequence_length-1, sequence_length - memory_size, -1)]
    expected_headers = ["Output_{:012d}_Expected_Batch-{}_Memory-{}".format(n_node, n_batch, j) for j in range(sequence_length-1, sequence_length - memory_size, -1)]
    input_headers.append(index_name)
    output_headers.append(index_name)
    expected_headers.append(index_name)
    return input_headers, output_headers, expected_headers

def main(data_dir, data_filename, nodes_filename, index_name):
    """Run main script"""

    # read in the input files
    filenames = pd.read_csv(data_filename)
    filenames = filenames[filenames["used_"]==True] # filter on used
    nodes_to_labels = pd.read_csv(nodes_filename)
    nodes_to_labels = nodes_to_labels[nodes_to_labels["used_"]==True] # filter on used

    # extract the memory size
    memory_size = np.min(filenames["memory_sizes"].to_numpy())
    sequence_length = np.min(filenames["sequence_length"].to_numpy())

    # make the initial figure
    n_nodes = len(nodes_to_labels)
    fig, axs = plt.subplots(n_nodes, 3, sharex=True, sharey=False)
    print("Preparing the plot...")

    # read in the data and anlayze each train/test metric
    all_colors = ["b","r","g","m","c","y","k"]
    all_markers = [".","+",",","o","x","v"]
    subplot_titles = ["Input", "Output", "Expected"]
    marker_iter = 0
    color_iter = 0
    for n_row in range(len(nodes_to_labels)):
        print("adding node {}...".format(nodes_to_labels.iloc[n_row].loc["nodes"]))
        for index, row in filenames.iterrows():
            # make the expected column headers
            input_headers, output_headers, expected_headers = makeColumnHeaders(row['n_batch'], nodes_to_labels.iloc[n_row].loc["nodes"], memory_size, sequence_length, index_name)

            # read in and trim the data
            input_data = pd.read_csv(row['input_filenames'], usecols = input_headers, dtype=np.float32)
            input_data = input_data[input_data[index_name]==row['n_epoch']] # filter on epoch
            output_data = pd.read_csv(row["output_filenames"], usecols = output_headers, dtype=np.float32)
            output_data = output_data[output_data[index_name]==row['n_epoch']] # filter on epoch
            expected_data = pd.read_csv(row["expected_filenames"], usecols = expected_headers, dtype=np.float32)
            expected_data = expected_data[expected_data[index_name]==row['n_epoch']] # filter on epoch

            # plot each node time-course
            plotTimeCourse(nodes_to_labels.iloc[n_row].loc["nodes"], n_row, memory_size, sequence_length, subplot_titles, input_data, output_data, expected_data, nodes_to_labels.iloc[n_row].loc["labels"],
                            axs, all_colors[color_iter], all_markers[marker_iter], row['series'])
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
    data_dir = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/KineticModel/"
    data_filename = data_dir + "TimeCourseFilenames.csv"
    nodes_filename = data_dir + "TimeCourseNodes.csv"

    # Input parameters
    index_name = "Epoch"
    main(data_dir, data_filename, nodes_filename, index_name)