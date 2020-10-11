from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def aggregateCategoryLabel(headers, traversals, index_name, data, label, agg_stats):
    """Creates a pandas data frame with headers according to headers
    and row labels according to labels where the column data from the
    specified header is aggregated according to the specified aggregation function.
    In addition, the number of iterations required to reach the aggregation function
    100 times is also reported.

    Args:
        headers: dataframe of train/test metrics to use to identify the min or max point
        categories: dataframe of mapping from index to the category, label, and input
        index_name: name of the index header
        data: the actual data frame of values to aggregate according to the specified statistics

    Returns:
        dataframe
    """

    unique_input = list(set(traversals['input']))
    unique_input.sort()
    for index, row in headers.iterrows():
        for input in unique_input:
            traversals_input = traversals[traversals['input']==input]
            data_subset = data[data[index_name].isin(traversals_input['index'])]
            if row['min_or_max'] == 'min':
                data_subset = data_subset[data_subset[row['headers']] == data_subset[row['headers']].min()]
            else:
                data_subset = data_subset[data_subset[row['headers']] == data_subset[row['headers']].max()]
            traversals_filtered = traversals_input[traversals_input['index'] == data_subset[index_name].iloc[0]]
            traversals_filtered.insert(0, "metric_value", data_subset[row['headers']])
            traversals_filtered.insert(0, "metric_name", row['headers'])
            traversals_filtered.insert(0, "label", label) 
            agg_stats = agg_stats.append(traversals_filtered, ignore_index=True)
    return agg_stats

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
    return filenames, labels

def readDataHeaders(filename):
    """Creates a pandas data frame with headers for 'headers','min_or_max'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'headers','min_or_max'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    return data_used

def readLatentTraversalSteps(filename):
    """Creates a pandas data frame with headers for 'step','gaussian_node','categorical_node','input','index'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'step','gaussian_node','categorical_node','input','index'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    return data_used

def main(data_dir, data_filename, headers_filename, traversals_filename, index_name, n_rows):
    """Run main script"""

    # read in the input files
    filenames, labels = readDataFilenamesCsv(data_filename)
    headers = readDataHeaders(headers_filename)
    traversals = readLatentTraversalSteps(traversals_filename)

    # make the empty data frame for the aggregate statistics
    flat_headers = list(headers.loc[:, "headers"])
    flat_headers.insert(0, index_name)
    columns_custom = ['label', 'step','gaussian_node','categorical_node','input','index','metric_value','metric_name']
    agg_stats = pd.DataFrame(columns=columns_custom)

    # make the initial figure
    n_data = len(filenames)

    # read in the data and anlayze each train/test metric
    for n in range(0, n_data):
        print("processing data {}...".format(n))
        # trim the data
        data = pd.read_csv(filenames[n], usecols=flat_headers, dtype=np.float32).iloc[:n_rows,:]

        # calculate the aggregate statistics
        print("Aggregating statistics for {}...".format(n))
        agg_stats = aggregateCategoryLabel(headers, traversals, index_name, data, labels[n], agg_stats)

    # store the aggregate statistics
    print("Storing aggregate statistics...")
    agg_stats.to_csv(data_dir + "LatentTraversal.csv")

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = ""
    data_filename = data_dir + "LatentTraversalInput.csv"
    headers_filename = data_dir + "LatentTraversalHeaders.csv"
    traversals_filename = data_dir + "LatentTraversalSteps.csv"

    # Name of the index
    index_name = "Epoch"; n_rows = 100000;
    main(data_dir, data_filename, headers_filename, traversals_filename, index_name, n_rows)