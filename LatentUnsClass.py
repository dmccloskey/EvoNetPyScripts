from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def aggregateCategoryLabel(thresholds, categories, index_name, data, label, n_agg_values = 10):
    """Creates a pandas data frame with headers according to headers
    and row labels according to labels where the column data from the
    specified header is aggregated according to the specified aggregation function.
    In addition, the number of iterations required to reach the aggregation function
    100 times is also reported.

    Args:
        headers: dataframe of train/test metrics to filter the data by based on the thresholds column
        categories: dataframe of mapping from index to the category, label, and input
        index_name: name of the index header
        data: the actual data frame of values to aggregate according to the specified statistics

    Returns:
        dataframe
    """
    row_data = {"label":label}

    # iteratively filter the data
    df_filtered = data
    for index, row in thresholds.iterrows():
        # filter the data
        df_filtered = df_filtered[df_filtered[row['headers']]>row['thresholds']]

    # use the data index to filter the categories
    categories_filtered = categories[categories['index'].isin(df_filtered[index_name])]

    # add in the labels as a new column
    categories_filtered.insert(0, "label", label)
    categories_filtered.drop(columns=['used_', 'index'])
    return categories_filtered

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
    """Creates a pandas data frame with headers for 'headers','thresholds'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'headers','thresholds'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    return data_used

def readCategoryLabel(filename):
    """Creates a pandas data frame with headers for 'category','predicted','input','index'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'category','predicted','input'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    return data_used

def main(data_dir, data_filename, headers_filename, categories_filename, index_name, n_rows):
    """Run main script"""

    # read in the input files
    filenames, labels = readDataFilenamesCsv(data_filename)
    headers = readDataHeaders(headers_filename)
    categories = readCategoryLabel(categories_filename)

    # make the empty data frame for the aggregate statistics
    columns_custom = ["label", "category", "predicted", "input"]
    agg_stats = pd.DataFrame(columns=columns_custom)
    flat_headers = list(headers.loc[:, "headers"])
    flat_headers.insert(0, index_name)

    # make the initial figure
    n_data = len(filenames)

    # read in the data and anlayze each train/test metric
    for n in range(0, n_data):
        print("processing data {}...".format(n))
        # trim the data
        data = pd.read_csv(filenames[n], usecols=flat_headers, dtype=np.float32).iloc[:n_rows,:]

        # calculate the aggregate statistics
        print("Aggregating statistics for {}...".format(n))
        row_data = aggregateCategoryLabel(headers, categories, index_name, data, labels[n])
        agg_stats = agg_stats.append(row_data, ignore_index=True)

    # store the aggregate statistics
    print("Storing aggregate statistics...")
    agg_stats.to_csv(data_dir + "LatentUnsClass.csv")

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = ""
    data_filename = data_dir + "LatentUnsClassInput.csv"
    headers_filename = data_dir + "LatentUnsClassHeaders.csv"
    categories_filename = data_dir + "LatentUnsClassCategories.csv"

    # Name of the index
    index_name = "Epoch"; n_rows = 100000;
    main(data_dir, data_filename, headers_filename, categories_filename, index_name, n_rows)