from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotTrainTest(index_name, train_test_headers, train_test_data, axs, color, marker1, marker2, label1, label2):
    """Generate a side by side plot of expected and predicted

    Args:
        train_test_headers: list of train/test metric names
        train_test_data: data frame where each frow is an iteration and each column is a train/test metric
        fig
    """
    n_metric_pairs = len(train_test_headers)
    for n in range(0,n_metric_pairs):
        if n==0:
            axs[n].scatter(train_test_data.loc[:, index_name],  train_test_data.loc[:, train_test_headers[n][0]],
                       alpha=0.5, c=color, marker=marker1, edgecolors='none', s=20, label=label1)
            axs[n].scatter(train_test_data.loc[:, index_name],  train_test_data.loc[:, train_test_headers[n][1]],
                       alpha=0.5, c=color, marker=marker2, edgecolors='none', s=20, label=label2)
        else:
            axs[n].scatter(train_test_data.loc[:, index_name],  train_test_data.loc[:, train_test_headers[n][0]],
                       alpha=0.5, c=color, marker=marker1, edgecolors='none', s=20)
            axs[n].scatter(train_test_data.loc[:, index_name],  train_test_data.loc[:, train_test_headers[n][1]],
                       alpha=0.5, c=color, marker=marker2, edgecolors='none', s=20)

def main(train_test_headers, train_test_filenames, train_test_labels, index_name, n_rows):
    """Run main script"""

    # read in the data and plot each train/test metric
    n_metric_pairs = len(train_test_headers)
    n_data = len(train_test_filenames)
    fig, axs = plt.subplots(n_metric_pairs, 1, sharex=True, sharey=False)
    all_colors = ["b","g","r","c","m","y","k"]
    all_markers = [".","*","+","o","x","v"]
    marker_iter = 0
    color_iter = 0
    for n in range(0, n_data):
        data = pd.read_csv(train_test_filenames[n])

        # plot each train/test metric
        train_label = train_test_labels[n] + "_train"
        test_label = train_test_labels[n] + "_test"
        plotTrainTest(index_name, train_test_headers, data.iloc[:n_rows,:], 
                      axs, all_colors[color_iter], all_markers[marker_iter], all_markers[marker_iter+1], train_label, test_label)
        color_iter += 1
        if color_iter > len(all_colors):
            color_iter = 0;
            marker_iter += 2

        # calculate the aggregate statistics

    # store the aggregate statistics

    # make the legend
    axs[0].legend(loc="upper left", markerscale=2)

    # show the image
    plt.show()

# Run main# Run main
if __name__ == "__main__":
    # Input files
	train_test_filenames = ["C:/Users/dmccloskey/Documents/MetabolomicsNormalization/ALEsKOs01/ConcsBN/FC-0_Sampled_OffProj/Classifier_TrainValMetricsPerEpoch.csv",
                  "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/ALEsKOs01/ConcsBN/FC-0_Sampled_OnProj/Classifier_TrainValMetricsPerEpoch.csv",
                  "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/ALEsKOs01/ConcsBN/FC-2_Sampled_OffProj/Classifier_TrainValMetricsPerEpoch.csv",
                  "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/ALEsKOs01/ConcsBN/FC-2_Sampled_OnProj/Classifier_TrainValMetricsPerEpoch.csv",
                  "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/ALEsKOs01/ConcsBN/FC-8_Sampled_OffProj/Classifier_TrainValMetricsPerEpoch.csv",
                  "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/ALEsKOs01/ConcsBN/FC-8_Sampled_OnProj/Classifier_TrainValMetricsPerEpoch.csv"]
	train_test_labels= ["FC-0_Sampled_OffProj",
                  "FC-0_Sampled_OnProj",
                  "FC-2_Sampled_OffProj",
                  "FC-2_Sampled_OnProj",
                  "FC-8_Sampled_OffProj",
                  "FC-8_Sampled_OnProj"]

    # Input headers
	loss_headers = ["Training_Train_Error","Validation_Test_Error"]
	accuracy_headers = ["Training_AccuracyMCMicro","Validation_AccuracyMCMicro"]
	precision_headers = ["Training_PrecisionMCMicro","Validation_PrecisionMCMicro"]
	train_test_headers = [loss_headers, accuracy_headers, precision_headers]

    # Name of the index
	index_name = "Epoch"; n_rows = 90000
	main(train_test_headers, train_test_filenames, train_test_labels, index_name, n_rows)