from matplotlib import pyplot as plt
import numpy as np
import csv
import sys
                   
def read_csv(filename, delimiter=','):
    """read table data from csv file"""
    data = []
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            try:
                keys = reader.fieldnames
                for row in reader:
                    data.append(row)
            except csv.Error as e:
                sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))
    except IOError as e:
        sys.exit('%s does not exist' % e) 
    return data

def plotExpectedPredicted(expected, predicted, label, fig, axes):
    """Generate a side by side plot of expected and predicted

    Args:
        expected (np.array): expected pixels
        predicted (np.array): predicted pixels
        label (string): label of the image
        fig
        axes
    """
    expected_2d = (np.reshape(expected, (28, 28)) * 255)#.astype(np.uint8)

    ax1 = axes[0]    
    ax1.imshow(expected_2d, interpolation='nearest', cmap='gray')
    ax1.set_title('Digit Label: {}'.format(label))
    ax1.set_xbound([0,28])

    predicted_2d = (np.reshape(predicted, (28, 28)) * 255)#.astype(np.uint8)

    ax2 = axes[1]    
    ax2.imshow(predicted_2d, interpolation='nearest', cmap='gray')
    ax2.set_title('Digit Label: {}'.format(label))
    ax2.set_xbound([0,28])

def main(filename, expected_headers, predicted_headers):
    """Run main script"""

    # read in the data
    data = read_csv(filename)

    fig, axes = plt.subplots(1,2, 
        figsize=(10,10),
        sharex=True, sharey=True,
        subplot_kw=dict(adjustable='box-forced', aspect='equal')) #https://stackoverflow.com/q/44703433/1870832
    
    # parse each data row
    for row in data[200:204]:
        expected = np.array([row[h] for h in expected_headers]).astype(np.float)
        predicted = np.array([row[h] for h in predicted_headers]).astype(np.float)

        plotExpectedPredicted(expected, predicted, "", fig, axes)

        # show the image
        plt.tight_layout()
        plt.show()

# Run main
filename = "C:/Users/domccl/Desktop/MetabolomicsExample3/AAELatentZ_ExpectedPredictedPerEpoch.csv"
expected_headers = ["DE-Output_{}_Expected_Batch-0_Memory-0".format(i) for i in range(784)]
predicted_headers = ["DE-Output_{}_Predicted_Batch-0_Memory-0".format(i) for i in range(784)]
main(filename, expected_headers, predicted_headers)
print("")