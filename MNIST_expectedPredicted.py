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

def plotExpectedPredicted(input, output, expected, label, fig, axes):
    """Generate a side by side plot of expected and predicted

    Args:
        input (np.array): input pixels
        output (np.array): output pixels
        expected (np.array): expected pixels
        label (string): label of the image
        fig
        axes
    """
    input_2d = (np.reshape(input, (28, 28)) * 255)#.astype(np.uint8)
    expected_2d = (np.reshape(expected, (28, 28)) * 255)#.astype(np.uint8)
    output_2d = (np.reshape(output, (28, 28)) * 255)#.astype(np.uint8)

    ax0 = axes[0]    
    ax0.imshow(input_2d, interpolation='nearest', cmap='gray')
    ax0.set_title('Digit Label: {}'.format(label))
    ax0.set_xbound([0,28])

    ax1 = axes[1]    
    ax1.imshow(expected_2d, interpolation='nearest', cmap='gray')
    ax1.set_title('Digit Label: {}'.format(label))
    ax1.set_xbound([0,28])

    ax2 = axes[2]    
    ax2.imshow(output_2d, interpolation='nearest', cmap='gray')
    ax2.set_title('Digit Label: {}'.format(label))
    ax2.set_xbound([0,28])

def main(filename_input, filename_output, filename_expected, input_headers, output_headers, expected_headers):
    """Run main script"""

    # read in the data
    input_data = read_csv(filename_input)
    output_data = read_csv(filename_output)
    expected_data = read_csv(filename_expected)

    assert(len(input_data) == len(output_data) == len(expected_data))
    n_data = len(input_data)
    
    # parse each data row
    for n in range(0, n_data):
        input = np.array([input_data[n][h] for h in input_headers]).astype(np.float)
        expected = np.array([expected_data[n][h] for h in expected_headers]).astype(np.float)
        output = np.array([output_data[n][h] for h in output_headers]).astype(np.float)

        fig, axes = plt.subplots(1,3, 
            figsize=(10,10),
            sharex=True, sharey=True,
            subplot_kw=dict(adjustable='box', aspect='equal')) #https://stackoverflow.com/q/44703433/1870832

        plotExpectedPredicted(input, output, expected, "", fig, axes)

        # show the image
        plt.tight_layout()
        plt.show()

# Run main# Run main
if __name__ == "__main__":
	filename_input = "C:/Users/dmccloskey/Documents/MNIST_examples/VAE/Gpu3/VAE_NodeInputsPerEpoch.csv"
	filename_expected = "C:/Users/dmccloskey/Documents/MNIST_examples/VAE/Gpu3/VAE_ExpectedPerEpoch.csv"
	filename_output = "C:/Users/dmccloskey/Documents/MNIST_examples/VAE/Gpu3/VAE_NodeOutputsPerEpoch.csv"
	input_headers = ["Input_{:012d}_Input_Batch-1_Memory-0".format(i) for i in range(784)]
	output_headers = ["Output_{:012d}_Output_Batch-1_Memory-0".format(i) for i in range(784)]
	expected_headers = ["Output_{:012d}_Expected_Batch-1_Memory-0".format(i) for i in range(784)]
	main(filename_input, filename_output, filename_expected, input_headers, output_headers, expected_headers)