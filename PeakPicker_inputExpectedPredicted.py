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

def plotExpectedPredicted(input, output, expected, label, fig):
    """Generate a side by side plot of expected and predicted

    Args:
        input (np.array): input pixels
        output (np.array): output pixels
        expected (np.array): expected pixels
        label (string): label of the image
        fig
    """
    N = 512
    g1 = (np.linspace(0, N, N), np.array(input))
    g2 = (np.linspace(0, N, N), np.array(expected))
    g3 = (np.linspace(0, N, N), np.array(output))

    data = (g1, g2, g3)
    colors = ("red", "green", "blue")
    groups = ("input", "expected", "output")

    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

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

        fig = plt.figure()
        plotExpectedPredicted(input, output, expected, "", fig)

        # show the image
        plt.legend(loc=2)
        plt.show()

# Run main# Run main
if __name__ == "__main__":
	filename_input = "C:/Users/dmccloskey/Documents/MNIST_examples/PeakIntegrator/Gpu1-0a/DenoisingAE_NodeInputsPerEpoch.csv"
	filename_expected = "C:/Users/dmccloskey/Documents/MNIST_examples/PeakIntegrator/Gpu1-0a/DenoisingAE_ExpectedPerEpoch.csv"
	filename_output = "C:/Users/dmccloskey/Documents/MNIST_examples/PeakIntegrator/Gpu1-0a/DenoisingAE_NodeOutputsPerEpoch.csv"
	input_headers = ["Intensity_{:012d}_Input_Batch-0_Memory-0".format(i) for i in range(512)]
	output_headers = ["Intensity_Out_{:012d}_Output_Batch-0_Memory-0".format(i) for i in range(512)]
	expected_headers = ["Intensity_Out_{:012d}_Expected_Batch-0_Memory-0".format(i) for i in range(512)]
	main(filename_input, filename_output, filename_expected, input_headers, output_headers, expected_headers)