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
    n_batches = len(input)
    for batch in range(0, n_batches):
        input_2d = np.clip(np.reshape(input[batch], (28, 28)), 0, 1)#.astype(np.uint8)
        expected_2d = np.clip(np.reshape(expected[batch], (28, 28)), 0, 1)#.astype(np.uint8)
        output_2d = np.clip(np.reshape(output[batch], (28, 28)), 0, 1)#.astype(np.uint8)
  
        axes[batch, 0].imshow(input_2d, interpolation='nearest', cmap='gray')
        axes[batch, 0].set_title('Digit Label: {}'.format(label))
        axes[batch, 0].set_xbound([0,28])
   
        axes[batch,1] .imshow(expected_2d, interpolation='nearest', cmap='gray')
        axes[batch,1] .set_title('Digit Label: {}'.format(label))
        axes[batch,1] .set_xbound([0,28])
   
        axes[batch,2] .imshow(output_2d, interpolation='nearest', cmap='gray')
        axes[batch,2] .set_title('Digit Label: {}'.format(label))
        axes[batch,2] .set_xbound([0,28])

def main(filename_input, filename_output, filename_expected, input_headers, output_headers, expected_headers):
    """Run main script"""

    # read in the data
    input_data = read_csv(filename_input)
    output_data = read_csv(filename_output)
    expected_data = read_csv(filename_expected)
    assert(len(input_data) == len(output_data) == len(expected_data))
    assert(len(input_headers[0]) == len(output_headers[0]) == len(expected_headers[0]))
    n_batches = len(input_headers)
    n_data = len(input_data)
    
    # parse each data row
    for n in range(50, n_data):
        input = []
        expected = []
        output = []
        for batch in range(0, n_batches):
            input.append(np.array([input_data[n][h] for h in input_headers[batch]]).astype(np.float))
            expected.append(np.array([expected_data[n][h] for h in expected_headers[batch]]).astype(np.float))
            output.append(np.array([output_data[n][h] for h in output_headers[batch]]).astype(np.float))

        fig, axes = plt.subplots(n_batches,3, 
            figsize=(10,10),
            sharex=True, sharey=True,
            subplot_kw=dict(adjustable='box', aspect='equal')) #https://stackoverflow.com/q/44703433/1870832

        plotExpectedPredicted(input, output, expected, "", fig, axes)

        # show the image
        plt.tight_layout()
        plt.show()

# Run main# Run main
if __name__ == "__main__":
    #filename_input = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu0-2c/CVAE_NodeInputsPerEpoch.csv"
    #filename_expected = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu0-2c/CVAE_ExpectedPerEpoch.csv"
    #filename_output = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu0-2c/CVAE_NodeOutputsPerEpoch.csv"
    # filename_input = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu2-0a/CVAE_NodeInputsPerEpoch.csv"
    # filename_expected = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu2-0a/CVAE_ExpectedPerEpoch.csv"
    # filename_output = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu2-0a/CVAE_NodeOutputsPerEpoch.csv"
    filename_input = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu3-0a/CVAE_NodeInputsPerEpoch.csv"
    filename_expected = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu3-0a/CVAE_ExpectedPerEpoch.csv"
    filename_output = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu3-0a/CVAE_NodeOutputsPerEpoch.csv"
    filename_input = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu6-1a/VAE_65000_NodeInputsPerEpoch.csv"
    filename_expected = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu6-1a/VAE_65000_ExpectedPerEpoch.csv"
    filename_output = "C:/Users/dmccloskey/Documents/GitHub/EvoNetData/MNIST_examples/CVAE/Gpu6-1a/VAE_65000_NodeOutputsPerEpoch.csv"
    input_headers = []
    output_headers = []
    expected_headers = []
    for batch in range(0, 4):
        input_headers.append(["Input_{:012d}_Input_Batch-{}_Memory-0".format(i, batch) for i in range(784)])
        output_headers.append(["Output_{:012d}_Output_Batch-{}_Memory-0".format(i, batch) for i in range(784)])
        expected_headers.append(["Output_{:012d}_Expected_Batch-{}_Memory-0".format(i, batch) for i in range(784)])
    main(filename_input, filename_output, filename_expected, input_headers, output_headers, expected_headers)