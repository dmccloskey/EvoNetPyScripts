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

def plotLatentTraversal(input, output, dimension, fig, axes):
    """Generate a plot of the latent traversals with the batch samples across
        the top axis and the dimensions along the side axis

    Args:
        input (np.array): input latent values
        output (np.array): output pixels
        fig
        axes
    """
    n_batches = len(input)
    for batch in range(0, n_batches):
        #input_2d = np.clip(np.reshape(input[batch], (28, 28)), 0, 1)#.astype(np.uint8)
        output_2d = np.clip(np.reshape(output[batch], (28, 28)), 0, 1)#.astype(np.uint8)
   
        axes[dimension,batch].imshow(output_2d, interpolation='nearest', cmap='gray')
        #axes[dimension,batch].set_title('Dimension {} and batch {}'.format(dimension,batch))
        axes[dimension,batch].set_xbound([0,28])
        axes[dimension,batch].set_axis_off

def main(filename_input, filename_output, input_headers, output_headers):
    """Run main script"""

    # read in the data
    input_data = read_csv(filename_input)
    output_data = read_csv(filename_output)
    assert(len(input_data) == len(output_data))
    n_batches = len(input_headers)
    n_dimensions = len(input_headers[0]);
    n_data = len(input_data)
    
    # parse each data row
    dimension = 0
    fig, axes = plt.subplots(n_dimensions,n_batches, 
        figsize=(10,10),
        sharex=True, sharey=True,
        subplot_kw=dict(adjustable='box', aspect='equal')) #https://stackoverflow.com/q/44703433/1870832
    for n in range(0, n_data):
        input = []
        output = []
        for batch in range(0, n_batches):
            input.append(np.array([input_data[n][h] for h in input_headers[batch]]).astype(np.float))
            output.append(np.array([output_data[n][h] for h in output_headers[batch]]).astype(np.float))

        plotLatentTraversal(input, output, dimension, fig, axes)

        # show the image
        dimension += 1
        if (dimension % n_dimensions == 0 and dimension != 0):
            plt.tight_layout()
            plt.show()
            dimension = 0
            fig, axes = plt.subplots(n_dimensions,n_batches, 
                figsize=(10,10),
                sharex=True, sharey=True,
                subplot_kw=dict(adjustable='box', aspect='equal')) #https://stackoverflow.com/q/44703433/1870832

# Run main# Run main
if __name__ == "__main__":
    filename_input = "C:/Users/dmccloskey/Documents/MNIST_examples/CVAE/Gpu0-2a/CVAEDecoder_NodeInputsPerEpoch.csv"
    filename_output = "C:/Users/dmccloskey/Documents/MNIST_examples/CVAE/Gpu0-2a/CVAEDecoder_NodeOutputsPerEpoch.csv"
    input_headers = []
    output_headers = []
    for batch in range(0, 8):
        input_headers.append(["Gaussian_encoding_{:012d}_Input_Batch-{}_Memory-0".format(i, batch) for i in range(8)])
        #input_headers.append(["Categorical_encoding-SoftMax-Out_{:012d}_Input_Batch-{}_Memory-0".format(i, batch) for i in range(8)])
        output_headers.append(["Output_{:012d}_Output_Batch-{}_Memory-0".format(i, batch) for i in range(784)])
    main(filename_input, filename_output, input_headers, output_headers)