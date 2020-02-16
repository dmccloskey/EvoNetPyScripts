from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from subprocess import Popen

def readCommandsCsv(filename):
    """Creates a pandas data frame with headers for 'commands'
    from a .csv file and returns each of the columns as seperate entities.  The column 'used_' is
    used to select what rows to use

    Args:
        filename: name of the file

    Returns:
        pandas data frame with headers for 'filenames','labels','colors','markers'
    """
    data = pd.read_csv(filename)
    data_used = data[data["used_"]==True]
    commands = list(data_used.loc[:, "commands"]) 
    return commands

def main(data_dir, filename):
    """Run main script"""

    # read in the input files
    commands = readCommandsCsv(filename)

    # make the empty data frame for the aggregate statistics
    DETACHED_PROCESS = 0x00000008
    procs = [ Popen(i, creationflags=DETACHED_PROCESS, shell=True) for i in commands ]
    for p in procs:
        p.wait()

# Run main
if __name__ == "__main__":
    # Input files
    data_dir = "C:/Users/dmccloskey/Documents/MetabolomicsNormalization/"
    filename = data_dir + "CommandsToRun.csv"
    main(data_dir, filename)